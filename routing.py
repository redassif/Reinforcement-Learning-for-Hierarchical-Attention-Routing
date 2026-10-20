"""Route each query through a tree of tokens."""
import torch
from torch import nn
import numpy as np
from scipy.optimize import linprog


def make_tree(length, leaf_size=4):
    """Split tokens into smaller groups."""
    if length < 1 or leaf_size < 1:
        raise ValueError("Tree length and leaf size must be positive")
    leaves, nodes, paths = [], [], []

    def build(ids, path):
        node = {'ids': ids, 'children': []}
        if len(ids) <= leaf_size:
            leaves.append(ids)
            paths.append(path)
        else:
            n = len(nodes)
            nodes.append(node)
            mid = len(ids) // 2
            node['children'] = [build(ids[:mid], path + [(n, 0)]),
                                build(ids[mid:], path + [(n, 1)])]
        return node

    tree = build(torch.arange(length), [])
    membership = torch.zeros(len(leaves), length, dtype=torch.bool)
    for j, ids in enumerate(leaves):
        membership[j, ids] = True
    return dict(tree=tree, leaves=leaves, nodes=nodes,
                paths=paths, membership=membership)

def fork_scores(Q,K,e):
    summaries=torch.stack([torch.stack([K[c['ids']].mean(0) for c in n['children']]) for n in e['nodes']])
    return torch.einsum('qd,nad->qna',Q[e['queries']],summaries)


def leaf_policy(Q,K,e):
    """Find the probability of reaching each leaf."""
    if not e["nodes"]:
        return torch.distributions.Categorical(logits=(Q[e["queries"]].sum(-1)*0)[:,None])
    forks=fork_scores(Q,K,e).log_softmax(-1)
    logs=torch.stack([torch.stack([forks[:,n,a] for n,a in path]).sum(0) for path in e['paths']],dim=-1)
    return torch.distributions.Categorical(logits=logs)


@torch.no_grad()
def evaluate(examples,q,k,save_rows=False):
    """Count target hits and measure how much attention is kept."""
    rows=[];accuracies=[]
    hits=total=fixed_hits=next_hits=last_hits=0
    mass_total=err_total=ref_total=0.
    for text_id,e in enumerate(examples):
        Q,K=q(e['X']),k(e['X'])
        decisions=fork_scores(Q,K,e).argmax(-1) if e["nodes"] else None
        # Choose the higher-scoring branch at each fork; ties go left.
        matching=torch.stack([(torch.stack([decisions[:,n]==a for n,a in path]).all(0) if path else torch.ones(len(e["queries"]),dtype=torch.bool)) for path in e['paths']],dim=-1)
        assert torch.all(matching.sum(-1)==1)
        chosen=matching.long().argmax(-1)
        query=e['queries'];target=e['needles'][query]
        mask=e['membership'][chosen]
        correct=mask.gather(1,target[:,None]).squeeze(1)
        attn=e['A'][query]*mask
        mass=attn.sum(-1)
        contexts=(attn/mass[:,None].clamp_min(1e-12))@e['V']
        reference=(e['A']@e['V'])[query]
        err_total+=float((contexts-reference).square().sum());ref_total+=float(reference.square().sum())
        count=len(query);right=int(correct.sum())
        hits+=right;total+=count;mass_total+=float(mass.double().sum())
        accuracies.append(right/count)
        fixed_hits+=int(e['membership'][:,target].sum(-1).max())
        last_hits+=int(e['membership'][-1,target].sum())
        next_ids=e['membership'][:,(query+1).clamp_max(len(Q)-1)].long().argmax(0)
        next_hits+=int(e['membership'][next_ids,target].sum())
        if save_rows:
            for j,i in enumerate(query.tolist()):
                rows.append(dict(text_id=text_id,source_row=e['source_row'],article=e['article'],query=i,
                                 target=int(target[j]),leaf=e['leaves'][int(chosen[j])].tolist(),
                                 hit=int(correct[j]),attention_mass=float(mass[j])))
    return dict(hits=hits,queries=total,accuracy=hits/total,
                mean_text_accuracy=sum(accuracies)/len(accuracies),
                next_token_leaf_accuracy=next_hits/total,last_leaf_accuracy=last_hits/total,
                best_fixed_leaf_accuracy=fixed_hits/total,
                attention_mass=mass_total/total,
                context_relative_error=(err_total/ref_total)**.5),rows


def make_models(seed, input_dim=768, projection_dim=64):
    torch.manual_seed(seed)
    return nn.Linear(input_dim,projection_dim,bias=False),nn.Linear(input_dim,projection_dim,bias=False)


def terminal_discounts(paths, gamma, reference):
    exponents = reference.new_tensor([max(len(path) - 1, 0) for path in paths])
    return reference.new_tensor(gamma).pow(exponents)


def loss_from_samples(policy, sampled, raw_rewards, discounts, entropy):
    """Compare each reward with the other sampled rewards."""
    returns = raw_rewards * discounts[sampled]
    count = sampled.shape[0]
    assert count > 1
    # Use the other samples as the baseline for each reward.
    baseline = (returns.sum(0, keepdim=True) - returns) / (count - 1)
    advantage = (returns - baseline).detach()
    policy_loss = -(policy.log_prob(sampled) * advantage).mean(0).sum()
    return policy_loss - entropy * policy.entropy().sum()


def update(example, query_projection, key_projection, optimizer, *, gamma, samples=32, entropy=.01):
    """Update the router weights using RL."""
    queries = query_projection(example['X'])
    keys = key_projection(example['X'])
    policy = leaf_policy(queries, keys, example)
    sampled = policy.sample((samples,))
    targets = example['needles'][example['queries']]
    rewards = example['membership'][sampled, targets[None, :]].float()
    discounts = terminal_discounts(example['paths'], gamma, rewards)
    loss = loss_from_samples(policy, sampled, rewards, discounts, entropy)

    optimizer.zero_grad()
    loss.backward()
    parameters = list(query_projection.parameters()) + list(key_projection.parameters())
    nn.utils.clip_grad_norm_(parameters, 1.)
    optimizer.step()

def shared_projection_feasibility(inputs, targets, tree, projection_dim=None):
    """Check all forks using the same matrix; the README explains the limits."""
    x=np.asarray(inputs,dtype=np.float64)
    targets=np.asarray(targets,dtype=int)
    n,d=x.shape
    if targets.shape!=(n,) or np.any(targets<0) or np.any(targets>=n):
        raise ValueError('Supply one valid target-token index per query')
    if projection_dim is None:projection_dim=d
    if projection_dim<1:raise ValueError('projection_dim must be positive')
    rows=[]
    for i,target in enumerate(targets):
        node=tree['tree']
        while node['children']:
            left,right=node['children']
            delta=x[right['ids'].tolist()].mean(0)-x[left['ids'].tolist()].mean(0)
            action=int(target in right['ids'].tolist())
            rows.append((2*action-1)*np.outer(x[i],delta).ravel())
            node=node['children'][action]
    if not rows:
        return dict(status='feasible',constraints=0,matrix=np.zeros((d,d)),minimum_margin=None,
                    rank_relaxation=projection_dim<d)
    a=np.stack(rows)
    solution=linprog(np.zeros(d*d),A_ub=-a,b_ub=-np.ones(len(a)),bounds=[(None,None)]*(d*d),method='highs')
    result=dict(status='feasible' if solution.success else 'infeasible' if solution.status==2 else 'unresolved',
                constraints=len(a),rank_relaxation=projection_dim<d,solver_message=solution.message)
    if solution.success:
        margins=a@solution.x
        if margins.min()<1-1e-6:raise RuntimeError('Returned solution failed margin verification')
        result.update(matrix=solution.x.reshape(d,d),minimum_margin=float(margins.min()))
    return result
