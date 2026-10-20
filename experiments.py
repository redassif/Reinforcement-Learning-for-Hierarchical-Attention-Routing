"""Run the synthetic, BERT routing, or sentiment experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import torch
from torch.nn import functional as F
from routing import make_tree, make_models, fork_scores, evaluate, update, shared_projection_feasibility

MODEL = 'google-bert/bert-base-uncased'
MODEL_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
WIKITEXT_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
SST2_REVISION = '8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb'
LAYER, HEAD, LEAF_SIZE = 5, 9, 4
LR, GAMMA, ENTROPY, SAMPLES = 3e-5, .8, .01, 32
INPUTS = torch.tensor([
    [.43,.15,.89], [.55,.87,.66], [.57,.85,.64], [.22,.58,.33],
    [.77,.25,.10], [.05,.80,.55], [.33,.44,.77], [.88,.12,.45],
])

# Query index -> target token (needle).
NEEDLE_MAP = {0:3, 1:3, 2:3, 3:3, 4:5, 5:5, 6:5, 7:5}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    if path.suffix == '.pt': torch.save(value, temp)
    else: temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def load_bert():
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    bert = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, attn_implementation='eager')
    return tokenizer, bert.eval().requires_grad_(False)


def dataset_rows(dataset, revision, filename):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(dataset, filename, repo_type='dataset', revision=revision)
    return pq.read_table(path).to_pylist()


def select_targets(x):
    """Pick two needles, then assign each query by cosine similarity."""
    normalized=F.normalize(x,dim=-1);similarity=normalized@normalized.T
    best=None
    for a in range(len(x)):
        for b in range(a+1,len(x)):
            choose_a=similarity[:,a]>=similarity[:,b]
            separated=((similarity[:,a]-similarity[:,b]).abs()>.05).float().mean().item()
            balance=min(int(choose_a.sum()),int((~choose_a).sum()))/(len(x)/2)
            score=-float(similarity[a,b])+separated+balance
            if best is None or score>best[0]:
                best=(score,(a,b),torch.where(choose_a,a,b))
    return best[1],best[2]


@torch.no_grad()
def greedy_hits(x,targets,tree,q,k):
    Q,K=q(x),k(x);hits=0
    for i in range(len(x)):
        node=tree['tree']
        while node['children']:
            summaries=torch.stack([K[c['ids']].mean(0) for c in node['children']])
            node=node['children'][int((Q[i]@summaries.T).argmax())]
        hits+=int(targets[i] in node['ids'])
    return hits


def synthetic(args):
    tree = make_tree(8, leaf_size=2)
    pair, selected = select_targets(INPUTS)
    maps = {'original': torch.tensor(list(NEEDLE_MAP.values())), 'cosine_selected': selected}
    results = []
    for name, targets in maps.items():
        print(name, 'needle map:', dict(enumerate(targets.tolist())), flush=True)
        check = shared_projection_feasibility(INPUTS.numpy(), targets.numpy(), tree)
        q, k = make_models(args.seed, input_dim=3, projection_dim=3)
        optimizer = torch.optim.Adam(list(q.parameters())+list(k.parameters()), lr=LR)
        example = dict(X=INPUTS, needles=targets, queries=torch.arange(8), **tree)
        for _ in range(args.updates or 3000):
            update(example, q, k, optimizer, gamma=GAMMA, samples=SAMPLES, entropy=ENTROPY)
        row = dict(map=name, targets=targets.tolist(), strict_feasibility=check['status'],
                   hits=greedy_hits(INPUTS, targets, tree, q, k), queries=8)
        results.append(row); print(row, flush=True)
    save(args.out/'synthetic.json', dict(pair=pair, results=results))


@torch.no_grad()
def prepare(args):
    tokenizer, bert = load_bert()
    counts = {'train':16, 'validation':4, 'test':4} if args.quick else {'train':10000, 'validation':100, 'test':200}
    selected = {}; used_tokens = set(); used_articles = set()
    for split in ['test', 'validation', 'train']:
        rows = dataset_rows('Salesforce/wikitext', WIKITEXT_REVISION,
                            f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet')
        candidates = []; article = None
        for row, record in enumerate(rows):
            text = ' '.join(record['text'].split())
            if not text: continue
            if text.startswith('='):
                if re.fullmatch(r'= [^=]+ =', text): article = text[2:-2].strip().casefold()
                continue
            if article is None or article in used_articles or len(text.split())<20: continue
            candidates.append(dict(row=row, article=article, text=text))
        random.Random(20261006+['train','validation','test'].index(split)).shuffle(candidates)
        chosen = []
        for record in candidates:
            tokens = tokenizer(record['text'], truncation=True, max_length=64)
            key = tuple(tokens['input_ids'])
            if key in used_tokens: continue
            used_tokens.add(key)
            chosen.append(dict(**record, **tokens, input_sha256=hashlib.sha256(json.dumps(key).encode()).hexdigest()))
            if len(chosen)==counts[split]: break
        assert len(chosen)==counts[split]
        used_articles.update(r['article'] for r in chosen); selected[split]=chosen
    save(args.data/'passages.json', selected)
    for split, records in selected.items():
        features = []; total = len(records)
        regions = ([(0,min(1000,total))]+[(s,min(s+100,total)) for s in range(1000,total,100)]
                   if split=='train' else [(0,total)])
        for start, end in regions:
            for offset in range(start,end,8):
                part = records[offset:min(offset+8,end)]
                batch = tokenizer.pad([{key:r[key] for key in ['input_ids','attention_mask','token_type_ids']}
                                       for r in part], return_tensors='pt')
                output = bert(**batch, output_hidden_states=True, output_attentions=True)
                hidden = output.hidden_states[LAYER]
                values = bert.encoder.layer[LAYER].attention.self.value(hidden).reshape(*hidden.shape[:2],12,64)[:,:,HEAD]
                for j, record in enumerate(part):
                    n = len(record['input_ids']); a = output.attentions[LAYER][j,HEAD,:n,:n].clone()
                    # Needle map: the highest-attention token for each query.
                    features.append(dict(X=hidden[j,:n].clone(), A=a, V=values[j,:n].clone(), needles=a.argmax(-1),
                        queries=torch.tensor([i for i,t in enumerate(record['input_ids']) if t not in tokenizer.all_special_ids]),
                        tokens=tokenizer.convert_ids_to_tokens(record['input_ids']), input_ids=record['input_ids'],
                        article=record['article'], source_row=record['row'], **make_tree(n)))
                if len(features)%100==0: print(split, len(features), '/', total, flush=True)
        save(args.data/f'{split}.pt', features)


def train(args):
    steps = args.updates or (20 if args.quick else 30000)
    interval = 10 if args.quick else 1000
    if steps%interval: raise ValueError(f'BERT updates must be a multiple of {interval}')
    if args.out.exists(): raise FileExistsError('Choose a new output folder for a fresh run')
    fit = torch.load(args.data/'train.pt', weights_only=True, mmap=True)
    val = torch.load(args.data/'validation.pt', weights_only=True)
    assert (len(fit),len(val))==((16,4) if args.quick else (10000,100))
    save(args.out/'plan.json', dict(quick_test=args.quick, training_passages=len(fit), end_step=steps, lr=LR, discount_factor=GAMMA,
        entropy_weight=ENTROPY, sampled_routes_per_query=SAMPLES, weight_decay=0., seeds=list(range(5)),
        fresh_initialization=True, bert_frozen=True))
    histories = {}
    for seed in range(5):
        q, k = make_models(seed)
        optimizer = torch.optim.Adam(list(q.parameters())+list(k.parameters()), lr=LR, weight_decay=0.)
        history = []; samples = 0
        for step in range(steps+1):
            if step%interval==0:
                row = dict(step=step, validation=evaluate(val,q,k)[0])
                history.append(row)
                save(args.out/f'seed_{seed}/step_{step}.pt', dict(seed=seed, step=step, size=len(fit),
                    W_Q=q.state_dict(), W_K=k.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(), sampled_routes=samples, discount_factor=GAMMA,
                    entropy_weight=ENTROPY, fresh_initialization=True))
                save(args.out/f'seed_{seed}/history.json', history)
                print('seed', seed, 'step', step, 'validation', row['validation']['accuracy'], flush=True)
            if step==steps: break
            example = fit[step%len(fit)]
            update(example,q,k,optimizer,gamma=GAMMA,samples=SAMPLES,entropy=ENTROPY)
            samples += SAMPLES*len(example['queries'])
        histories[seed]=history
    curve = {step:sum(histories[s][step//interval]['validation']['accuracy'] for s in range(5))/5
             for step in range(0,steps+1,interval)}
    selected = max(curve,key=lambda step:(curve[step],-step))
    save(args.out/'selection.json',dict(selected_step=selected,validation_accuracy=curve[selected]))
    # Test data is loaded only after validation selects the checkpoint.
    test = torch.load(args.data/'test.pt',weights_only=True)
    results = []
    for seed in range(5):
        q,k = load_router(args.out,selected,seed)
        results.append(dict(seed=seed,**evaluate(test,q,k)[0]))
    save(args.out/'results.json',dict(quick_test=args.quick,selected_step=selected,seeds=results,
        mean_test_accuracy=sum(r['accuracy'] for r in results)/5))
    print('Mean routing accuracy:',sum(r['accuracy'] for r in results)/5,flush=True)


def load_router(folder,step,seed):
    saved = torch.load(folder/f'seed_{seed}/step_{step}.pt',weights_only=True)
    q,k = make_models(seed);q.load_state_dict(saved['W_Q']);k.load_state_dict(saved['W_K'])
    return q.eval().requires_grad_(False),k.eval().requires_grad_(False)


def choose_mask(hidden,mode,q=None,k=None):
    n=len(hidden);e=dict(queries=torch.arange(n),**make_tree(n))
    if mode=='all':return torch.ones(n,n,dtype=torch.bool)
    if mode=='next':
        leaf=e['membership'][:,(torch.arange(n)+1).clamp_max(n-1)].long().argmax(0)
    elif mode=='rl':
        if not e['nodes']:return torch.ones(n,n,dtype=torch.bool)
        actions=fork_scores(q(hidden),k(hidden),e).argmax(-1)
        matches=torch.stack([torch.stack([actions[:,v]==a for v,a in path]).all(0)
                             for path in e['paths']],dim=-1)
        assert torch.all(matches.sum(-1)==1)
        leaf=matches.long().argmax(-1)
    else:raise ValueError(mode)
    return e['membership'][leaf]

class RoutedHead:
    """Change one attention head and keep the other heads unchanged."""
    def __init__(self,special_ids):
        self.special_ids=set(special_ids);self.mode='full';self.q=self.k=None
        self.check=False;self.checked=0

    def hook(self,module,args,output):
        if self.mode=='full':return output
        hidden=args[0];original,probs=output
        context=original.clone();changed_probs=probs.clone()
        h=module.attention_head_size;sl=slice(HEAD*h,(HEAD+1)*h)
        values=module.value(hidden).view(*hidden.shape[:2],module.num_attention_heads,h)[:,:,HEAD]
        for b in range(len(hidden)):
            n=int(self.batch['attention_mask'][b].sum())
            ids=self.batch['input_ids'][b,:n].tolist()
            queries=torch.tensor([i for i,t in enumerate(ids) if t not in self.special_ids],dtype=torch.long)
            if not len(queries):continue
            if self.mode=='zero':
                context[b,queries,sl]=0;changed_probs[b,HEAD,queries]=0
                continue
            allowed=choose_mask(hidden[b,:n],self.mode,self.q,self.k)[queries]
            a=probs[b,HEAD,queries,:n]*allowed
            mass=a.sum(-1,keepdim=True)
            assert torch.all(mass>0),'Selected leaf has numerically zero attention mass'
            a=a/mass
            context[b,queries,sl]=a@values[b,:n]
            changed_probs[b,HEAD,queries]=0
            changed_probs[b,HEAD,queries,:n]=a
            assert torch.all(a[~allowed]==0) and torch.allclose(a.sum(-1),torch.ones(len(queries)),atol=1e-6)
            if self.check:
                # Check attention by calculating it directly within the chosen leaf.
                Q=module.query(hidden[b,:n]).view(n,module.num_attention_heads,h)[:,HEAD]
                K=module.key(hidden[b,:n]).view(n,module.num_attention_heads,h)[:,HEAD]
                scores=Q[queries]@K.T/(h**.5)
                expected=scores.masked_fill(~allowed,-torch.inf).softmax(-1)@values[b,:n]
                assert torch.allclose(context[b,queries,sl],expected,atol=2e-5,rtol=2e-5)
                untouched=torch.ones_like(context[b],dtype=torch.bool)
                untouched[queries,sl]=False
                assert torch.equal(context[b][untouched],original[b][untouched])
                self.checked+=1
        return context,changed_probs


@torch.no_grad()
def sentiment(args):
    tokenizer, bert = load_bert()
    step = json.loads((args.routers/'selection.json').read_text())['selected_step']
    def encode(record):
        return dict(label=record['label'],**tokenizer(record['sentence'],truncation=True,max_length=64))
    evaluation = [encode(r) for r in dataset_rows('stanfordnlp/sst2',SST2_REVISION,'data/validation-00000-of-00001.parquet')]
    assert len(evaluation)==872
    if args.quick: evaluation=evaluation[:16]
    used = {tuple(r['input_ids']) for r in evaluation}
    rl_train = json.loads((args.data/'passages.json').read_text())['train']
    assert used.isdisjoint(tuple(r['input_ids']) for r in rl_train)
    rows = dataset_rows('stanfordnlp/sst2',SST2_REVISION,'data/train-00000-of-00001.parquet')
    random.Random(20261009).shuffle(rows); training=[];counts=[0,0]
    per_class = 16 if args.quick else 1000
    for row in rows:
        if counts[row['label']]>=per_class: continue
        record=encode(row);key=tuple(record['input_ids'])
        if key in used: continue
        used.add(key);training.append(record);counts[row['label']]+=1
        if len(training)==2*per_class: break
    assert counts==[per_class,per_class]
    router=RoutedHead(tokenizer.all_special_ids)
    handle=bert.encoder.layer[LAYER].attention.self.register_forward_hook(router.hook)
    def features(records):
        pooled=[]
        for start in range(0,len(records),16):
            batch=tokenizer.pad([{key:r[key] for key in ['input_ids','attention_mask','token_type_ids']}
                                 for r in records[start:start+16]],return_tensors='pt')
            router.batch=batch;hidden=bert(**batch).last_hidden_state
            valid=batch['attention_mask'].bool()
            for token in tokenizer.all_special_ids:valid=valid & (batch['input_ids']!=token)
            pooled.append((hidden*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1))
        return torch.cat(pooled).double()
    # Fit one classifier using normal BERT, then keep it fixed.
    x=features(training);y=torch.tensor([2*r['label']-1 for r in training],dtype=torch.float64)
    mean=x.mean(0);intercept=y.mean();centered=x-mean
    weights=torch.linalg.solve(centered.T@centered+torch.eye(768,dtype=torch.float64),centered.T@(y-intercept))
    labels=torch.tensor([r['label'] for r in evaluation]);results=[]
    for mode,seed in [('full',None),('next',None),('zero',None)]+[('rl',s) for s in range(5)]:
        router.mode=mode
        if mode=='rl':router.q,router.k=load_router(args.routers,step,seed)
        predicted=((features(evaluation)-mean)@weights+intercept>=0).long()
        row=dict(quick_test=args.quick,mode=mode,seed=seed,correct=int((predicted==labels).sum()),examples=len(labels),
                 accuracy=float((predicted==labels).double().mean()))
        results.append(row);save(args.out/f'{mode}_{seed}_predictions.json',dict(labels=labels.tolist(),predictions=predicted.tolist()))
        print(row,flush=True)
    handle.remove();save(args.out/'sentiment.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['synthetic','prepare','train','sentiment'])
    parser.add_argument('--updates',type=int)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--quick',action='store_true',help='Use tiny BERT datasets to check execution, not accuracy')
    parser.add_argument('--data',type=Path,default=Path('../routing-data'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--routers',type=Path,default=Path('../routing-runs'))
    args=parser.parse_args()
    if args.out is None:
        args.out=Path('../'+{'synthetic':'synthetic-results','prepare':'routing-data','train':'routing-runs','sentiment':'sentiment-results'}[args.mode])
    if args.quick: print('Quick execution test; these are not research accuracy results.',flush=True)
    if args.updates is not None and args.updates<1:parser.error('--updates must be positive')
    torch.set_num_threads(1)
    {'synthetic':synthetic,'prepare':prepare,'train':train,'sentiment':sentiment}[args.mode](args)
