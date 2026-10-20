# Hierarchical Routing with Reinforcement Learning

I study when a token router can follow a tree to reach the right tokens, and whether reinforcement learning (RL) can learn those routes.

## Files

`routing.py`: Builds the tree, trains the router with RL, and checks whether the required routes are mathematically possible.

`experiments.py`: Defines the target tokens, runs the small example, trains on BERT features, and tests sentiment predictions.

`requirements.txt`: Lists the required Python packages.

`README.md`: Explains the project.

## Run

Use Python 3.12. Run these commands from this folder:

```bash
python -m pip install -r requirements.txt
python experiments.py synthetic
```

`NEEDLE_MAP` in `experiments.py` assigns a target token to each query. The code also creates another map using cosine similarity. Changing the targets changes the task.

The small example uses a learning rate of `0.00003`. With seed 42 and 3,000 updates, it reached 4/8 targets with the original map and 0/8 with the similarity map. The similarity map is mathematically possible, but training did not learn it in this run. This does not reproduce the earlier successful toy experiments.

BERT training settings: 10,000 passages, 30,000 updates per seed, five seeds, learning rate `0.00003`, discount factor `0.8`, exploration bonus `0.01`, and 32 sampled routes per query.

Training saves checkpoints. Use a new output folder for each run.

Quick test option: Uses small datasets and 20 updates per seed to check that the BERT code runs. It is not an accuracy test.

## Recorded results

These results are from October 9, 2026. The smaller code version was checked against the original code. The full study was not repeated when preparing this repository.

RL routing on WikiText-2, average across five seeds: **84.40%**

Always routing to the leaf containing the next token: **80.66%**

Frozen BERT sentiment classifier: **80.85%**

Same classifier with one attention head using RL routing, average across five seeds: **80.99%**

Same classifier with that head's output set to zero for ordinary query tokens: **81.08%**

Routing was tested on 200 passages. Sentiment was tested on 872 SST-2 validation examples. The small sentiment difference is not enough to show an improvement. The WikiText test data was examined during earlier experiments, so these results are preliminary.

## Scope

The math check asks whether one shared matrix can make every required branch decision with a clear score difference. It does not check whether that solution fits smaller projection dimensions or whether routing could succeed with equal branch scores.

A possible solution does not guarantee that RL will learn it. BERT's weights stay fixed in these experiments. The sentiment test still computes full attention before keeping the selected leaf, so it does not show a speed improvement. Training BERT and the router together is future work.

I used AI to help write some of the code. I also ran the training experiments myself.
