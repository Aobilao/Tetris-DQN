# tetris-rl

## Install

```
pip install -e .
pip install torch wandb
wandb login
```

## Train

```
python -m dqn.train [--topout-mask] [--resume]
python -m rainbow_dqn.train [--topout-mask] [--resume]
```

Saves to `agent.pt` / `rainbow_agent.pt` (overwritten each checkpoint, used by `--resume`) plus periodic `*_step<N>.pt` snapshots that are never overwritten. Metrics log to the `tetris-dqn` Weights & Biases project.

## Play

```
python -m dqn.play [--render]
python -m rainbow_dqn.play [--render]
```

Loads `agent.pt` / `rainbow_agent.pt` from the current directory.
