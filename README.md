# tetris-rl

## Install

```
pip install -r requirements.txt
pip install -e .
```

## Train

```
python -m dqn.train [--topout-mask] [--resume]
python -m rainbow_dqn.train [--topout-mask] [--resume]
```

Saves to `agent.pt` / `rainbow_agent.pt` (overwritten each checkpoint, used by `--resume`) plus periodic `*_step<N>.pt` snapshots that are never overwritten. `agent_best.pt` / `rainbow_agent_best.pt` is overwritten whenever an eval beats the best `avg_reward` so far. Progress prints to stdout every eval period.

## Play

```
python -m dqn.play [--render]
python -m rainbow_dqn.play [--render]
```

Loads `agent.pt` / `rainbow_agent.pt` from the current directory.
