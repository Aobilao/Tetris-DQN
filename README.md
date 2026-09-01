# tetris-rl

## Install

```
pip install -r requirements.txt
pip install -e .
```

## Train

```
python -m dqn.train [--topout-mask] [--resume]
```

Saves to `agent.pt` (overwritten each checkpoint, used by `--resume`) plus periodic `agent_step<N>.pt` snapshots that are never overwritten. `agent_best.pt` is overwritten whenever an eval beats the best `avg_reward` so far. Progress prints to stdout every eval period.

- `--resume` continues from `agent.pt` in the current directory, restoring the step count, replay buffer and RNG state. Config is taken from the checkpoint, so `--topout-mask` is ignored.
- `--topout-mask` makes the agent skip placements that would top out, unless every placement does. Fresh runs only.

## Play

```
python -m dqn.play <checkpoint> [--render]
```

Example: `python -m dqn.play agent_ckpt.pt --render`

- `--render` prints the board to stdout after each placement, clearing the screen between frames.

## Web

```
python scripts/export_web_model.py <checkpoint> [--page PATH]
```

Example: `python scripts/export_web_model.py agent_ckpt.pt`

Bakes the checkpoint's weights and normalizer stats into `web/index.html` as its `const MODEL = {...}` line, then prints what it wrote. Open that file in a browser to play the game yourself or watch the agent play, no server or build step.

- `--page` targets a different page file. Defaults to `web/index.html` beside the repo root, so the command works from any directory.
- The page is rewritten in place with no backup; only the `const MODEL = ...;` line changes. Reload with a hard refresh or the browser serves the old weights.
