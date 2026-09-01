from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool

from dqn.config import DQNConfig
from dqn.network import QNetwork
from dqn.normalization import RunningNorm
from dqn.replay_buffer import ReplayBuffer
from tetris_rl.env import Observation

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class DQNAgent:
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        config: DQNConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.config = config
        self.device = device

        torch.manual_seed(config.seed)
        self.online = QNetwork(n_features, config.d_ff).to(device)
        self.target = QNetwork(n_features, config.d_ff).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.rng = np.random.default_rng(seed=config.seed)
        self.buffer = ReplayBuffer(config.capacity, n_features, n_actions, self.rng)
        self.normalizer = RunningNorm(n_features)
        self.epsilon = config.epsilon_start
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), config.learning_rate, eps=1e-4
        )

        self.cached_epsilon = 0.0
        self.training = True

    def _safe_mask(self, obs: Observation) -> Bool[np.ndarray, " n_actions"]:
        action_mask = obs["action_mask"]
        if not self.config.use_topout_mask:
            return action_mask
        safe_mask = action_mask & ~obs["afterstate_topout"]
        return safe_mask if safe_mask.any() else action_mask

    def act(self, obs: Observation) -> int:
        action_mask = self._safe_mask(obs)
        valid_actions = np.flatnonzero(action_mask)
        if self.rng.random() <= self.epsilon:
            return int(valid_actions[self.rng.integers(0, valid_actions.size)])

        afterstates = obs["afterstate_features"]
        afterstates = torch.from_numpy(afterstates).float().to(self.device)
        afterstates = self.normalizer.normalize(afterstates)

        with torch.no_grad():
            values = self.online(afterstates).squeeze(-1)

        mask = torch.from_numpy(action_mask).to(self.device)
        values = values.masked_fill(~mask, float("-inf"))
        action = values.argmax().item()
        return action

    def step(
        self, obs: Observation, env: gym.Env
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        mask = obs["action_mask"]
        self.normalizer.update(obs["afterstate_features"][mask])

        action = self.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)
        self.buffer.push(
            afterstate=obs["afterstate_features"][action],
            next_state=next_obs["afterstate_features"],
            next_mask=self._safe_mask(next_obs),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        return next_obs, reward, terminated, truncated, info

    def update(self) -> float:
        afterstates, next_states, next_masks, rewards, terminated, _ = (
            self.buffer.sample(self.config.batch_size)
        )

        afterstates = torch.from_numpy(afterstates).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        next_masks = torch.from_numpy(next_masks).bool().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        terminated = torch.from_numpy(terminated).bool().to(self.device)

        afterstates = self.normalizer.normalize(afterstates)
        next_states = self.normalizer.normalize(next_states)

        with torch.no_grad():
            next_values = torch.zeros(next_states.shape[0], device=self.device)
            active = ~terminated

            active_next_states = next_states[active]
            active_next_masks = next_masks[active]

            online_values = self.online(active_next_states).squeeze(-1)
            online_values = online_values.masked_fill(~active_next_masks, float("-inf"))
            best = online_values.argmax(dim=-1)

            batch_idxs = torch.arange(active_next_states.shape[0], device=self.device)
            best_states = active_next_states[batch_idxs, best]
            next_values[active] = self.target(best_states).squeeze(-1)

            td_targets = rewards + self.config.gamma * next_values

        predicted = self.online(afterstates).squeeze(-1)
        self.optimizer.zero_grad()
        loss = F.huber_loss(predicted, td_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=self.config.max_grad_norm
        )
        self.optimizer.step()
        return loss.item()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def eval(self) -> None:
        if not self.training:
            return
        self.cached_epsilon = self.epsilon
        self.epsilon = 0.0
        self.online.eval()
        self.training = False

    def train(self) -> None:
        if self.training:
            return
        self.epsilon = self.cached_epsilon
        self.online.train()
        self.training = True
