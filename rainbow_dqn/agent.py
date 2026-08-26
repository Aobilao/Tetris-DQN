from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32

from rainbow_dqn.config import RDQNConfig
from rainbow_dqn.network import QNetwork
from rainbow_dqn.normalization import RunningNorm
from rainbow_dqn.replay_buffer import ReplayBuffer
from tetris_rl.env import Observation

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class RDQNAgent:
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        config: RDQNConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.config = config
        self.device = device

        self.build_support()

        self.online = QNetwork(
            n_features, config.d_ff, config.n_atoms, config.noisy_sigma
        ).to(device)
        self.target = QNetwork(
            n_features, config.d_ff, config.n_atoms, config.noisy_sigma
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.nstep: deque[tuple[Float32[np.ndarray, " n_features"], float]] = deque(
            maxlen=config.n_step
        )
        self.rng = np.random.default_rng(seed=config.seed)
        self.buffer = ReplayBuffer(
            config.capacity,
            n_features,
            n_actions,
            self.rng,
            config.alpha,
            config.priority_eps,
        )
        self.normalizer = RunningNorm(n_features)
        self.beta = config.beta_start
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), config.learning_rate
        )

    def build_support(self) -> None:
        config = self.config
        self.support = torch.linspace(
            config.v_min, config.v_max, config.n_atoms, device=self.device
        )
        self.delta_z = (config.v_max - config.v_min) / (config.n_atoms - 1)

    def _safe_mask(self, obs: Observation) -> Bool[np.ndarray, " n_actions"]:
        action_mask = obs["action_mask"]
        if not self.config.use_topout_mask:
            return action_mask
        safe_mask = action_mask & ~obs["afterstate_topout"]
        return safe_mask if safe_mask.any() else action_mask

    def act(self, obs: Observation) -> int:
        action_mask = self._safe_mask(obs)
        afterstates = obs["afterstate_features"]
        afterstates = torch.from_numpy(afterstates).float().to(self.device)
        afterstates = self.normalizer.normalize(afterstates)

        with torch.no_grad():
            probs = F.softmax(self.online(afterstates), dim=-1)
            values = (probs * self.support).sum(-1)

        mask = torch.from_numpy(action_mask).to(self.device)
        values = values.masked_fill(~mask, float("-inf"))
        return int(values.argmax().item())

    def _emit(
        self,
        start: int,
        next_state: Float32[np.ndarray, "n_actions n_features"],
        next_mask: Bool[np.ndarray, " n_actions"],
        terminated: bool,
    ) -> None:
        gamma = self.config.gamma
        reward = 0.0
        for offset in range(start, len(self.nstep)):
            reward += gamma ** (offset - start) * self.nstep[offset][1]
        self.buffer.push(
            afterstate=self.nstep[start][0],
            next_state=next_state,
            next_mask=next_mask,
            reward=reward,
            discount=gamma ** (len(self.nstep) - start),
            terminated=terminated,
        )

    def step(
        self, obs: Observation, env: gym.Env
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        mask = obs["action_mask"]
        self.normalizer.update(obs["afterstate_features"][mask])

        self.online.reset_noise()
        action = self.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)

        self.nstep.append((obs["afterstate_features"][action], reward))
        next_state = next_obs["afterstate_features"]
        next_mask = self._safe_mask(next_obs)

        if len(self.nstep) == self.config.n_step:
            self._emit(0, next_state, next_mask, terminated)

        if terminated or truncated:
            start = 1 if len(self.nstep) == self.config.n_step else 0
            for i in range(start, len(self.nstep)):
                self._emit(i, next_state, next_mask, terminated)
            self.nstep.clear()

        return next_obs, reward, terminated, truncated, info

    def update(self) -> float:
        self.online.reset_noise()
        self.target.reset_noise()

        (
            afterstates,
            next_states,
            next_masks,
            rewards,
            discounts,
            terminated,
            idxs,
            weights,
        ) = self.buffer.sample(self.config.batch_size, self.beta)

        afterstates = torch.from_numpy(afterstates).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        next_masks = torch.from_numpy(next_masks).bool().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        discounts = torch.from_numpy(discounts).float().to(self.device)
        terminated = torch.from_numpy(terminated).bool().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)

        afterstates = self.normalizer.normalize(afterstates)
        next_states = self.normalizer.normalize(next_states)

        with torch.no_grad():
            next_probs = F.softmax(self.online(next_states), dim=-1)
            next_values = (next_probs * self.support).sum(-1)
            next_values = next_values.masked_fill(~next_masks, float("-inf"))
            best = next_values.argmax(dim=-1)

            batch_idxs = torch.arange(next_states.shape[0], device=self.device)
            best_states = next_states[batch_idxs, best]
            target_probs = F.softmax(self.target(best_states), dim=-1)

            bootstrap = (discounts * ~terminated).unsqueeze(-1)
            tz = rewards.unsqueeze(-1) + bootstrap * self.support
            tz = tz.clamp(self.config.v_min, self.config.v_max)

            b = (tz - self.config.v_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < self.config.n_atoms - 1) & (lower == upper)] += 1

            projected = torch.zeros_like(target_probs)
            projected.scatter_add_(1, lower, target_probs * (upper - b))
            projected.scatter_add_(1, upper, target_probs * (b - lower))

        log_probs = F.log_softmax(self.online(afterstates), dim=-1)
        losses = -(projected * log_probs).sum(dim=-1)

        self.optimizer.zero_grad()
        loss = (losses * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=self.config.max_grad_norm
        )
        self.optimizer.step()

        self.buffer.update_priorities(idxs, losses.detach().cpu().numpy())
        return loss.item()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def eval(self) -> None:
        self.online.eval()

    def train(self) -> None:
        self.online.train()
