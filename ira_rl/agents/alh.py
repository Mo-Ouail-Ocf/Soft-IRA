from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemFFW(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        ae_noise: float = 0.2,
        hidden_dim: int = 64,
        hypo_dim: int = 64,
        eps: float = 0.03,
        **kwargs,
    ):
        super().__init__()
        self.ae_noise = ae_noise
        self.state_dim = hypo_dim
        self.eps = eps
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim + action_dim + 1, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hypo_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(observation_dim + action_dim + 1 + hypo_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, observation_dim + action_dim + 1),
        )

    def _mk_input(self, observation, action, reward):
        return torch.cat((observation, action, reward.reshape(-1, 1)), dim=-1)

    def encode(self, observation, action, reward, prev_vec=None):
        encoded = F.normalize(self.encoder(self._mk_input(observation, action, reward)), eps=self.eps, dim=-1)
        encoded = encoded.mean(dim=-2, keepdim=False).reshape(-1)
        if prev_vec is not None:
            encoded = F.normalize(encoded + prev_vec.detach().reshape(-1), eps=self.eps, dim=-1)
        return encoded

    def sample_encode(self, observation, action, reward, mini_batch_size: int, prev_vec=None):
        batch_size = observation.size(0)
        if mini_batch_size is None:
            mini_batch_size = batch_size // 2
        source = self._mk_input(observation, action, reward)
        indices = torch.randint(0, batch_size, size=(batch_size, mini_batch_size), device=source.device)
        mini_batch_indices = (
            indices
            - torch.min(indices, dim=-1)[0][..., None]
            + torch.arange(batch_size, device=source.device)[..., None]
        ) % batch_size
        mini_batches = source[mini_batch_indices]

        encoded = F.normalize(self.encoder(mini_batches), eps=self.eps, dim=-1)
        encoded = encoded.mean(dim=-2, keepdim=False).reshape(batch_size, self.state_dim)
        if prev_vec is not None:
            assert prev_vec.shape == (batch_size, self.state_dim) or prev_vec.shape == (self.state_dim,)
            encoded = F.normalize(encoded + prev_vec, eps=self.eps, dim=-1)
        return encoded

    def decode(self, observation, action, reward, prev_vec):
        observations = self._mk_input(observation, action, reward)
        batch_size = observation.size(0)
        if prev_vec.shape != (batch_size, self.state_dim):
            prev_vec = prev_vec.reshape(1, -1).expand(batch_size, -1)
        original_x = observations
        noise_range = original_x.max(dim=0).values - original_x.min(dim=0).values
        noise = torch.randn_like(original_x) * noise_range * self.ae_noise
        denoised_x = self.decoder(torch.cat((original_x + noise, prev_vec), dim=-1))
        return denoised_x, F.mse_loss(original_x, denoised_x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim: int = 64, hypo_dim: int = 64):
        super().__init__()
        self.hypo_dim = hypo_dim
        self.l1 = nn.Linear(state_dim + hypo_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state, prev_vec):
        batch_size = state.shape[0]
        if prev_vec.shape != (batch_size, self.hypo_dim):
            prev_vec = prev_vec.reshape(1, -1).expand(batch_size, -1)
        hidden = torch.cat([state, prev_vec], 1)
        hidden = F.relu(self.l1(hidden))
        hidden = F.relu(self.l2(hidden))
        return self.max_action * torch.tanh(self.l3(hidden))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int = 64, hypo_dim: int = 64):
        super().__init__()
        self.hypo_dim = hypo_dim
        self.l1 = nn.Linear(state_dim + action_dim + hypo_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim + hypo_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, prev_vec):
        batch_size = state.shape[0]
        if prev_vec.shape != (batch_size, self.hypo_dim):
            prev_vec = prev_vec.reshape(1, -1).expand(batch_size, -1)
        state_action = torch.cat([state, action, prev_vec], 1)

        q1 = F.relu(self.l1(state_action))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(state_action))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action, prev_vec):
        batch_size = state.shape[0]
        if prev_vec.shape != (batch_size, self.hypo_dim):
            prev_vec = prev_vec.reshape(1, -1).expand(batch_size, -1)
        state_action = torch.cat([state, action, prev_vec], 1)
        q1 = F.relu(self.l1(state_action))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class memTD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device: str = "cpu",
        hidden_dim: int = 64,
        hypo_dim: int = 64,
        mini_batch_size=None,
        **kwargs,
    ):
        self.device = device
        self.mini_batch_size = mini_batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.mem = MemFFW(state_dim, action_dim, hidden_dim=hidden_dim, hypo_dim=hypo_dim).to(device)
        self.mem_optimizer = torch.optim.Adam(self.mem.parameters(), lr=3e-4)

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim=hidden_dim, hypo_dim=hypo_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim, hypo_dim=hypo_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.initial_state = torch.nn.Parameter(torch.zeros((hypo_dim,), dtype=torch.float32, device=device))
        self.initial_state_optimizer = torch.optim.Adam([self.initial_state], lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self._test_state = None

    def train_mem_step(self, observation, action, reward) -> dict:
        split = np.random.randint(1, observation.shape[0] - 1)
        metrics = {}

        batch = {"observation": observation, "action": action, "reward": reward}
        first = {
            "observation": batch["observation"][:split],
            "action": batch["action"][:split],
            "reward": batch["reward"][:split],
        }

        h1 = self.mem.encode(**first, prev_vec=None)
        _, loss_h1 = self.mem.decode(**first, prev_vec=h1)
        second = {
            "observation": batch["observation"][split:],
            "action": batch["action"][split:],
            "reward": batch["reward"][split:],
        }
        hypothesis = self.mem.encode(**second, prev_vec=h1)

        _, loss_h = self.mem.decode(**batch, prev_vec=hypothesis)
        diversity_loss = -F.mse_loss(h1, self.initial_state.detach()) - F.mse_loss(hypothesis, self.initial_state.detach())
        loss = loss_h1 + loss_h + diversity_loss
        metrics.update(
            {
                "internal_mem_loss": loss,
                "loss_h_1": loss_h1,
                "loss_h": loss_h,
                "diversity_loss": diversity_loss,
            }
        )

        self.mem_optimizer.zero_grad()
        loss.backward()
        self.mem_optimizer.step()

        _, train_mem_loss = self.mem.decode(**batch, prev_vec=self.initial_state)
        self.initial_state_optimizer.zero_grad()
        train_mem_loss.backward()
        self.initial_state_optimizer.step()
        metrics.update({"D_train_mem_loss": train_mem_loss})
        return metrics

    def forget(self):
        self._test_state = None

    @property
    def prev_state(self):
        if self._test_state is None:
            self._test_state = self.initial_state.detach()
        return self._test_state

    @prev_state.setter
    def prev_state(self, value):
        self._test_state = value

    def watch(self, observation, action, reward) -> dict:
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        observation = observation.reshape(-1, self.state_dim)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        action = action.reshape(-1, self.action_dim)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        reward = reward.reshape(-1)
        with torch.no_grad():
            self.prev_state = self.mem.encode(observation, action, reward, self.prev_state)
        return {}

    def select_action(self, observation, return_batch=False):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        observation = observation.reshape(-1, self.state_dim)
        output = self.actor(observation, getattr(self, "prev_state", None)).cpu().data.numpy()
        if return_batch:
            return output
        return output.flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        if self.total_it % 10 == 0:
            self.train_mem_step(state, action, reward)

        with torch.no_grad():
            mini_batch_size = self.mini_batch_size if self.mini_batch_size is not None else batch_size // 2
            hypothesis = self.mem.sample_encode(
                state,
                action,
                reward,
                mini_batch_size=mini_batch_size,
                prev_vec=None,
            )

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, prev_vec=hypothesis) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action, prev_vec=hypothesis)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action, prev_vec=hypothesis)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state, prev_vec=hypothesis), prev_vec=hypothesis).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.mem.state_dict(), filename + "_mem")
        torch.save(self.mem_optimizer.state_dict(), filename + "_mem_optimizer")
        torch.save(self.initial_state, filename + "_initial_state")
        torch.save(self.initial_state_optimizer.state_dict(), filename + "_initial_state_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.mem.load_state_dict(torch.load(filename + "_mem"))
        self.mem_optimizer.load_state_dict(torch.load(filename + "_mem_optimizer"))
        self.initial_state = torch.load(filename + "_initial_state").to(self.device)
        self.initial_state_optimizer.load_state_dict(torch.load(filename + "_initial_state_optimizer"))
