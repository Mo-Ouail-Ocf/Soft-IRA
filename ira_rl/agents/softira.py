import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from faiss import StandardGpuResources, METRIC_Linf
from faiss.contrib.torch_utils import torch_replacement_knn_gpu

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter notes (mapped to paper symbols)
#
#   alpha_rde   : α_RDE  — RDE inner-product coefficient             (5e-4)
#   k           : k      — number of Chebyshev nearest-neighbours   (10)
#   d_max       : D_max  — rubber-band deviation target, auto-set to 0.05√d
#   target_entropy        — H_target for SAC α, auto-set to −dim(A)
#   warmup_timestamps     — steps before kNN lookup activates        (4000)
#   log_alpha, log_beta   — both initialised to 0  (α=β=1 at t=0)
# ──────────────────────────────────────────────────────────────────────────────

LOG_STD_MAX = 2
LOG_STD_MIN = -5


# ══════════════════════════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════════════════════════

class GaussianActor(nn.Module):
    """SAC squashed-Gaussian actor.

    sample(s) returns
        action   — reparameterised sample in [−max_action, max_action]
        log_prob — log π_φ(a|s),  shape (B,1)
        a_bar    — ā = tanh(μ_φ(s))·max_action  (deterministic squashed mean)

    a_bar is the anchor point for all IRA components (GAG query, RDE input).
    It carries live gradients through μ_φ — caller decides when to detach.
    """

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = max_action

    def _trunk(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, state):
        mean, std = self._trunk(state)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        a_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - a_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        a_bar = torch.tanh(mean) * self.max_action
        return a_t * self.max_action, log_prob, a_bar

    def log_prob_from_action(self, state, action):
        """Evaluate log π_φ(a|s) for pre-collected buffer actions.

        Used inside the kNN soft-Q ranking; a constant log(max_action) offset
        is dropped because ranking is invariant to additive constants.

        action : tensor already in [−max_action, max_action], shape (B,A)
        returns: log_prob, shape (B,1)
        """
        mean, std = self._trunk(state)
        a_norm = (action / self.max_action).clamp(-1 + 1e-6, 1 - 1e-6)
        x_t = torch.atanh(a_norm)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(x_t) - torch.log(1 - a_norm.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """Twin soft Q-networks.

    forward() and Q1()/Q2() each return (Q_value, penultimate_features).
    Penultimate features φ(s,a;θ⁺) are used by the RDE loss.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        f1 = F.relu(self.l2(F.relu(self.l1(sa))))
        f2 = F.relu(self.l5(F.relu(self.l4(sa))))
        return self.l3(f1), self.l6(f2), f1, f2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        f = F.relu(self.l2(F.relu(self.l1(sa))))
        return self.l3(f), f

    def Q2(self, state, action):
        sa = torch.cat([state, action], dim=1)
        f = F.relu(self.l5(F.relu(self.l4(sa))))
        return self.l6(f), f


# ══════════════════════════════════════════════════════════════════════════════
# Soft-IRA
# ══════════════════════════════════════════════════════════════════════════════

class SoftIRA(object):
    """Soft-IRA: IRA (RDE + GAG) integrated into SAC.

    Key design points
    ─────────────────
    • Actor is a squashed Gaussian (SAC).  a_bar = tanh(μ_φ(s)) is the anchor.
    • Buffer stores (s,a) pairs without Q-values; soft-Q re-evaluated on-the-fly.
    • RDE uses a_bar.detach() inside the critic loss — no critic→actor gradient.
    • GAG uses a_bar (live) inside the actor loss — gradient flows into μ_φ.
    • β is auto-tuned via a rubber-band dual loss, exactly mirroring SAC's α.
    • kNN lookup activates after warmup_timestamps steps.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha_rde=5e-4,
        k=10,
        device="cuda:0",
        warmup_timestamps=4000,
        action_buffer_size=200_000,
        d_max=None,
        target_entropy=None,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        beta_lr=3e-4,
        warmup_timesteps=None,
        **kwargs,
    ):
        if hidden_dim != 256:
            raise ValueError("Legacy SoftIRA uses a fixed hidden_dim of 256.")
        if warmup_timesteps is not None:
            warmup_timestamps = warmup_timesteps

        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha_rde = alpha_rde
        self.k = k
        self.warmup_timestamps = warmup_timestamps
        self.action_buffer_size = int(action_buffer_size)
        self.total_it = 0
        self.buf_idx = 0

        self.actor = GaussianActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_entropy = float(-action_dim) if target_entropy is None else float(target_entropy)
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.d_max = 0.05 * math.sqrt(action_dim) if d_max is None else float(d_max)
        self.log_beta = torch.tensor([0.0], requires_grad=True, device=device)
        self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr)

        self.buff_actions = torch.zeros((self.action_buffer_size, action_dim), device=device)
        self.buff_actions.requires_grad = False

        torch.cuda.set_device(int(str(device).replace("cuda:", "")))
        self.resource = StandardGpuResources()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    def _update_beta(self, deviation: torch.Tensor) -> float:
        beta_loss = -(self.log_beta * (deviation - self.d_max))
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()
        return beta_loss.item()

    def select_action(self, state):
        """Deterministic mean action for evaluation (no noise)."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            _, _, a_bar = self.actor.sample(state)
        return a_bar.cpu().numpy().flatten()

    def estimate_q(self, state, action):
        """Estimate the clipped twin-Q value for logging/evaluation."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            q1, q2, _, _ = self.critic(state, action)
        return float(torch.min(q1, q2).item())

    def _knn(self, query, memory):
        """Chebyshev (L∞) kNN on GPU.  No gradients computed."""
        with torch.no_grad():
            _, idx = torch_replacement_knn_gpu(
                self.resource,
                query,
                memory,
                self.k,
                metric=METRIC_Linf,
            )
        return idx

    def _retrospective_lookup(self, state, a_bar):
        """Compute ã_opt, ã_sub and target features for the batch.

        Query   : a_bar.detach()  (squashed deterministic mean, no gradient)
        Metric  : Chebyshev kNN
        Ranking : soft Q-value  Q^soft = min(Q1′,Q2′) − α · log π_φ(a′|s)
                  computed on-the-fly with current target critics & policy.

        Returns (all detached, no gradients)
            a_opt      : (B, A)  highest soft-Q neighbour  → GAG anchor
            a_sub      : (B, A)  second-highest soft-Q neighbour → RDE anchor
            fea_sub_1  : (B, 256)  target-critic Q1 features at a_sub
            fea_sub_2  : (B, 256)  target-critic Q2 features at a_sub

        Note: ã_sub is the second-best (not worst) neighbour — mirroring IRA's
        original choice of using the most-confusable suboptimal action for RDE.

        Per-step vs per-batch approximation: anchors are computed for the
        current environment state, not per batch sample. This is inherited
        directly from IRA Algorithm 1 and validated empirically therein.
        """
        B = state.shape[0]
        idx = self._knn(a_bar.detach(), self.buff_actions)

        neighbor_actions = self.buff_actions[idx]

        S = state.shape[1]
        state_exp = state.unsqueeze(1).expand(-1, self.k, -1).reshape(B * self.k, S)
        actions_flat = neighbor_actions.reshape(B * self.k, self.action_dim)

        q1_all, _ = self.critic_target.Q1(state_exp, actions_flat)
        q2_all, _ = self.critic_target.Q2(state_exp, actions_flat)
        lp_all = self.actor.log_prob_from_action(state_exp, actions_flat)

        q_soft = (
            torch.min(q1_all, q2_all) - self.alpha.detach() * lp_all
        ).reshape(B, self.k)

        _, sorted_idx = torch.sort(q_soft, descending=True, dim=-1)
        opt_idx = sorted_idx[:, 0]
        sub_idx = sorted_idx[:, 1]

        rows = torch.arange(B, device=self.device)
        a_opt = neighbor_actions[rows, opt_idx, :]
        a_sub = neighbor_actions[rows, sub_idx, :]

        _, fea_sub_1 = self.critic_target.Q1(state, a_sub)
        _, fea_sub_2 = self.critic_target.Q2(state, a_sub)

        return a_opt, a_sub, fea_sub_1, fea_sub_2

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            end = self.buf_idx + batch_size
            if end > self.action_buffer_size:
                left = end - self.action_buffer_size
                self.buff_actions[self.buf_idx:] = action[: batch_size - left]
                self.buff_actions[:left] = action[batch_size - left :]
                self.buf_idx = left
            else:
                self.buff_actions[self.buf_idx:end] = action
                self.buf_idx = end % self.action_buffer_size

        post_warmup = self.total_it > self.warmup_timestamps

        sampled_action, log_prob, a_bar = self.actor.sample(state)
        rde_loss_value = 0.0
        gag_loss_value = 0.0
        beta_loss_value = 0.0
        anchor_deviation = 0.0

        if post_warmup:
            with torch.no_grad():
                a_opt, a_sub, fea_sub_1, fea_sub_2 = self._retrospective_lookup(state, a_bar)

        with torch.no_grad():
            next_a, next_lp, _ = self.actor.sample(next_state)
            tQ1, tQ2, _, _ = self.critic_target(next_state, next_a)
            target_Q = torch.min(tQ1, tQ2) - self.alpha * next_lp
            target_Q = reward + not_done * self.discount * target_Q

        Q1, Q2, _, _ = self.critic(state, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if post_warmup:
            a_bar_det = a_bar.detach()
            _, fea_abar_1 = self.critic.Q1(state, a_bar_det)
            _, fea_abar_2 = self.critic.Q2(state, a_bar_det)

            rde_loss = (
                torch.einsum("ij,ij->i", fea_abar_1, fea_sub_1).mean()
                + torch.einsum("ij,ij->i", fea_abar_2, fea_sub_2).mean()
            ) * self.alpha_rde

            critic_loss = critic_loss + rde_loss
            rde_loss_value = rde_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        q1_pi, _ = self.critic.Q1(state, sampled_action)
        q2_pi, _ = self.critic.Q2(state, sampled_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        if post_warmup:
            gag_loss = self.beta.detach() * (a_bar - a_opt).pow(2).sum(dim=-1).mean()
            actor_loss = actor_loss + gag_loss
            gag_loss_value = gag_loss.item()
            anchor_deviation = (a_bar.detach() - a_opt).pow(2).sum(dim=-1).mean().item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha_loss_value = alpha_loss.item()

        if post_warmup:
            deviation = (a_bar.detach() - a_opt).pow(2).sum(dim=-1).mean()
            beta_loss_value = self._update_beta(deviation)

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        filled_actions = min(self.total_it * batch_size, self.action_buffer_size)
        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/alpha": self.alpha.item(),
            "train/alpha_loss": alpha_loss_value,
            "train/beta": self.beta.item(),
            "train/beta_loss": beta_loss_value,
            "train/rde_loss": rde_loss_value,
            "train/gag_loss": gag_loss_value,
            "train/anchor_deviation": anchor_deviation,
            "train/entropy": (-log_prob.mean()).item(),
            "train/q1_mean": Q1.mean().item(),
            "train/q2_mean": Q2.mean().item(),
            "train/target_q_mean": target_Q.mean().item(),
            "train/post_warmup": float(post_warmup),
            "train/action_buffer_fill": float(filled_actions),
        }

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.log_alpha.data, filename + "_log_alpha")
        torch.save(self.log_beta.data, filename + "_log_beta")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.log_alpha.data = torch.load(filename + "_log_alpha")
        self.log_beta.data = torch.load(filename + "_log_beta")

    def state_dict(self) -> dict:
        return {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_beta": self.log_beta.detach().cpu(),
            "beta_optimizer": self.beta_optimizer.state_dict(),
            "buff_actions": self.buff_actions.detach().cpu(),
            "buf_idx": self.buf_idx,
            "total_it": self.total_it,
        }

    def load_state_dict(self, state: dict) -> None:
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.log_alpha.data = state["log_alpha"].to(self.device)
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.log_beta.data = state["log_beta"].to(self.device)
        self.beta_optimizer.load_state_dict(state["beta_optimizer"])
        self.buff_actions.copy_(state["buff_actions"].to(self.device))
        self.buf_idx = state.get("buf_idx", 0)
        self.total_it = state.get("total_it", 0)
