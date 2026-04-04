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

LOG_STD_MAX =  2
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
        self.l1            = nn.Linear(state_dim, 256)
        self.l2            = nn.Linear(256, 256)
        self.mean_layer    = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action    = max_action

    def _trunk(self, state):
        x       = F.relu(self.l1(state))
        x       = F.relu(self.l2(x))
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, state):
        mean, std = self._trunk(state)
        dist      = torch.distributions.Normal(mean, std)
        x_t       = dist.rsample()                              # reparameterised
        a_t       = torch.tanh(x_t)
        # tanh-squashing correction to log prob
        log_prob  = dist.log_prob(x_t) - torch.log(1 - a_t.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)          # (B,1)
        a_bar     = torch.tanh(mean) * self.max_action          # ā  (no noise)
        return a_t * self.max_action, log_prob, a_bar

    def log_prob_from_action(self, state, action):
        """Evaluate log π_φ(a|s) for pre-collected buffer actions.

        Used inside the kNN soft-Q ranking; a constant log(max_action) offset
        is dropped because ranking is invariant to additive constants.

        action : tensor already in [−max_action, max_action], shape (B,A)
        returns: log_prob, shape (B,1)
        """
        mean, std = self._trunk(state)
        a_norm    = (action / self.max_action).clamp(-1 + 1e-6, 1 - 1e-6)
        x_t       = torch.atanh(a_norm)
        dist      = torch.distributions.Normal(mean, std)
        log_prob  = dist.log_prob(x_t) - torch.log(1 - a_norm.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)               # (B,1)


class Critic(nn.Module):
    """Twin soft Q-networks.

    forward() and Q1()/Q2() each return (Q_value, penultimate_features).
    Penultimate features φ(s,a;θ⁺) are used by the RDE loss.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa   = torch.cat([state, action], dim=1)
        f1   = F.relu(self.l2(F.relu(self.l1(sa))))
        f2   = F.relu(self.l5(F.relu(self.l4(sa))))
        return self.l3(f1), self.l6(f2), f1, f2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        f  = F.relu(self.l2(F.relu(self.l1(sa))))
        return self.l3(f), f

    def Q2(self, state, action):
        sa = torch.cat([state, action], dim=1)
        f  = F.relu(self.l5(F.relu(self.l4(sa))))
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
        discount          = 0.99,
        tau               = 0.005,
        alpha_rde         = 5e-4,       # α_RDE
        k                 = 10,         # Chebyshev kNN neighbours
        device            = 'cuda:0',
        warmup_timestamps = 4000,
        action_buffer_size= 200_000,
        d_max             = None,       # None → 0.05√(action_dim)
        target_entropy    = None,       # None → −action_dim
    ):
        self.device            = device
        self.action_dim        = action_dim
        self.max_action        = max_action
        self.discount          = discount
        self.tau               = tau
        self.alpha_rde         = alpha_rde
        self.k                 = k
        self.warmup_timestamps = warmup_timestamps
        self.action_buffer_size= int(action_buffer_size)
        self.total_it          = 0
        self.buf_idx           = 0

        # ── Networks ──────────────────────────────────────────────────────
        self.actor           = GaussianActor(state_dim, action_dim, max_action).to(device)
        # No target actor: SAC's stochasticity is its own stability mechanism.
        # The entropy term α·log π_φ(a'|s') must be evaluated under the same
        # policy that generates a' — must be the current actor φ, not a lagged copy.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic           = Critic(state_dim, action_dim).to(device)
        self.critic_target    = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # ── Auto-tuned α (entropy temperature) ───────────────────────────
        # L_logα = log α · (−log π_φ(a|s) − H_target)
        # Initialised at log α = 0  ↔  α = 1
        self.target_entropy  = float(-action_dim) if target_entropy is None else float(target_entropy)
        self.log_alpha       = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # ── Auto-tuned β (GAG rubber-band) ────────────────────────────────
        # L_logβ = −log β · (‖ā − ã_opt‖² − D_max)
        # Initialised at log β = 0  ↔  β = 1  (mirrors IRA's μ=1.0 start)
        self.d_max           = 0.05 * math.sqrt(action_dim) if d_max is None else float(d_max)
        self.log_beta        = torch.zeros(1, requires_grad=True, device=device)
        self.beta_optimizer  = torch.optim.Adam([self.log_beta], lr=3e-4)

        # ── Action buffer ─────────────────────────────────────────────────
        # Stores only actions (FIFO ring-buffer).
        # States are NOT stored; log π_φ(a'|s) is evaluated on-the-fly.
        self.buff_actions = torch.zeros(
            (self.action_buffer_size, action_dim), device=device
        )
        self.buff_actions.requires_grad = False

        torch.cuda.set_device(int(str(device).replace('cuda:', '')))
        self.resource = StandardGpuResources()

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state):
        """Deterministic mean action for evaluation (no noise)."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            _, _, a_bar = self.actor.sample(state)
        return a_bar.cpu().numpy().flatten()

    # ── FAISS kNN ─────────────────────────────────────────────────────────────

    def _knn(self, query, memory):
        """Chebyshev (L∞) kNN on GPU.  No gradients computed."""
        with torch.no_grad():
            _, idx = torch_replacement_knn_gpu(
                self.resource, query, memory, self.k, metric=METRIC_Linf
            )
        return idx  # (B, k)

    # ── Retrospective lookup ──────────────────────────────────────────────────

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

        # ── kNN retrieval ──────────────────────────────────────────────
        # No deduplication needed: SAC samples from a continuous Gaussian,
        # so exact floating-point duplicates in the buffer are impossible.
        # Deduplication (O(N log N) on 200k rows every step) is only needed
        # for deterministic actors like TD3 that can repeat identical actions.
        idx              = self._knn(a_bar.detach(), self.buff_actions)  # (B, k)

        # Gather neighbour actions → (B, k, A)
        neighbor_actions = self.buff_actions[idx]

        # ── Batched soft-Q evaluation (single forward pass per network) ──
        # Flatten: (B, k, A) → (B·k, A)
        S           = state.shape[1]
        state_exp   = state.unsqueeze(1).expand(-1, self.k, -1).reshape(B * self.k, S)
        actions_flat = neighbor_actions.reshape(B * self.k, self.action_dim)

        q1_all, _  = self.critic_target.Q1(state_exp, actions_flat)    # (B·k, 1)
        q2_all, _  = self.critic_target.Q2(state_exp, actions_flat)    # (B·k, 1)
        lp_all     = self.actor.log_prob_from_action(state_exp, actions_flat)  # (B·k, 1)

        # Q^soft(s,a') = min(Q1',Q2') − α · log π_φ(a'|s)
        q_soft = (
            torch.min(q1_all, q2_all) - self.alpha.detach() * lp_all
        ).reshape(B, self.k)                                            # (B, k)

        # ── Sort descending: index 0 = best, index 1 = second-best ─────
        _, sorted_idx = torch.sort(q_soft, descending=True, dim=-1)    # (B, k)
        opt_idx = sorted_idx[:, 0]   # best     → GAG anchor
        sub_idx = sorted_idx[:, 1]   # 2nd-best → RDE anchor (most confusable)

        rows    = torch.arange(B, device=self.device)
        a_opt   = neighbor_actions[rows, opt_idx, :]    # (B, A)
        a_sub   = neighbor_actions[rows, sub_idx, :]    # (B, A)

        # ── Target features for RDE ──────────────────────────────────────
        _, fea_sub_1 = self.critic_target.Q1(state, a_sub)
        _, fea_sub_2 = self.critic_target.Q2(state, a_sub)

        return a_opt, a_sub, fea_sub_1, fea_sub_2

    # ── Main training step ────────────────────────────────────────────────────

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ══════════════════════════════════════════════════════════════════
        # Buffer maintenance — FIFO ring-buffer
        #
        # Filled from replay_buffer.sample(), not raw environment steps.
        # This is intentional: sampling from the full replay buffer gives
        # diverse historical coverage across all of training, which maximises
        # the chance of finding a high-quality ã_opt in the kNN lookup.
        # A fresh-env-step-only buffer would narrow the pool to recent actions,
        # losing coverage of historically good regions — contrary to IRA's
        # retrospective exploitation goal.
        # ══════════════════════════════════════════════════════════════════
        with torch.no_grad():
            end = self.buf_idx + batch_size
            if end > self.action_buffer_size:
                left = end - self.action_buffer_size
                self.buff_actions[self.buf_idx:] = action[:batch_size - left]
                self.buff_actions[:left]         = action[batch_size - left:]
                self.buf_idx = left
            else:
                self.buff_actions[self.buf_idx:end] = action
                self.buf_idx = end % self.action_buffer_size

        post_warmup = self.total_it > self.warmup_timestamps

        # ══════════════════════════════════════════════════════════════════
        # Step 1 — Actor forward  (WITH gradients)
        #
        # sampled_action : a ~ π_φ(·|s)  via reparameterisation + tanh
        # log_prob       : log π_φ(a|s)  with squashing correction
        # a_bar          : ā = tanh(μ_φ(s))·max_action  — LIVE GRAPH
        #
        # a_bar is the anchor for GAG and RDE.
        # Whether it is detached depends on the loss being computed:
        #   • critic loss / RDE : a_bar.detach()   (must NOT modify φ)
        #   • actor  loss / GAG : a_bar             (MUST flow into μ_φ)
        # ══════════════════════════════════════════════════════════════════
        sampled_action, log_prob, a_bar = self.actor.sample(state)

        # ══════════════════════════════════════════════════════════════════
        # Step 2 — Retrospective kNN lookup  (NO gradients anywhere)
        #
        # Lookup is computed for all B batch states simultaneously (per-batch),
        # so each batch sample gets its own optimal and suboptimal anchors.
        # This is better than the per-step simplification in the pseudoalgorithm,
        # which described anchors for a single environment state only.
        #
        # Early-training note: for the first warmup_timestamps steps, buffer
        # actions were collected under a near-random policy. log π_φ(a'|s)
        # for off-policy buffer actions could be severely negative, distorting
        # the soft-Q ranking. The warmup guard avoids this; once the policy has
        # stabilised, soft-Q ranking is well-behaved.
        # ══════════════════════════════════════════════════════════════════
        if post_warmup:
            with torch.no_grad():
                a_opt, a_sub, fea_sub_1, fea_sub_2 = \
                    self._retrospective_lookup(state, a_bar)

        # ══════════════════════════════════════════════════════════════════
        # Step 3 — Soft Bellman target  y = r + γ(min Q′ − α log π)
        # ══════════════════════════════════════════════════════════════════
        with torch.no_grad():
            next_a, next_lp, _ = self.actor.sample(next_state)
            tQ1, tQ2, _, _     = self.critic_target(next_state, next_a)
            target_Q = torch.min(tQ1, tQ2) - self.alpha * next_lp
            target_Q = reward + not_done * self.discount * target_Q

        # ══════════════════════════════════════════════════════════════════
        # Step 4 — Critic update
        #
        # L_Q = (Q(s,a) − y)²
        #      + α_RDE · ⟨φ(s, ā_⊥; θ⁺), φ(s, ã_sub; θ′⁺)⟩
        #
        # ā_⊥ = a_bar.detach()  ← CRITICAL: critic must not modify φ.
        #   a_bar tells the encoder *where* to compute a representation.
        #   Without .detach(), critic.backward() would silently update the
        #   actor parameters through the RDE term.
        #
        # fea_sub from target critic (θ′⁺) — also detached, fixed target.
        # ══════════════════════════════════════════════════════════════════
        Q1, Q2, _, _ = self.critic(state, action)
        critic_loss  = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if post_warmup:
            a_bar_det = a_bar.detach()                     # ā_⊥ — no actor grad
            _, fea_abar_1 = self.critic.Q1(state, a_bar_det)
            _, fea_abar_2 = self.critic.Q2(state, a_bar_det)

            # Minimise inner product  ↔  maximise angular separation
            rde_loss = (
                torch.einsum('ij,ij->i', fea_abar_1, fea_sub_1).mean() +
                torch.einsum('ij,ij->i', fea_abar_2, fea_sub_2).mean()
            ) * self.alpha_rde

            critic_loss = critic_loss + rde_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ══════════════════════════════════════════════════════════════════
        # Step 5 — Actor update
        #
        # L_π = α log π_φ(a|s) − min(Q1,Q2)(s,a) + β‖ā − ã_opt‖²
        #
        # Three responsibilities:
        #   • −Q term      : exploit high-value regions (both μ and σ)
        #   • α log π term : maintain entropy   (primarily σ)
        #   • β‖ā−ã_opt‖² : steer mean toward best anchor  (only μ via ā)
        #
        # Governance is primarily decoupled: GAG controls the mean direction;
        # entropy controls the spread. Q-gradients touch both μ and σ weakly
        # through reparameterisation (see paper §4.3 for full discussion).
        #
        # α and β are detached here — their optimisers run separately in Step 6.
        # ══════════════════════════════════════════════════════════════════
        q1_pi, _ = self.critic.Q1(state, sampled_action)
        q2_pi, _ = self.critic.Q2(state, sampled_action)
        min_q_pi  = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        if post_warmup:
            # a_bar NOT detached — gradient flows into μ_φ (that is the point)
            gag_loss   = self.beta.detach() * (a_bar - a_opt).pow(2).sum(dim=-1).mean()
            actor_loss = actor_loss + gag_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ══════════════════════════════════════════════════════════════════
        # Step 6 — Dual updates
        #
        # Both use detached quantities (log_prob.detach(), a_bar.detach())
        # so backward() only touches log_alpha / log_beta respectively.
        #
        # α update:
        #   L_logα = log α · (−log π_φ(a|s) − H_target)
        #   When entropy < H_target: (−log π + H_target) < 0 → α decreases
        #   When entropy > H_target: (−log π + H_target) > 0 → α increases
        #   (opposite sign convention to some implementations — verify with
        #    your replay buffer's log_prob sign before running)
        #
        # β update:
        #   L_logβ = −log β · (‖ā − ã_opt‖² − D_max)
        #   ∂L/∂logβ = −(deviation − D_max)
        #   Gradient-descent step: logβ ← logβ + η·(deviation − D_max)  ✓
        #   Deviation > D_max → β grows  (tighter rubber band)
        #   Deviation < D_max → β shrinks (entropy drives local exploration)
        # ══════════════════════════════════════════════════════════════════

        # α
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # β
        if post_warmup:
            deviation = (a_bar.detach() - a_opt).pow(2).sum(dim=-1).mean()
            beta_loss = -(self.log_beta * (deviation - self.d_max))
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()

        # ══════════════════════════════════════════════════════════════════
        # Step 7 — Soft target update (critic only)
        # Actor has no target network — see constructor note.
        # ══════════════════════════════════════════════════════════════════
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, filename):
        torch.save(self.critic.state_dict(),            filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),  filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(),             filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),   filename + "_actor_optimizer")
        torch.save(self.log_alpha.data,                 filename + "_log_alpha")
        torch.save(self.log_beta.data,                  filename + "_log_beta")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.log_alpha.data = torch.load(filename + "_log_alpha")
        self.log_beta.data  = torch.load(filename + "_log_beta")