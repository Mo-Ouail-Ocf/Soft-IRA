from __future__ import annotations

from pathlib import Path

import numpy as np
import tqdm.auto

from ira_rl.agents.factory import build_agent
from ira_rl.training.checkpointing import CheckpointManager
from ira_rl.training.envs import (
    compute_discounted_returns,
    make_env,
    reset_env,
    seed_action_space,
    seed_everything,
    step_env,
)
from ira_rl.training.logging_utils import RunLogger
from ira_rl.utils import ReplayBuffer


def build_run_name(cfg) -> str:
    return f"{cfg.algorithm.name}_{cfg.environment.name}_seed_{cfg.run.seed}_{cfg.run.experiment_name}"


def evaluate_agent(agent, cfg, step: int) -> dict[str, float]:
    env = make_env(cfg.environment.name)
    seed_action_space(env, cfg.run.seed + 100)

    episode_rewards = []
    episode_lengths = []
    predicted_q_values: list[float] = []
    monte_carlo_returns: list[float] = []

    for episode in range(cfg.run.eval_episodes):
        state = reset_env(env, seed=cfg.run.seed + 100 + episode)
        done = False
        rewards = []
        q_predictions = []

        while not done:
            action = agent.select_action(np.asarray(state, dtype=np.float32))
            if cfg.run.track_overestimation and hasattr(agent, "estimate_q"):
                q_predictions.append(agent.estimate_q(np.asarray(state, dtype=np.float32), action))
            next_state, reward, done, _ = step_env(env, action)
            rewards.append(float(reward))
            state = next_state

        episode_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))
        if cfg.run.track_overestimation and q_predictions:
            discounted_returns = compute_discounted_returns(rewards, cfg.algorithm.discount)
            predicted_q_values.extend(q_predictions)
            monte_carlo_returns.extend(discounted_returns)

    env.close()

    metrics = {
        "eval/avg_reward": float(np.mean(episode_rewards)),
        "eval/std_reward": float(np.std(episode_rewards)),
        "eval/avg_episode_length": float(np.mean(episode_lengths)),
        "eval/step": float(step),
    }
    if predicted_q_values and monte_carlo_returns:
        predicted = np.asarray(predicted_q_values, dtype=np.float32)
        actual = np.asarray(monte_carlo_returns, dtype=np.float32)
        metrics.update(
            {
                "eval/predicted_q_mean": float(predicted.mean()),
                "eval/monte_carlo_return_mean": float(actual.mean()),
                "eval/overestimation_bias": float((predicted - actual).mean()),
                "eval/q_return_mse": float(np.square(predicted - actual).mean()),
            }
        )
    return metrics


def _select_training_action(agent, state: np.ndarray, cfg, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    if getattr(agent, "is_stochastic", False):
        return agent.sample_action(state)
    action = agent.select_action(state)
    if cfg.algorithm.exploration_noise > 0:
        noise_scale = (action_high - action_low) / 2.0
        noise = np.random.normal(0.0, cfg.algorithm.exploration_noise * noise_scale, size=action.shape)
        action = np.clip(action + noise, action_low, action_high)
    return action


def train(cfg) -> None:
    run_name = build_run_name(cfg)
    output_dir = Path(cfg.paths.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.run.seed)

    env = make_env(cfg.environment.name)
    eval_env = None
    seed_action_space(env, cfg.run.seed)
    state = reset_env(env, seed=cfg.run.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    agent = build_agent(
        cfg.algorithm,
        state_dim,
        action_dim,
        max_action,
        cfg.run.device,
        total_timesteps=int(cfg.run.total_timesteps),
    )
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=cfg.run.replay_buffer_size,
        device=cfg.run.device,
    )
    logger = RunLogger(cfg, run_name=run_name, output_dir=output_dir)
    checkpoint_manager = CheckpointManager(output_dir / "checkpoints")

    evaluations: list[float] = []
    initial_metrics = evaluate_agent(agent, cfg, step=0)
    logger.log_metrics(initial_metrics, step=0)
    evaluations.append(initial_metrics["eval/avg_reward"])

    episode_reward = 0.0
    episode_steps = 0
    episode_number = 0

    progress = tqdm.auto.tqdm(range(int(cfg.run.total_timesteps)), desc=f"Training {run_name}")
    for step in progress:
        episode_steps += 1
        if step < cfg.run.start_timesteps:
            action = env.action_space.sample()
        else:
            action = _select_training_action(
                agent,
                np.asarray(state, dtype=np.float32),
                cfg,
                action_low,
                action_high,
            )

        next_state, reward, done, _ = step_env(env, action)
        replay_buffer.add(state, action, next_state, reward, float(done))
        state = next_state
        episode_reward += float(reward)

        if replay_buffer.size >= cfg.algorithm.batch_size and step >= cfg.run.start_timesteps:
            train_metrics = agent.train(replay_buffer, cfg.algorithm.batch_size)
            logger.log_metrics(train_metrics, step=step + 1)

        if done:
            logger.log_metrics(
                {
                    "episode/reward": episode_reward,
                    "episode/length": episode_steps,
                    "episode/index": episode_number,
                },
                step=step + 1,
            )
            state = reset_env(env)
            episode_reward = 0.0
            episode_steps = 0
            episode_number += 1

        if (step + 1) % cfg.run.eval_frequency == 0:
            eval_metrics = evaluate_agent(agent, cfg, step=step + 1)
            logger.log_metrics(eval_metrics, step=step + 1)
            evaluations.append(eval_metrics["eval/avg_reward"])
            np.save(output_dir / "evaluations.npy", np.asarray(evaluations, dtype=np.float32))
            checkpoint_manager.save_last(
                agent,
                step=step + 1,
                score=eval_metrics["eval/avg_reward"],
                metadata={"run_name": run_name},
            )
            checkpoint_manager.maybe_save_best(
                agent,
                step=step + 1,
                score=eval_metrics["eval/avg_reward"],
                metadata={"run_name": run_name},
            )
            progress.set_postfix(
                eval_reward=f"{eval_metrics['eval/avg_reward']:.2f}",
                best=f"{checkpoint_manager.best_score:.2f}",
            )

    if evaluations:
        checkpoint_manager.save_last(
            agent,
            step=int(cfg.run.total_timesteps),
            score=evaluations[-1],
            metadata={"run_name": run_name},
        )

    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()
