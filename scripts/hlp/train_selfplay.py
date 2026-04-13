from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hlp.env import HyperGymSelfPlayConfig, HyperGymTeamEnv
from scripts.hlp.fsp import FSPOpponentPool
from scripts.hlp.policy import TeamHybridActorCritic
from scripts.hlp.reward import TeamRewardConfig, TeamRewardShaper


@dataclass
class RolloutBatch:
    obs: List[np.ndarray]
    skill: List[np.ndarray]
    target: List[np.ndarray]
    log_prob: List[float]
    value: List[float]
    reward: List[float]
    done: List[bool]
    next_obs: List[np.ndarray]


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - float(dones[t])
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def save_checkpoint(path: Path, policy, optimizer, history, config):
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a high-level HyperGym self-play policy with FSP-style opponents.")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="core/checkpoints/high_level/rl/run")
    parser.add_argument("--bc-init-checkpoint", type=str, default="")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--scripted-prob", type=float, default=0.2)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = HyperGymTeamEnv(
        HyperGymSelfPlayConfig(max_steps=args.max_steps, base_seed=args.seed),
        reward_shaper=TeamRewardShaper(TeamRewardConfig()),
    )
    obs_dim = 29
    policy = TeamHybridActorCritic(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    if args.bc_init_checkpoint:
        policy.load_bc_actor(args.bc_init_checkpoint, device=device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opponent_pool = FSPOpponentPool(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
        scripted_prob=args.scripted_prob,
        seed=args.seed,
    )
    opponent_pool.add_snapshot(policy, label="init")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []

    total_env_steps = 0
    best_goal_diff = -10**9
    episode_cursor = 0
    for update_idx in range(args.updates):
        batch = RolloutBatch([], [], [], [], [], [], [], [])
        update_episode_returns: List[float] = []
        update_episode_steps: List[int] = []
        update_goal_for = 0
        update_goal_against = 0

        while len(batch.obs) < args.rollout_steps:
            controlled_team = "home" if episode_cursor % 2 == 0 else "away"
            opponent_team = "away" if controlled_team == "home" else "home"
            opponent = opponent_pool.sample_opponent(team=opponent_team)
            obs = env.reset(
                controlled_team=controlled_team,
                opponent_policy=opponent,
                episode_seed=args.seed + episode_cursor,
            )

            episode_return = 0.0
            episode_steps = 0
            episode_goal_for = 0
            episode_goal_against = 0

            while episode_steps < args.max_steps and len(batch.obs) < args.rollout_steps:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = policy.act(obs_t, deterministic=False)
                skill = action.skill.squeeze(0).cpu().numpy()
                target = action.target.squeeze(0).cpu().numpy()
                next_obs, reward, done, info = env.step(skill, target)

                batch.obs.append(obs)
                batch.skill.append(skill)
                batch.target.append(target)
                batch.log_prob.append(float(action.log_prob.item()))
                batch.value.append(float(action.value.item()))
                batch.reward.append(float(reward))
                batch.done.append(bool(done))
                batch.next_obs.append(next_obs)

                if info.get("winner") == controlled_team:
                    episode_goal_for = 1
                elif info.get("winner") == opponent_team:
                    episode_goal_against = 1

                obs = next_obs
                episode_return += reward
                episode_steps += 1
                total_env_steps += 1
                if done:
                    break

            update_episode_returns.append(episode_return)
            update_episode_steps.append(episode_steps)
            update_goal_for += episode_goal_for
            update_goal_against += episode_goal_against
            episode_cursor += 1

        with torch.no_grad():
            next_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = 0.0 if (batch.done and batch.done[-1]) else float(policy.forward(next_obs_t)["value"].item())

        values = np.asarray(batch.value, dtype=np.float32)
        rewards = np.asarray(batch.reward, dtype=np.float32)
        dones = np.asarray(batch.done, dtype=np.bool_)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
        skill_tensor = torch.as_tensor(np.asarray(batch.skill), dtype=torch.long, device=device)
        target_tensor = torch.as_tensor(np.asarray(batch.target), dtype=torch.float32, device=device)
        old_log_prob_tensor = torch.as_tensor(np.asarray(batch.log_prob), dtype=torch.float32, device=device)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

        num_samples = obs_tensor.shape[0]
        indices = np.arange(num_samples)
        update_stats: Dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        update_count = 0
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, args.minibatch_size):
                mb_idx = indices[start:start + args.minibatch_size]
                batch_eval = policy.evaluate_actions(
                    obs_tensor[mb_idx],
                    skill_tensor[mb_idx],
                    target_tensor[mb_idx],
                )
                log_ratio = batch_eval["log_prob"] - old_log_prob_tensor[mb_idx]
                ratio = log_ratio.exp()
                adv = advantage_tensor[mb_idx]
                pg_loss_1 = -adv * ratio
                pg_loss_2 = -adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                value_loss = F.mse_loss(batch_eval["value"], return_tensor[mb_idx])
                entropy = batch_eval["entropy"].mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                update_stats["policy_loss"] += float(policy_loss.item())
                update_stats["value_loss"] += float(value_loss.item())
                update_stats["entropy"] += float(entropy.item())
                update_count += 1

        for key in update_stats:
            update_stats[key] /= max(update_count, 1)

        row = {
            "update": update_idx + 1,
            "episodes_in_batch": len(update_episode_returns),
            "avg_episode_return": float(np.mean(update_episode_returns)) if update_episode_returns else 0.0,
            "avg_episode_steps": float(np.mean(update_episode_steps)) if update_episode_steps else 0.0,
            "goal_for": update_goal_for,
            "goal_against": update_goal_against,
            "policy_loss": update_stats["policy_loss"],
            "value_loss": update_stats["value_loss"],
            "entropy": update_stats["entropy"],
            "pool_size": len(opponent_pool.snapshots),
            "total_env_steps": total_env_steps,
        }
        history.append(row)
        print(
            f"[hlp] update={row['update']} episodes={row['episodes_in_batch']} "
            f"avg_return={row['avg_episode_return']:.3f} avg_steps={row['avg_episode_steps']:.1f} "
            f"gf={update_goal_for} ga={update_goal_against} "
            f"pi={row['policy_loss']:.4f} vf={row['value_loss']:.4f} ent={row['entropy']:.4f} "
            f"pool={row['pool_size']}"
        )

        if (update_idx + 1) % args.snapshot_interval == 0:
            opponent_pool.add_snapshot(policy, label=f"update_{update_idx + 1}")

        save_checkpoint(save_dir / "last.pt", policy, optimizer, history, vars(args))
        goal_diff = update_goal_for - update_goal_against
        if goal_diff > best_goal_diff:
            best_goal_diff = goal_diff
            save_checkpoint(save_dir / "best_goal_diff.pt", policy, optimizer, history, vars(args))

    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
