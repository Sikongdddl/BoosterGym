from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.imitation.datasets.hypergym_vlm_dataset import (
    TeamActionBCDataset,
    load_bc_samples,
    split_bc_samples,
    summarize_bc_samples,
)
from core.imitation.models.bc_policy import TeamActionBCPolicy


def _compute_metrics(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    skill_logits = outputs["skill_logits"]
    target_pred = outputs["target_pred"]
    skill_targets = batch["skill_targets"]
    target_targets = batch["target_targets"]

    skill_loss = F.cross_entropy(skill_logits.reshape(-1, skill_logits.shape[-1]), skill_targets.reshape(-1))
    target_loss = F.mse_loss(target_pred, target_targets)
    total_loss = skill_loss + target_loss

    pred_skills = skill_logits.argmax(dim=-1)
    skill_acc = (pred_skills == skill_targets).float().mean().item()
    target_mae = (target_pred - target_targets).abs().mean().item()
    return {
        "loss": float(total_loss.item()),
        "skill_loss": float(skill_loss.item()),
        "target_loss": float(target_loss.item()),
        "skill_acc": float(skill_acc),
        "target_mae": float(target_mae),
    }


def _run_epoch(
    model: TeamActionBCPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "skill_loss": 0.0, "target_loss": 0.0, "skill_acc": 0.0, "target_mae": 0.0}
    steps = 0
    for batch in loader:
        obs = batch["obs"].to(device)
        skill_targets = batch["skill_targets"].to(device)
        target_targets = batch["target_targets"].to(device)
        outputs = model(obs)
        skill_loss = F.cross_entropy(outputs["skill_logits"].reshape(-1, outputs["skill_logits"].shape[-1]), skill_targets.reshape(-1))
        target_loss = F.mse_loss(outputs["target_pred"], target_targets)
        loss = skill_loss + target_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            metrics = _compute_metrics(
                {"skill_targets": skill_targets, "target_targets": target_targets},
                outputs,
            )
        for key in totals:
            totals[key] += metrics[key]
        steps += 1

    if steps == 0:
        raise RuntimeError("DataLoader yielded zero batches.")
    return {key: value / steps for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a state-only behavior cloning policy for HyperGym.")
    parser.add_argument("--dataset-roots", type=str, required=True, help="Comma-separated dataset root directories.")
    parser.add_argument("--include-teams", type=str, default="home,away")
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--only-query-steps", action="store_true")
    parser.add_argument("--only-vlm", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="core/checkpoints/high_level/bc/run")
    parser.add_argument("--init-checkpoint", type=str, default="", help="Optional checkpoint path to resume or initialize from.")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    dataset_roots = [item.strip() for item in args.dataset_roots.split(",") if item.strip()]
    include_teams = [item.strip() for item in args.include_teams.split(",") if item.strip()]
    samples = load_bc_samples(
        dataset_roots,
        include_teams=include_teams,
        include_fallback=bool(args.include_fallback),
        only_query_steps=bool(args.only_query_steps),
        only_vlm=bool(args.only_vlm),
    )
    train_samples, val_samples = split_bc_samples(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))
    train_summary = summarize_bc_samples(train_samples)
    val_summary = summarize_bc_samples(val_samples) if val_samples else {"num_samples": 0}

    train_dataset = TeamActionBCDataset(train_samples)
    val_dataset = TeamActionBCDataset(val_samples) if val_samples else None
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers)) if val_dataset is not None else None

    obs_dim = train_summary["obs_dim"]
    model = TeamActionBCPolicy(
        obs_dim=obs_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_players=2,
        num_skills=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    best_path = save_dir / "best.pt"
    start_epoch = 1

    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        if history:
            start_epoch = int(history[-1]["epoch"]) + 1
        best_val_loss = float(
            min(
                [row["val"]["loss"] for row in history if row.get("val") and "loss" in row["val"]],
                default=float("inf"),
            )
        )
        print(f"[bc] resumed from {checkpoint_path} at epoch={start_epoch}")

    for epoch in range(start_epoch, start_epoch + int(args.epochs)):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        val_metrics = _run_epoch(model, val_loader, None, device) if val_loader is not None else {}
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        message = (
            f"[bc] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_skill_acc={train_metrics['skill_acc']:.4f}"
        )
        if val_metrics:
            message += (
                f" val_loss={val_metrics['loss']:.4f} val_skill_acc={val_metrics['skill_acc']:.4f}"
            )
        print(message)

        if val_metrics:
            current_val_loss = float(val_metrics["loss"])
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": vars(args),
                        "obs_dim": obs_dim,
                        "train_summary": train_summary,
                        "val_summary": val_summary,
                        "history": history,
                    },
                    best_path,
                )

    final_path = save_dir / "last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "obs_dim": obs_dim,
            "train_summary": train_summary,
            "val_summary": val_summary,
            "history": history,
        },
        final_path,
    )
    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "train_summary": train_summary,
                "val_summary": val_summary,
                "dataset_roots": dataset_roots,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
