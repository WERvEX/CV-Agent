"""Checkpoint registry: Top-N leaderboard, manual saves, and cross-run discovery."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from cv_agent.core.config import CheckpointConfig, HyperParams
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

CHECKPOINTS_DIR = "checkpoints"
LEADERBOARD_FILE = "leaderboard.json"
MANIFEST_FILE = "manifest.json"
WEIGHTS_FILE = "weights.pt"


@dataclass
class CheckpointInfo:
    """A discoverable checkpoint for fork or listing."""

    id: str
    run_dir: Path
    weights_path: Path
    score: float
    round: int
    hyperparams: dict[str, Any]
    kind: Literal["top", "manual", "resume"]
    label: str
    saved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_dir": str(self.run_dir),
            "weights_path": str(self.weights_path),
            "score": self.score,
            "round": self.round,
            "hyperparams": self.hyperparams,
            "kind": self.kind,
            "label": self.label,
            "saved_at": self.saved_at,
        }


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("Checkpoint name cannot be empty.")
    return cleaned[:64]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    """Manage Top-N and manual checkpoints within a run directory."""

    def __init__(self, run_dir: Path, config: CheckpointConfig) -> None:
        self.run_dir = run_dir
        self.config = config
        self.root = run_dir / CHECKPOINTS_DIR
        self.top_dir = self.root / "top"
        self.manual_dir = self.root / config.manual_save_dir
        self.leaderboard_path = self.root / LEADERBOARD_FILE

    def _ensure_dirs(self) -> None:
        self.top_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)

    def _load_leaderboard(self) -> dict[str, Any]:
        if not self.leaderboard_path.exists():
            return {"top_n": self.config.top_n, "entries": []}
        with open(self.leaderboard_path, encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("entries", [])
        data["top_n"] = self.config.top_n
        return data

    def _save_leaderboard(self, data: dict[str, Any]) -> None:
        self._ensure_dirs()
        data["top_n"] = self.config.top_n
        with open(self.leaderboard_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)

    def _rank_filename(self, rank: int, score: float, round_num: int) -> str:
        return f"rank{rank:02d}_score{score:.4f}_round{round_num}.pt"

    def record_score(
        self,
        weights_src: Path,
        score: float,
        round_num: int,
        hyperparams: dict[str, Any],
    ) -> bool:
        """Insert into Top-N if score qualifies; prune and rewrite files."""
        if not self.config.auto_save_top:
            return False
        if not weights_src.exists():
            logger.warning(f"Cannot record Top-N checkpoint — weights missing: {weights_src}")
            return False

        self._ensure_dirs()
        board = self._load_leaderboard()
        entries: list[dict[str, Any]] = list(board.get("entries", []))
        top_n = self.config.top_n

        old_paths_by_round: dict[int, Path] = {}
        for e in entries:
            rel = e.get("weights_path", "")
            if rel:
                p = self.run_dir / rel
                if p.exists():
                    old_paths_by_round[int(e.get("round", -1))] = p

        qualifies = len(entries) < top_n
        if not qualifies and entries:
            min_score = min(float(e["score"]) for e in entries)
            qualifies = score > min_score

        if not qualifies:
            logger.debug(f"Score {score:.4f} did not enter Top-{top_n}.")
            return False

        entries = [e for e in entries if int(e.get("round", -1)) != round_num]
        entries.append({
            "score": float(score),
            "round": int(round_num),
            "hyperparams": hyperparams,
            "saved_at": _utc_now_iso(),
            "weights_path": "",
        })
        entries.sort(key=lambda e: float(e["score"]), reverse=True)
        entries = entries[:top_n]

        for old in self.top_dir.glob("rank*.pt"):
            try:
                old.unlink()
            except OSError as exc:
                logger.warning(f"Failed to remove old top checkpoint {old}: {exc}")

        for rank, entry in enumerate(entries, start=1):
            rel = self._rank_filename(rank, float(entry["score"]), int(entry["round"]))
            dst = self.top_dir / rel
            src_round = int(entry["round"])
            src_path = weights_src if src_round == round_num else old_paths_by_round.get(src_round)
            if src_path is None or not src_path.exists():
                logger.warning(f"No weights source for round {src_round} in Top-N rewrite.")
                continue
            shutil.copy2(src_path, dst)
            entry["rank"] = rank
            entry["weights_path"] = f"{CHECKPOINTS_DIR}/top/{rel}"

        self._save_leaderboard({"top_n": top_n, "entries": entries})
        logger.info(f"Top-{top_n} updated — score {score:.4f} round {round_num} recorded.")
        return True

    def save_manual(
        self,
        name: str,
        weights_src: Path,
        score: float,
        round_num: int,
        hyperparams: dict[str, Any],
    ) -> Path:
        """Save a user-named checkpoint package under checkpoints/manual/."""
        if not weights_src.exists():
            raise FileNotFoundError(f"Weights not found: {weights_src}")

        safe_name = _sanitize_name(name)
        self._ensure_dirs()
        dest_dir = self.manual_dir / safe_name
        if dest_dir.exists():
            raise ValueError(f"Manual checkpoint '{safe_name}' already exists.")
        dest_dir.mkdir(parents=True)

        weights_dst = dest_dir / WEIGHTS_FILE
        shutil.copy2(weights_src, weights_dst)

        manifest = {
            "name": safe_name,
            "score": float(score),
            "round": int(round_num),
            "hyperparams": hyperparams,
            "weights_path": f"{CHECKPOINTS_DIR}/{self.config.manual_save_dir}/{safe_name}/{WEIGHTS_FILE}",
            "saved_at": _utc_now_iso(),
            "kind": "manual",
        }
        manifest_path = dest_dir / MANIFEST_FILE
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Manual checkpoint saved: {dest_dir}")
        return dest_dir

    @staticmethod
    def load_manifest(manifest_path: Path) -> dict[str, Any]:
        """Load and validate a manifest.json; ensure weights exist."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)

        run_dir = manifest_path.parent.parent.parent
        weights_rel = data.get("weights_path", "")
        weights_path = run_dir / weights_rel if weights_rel else manifest_path.parent / WEIGHTS_FILE
        if not weights_path.exists():
            weights_path = manifest_path.parent / WEIGHTS_FILE
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights missing for manifest: {manifest_path}")

        data["_resolved_weights_path"] = str(weights_path)
        return data


def list_resumable_runs(output_root: Path) -> list[Path]:
    """Experiment directories that can be resumed (session_state or legacy artifacts)."""
    if not output_root.exists():
        return []
    runs: list[Path] = []
    for d in sorted(output_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("exp_"):
            continue
        if (d / "session_state.json").exists() or (d / "decision_log.json").exists():
            runs.append(d)
    return runs


def list_checkpoints(output_root: Path) -> list[CheckpointInfo]:
    """Aggregate forkable checkpoints and resumable runs across experiments."""
    results: list[CheckpointInfo] = []
    if not output_root.exists():
        return results

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("exp_"):
            continue

        # Resumable session (not fork — listed for resume wizard)
        session_path = run_dir / "session_state.json"
        if session_path.exists():
            try:
                with open(session_path, encoding="utf-8") as fh:
                    session = json.load(fh)
                best_ckpt = session.get("best_checkpoint")
                weights = run_dir / best_ckpt if best_ckpt else run_dir / "weights" / "best.pt"
                if not weights.exists():
                    weights = run_dir / "weights" / "last.pt"
                if weights.exists():
                    results.append(
                        CheckpointInfo(
                            id=f"{run_dir.name}:resume",
                            run_dir=run_dir,
                            weights_path=weights,
                            score=float(session.get("best_score", 0.0)),
                            round=int(session.get("round_num", 0)),
                            hyperparams=session.get("current_params") or {},
                            kind="resume",
                            label=f"Resume {run_dir.name} (round {session.get('round_num', '?')})",
                            saved_at=None,
                        )
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Skipping resume entry for {run_dir}: {exc}")

        ckpt_root = run_dir / CHECKPOINTS_DIR
        if not ckpt_root.exists():
            continue

        board_path = ckpt_root / LEADERBOARD_FILE
        if board_path.exists():
            try:
                with open(board_path, encoding="utf-8") as fh:
                    board = json.load(fh)
                for entry in board.get("entries", []):
                    rel = entry.get("weights_path", "")
                    weights = run_dir / rel if rel else None
                    if weights is None or not weights.exists():
                        continue
                    rank = entry.get("rank", "?")
                    results.append(
                        CheckpointInfo(
                            id=f"{run_dir.name}:top:{rank}",
                            run_dir=run_dir,
                            weights_path=weights,
                            score=float(entry.get("score", 0.0)),
                            round=int(entry.get("round", 0)),
                            hyperparams=entry.get("hyperparams") or {},
                            kind="top",
                            label=f"Top-{rank} {run_dir.name} score={entry.get('score', 0):.4f}",
                            saved_at=entry.get("saved_at"),
                        )
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Skipping leaderboard for {run_dir}: {exc}")

        manual_root = ckpt_root / "manual"
        if manual_root.exists():
            for manifest_path in sorted(manual_root.glob(f"*/{MANIFEST_FILE}")):
                try:
                    data = CheckpointManager.load_manifest(manifest_path)
                    weights = Path(data["_resolved_weights_path"])
                    name = data.get("name", manifest_path.parent.name)
                    results.append(
                        CheckpointInfo(
                            id=f"{run_dir.name}:manual:{name}",
                            run_dir=run_dir,
                            weights_path=weights,
                            score=float(data.get("score", 0.0)),
                            round=int(data.get("round", 0)),
                            hyperparams=data.get("hyperparams") or {},
                            kind="manual",
                            label=f"Manual '{name}' ({run_dir.name}) score={data.get('score', 0):.4f}",
                            saved_at=data.get("saved_at"),
                        )
                    )
                except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
                    logger.warning(f"Skipping manual checkpoint {manifest_path}: {exc}")

    return results


def find_checkpoint_by_id(output_root: Path, checkpoint_id: str) -> CheckpointInfo | None:
    """Resolve a checkpoint id from list_checkpoints."""
    for info in list_checkpoints(output_root):
        if info.id == checkpoint_id:
            return info
    return None


def hyperparams_from_manifest(hyperparams: dict[str, Any]) -> HyperParams:
    """Build HyperParams from manifest dict, ignoring unknown keys."""
    keys = set(HyperParams.model_fields.keys())
    filtered = {k: v for k, v in hyperparams.items() if k in keys}
    return HyperParams(**filtered)
