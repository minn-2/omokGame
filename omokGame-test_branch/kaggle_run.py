"""
Kaggle bootstrap runner for Omok PPO training.

Usage (inside Kaggle notebook cell):
    !python kaggle_run.py --resume
"""

import argparse
import os
import shutil
from pathlib import Path


WORKING_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
CKPT_DIR = Path("checkpoints")
P2_FILE = CKPT_DIR / "ppo_p2.pt"
LOG_FILE = CKPT_DIR / "train_log.json"


def find_latest_checkpoint_dataset():
    """
    Locate a dataset folder under /kaggle/input that contains checkpoints.
    Returns (pt_path, log_path) or (None, None).
    """
    if not INPUT_DIR.exists():
        return None, None

    candidates = []
    for ds in INPUT_DIR.iterdir():
        if not ds.is_dir():
            continue
        pt = ds / "ppo_p2.pt"
        lg = ds / "train_log.json"
        if pt.exists():
            mtime = pt.stat().st_mtime
            candidates.append((mtime, pt, lg if lg.exists() else None))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, pt, lg = candidates[0]
    return pt, lg


def restore_from_input(force: bool = False):
    """
    Restore checkpoint files from /kaggle/input to local checkpoints/.
    If force=False, only restore when local file is missing.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    src_pt, src_log = find_latest_checkpoint_dataset()
    if src_pt is None:
        print("[KAGGLE] No checkpoint dataset found in /kaggle/input.")
        return

    if force or not P2_FILE.exists():
        shutil.copy(src_pt, P2_FILE)
        print(f"[KAGGLE] Restored checkpoint: {src_pt} -> {P2_FILE}")
    else:
        print(f"[KAGGLE] Local checkpoint exists, skip restore: {P2_FILE}")

    if src_log is not None and (force or not LOG_FILE.exists()):
        shutil.copy(src_log, LOG_FILE)
        print(f"[KAGGLE] Restored log: {src_log} -> {LOG_FILE}")


def backup_to_working():
    """
    Copy current checkpoint files to /kaggle/working for easy download/output capture.
    """
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    for src in (P2_FILE, LOG_FILE):
        if src.exists():
            dst = WORKING_DIR / src.name
            shutil.copy(src, dst)
            print(f"[KAGGLE] Backed up: {src} -> {dst}")


def run_train(resume: bool):
    from PPO_colab import train

    train(resume=resume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument(
        "--force-restore",
        action="store_true",
        help="always restore checkpoint from /kaggle/input if present",
    )
    args = parser.parse_args()

    print("[KAGGLE] Bootstrapping...")
    print(f"[KAGGLE] CWD: {os.getcwd()}")
    restore_from_input(force=args.force_restore)

    print("[KAGGLE] Start training...")
    run_train(resume=args.resume)

    print("[KAGGLE] Final backup...")
    backup_to_working()
    print("[KAGGLE] Done.")


if __name__ == "__main__":
    main()
