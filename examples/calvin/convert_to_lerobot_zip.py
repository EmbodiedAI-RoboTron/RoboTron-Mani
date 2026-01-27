"""
Convert CALVIN `.zip` dataset (e.g. `task_D_D.zip`) to LeRobot format **without extracting**.

This script reads `episode_XXXXXXX.npz` step files directly from the zip via `zipfile`,
and uses `ep_start_end_ids.npy` to group steps into LeRobot episodes.
Each (start_id, end_id) pair becomes one complete episode.

Usage:
python3 examples/calvin/convert_to_lerobot.py --zip-path /path/to/task_D_D.zip

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
python3 examples/calvin/convert_to_lerobot.py --zip-path /path/to/task_D_D.zip --push-to-hub

The resulting dataset will get saved to `$LEROBOT_HOME/<repo_id>`.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
import numpy as np
# from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_HOME = Path("/mnt/data/datasets/lerobot")

@dataclass(frozen=True)
class Args:
    zip_path: str
    repo_id: str = "calvin/task_D_D"
    fps: int = 10
    splits: Literal["training", "validation", "both"] = "training"
    action_key: Literal["rel_actions", "actions"] = "rel_actions"
    max_episodes: int | None = None  # for debugging
    push_to_hub: bool = False


def _load_npy_from_zip(z: zipfile.ZipFile, name: str) -> np.ndarray:
    # `np.load` supports file-like objects; ZipExtFile works, and avoids extra copies.
    with z.open(name, "r") as f:
        return np.load(f, allow_pickle=True)


def _iter_episode_lang(
    z: zipfile.ZipFile, split: Literal["training", "validation"]
) -> list[tuple[int, int]]:
    base = f"task_D_D/{split}"
    lang_data = _load_npy_from_zip(z, f"{base}/lang_annotations/auto_lang_ann.npy").item()
    # shape: (num_eps, 2), inclusive bounds
    return lang_data


def _load_step_npz_from_zip(
    z: zipfile.ZipFile, split: Literal["training", "validation"], step_id: int
) -> dict[str, np.ndarray]:
    name = f"task_D_D/{split}/episode_{step_id:07d}.npz"
    with z.open(name, "r") as f:
        npz = np.load(f, allow_pickle=True)
        try:
            return {k: npz[k] for k in npz.files}
        finally:
            npz.close()


def main(args: Args):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (15,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    splits: list[Literal["training", "validation"]]
    if args.splits == "both":
        splits = ["training", "validation"]
    else:
        splits = [args.splits]

    with zipfile.ZipFile(args.zip_path, "r") as z:
        for split in splits:
            lang_data = _iter_episode_lang(z, split)
            ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
            lang_ann = lang_data["language"]["ann"]  # length total number of annotations
            lang_task = lang_data["language"]["task"]
            for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
                task = lang_ann[i]
                for idx in range(start_idx, end_idx + 1):
                    step = _load_step_npz_from_zip(z, split, idx)

                    # state = np.concatenate([step["robot_obs"], step["scene_obs"]], axis=-1).astype(np.float32)
                    dataset.add_frame(
                        {
                            "image": step["rgb_static"],
                            "wrist_image": step["rgb_gripper"],
                            "state": step["robot_obs"].astype(np.float32),  # 15个
                            "actions": step[args.action_key].astype(np.float32),
                            "task": task,
                        }
                    )

                dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["calvin", "task_D_D"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
