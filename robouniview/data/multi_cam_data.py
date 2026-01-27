"""
Multi-camera dataset loading and processing module.

This module provides dataset classes for loading and processing multi-camera
robotics data from various sources (Calvin, MetaWorld, Libero, RoboCasa, etc.).
"""

# Standard library imports
import copy
import functools
import glob
import io
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from multiprocessing import Value
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import braceexpand
import cv2
import h5py
import hydra
import numpy as np
import open3d as o3d
import pybullet
import pyhash
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation, Rotation as scipyR
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# Local application imports
from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    lookup_naming_pattern,
    process_actions,
    process_depth,
    process_state,
)
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    print_and_save,
)
from calvin_env.envs.play_table_env import get_env
from robouniview.data.data_utils import (
    ColorJitter_ctm,
    OccupancyVFE,
    RandomShiftsAug,
    cam,
    deproject,
    get_gripper_camera_view_matrix,
)
from robouniview.data.vl_dataset import CaptionDataset, VQADataset
import robouniview.data.transform_utils as robosuite_trans

# Optional imports
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# Module-level constants
Image.MAX_IMAGE_PIXELS = 1000000000

# Module-level variables
hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)




class LeRobotDataset(lerobot_dataset.LeRobotDataset):
    def __init__(self, repo_id: str, *args, **kwargs):
        super().__init__(repo_id, *args, **kwargs)

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item


def load_from_zip(zip_ref, filename):
    with zip_ref.open(filename) as file:
        return io.BytesIO(file.read())


def get_validation_window_size(
    idx: int, min_window_size: int, max_window_size: int
) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }      
)

prop_state = DictConfig(
    {
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)


keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
                'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix','static_fov', 'gripper_fov', 'gripper_cam2gripper'] 

# Common keys/constants
RGB_KEYS = ['rgb_static', 'rgb_gripper']
# Flip signs on Y/Z axes in homogeneous 4x4 matrices to match deproject conventions
SIGN_FLIP_4X4 = np.array([[1, 1, 1, 1],
                          [-1, -1, -1, -1],
                          [-1, -1, -1, -1],
                          [1, 1, 1, 1]])
# Axis swap rotation used to align axis conventions, with homogeneous coordinate preserved
AXIS_SWAP_R = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

# ---- Helpers (keep minimal to avoid broad behavior changes) ----
def invert_and_flip(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix) * SIGN_FLIP_4X4

def apply_translate(matrix: np.ndarray, translate_4x4: np.ndarray) -> np.ndarray:
    return np.dot(matrix, translate_4x4)

def apply_rotate(matrix: np.ndarray, rotate_4x4: np.ndarray) -> np.ndarray:
    return np.dot(matrix, rotate_4x4)

def compute_fov_from_intrinsics(intrinsic_matrix: np.ndarray, image_width: int) -> float:
    fx = intrinsic_matrix[0, 0]
    return 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi

def depth_to_meters(depth: np.ndarray, near: float, far: float, mode: str = "linear") -> np.ndarray:
    """
    Convert depth to metric distance.
    mode:
      - 'linear': near + depth * (far - near)
      - 'gl': near / (1 - depth * (1 - near / far))  # typical OpenGL clip mapping
    """
    if mode == "linear":
        return near + depth * (far - near)
    elif mode == "gl":
        return near / (1 - depth * (1 - near / far))
    else:
        return depth

def apply_color_jitter_pair(jitter_fn, img_static, img_gripper, jitter_state=None):
    """
    Apply identical color jitter augmentation to a pair of images, tracking state.
    jitter_fn should return (img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
    """
    if jitter_state is None:
        img_static, *jitter_state = jitter_fn(img_static)
    else:
        img_static, *jitter_state = jitter_fn(img_static, *jitter_state)
    img_gripper, *jitter_state = jitter_fn(img_gripper, *jitter_state)
    return img_static, img_gripper, tuple(jitter_state)

class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        with open('/mnt/data/yanfeng/project/RoboTron-Mani/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict[str, Callable],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
       
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            if window_size == 0 and seq_idx == 0:  # single file loader
                seq_rgb_obs_ = rgb_obs
            else:  # episode loader 
                seq_rgb_obs_ = rgb_obs[seq_idx : seq_idx + window_size]
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = torch.stack([transforms[rgb_obs_key](img) for img in seq_rgb_obs_])
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        return {"rgb_obs": seq_rgb_obs_dict}
    
    def process_calib(
        self,
        episode: Dict[str, np.ndarray],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        seq_calib_obs_dict = {}
        for _, calib_obs_key in enumerate(keys_calib):
            calib_obs = episode[calib_obs_key]
            if window_size == 0 and seq_idx == 0:  # single file loader
                seq_calib_obs_ = torch.from_numpy(calib_obs)
            else:
                seq_calib_obs_ = torch.from_numpy(calib_obs[seq_idx : seq_idx + window_size])
            seq_calib_obs_dict[calib_obs_key] = seq_calib_obs_
        
        return {"calib_obs": seq_calib_obs_dict}
    
    def process_pcd(
        self,
        episode: Dict[str, np.ndarray],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        keys_pcd = ['pcd'] 
        seq_pcd_obs_dict = {}
        for _, pcd_obs_key in enumerate(keys_pcd):
            pcd_obs = episode[pcd_obs_key]
            if window_size == 0 and seq_idx == 0:  # single file loader
                seq_pcd_obs_ = torch.from_numpy(pcd_obs)
            else:
                seq_pcd_obs_ = torch.from_numpy(pcd_obs[seq_idx : seq_idx + window_size])
            seq_pcd_obs_dict[pcd_obs_key] = seq_pcd_obs_
        
        return {"pcd_obs": seq_pcd_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        resample = True
        max_data_fetch_iteration = 300

        for fetch_iteration in range(max_data_fetch_iteration):
            try:
            # if 1:
                if not isinstance(idx, Tuple):
                    # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
                    # acts like Constant dataset. Currently, used for language data
                    if self.min_window_size == self.max_window_size:
                        window_size = self.max_window_size
                    elif self.min_window_size < self.max_window_size:
                        window_size = self._get_window_size(idx)
                    else:
                        logger.error(
                            f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                        )
                        raise ValueError
                else:
                    idx, window_size = idx
        
                head = False
                sequence = self._get_sequences(idx, window_size, head=head)
                if sequence == 0:
                    idx = random.randint(0, len(self) - 1)
                    logger.info(
                            f"env_resample"
                        )
                else:
                    if self.pad:
                        pad_size = self._get_pad_size(sequence)
                        sequence = self._pad_sequence(sequence, pad_size, head=head)
                    return sequence
            
            except Exception as e:
                print(f"Resample warning:dataset warning{fetch_iteration}, {e}")
            if resample:
                idx = random.randint(0, len(self) - 1)
            else:
                return None

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        episode = self._load_episode(idx, window_size)
        if episode == 0:
            return 0
        
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )  # 保持不变，取episode['robot_obs']，并截取前15个
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)  # 执行self.transforms[rgb_obs_key]
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)  # 直接取depth
        seq_acts = process_actions(episode, self.observation_space, self.transforms)  # 直接取rel_actions
        info = get_state_info_dict(episode)
        seq_calib_obs = self.process_calib(episode)
        seq_pcd_obs = self.process_pcd(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_calib_obs,
            **seq_pcd_obs,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        # 'robot_obs' =
        # tensor([[-0.2471, -0.1064,  0.4983,  3.1055,  0.0572,  1.7654,  0.0405, -0.7534,
        #         0.8667,  2.2012, -2.5975, -0.7346,  1.8360,  0.9573, -1.0000],
        #         [-0.2430, -0.1086,  0.5053,  3.1127,  0.0481,  1.7694,  0.0405, -0.7696,
        #         0.8897,  2.2128, -2.5932, -0.7307,  1.8190,  0.9407, -1.0000],
        #         [-0.2374, -0.1108,  0.5122,  3.1193,  0.0386,  1.7716,  0.0405, -0.7866,
        #         0.9118,  2.2220, -2.5871, -0.7281,  1.8022,  0.9227, -1.0000]])
        # 'rgb_obs' =
        # {'rgb_static': tensor([[[[1.4340, 1.4340, 1.4340,  ..., 1.9157, 1.9157, 1.9157],
        #         [1.4340, ... 1.0367,  ..., 1.5913, 1.5913, 1.5913]]]]), 'rgb_gripper': tensor([[[[ 0.5435,  0.5435,  0.5435,  ...,  0.6895,  0.6895,  0.6895],
        #         [ 0...2074,  ..., -0.5275, -0.5133, -0.5133]]]])}
        # 'depth_obs' =
        # {}
        # 'actions' =
        # tensor([[-0.1116, -0.2023,  0.3489,  0.1768, -0.1510,  0.0718, -1.0000],
        #         [-0.1072, -0.2796,  0.3480,  0.1674, -0.1596,  0.0377, -1.0000],
        #         [-0.1121, -0.2847,  0.3103,  0.1461, -0.1293,  0.0271, -1.0000]])
        # 'state_info' =
        # {'robot_obs': tensor([[-0.2471, -0.1064,  0.4983,  3.1055,  0.0572,  1.7654,  0.0405, -0.7534,
        #     ...27, -1.0000]],
        #     dtype=torch.float64), 'scene_obs': tensor([[ 2.5973e-01,  1.9946e-01,  1.4407e-19,  1.6662e-16,  0.0000e+00,
        #         0...-1.7141e+00]],
        #     dtype=torch.float64)}
        # 'use_for_aux_lang_loss' =
        # False
        # 'lang' =
        # 'grasp the red block, then lift it up'
        # 'calib_obs' =
        # {'static_extrinsic_matrix': tensor([[[ 0.7237,  0.5135, -0.4611,  0.2127],
        #         [ 0.0299, -0.6909, -0.7224,  ...  0.0000,  1.0000]]], dtype=torch.float64), 'static_intrinsic_matrix': tensor([[[1.1430e+03, 0.0000e+00, 1.0000e+02],
        #         [0.0000e+00, 1.1430e+03, 1.00...0e+00, 1.0000e+00]]], dtype=torch.float64), 'static_distCoeffs_matrix': tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0.],
        # ...0., 0., 0., 0., 0.]], dtype=torch.float64), 'gripper_extrinsic_matrix': tensor([[[ 0.9802,  0.1949,  0.0356,  0.2449],
        #         [ 0.1980, -0.9569, -0.2124, -...  0.0000,  1.0000]]], dtype=torch.float64), 'gripper_intrinsic_matrix': tensor([[[54.7355,  0.0000, 42.0000],
        #         [ 0.0000, 54.7355, 42.0000],
        #         ...  0.0000,  1.0000]]], dtype=torch.float64), 'gripper_distCoeffs_matrix': tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0.],
        # ...0., 0., 0., 0., 0.]], dtype=torch.float64), 'state_matrix': tensor([[[ 0.9802,  0.1949,  0.0356,  0.2449],
        # ...
        # 'pcd_obs' =
        # {'pcd': tensor([[[[[0., 0., 0., 0.],
        #         [0., 0., 0., 0.],
        #         [0., 0., 0., 0.]... 0., 0.],
        #         [0., 0., 0., 0.]]]]])}
        # 'idx' =
        # 5462
        # len() =
        # 10
        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int: #
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int: #yiyang_question
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)}) # YF：原来不传入head参数，为啥robot_obs不使用head呢？
        seq.update(
            {
                "calib_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["calib_obs"].items()
                }
            }
        )
        seq.update(
            {
                "pcd_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["pcd_obs"].items()
                }
            }
        )
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head), # XYZ补0，爪子不动
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.
        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info


class DebugDataset(Dataset):
    def __init__(self, **kwargs: Any,):
        super().__init__()

    def __len__(self) -> int:
        return 10000
    
    def __getitem__(self, index):
        window_size = 8
        rgb = torch.randn(window_size, 3, 200, 200)
        gripper = torch.randn(window_size, 84, 84)
        state = torch.randn(window_size, 15)


class BaseMultiDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        colour_aug=[0,0,0,0],
        data_path_list = [],
        state_matrixs_path = '',
        data_tasks_groups = None,
        env_resample = False,
        only_single_task=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.env_resample = env_resample
        self.data_tasks_groups = data_tasks_groups
        self.state_matrixs_path = state_matrixs_path
        self.save_format = save_format
        self.data_path_list = data_path_list
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.colour_aug = colour_aug
        if sum(colour_aug) ==0:
            self.use_colour_aug = False
        else:
            self.use_colour_aug = True
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.only_single_task = only_single_task

        self.ColorJitter = ColorJitter_ctm(colour_aug[0], colour_aug[1], colour_aug[2], colour_aug[3])
        voxel_range = [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]
        voxel_size = [0.0125, 0.0125, 0.0125*2]
        self.vfe_generator = OccupancyVFE(voxel_range, voxel_size)
    
        self.tmp=[] 
        self.data_name = None

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = False
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    def _prepare_eposides(self, episodes):
        """
        Build a single episode dict by fusing per-frame fields, computing point clouds,
        optional color augmentation, and packing calibration matrices consistently.
        """
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        # rgb handled separately
        keys.remove("rgb_static")
        keys.remove("rgb_gripper")

        calibs = []
        pcds = []
        rgbs = []
        colour_aug_random = random.randint(0, 10)
        # Track color jitter state so both cameras receive identical transform per frame
        jitter_state = None

        for i, ep in enumerate(episodes):
            rgb = {}
            calib = ep['calib']
            cam_config = ep['cam_config']
            static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
            gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']

            static_cam = cam(static_extrinsic_matrix, cam_config['static']['height'], cam_config['static']['width'], cam_config['static']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix, cam_config['gripper']['height'], cam_config['gripper']['width'], cam_config['gripper']['fov'])
            static_pcd = deproject(static_cam, ep['depth_static'], homogeneous=False, sanity_check=False).transpose(1, 0)
            gripper_pcd = deproject(gripper_cam, ep['depth_gripper'], homogeneous=False, sanity_check=False).transpose(1, 0)
            cloud = np.concatenate([static_pcd, gripper_pcd], axis=0)

            rgb['rgb_static'] = ep['rgb_static']
            rgb['rgb_gripper'] = ep['rgb_gripper']

            if colour_aug_random > 2 and self.use_colour_aug:
                rgb['rgb_static'], rgb['rgb_gripper'], jitter_state = apply_color_jitter_pair(
                    self.ColorJitter, rgb['rgb_static'], rgb['rgb_gripper'], jitter_state
                )

            static_rgb = np.reshape(np.array(rgb['rgb_static']), (-1, 3))
            gripper_rgb = np.reshape(np.array(rgb['rgb_gripper']), (-1, 3))
            pcd_rgb = np.concatenate([static_rgb, gripper_rgb], axis=0) / 255
            pcd = self.vfe_generator.generate(cloud[:, :3], pcd_rgb)
            if 0:
                _pcd = o3d.geometry.PointCloud()
                point_tmp, rgb_tmp = self.vfe_generator.decode_occupied_grid_with_range(pcd)
                _pcd.points = o3d.utility.Vector3dVector(point_tmp)
                _pcd.colors = o3d.utility.Vector3dVector(rgb_tmp)
                o3d.io.write_point_cloud("tmp.pcd", _pcd)
                _pcd.points = o3d.utility.Vector3dVector(static_pcd)
                _pcd.colors = o3d.utility.Vector3dVector(static_rgb/255)
                o3d.io.write_point_cloud("tmp1.pcd", _pcd)
                _pcd.points = o3d.utility.Vector3dVector(gripper_pcd)
                _pcd.colors = o3d.utility.Vector3dVector(gripper_rgb/255)
                o3d.io.write_point_cloud("tmp2.pcd", _pcd)
            
            calib['static_extrinsic_matrix'] = static_extrinsic_matrix * SIGN_FLIP_4X4
            calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
            calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']
            calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix * SIGN_FLIP_4X4
            calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
            calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']
            calib['state_matrix'] = ep['state_matrix'] * SIGN_FLIP_4X4
            calib['gripper_cam2gripper'] = ep['gripper_cam2gripper'] * SIGN_FLIP_4X4
            calib['static_fov'] = np.array([cam_config['static']['fov'], cam_config['static']['height'], cam_config['static']['width']])
            calib['gripper_fov'] = np.array([cam_config['gripper']['fov'], cam_config['gripper']['height'], cam_config['gripper']['width']])
            rgbs.append(rgb)
            pcds.append(pcd)
            calibs.append(calib)

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode.update({key: np.stack([calib[key] for calib in calibs]) for key in keys_calib})
        episode.update({key: [rgb[key] for rgb in rgbs] for key in RGB_KEYS})
        episode['pcd'] = np.stack(pcds)
        return episode


class DiskCalvinDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task,
                self.ep_start_end_ids
            ) = self._build_file_indices_lang(self.abs_datasets_dir) 
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)
        
        if self.env_resample and 'training' in str(self.abs_datasets_dir):
            self.scene_info = np.load(
                    f'{self.abs_datasets_dir}/scene_info.npy',
                    allow_pickle=True
                ).item()
            
        if 'task_D_D' in str(self.abs_datasets_dir):
            pass
        else:
            self.data_path_dic = {}
            for data_path in self.data_path_list:
                self.data_path_dic[data_path] = {}
                files = glob.glob(f"{data_path}/training_npz_pcd_new/*/episode_*")
                for file in files:
                    self.data_path_dic[data_path][file.split('/')[-1]] = file

    def _get_episode_name_split(self, file_idx: int, data_path) -> Path:

        if len(self.data_path_list)>0:
            #data_path = self.data_path_list[1]
            #data_path = random.choice(self.data_path_list)
            if self.validation:
                if 'task_D_D' in str(self.abs_datasets_dir):
                    return Path(f"{data_path}/validation/episode_{file_idx:0{7}d}.npz")
                else:
                    files = glob.glob(f"{data_path}/validation/*/episode_{file_idx:0{7}d}.npz")
                    file = files[0]
                    return Path(file)
            else:
                if 'task_D_D' in str(self.abs_datasets_dir):
                    return Path(f"{data_path}/training/episode_{file_idx:0{7}d}.npz")
                else:
                   
                    file = self.data_path_dic[data_path][f"episode_{file_idx:0{7}d}.npz"]
                    return Path(file)
                    
        if 'task_D_D' in str(self.abs_datasets_dir):
            return Path(
                f"{self.abs_datasets_dir}/episode_{file_idx:0{7}d}.npz"
            )
        else:
            files = glob.glob(f"{data_path}/training_npz_pcd_new/*/episode_{file_idx:0{7}d}.npz")
            file = files[0]
            return Path(file)

    def load_pcd(self,pcd_name):
        return o3d.io.read_point_cloud(str(pcd_name))

    def _get_episode(self, file_idx, data_path):

        episode = self.load_file(self._get_episode_name_split(file_idx, data_path))
        episode = {key: episode[key] for key in episode}
        # state_matrix = self.load_file(self._get_episode_name_split(file_idx, self.state_matrixs_path))
        # state_matrix = state_matrix['calib'].item()
        # state_matrix = state_matrix['rgb_gripper']['extrinsic_matrix']

        episode['rgb_static'] = Image.fromarray(episode['rgb_static'])
        episode['rgb_gripper'] = Image.fromarray(episode['rgb_gripper'])
        episode['state_matrix'] = episode['calib'].item()['rgb_gripper']['extrinsic_matrix'] #  state_matrix
        episode['calib'] = episode['calib'].item()
        episode['cam_config'] = episode['cam_config'].item()
        # episode['gripper_cam2gripper'] = np.array([[-7.96325957e-04, 9.99999676e-01, 6.63332084e-09, -7.96272678e-05],
        #                                            [9.63557976e-01, 7.67306153e-04, -2.67498860e-01, 5.89059483e-02],
        #                                            [-2.67498777e-01, -2.13013621e-04, -9.63558266e-01, -1.61648083e-01],
        #                                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        episode['gripper_cam2gripper'] = np.array([[-0.0008, 0.9999, 0., -0.0001],
                                                   [0.9635, 0.0008, -0.2674, 0.0589],
                                                   [-0.2674, 0.0000, -0.9635, -0.1616],
                                                   [0., 0., 0., 1.]])
        if 1:
            robot_obs_ort = np.array(pybullet.getMatrixFromQuaternion(pybullet.getQuaternionFromEuler(episode['robot_obs'][3:6]))).reshape(3,3)
            actions_ort = np.array(pybullet.getMatrixFromQuaternion(pybullet.getQuaternionFromEuler(episode['actions'][3:6]))).reshape(3,3)
            # 此处注意假如坐标系旋转的话需要乘旋转矩阵
            B = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            robot_obs_ort = np.dot(B, robot_obs_ort) # 计算新的旋转矩阵A_new
            actions_ort = np.dot(B, actions_ort) # 计算新的旋转矩阵A_new
            robot_obs_ort_inv = np.linalg.inv(robot_obs_ort)
            rel_actions_ort = np.dot(actions_ort, robot_obs_ort_inv)
            quat_error = robosuite_trans.mat2quat(rel_actions_ort)
            delta_ort = robosuite_trans.quat2axisangle(quat_error) / 0.05
            delta_pos = (episode['actions'][:3] - episode['robot_obs'][:3]) / 0.02
            action = np.array([*delta_pos, *delta_ort, episode['rel_actions'][-1]])
            episode['rel_actions'] = np.array([action[1], -action[0], action[2], action[3], action[4], action[5], action[6]]) # 见https://km.sankuai.com/collabpage/2537354641

        return episode

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        
        # end_idx = self.episode_lookup[idx]+1 # 注意self.episode_lookup[idx]返回的是索引值，range()的话需要+1
        # start_idx_eposide, end_idx_eposide = self.ep_start_end_ids[self.lang_lookup[idx]]
        # start_idx = max(end_idx - window_size, start_idx_eposide)
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size

        if self.env_resample:
            if "task_ABC_D" in  str(self.abs_datasets_dir) :
                if ("calvin_scene_B" in self.scene_info and
                    start_idx <= self.scene_info["calvin_scene_B"][1]):
                    scene = "B"
                elif ("calvin_scene_C" in self.scene_info and
                    start_idx <= self.scene_info["calvin_scene_C"][1]):
                    scene = "C"
                elif ("calvin_scene_A" in self.scene_info and
                    start_idx <= self.scene_info["calvin_scene_A"][1]):
                    scene = "A"
                else:
                    scene = "D"
                    
                task = self.lang_task[self.lang_lookup[idx]]
                
                if ('slider' in task) or ('lightbulb' in task):
                    if scene == "A":
                        pass
                    else:
                        return 0
 
        # keys_calib = ['static_extrinsic_matrix','static_intrinsic_matrix','static_distCoeffs_matrix',
        #               'gripper_extrinsic_matrix','gripper_intrinsic_matrix','gripper_distCoeffs_matrix','state_matrix'] 
        keys_rgb = ['rgb_static','rgb_gripper'] 
        data_path = random.choice(self.data_path_list)
        if self.data_tasks_groups is not None:
            task = self.lang_task[self.lang_lookup[idx]]
            for i, data_tasks_group in enumerate(self.data_tasks_groups):
                if task in data_tasks_group:
                    data_path = self.data_path_list[i]
                    continue
        
        # episodes = [
        #     self.load_file(self._get_episode_name_split(file_idx, data_path))
        #     for file_idx in range(start_idx, end_idx)
        # ]
        # state_matrixs = [
        #     self.load_file(self._get_episode_name_split(file_idx, self.state_matrixs_path))
        #     for file_idx in range(start_idx, end_idx)
        # ]
        episodes = [self._get_episode(file_idx, data_path) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64, me
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"] 
        lang_lookup = []
        # partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.only_single_task and "rotate_blue_block_right" not in lang_task[i]: continue # 仅测试使用
            # if self.partial_data:
            #     if (start_idx, end_idx) not in partial_st_ed_list:
            #         continue
            if self.pretrain: 
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
            # for idx in range(start_idx+1, end_idx): # YF：end_idx表示长度，而不是下标索引,注意此处start_idx+1是为了与window_size+1对应，生成下一张图片
                if cnt % self.skip_frames == 0: 
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task, ep_start_end_ids

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()
        assert False
        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)


class DiskMetaWorldDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        
        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, zip_file, frames):
        pos, grasp = np.split(frames[file_idx]['action'], [3]) # YF：pos乘了40变为了[-1,1]
        action = np.concatenate([pos, np.zeros_like(pos), [-1 if grasp>0 else 1]]) # grasp:[-0.5,0.5],0.5是关，期望-1是关
        action = np.array([action[1], -action[0], action[2], action[3], action[4], action[5], action[6]]) # 见https://km.sankuai.com/collabpage/2537354641

        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     rgb = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "corner2", f'image_save_{file_idx}.jpg')))
        #     rgb2 = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "behindGripper", f'image_save_{file_idx}.jpg')))
        #     rgb_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "corner2", f'depth_save_{file_idx}.npy')))
        #     rgb2_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "behindGripper", f'depth_save_{file_idx}.npy')))
        #     calib = np.load(load_from_zip(zip_ref, os.path.join('rgb', "corner2", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        #     calib2 = np.load(load_from_zip(zip_ref, os.path.join('rgb', "behindGripper", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        rgb = Image.open(os.path.join(zip_file, os.path.join('rgb', "corner2", f'image_save_{file_idx}.jpg')))
        rgb2 = Image.open(os.path.join(zip_file, os.path.join('rgb', "behindGripper", f'image_save_{file_idx}.jpg')))
        rgb_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "corner2", f'depth_save_{file_idx}.npy')))
        rgb2_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "behindGripper", f'depth_save_{file_idx}.npy')))
        calib = np.load(os.path.join(zip_file, os.path.join('rgb', "corner2", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        calib2 = np.load(os.path.join(zip_file, os.path.join('rgb', "behindGripper", f'config_save_{file_idx}.npy')), allow_pickle=True).item()
        rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM) # rgb = np.flip(rgb, axis=0) # 注意此处需要添加flip
        rgb2 = rgb2.transpose(Image.FLIP_TOP_BOTTOM) # rgb2 = np.flip(rgb2, axis=0)
        # Convert depth using OpenGL-style mapping (scaled by extent)
        extent = calib['cam_config']['extent']
        near = calib['cam_config']['nearval'] * extent
        far = calib['cam_config']['farval'] * extent
        rgb_depth = np.flip(rgb_depth, axis=0)
        rgb_depth = depth_to_meters(rgb_depth, near, far, mode="gl")
        rgb2_depth = np.flip(rgb2_depth, axis=0)
        near2 = calib2['cam_config']['nearval'] * extent
        far2 = calib2['cam_config']['farval'] * extent
        rgb2_depth = depth_to_meters(rgb2_depth, near2, far2, mode="gl")
        calib['extrinsic_matrix'] = np.linalg.inv(calib['extrinsic_matrix']) * np.array([[1,1,1,1], [-1,-1,-1,-1], [-1,-1,-1,-1], [1,1,1,1]]) # 主要原因是deproject，对YZ加了一个符号
        calib2['extrinsic_matrix'] = np.linalg.inv(calib2['extrinsic_matrix']) * np.array([[1,1,1,1], [-1,-1,-1,-1], [-1,-1,-1,-1], [1,1,1,1]])
        T_translate = np.array([[1, 0, 0, 0],   # [-0.5038039  -0.10849035  0.12061001  1.        ] [0.48056049 0.41590198 0.60125949 1.        ]
                                [0, 1, 0, 0.25],
                                [0, 0, 1, -0.25],
                                [0, 0, 0, 1]])
        calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], T_translate)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], T_translate) # 为了与calvin点云坐标对齐
        # gripper_cam2gripper = np.array([[7.95685786e-04, 9.99998403e-01, 1.59997833e-03, -5.66836070e-17],
        #                                 [1.73645314e-01, 1.43750538e-03, -9.84807209e-01, -1.40918499e-17],
        #                                 [-9.84807937e-01, 1.06142584e-03, -1.73643893e-01, -5.00000000e-01],
        #                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        #                             ])
        gripper_cam2gripper = np.array([[ 0.0008, 1.0000, 0.0016, -0.0000],
                                        [ 0.1736, 0.0014, -0.9848, -0.0000],
                                        [-0.9848, 0.0011, -0.1736, -0.5000],
                                        [ 0.0000, 0.0000, 0.0000, 1.0000]
                                    ])
        return {
            "rgb_static": rgb, # np.array(rgb),
            "rgb_gripper": rgb2, # np.array(rgb2),
            "rel_actions": action,  # action_scale: float = 1.0 / 100
            "robot_obs": frames[file_idx]['obs'],
            "scene_obs": frames[file_idx]['obs'],
            "depth_static": rgb_depth,
            "depth_gripper": rgb2_depth,
            "calib": {  "rgb_static": calib,
                        "rgb_gripper": calib2},
            "cam_config": { 'static': calib['cam_config'],
                            'gripper': calib2['cam_config']},
            "state_matrix": calib2['extrinsic_matrix'], # 注意此处是爪子矩阵,
            "gripper_cam2gripper": gripper_cam2gripper,
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # zip_file, end_idx, video_len, n_task = self.episode_lookup[idx]
        # end_idx += 1 # 注意返回的是索引值，range()的话需要+1
        zip_file, start_idx, video_len, n_task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        zip_file = os.path.join(self.abs_datasets_dir, zip_file)
        # start_idx = max(end_idx - window_size, 0)

        zip_file = os.path.splitext(zip_file)[0]
        with open(os.path.join(zip_file, 'param/param.pickle'), 'rb') as file:
            frames = pickle.load(file)
            episodes = [self._get_episode(file_idx, zip_file, frames) for file_idx in range(start_idx, end_idx)]
        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     frames = pickle.load(load_from_zip(zip_ref, 'param/param.pickle'))
        #     episodes = [self._get_episode(file_idx, zip_file, frames) for file_idx in range(start_idx, end_idx)]
        
        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[n_task]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for data_slice in data:
            zip_file = data_slice[0]
            video_len = data_slice[1]
            if self.only_single_task and "basketball-v2-goal-observable" not in zip_file: continue
            assert video_len <= 500
            lang_ann[data_slice[2]] = data_slice[3]
            for start_frame_index in range(0, video_len+1-self.min_window_size):
            # for start_frame_index in range(0+1, video_len): # YF: 看calvin
                episode_lookup.append(
                    (zip_file, start_frame_index, video_len, data_slice[2]))
        return episode_lookup, lang_ann
        

class DiskLiberoDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, demo_path):
        
        def rotMatList2NPRotMat(rot_mat_arr):
            np_rot_arr = np.array(rot_mat_arr)
            np_rot_mat = np_rot_arr.reshape((3, 3))
            return np_rot_mat
        def posRotMat2Mat(pos, rot_mat):
            t_mat = np.eye(4)
            t_mat[:3, :3] = rot_mat
            t_mat[:3, 3] = np.array(pos)
            return t_mat

        file_path = demo_path / f"{file_idx}.npz"
        episode = np.load(file_path, allow_pickle=True)
        action=np.concatenate([episode['action'][:6], [-1 if episode['action'][-1]>0 else 1]]) # YF: libero episode['action'][-1]为-1/1，其中1表示关; 
        # action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        agentview = episode['agentview'].item()
        robot0_eye_in_hand = episode['robot0_eye_in_hand'].item()
        agentview['calib']['extrinsic_matrix'] = np.linalg.inv(agentview['calib']['extrinsic_matrix']) * SIGN_FLIP_4X4  # 主要原因是deproject，对YZ加了一个符号
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.linalg.inv(robot0_eye_in_hand['calib']['extrinsic_matrix']) * SIGN_FLIP_4X4
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        base_pose = posRotMat2Mat(episode['base_p'], rotMatList2NPRotMat(episode['base_r']))
        agentview['calib']['extrinsic_matrix'] = np.dot(agentview['calib']['extrinsic_matrix'], base_pose)
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.dot(robot0_eye_in_hand['calib']['extrinsic_matrix'], base_pose)
        # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.3],    # 
            [0, 1, 0, 0.0],     # 
            [0, 0, 1, -0.1],    # [-0.24398557 -0.43166057  0.01677953  1.        ] [0.86626239 0.57459427 0.90260449 1.        ]  
            [0, 0, 0, 1] 
        ])
        agentview['calib']['extrinsic_matrix'] = np.dot(agentview['calib']['extrinsic_matrix'], T_translate)
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.dot(robot0_eye_in_hand['calib']['extrinsic_matrix'], T_translate) # 为了与calvin点云坐标对齐
        R = AXIS_SWAP_R
        agentview['calib']['extrinsic_matrix']=np.dot(agentview['calib']['extrinsic_matrix'], R)
        robot0_eye_in_hand['calib']['extrinsic_matrix'] = np.dot(robot0_eye_in_hand['calib']['extrinsic_matrix'], R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        # gripper_cam2gripper = np.array([[1.73745673e-06, 1.00000000e+00, 4.98534343e-06, -9.65734593e-05],
        #                                 [1.00000000e+00, -1.73746611e-06, 1.88231867e-06, -4.99967197e-02],
        #                                 [1.88232733e-06, 4.98534016e-06, -1.00000000e+00, -9.69942531e-02],
        #                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        #                             ])
        gripper_cam2gripper = np.array([[ 0.0, 1.0, 0.0, -0.0001],
                                        [ 1.0, -0.0, 0.0, -0.05],
                                        [ 0.0, 0.0, -1.0, -0.097],
                                        [ 0.0, 0.0, 0.0, 1.0]
                                    ])
        return {
            "rgb_static": Image.fromarray(agentview['image']),
            "rgb_gripper": Image.fromarray(robot0_eye_in_hand['image']),
            "rel_actions": action,
            "robot_obs": np.zeros(20), # 由于没有保存，且没有使用，直接复制为0了。
            "scene_obs": np.zeros(20),
            "depth_static": agentview['depth'],
            "depth_gripper": robot0_eye_in_hand['depth'],
            "calib": {  'rgb_static': agentview['calib'],
                        'rgb_gripper': robot0_eye_in_hand['calib']},
            "cam_config": { "static": {"height": 256, "width": 256, "fov": 45},
                            "gripper": {"height": 256, "width": 256, "fov": 75}},
            "state_matrix": robot0_eye_in_hand['calib']['extrinsic_matrix'],  # 注意此处是爪子矩阵
            "gripper_cam2gripper": gripper_cam2gripper,
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # task, demo_id, end_idx = self.episode_lookup[idx]
        # end_idx+=1
        # demo_path = self.abs_datasets_dir / task / f"demo_{demo_id}"
        # start_idx = max(end_idx - window_size, 0)
        task, demo_id, start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        demo_path = self.abs_datasets_dir / task / f"demo_{demo_id}"
        episodes = [self._get_episode(file_idx, demo_path) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[task]
            # if self.text_aug:
            #     task = self.lang_task[self.lang_lookup[idx]]
            #     enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
            #     episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup, lang_ann = [], {}
        task_emb_list = np.load(abs_datasets_dir/'emb.npz', allow_pickle=True)
        
        _task_list = os.listdir(abs_datasets_dir)
        for task in _task_list:
            data_dir = abs_datasets_dir / task
            if not data_dir.is_dir(): continue # 样本均在文件夹内，假如非文件夹则无效
            if self.only_single_task and task not in ['task_86', 'task_87', 'task_88', 'task_89'] : continue # 调试使用
            lang_ann[task] = task_emb_list['arr_0'][int(task.split('_')[-1])]
            demo_num = len(os.listdir(data_dir))
            for i in range(demo_num):
                step_num = len(os.listdir(data_dir / f"demo_{i}"))
                for j in range(step_num+1 - self.min_window_size):
                # for j in range(0+1, step_num):
                    episode_lookup.append((task, i, j))

        return episode_lookup, lang_ann
        

class DiskRobocasaDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann,
                self.base_extrinsics
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, zip_file, frames, base_extrinsic):

        action = frames[file_idx]['action'][:7] # 注意这里移动12个值，前6个是爪子位姿，第7个是爪子张合1表示关，第8-10是移动，11是躯干，12是底盘移动还是夹爪
        action[-1] = -1 if action[-1] > 0 else 1  # 1表示关
        # action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        # ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand", "robot0_agentview_center", "robot0_frontview"],
        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     rgb = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'image_save_' + str(file_idx) + ".jpg")))
        #     rgb2 = Image.open(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg")))
        #     rgb_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'depth_save_' + str(file_idx) + ".npy")))
        #     rgb2_depth = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy")))
        #     calib = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_agentview_left", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        #     calib2 = np.load(load_from_zip(zip_ref, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        rgb = Image.open(os.path.join(zip_file, os.path.join('rgb', "robot0_agentview_left", 'image_save_' + str(file_idx) + ".jpg")))
        rgb2 = Image.open(os.path.join(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'image_save_' + str(file_idx) + ".jpg")))
        rgb_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_agentview_left", 'depth_save_' + str(file_idx) + ".npy")))
        rgb2_depth = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'depth_save_' + str(file_idx) + ".npy")))
        calib = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_agentview_left", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        calib2 = np.load(os.path.join(zip_file, os.path.join('rgb', "robot0_eye_in_hand", 'config_save_' + str(file_idx) + ".npy")), allow_pickle=True).item()
        rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM) # rgb = np.flip(np.array(rgb), axis=0 # 注意此处需要添加flip
        rgb2 = rgb2.transpose(Image.FLIP_TOP_BOTTOM) # rgb2 = np.flip(np.array(rgb2), axis=0)
        extent = calib['cam_config']['extent']
        near = calib['cam_config']['nearval'] * extent
        far = calib['cam_config']['farval'] * extent
        rgb_depth = np.flip(rgb_depth, axis=0)
        rgb_depth = depth_to_meters(rgb_depth, near, far, mode="gl")
        rgb2_depth = np.flip(rgb2_depth, axis=0)
        near2 = calib2['cam_config']['nearval'] * extent
        far2 = calib2['cam_config']['farval'] * extent
        rgb2_depth = depth_to_meters(rgb2_depth, near2, far2, mode="gl")
        calib['extrinsic_matrix'] = np.linalg.inv(calib['extrinsic_matrix']) * SIGN_FLIP_4X4
        calib2['extrinsic_matrix'] = np.linalg.inv(calib2['extrinsic_matrix']) * SIGN_FLIP_4X4 # 主要原因是deproject，对YZ加了一个符号
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], base_extrinsic)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], base_extrinsic)
        # 平移矩阵 T_translate single_stage_kitchen_pnp_PnPMicrowaveToCounter 0.6-1.0
        T_translate = np.array([
            [1, 0, 0, 0.3], #  [-0.81904223 -1.35147707  0.70431699  1.        ] [0.8594698  0.75282846 1.83341838 1.        ]
            [0, 1, 0, 0.0], # 
            [0, 0, 1, 0.7], # 
            [0, 0, 0, 1]
        ])
        calib['extrinsic_matrix'] = np.dot(calib['extrinsic_matrix'], T_translate)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], T_translate) # 为了与calvin点云坐标对齐
        R = AXIS_SWAP_R
        calib['extrinsic_matrix']=np.dot(calib['extrinsic_matrix'], R)
        calib2['extrinsic_matrix'] = np.dot(calib2['extrinsic_matrix'], R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        # gripper_cam2gripper = np.array([[-1.26022068e-16, 1.00000000e+00, -1.58006978e-16, -9.02557353e-18],
        #                                 [1.00000000e+00, 9.37418353e-17, -1.75308039e-16, -5.00000000e-02],
        #                                 [-2.38202292e-17, -1.35668470e-16, -1.00000000e+00, -9.70000000e-02],
        #                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        #                             ])
        gripper_cam2gripper = np.array([[-0.0, 1.0, -0.0, -0.0],
                                        [ 1.0, 0.0, -0.0, -0.05],
                                        [-0.0, -0.0, -1.0, -0.097],
                                        [ 0.0, 0.0, 0.0, 1.0]
                                    ])
        return {
            "rgb_static": rgb,
            "rgb_gripper": rgb2,
            "rel_actions": action,
            "robot_obs": frames[file_idx]['states'],
            "scene_obs": frames[file_idx]['states'],
            "depth_static": rgb_depth,
            "depth_gripper": rgb2_depth,
            "calib": {  "rgb_static": calib,
                        "rgb_gripper": calib2},
            "cam_config":{  "static": calib['cam_config'],
                            "gripper": calib2['cam_config']},
            "state_matrix": calib2['extrinsic_matrix'], # 注意此处是爪子矩阵
            "gripper_cam2gripper": gripper_cam2gripper,
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # zip_file, end_idx, video_len, n_task = self.episode_lookup[idx]
        # end_idx += 1 # YF: 看calvin
        # start_idx = max(end_idx - window_size, 0)
        zip_file, start_idx, video_len, n_task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        base_extrinsic = self.base_extrinsics[zip_file]

        with open(os.path.join(os.path.splitext(os.path.join(self.abs_datasets_dir, zip_file))[0], 'param/param.pickle'), 'rb') as file:
            frames = pickle.load(file)
            episodes = [self._get_episode(file_idx, os.path.splitext(os.path.join(self.abs_datasets_dir, zip_file))[0], frames, base_extrinsic) for file_idx in range(start_idx, end_idx)]
        # with zipfile.ZipFile(os.path.join(self.abs_datasets_dir, zip_file), 'r') as zip_ref:
        #     frames = pickle.load(load_from_zip(zip_ref, 'param/param.pickle'))
        #     episodes = [self._get_episode(file_idx, os.path.join(self.abs_datasets_dir, zip_file), frames, base_extrinsic) for file_idx in range(start_idx, end_idx)]
        
        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[zip_file]
            # if self.text_aug:
            #     task = self.lang_task[self.lang_lookup[idx]]
            #     enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
            #     episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, base_extrinsics, episode_lookup = {}, {}, []
        json_path = os.path.join(abs_datasets_dir, "train_metaworld.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for data_slice in data:
            if 'navigat' in data_slice[2]: continue
            if self.only_single_task and 'CloseDrawer' not in data_slice[2]: continue # 仅调试使用
            zip_file = data_slice[0]
            video_len = data_slice[1]
            # if "basketball-v2-goal-observable" not in zip_file: continue
            # assert video_len <= 500
            lang_ann[zip_file] = data_slice[3]
            base_extrinsics[zip_file] = np.array(data_slice[4])
            for start_frame_index in range(0, video_len+1-self.min_window_size):
            # for start_frame_index in range(0+1, video_len): # YF: 看calvin
                episode_lookup.append(
                    (zip_file, start_frame_index, video_len, data_slice[2]))
        return episode_lookup, lang_ann, base_extrinsics


class DiskRoboMimicDataset(BaseMultiDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.tasks = ["lift", "can", "square"] # transport为双臂，["lift", "can", "square", "transport", "tool_hang"]
        self.lang_ann = {
            "lift" : "The robot arm must lift a small cube.",
            "can" : "The robot must place a coke can from a large bin into a smaller target bin.",
            "square" : "The robot must pick a square nut and place it on a rod.",
            "transport" : "Two robot arms must transfer a hammer from a closed container on a shelf to a target bin on another shelf.",
            "tool_hang" : "Insert the hook into the base, assemble the frame, and hang the wrench on the hook.",
        }
        if self.with_lang:
            (
                self.episode_lookup
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx, frames, cam_infos): 
        # action: [-1.         -1.         -1.         -0.55634028 -1.         -1. -1.        ] [1.         1.         1.         0.72973686 0.45003703 1. 1.        ]
        action = frames["actions"][file_idx][()]
        action[-1] = -1 if action[-1] > 0 else 1 # robomimic中1表示关
        # action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        rgb_static = frames["obs"]['agentview_image'][file_idx][()] # 256 256 3
        rgb_gripper = frames["obs"]['robot0_eye_in_hand_image'][file_idx][()] # 256 256 3
        depth_static = np.squeeze(frames["obs"]['agentview_depth'][file_idx][()])
        depth_gripper = np.squeeze(frames["obs"]['robot0_eye_in_hand_depth'][file_idx][()])
        camera_info = cam_infos[f"{file_idx}"]

        static_extrinsic_matrix = np.linalg.inv(np.array(camera_info["agentview"]['extrinsics'])) * SIGN_FLIP_4X4   #注意这里没有求逆
        gripper_extrinsic_matrix = np.linalg.inv(np.array(camera_info['robot0_eye_in_hand']['fix_extrinsics'])) * SIGN_FLIP_4X4 # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
        T_translate = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.4], # [-0.17879518 -0.40093808  0.90953124  1.        ] [0.33830822 0.33967275 1.29464515 1.        ]
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = AXIS_SWAP_R
        static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        calib = {}
        calib['rgb_static'] = {} 
        calib['rgb_static']['extrinsic_matrix'] = static_extrinsic_matrix # 4 4 
        calib['rgb_static']['intrinsic_matrix'] = np.array(camera_info["agentview"]["intrinsics"]) # 3 3
        calib['rgb_static']['distCoeffs_matrix'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]) # 8

        calib['rgb_gripper'] = {}
        calib['rgb_gripper']['extrinsic_matrix'] = gripper_extrinsic_matrix # 4 4 
        calib['rgb_gripper']['intrinsic_matrix'] = np.array(camera_info['robot0_eye_in_hand']["intrinsics"]) # 3 3
        calib['rgb_gripper']['distCoeffs_matrix'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]) # 8

        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        cam_config = {}
        cam_config['static'] = {}
        cam_config['static']['height'] = rgb_static.shape[0]
        cam_config['static']['width'] = rgb_static.shape[1]
        cam_config['static']['fov'] = calculate_fov(calib['rgb_static']['intrinsic_matrix'], rgb_static.shape[0])
        cam_config['gripper'] = {}
        cam_config['gripper']['height'] = rgb_gripper.shape[0]
        cam_config['gripper']['width'] = rgb_gripper.shape[1]
        cam_config['gripper']['fov'] = calculate_fov(calib['rgb_gripper']['intrinsic_matrix'], rgb_gripper.shape[0])
        # joints = frames["obs"]["robot0_joint_pos"][file_idx]
        # gripper_cam2gripper = np.array([[-1.01359607e-17, 1.00000000e+00, -1.77379791e-17, 5.72510797e-18],
        #                                 [1.00000000e+00, 6.34456985e-17, -1.71448900e-16, -5.00000000e-02],
        #                                 [-1.52043694e-16, -3.18033167e-17, -1.00000000e+00, -9.70000000e-02],
        #                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        #                             ]) 
        gripper_cam2gripper = np.array([[-0.0, 1.0, -0.0, 0.0],
                                        [ 1.0, 0.0, -0.0, -0.05],
                                        [-0.0, -0.0, -1.0, -0.097],
                                        [ 0.0, 0.0, 0.0, 1.0]
                                    ])

        return {
            "rgb_static": Image.fromarray(rgb_static),
            "rgb_gripper": Image.fromarray(rgb_gripper),
            "depth_static": depth_static,
            "depth_gripper": depth_gripper, # 单位m
            "rel_actions": action,
            "robot_obs": np.zeros(20),
            "scene_obs": np.zeros(20),
            "calib": calib,
            "cam_config": cam_config,
            "state_matrix": gripper_extrinsic_matrix,  # 注意此处是爪子矩阵
            "gripper_cam2gripper": gripper_cam2gripper,
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        
        with h5py.File(dataset_file, "r") as data:       
            def load_h5_data(data):
                out = dict()
                for k in data.keys():
                    if isinstance(data[k], h5py.Dataset): out[k] = data[k][:]
                    else: out[k] = load_h5_data(data[k])
                return out
            frames = data["data"][episode_id]
            cam_infos = json.loads(frames.attrs['camera_info_realtime'])
            # frames = load_h5_data(frames)  # 暂时不使用
            episodes = [self._get_episode(file_idx, frames, cam_infos) for file_idx in range(start_idx, end_idx)]

            episode = self._prepare_eposides(episodes)

            if self.with_lang:
                episode["language"] = self.lang_ann[task]
                if self.text_aug:
                    task = self.lang_task[self.lang_lookup[idx]]
                    enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                    episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
        """
        episode_lookup = []
        # 加载不同任务
        for task in self.tasks:
            if self.only_single_task and "lift" not in task: continue
            dataset_file = os.path.join(abs_datasets_dir, task, "ph", "depth.hdf5")
            data = h5py.File(dataset_file, "r")
            episodes = data["data"]             # 记录每一条轨迹的信息
            for i in range(len(episodes)):             # 构建episode_lookup
                episode_length = len(episodes["demo_{0}".format(i)]["actions"])
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                    episode_lookup.append((dataset_file, "demo_{0}".format(i), start_frame_index, episode_length, task))
            data.close()
        return episode_lookup # zyf
        

class DiskManiSkill2Dataset(BaseMultiDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, start_idx, end_idx, frames): 
        from scipy.spatial.transform import Rotation as R
        def base_pose_to_matrix(base_pose):
            position = base_pose[:, :3] # 提取位置和四元数
            orientation = base_pose[:, 3:]
            rotation = R.from_quat(orientation).as_matrix() # 将四元数转换为旋转矩阵
            transform_matrix = np.eye(4) # 创建4x4的同质变换矩阵
            transform_matrix = np.tile(transform_matrix, (13, 1, 1))
            transform_matrix[:, :3, :3] = rotation
            transform_matrix[:, :3, 3] = position
            return transform_matrix

        action = frames["actions"][start_idx:end_idx][()]
        action = np.hstack([action[:, 1:2], action[:, 0:1], -action[:, 2:3], action[:, 4:5], action[:, 3:4], -action[:, 5:6], action[:, 6:7]])  # 对齐坐标系[1, 0,-2,4,3,-5,6]

        rgb_static = frames["obs"]["image"]["base_camera"]["rgb"][start_idx:end_idx][()]
        rgb_gripper = frames["obs"]["image"]["hand_camera"]["rgb"][start_idx:end_idx][()]
        depth_static = np.squeeze(frames["obs"]["image"]["base_camera"]["depth"][start_idx:end_idx][()])
        depth_gripper = np.squeeze(frames["obs"]["image"]["hand_camera"]["depth"][start_idx:end_idx][()])
        if len(depth_static.shape) == 2:
            depth_static = depth_static[None]
            depth_gripper = depth_gripper[None]
        static_extrinsic_matrix = frames["obs"]["camera_param"]["base_camera"]['extrinsic_cv'][start_idx:end_idx][()] * SIGN_FLIP_4X4   # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
        gripper_extrinsic_matrix = frames["obs"]["camera_param"]["hand_camera"]['extrinsic_cv'][start_idx:end_idx][()] * SIGN_FLIP_4X4
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        base_pose = base_pose_to_matrix(frames['obs']['agent']['base_pose'][start_idx:end_idx][()])
        static_extrinsic_matrix = np.einsum('mij,mjk->mik', static_extrinsic_matrix, base_pose)
        gripper_extrinsic_matrix = np.einsum('mij,mjk->mik', gripper_extrinsic_matrix, base_pose)
        # 平移矩阵 T_translate 待修改
        T_translate = np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.0], # [-0.2615632  -0.79399014 -1.17142143  1. ] [8.50e-01 7.642e-01 8.9469e-04 1.00e+00]
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, -1, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, R) # 往前为X轴；往右为Y轴；往下为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴
        static_intrinsic_matrix = frames["obs"]["camera_param"]["base_camera"]["intrinsic_cv"][start_idx:end_idx][()]
        gripper_intrinsic_matrix = frames["obs"]["camera_param"]["hand_camera"]["intrinsic_cv"][start_idx:end_idx][()]
        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        cam_config = {'static':{'height': rgb_static.shape[1],
                                'width': rgb_static.shape[2],
                                'fov': calculate_fov(static_intrinsic_matrix[0], rgb_static.shape[1])}, #90
                     'gripper':{'height': rgb_gripper.shape[1],
                                'width': rgb_gripper.shape[2],
                                'fov': calculate_fov(gripper_intrinsic_matrix[0], rgb_gripper.shape[1])}}
        return [{   "rgb_static": Image.fromarray(rgb_static_),
                    "rgb_gripper": Image.fromarray(rgb_gripper_),
                    "depth_static": depth_static_ / 1000,
                    "depth_gripper": depth_gripper_ / 1000, # 单位mm
                    "rel_actions": action_,
                    "robot_obs": np.zeros(20),
                    "scene_obs": np.zeros(20),
                    "calib": {  'rgb_static': { 'extrinsic_matrix': static_extrinsic_matrix_,         
                                                'intrinsic_matrix': static_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                                'rgb_gripper': {'extrinsic_matrix': gripper_extrinsic_matrix_,
                                                'intrinsic_matrix': gripper_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
                    "cam_config": cam_config,
                    "state_matrix": gripper_extrinsic_matrix_,  # 注意此处是爪子矩阵
        }for (rgb_static_, rgb_gripper_, depth_static_, depth_gripper_, action_, static_extrinsic_matrix_, static_intrinsic_matrix_, gripper_extrinsic_matrix_, gripper_intrinsic_matrix_) in zip(rgb_static, rgb_gripper, depth_static, depth_gripper, action, static_extrinsic_matrix, static_intrinsic_matrix, gripper_extrinsic_matrix, gripper_intrinsic_matrix)]

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # dataset_file, episode_id, end_idx, episode_length, task = self.episode_lookup[idx]
        # end_idx +=1
        # data = h5py.File(dataset_file, "r")
        # frames = data[f"traj_{episode_id}"]
        # start_idx = max(end_idx - window_size, 0)
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size

        with h5py.File(dataset_file, "r") as data:
            frames = data[f"traj_{episode_id}"]
            def load_h5_data(data):
                out = dict()
                for k in data.keys():
                    if isinstance(data[k], h5py.Dataset): out[k] = data[k][:]
                    else: out[k] = load_h5_data(data[k])
                return out
            # frames = load_h5_data(frames) # 暂时不使用
            # episodes = [self._get_episode(file_idx, frames)for file_idx in range(start_idx, end_idx)]
            episodes = self._get_episode(start_idx, end_idx, frames)

            episode = self._prepare_eposides(episodes)

            if self.with_lang:
                episode["language"] = self.lang_ann[task]
                if self.text_aug:
                    task = self.lang_task[self.lang_lookup[idx]]
                    enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                    episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        task_emb_list = np.load(os.path.join(abs_datasets_dir, 'maniskill2_lang.py.npy'), allow_pickle=True).item()
        abs_datasets_dir = os.path.join(abs_datasets_dir, "v0")

        def find_files_with_suffix(root_path, suffix):
            import glob
            search_pattern = os.path.join(root_path, '**', f'*{suffix}') # 构建搜索模式
            matching_files = glob.glob(search_pattern, recursive=True) # 在指定路径下递归搜索所有符合后缀的文件
            absolute_paths = [os.path.abspath(file) for file in matching_files] # 获取并返回这些文件的绝对路径
            return absolute_paths
        lang_ann, episode_lookup = {}, []
        h5_files = find_files_with_suffix(abs_datasets_dir, 'rgbd.pd_ee_delta_pose.h5')
        for h5_f in h5_files:
            task = [elem for elem in list(task_emb_list.keys()) if elem in h5_f]
            assert len(task)==1
            if "soft_body" in h5_f: continue # 暂时不使用soft_body，task是Excavate-v0/Fill-v0/Hang-v0
            if self.only_single_task and 'LiftCube' not in task[0]: continue # 仅调试使用
            lang_ann[task[0]] = task_emb_list[task[0]]

            json_path = h5_f.replace(".h5", ".json")
            json_data = json.load(open(json_path, 'rb'))
            episodes = json_data["episodes"]
            for i in range(len(episodes)):
                assert i == episodes[i]["episode_id"]
                episode_length = episodes[i]["elapsed_steps"]
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                # for start_frame_index in range(0+1, episode_length): # YF:看calvin
                    episode_lookup.append((h5_f, episodes[i]["episode_id"], start_frame_index, episode_length, task[0]))
                
        return episode_lookup, lang_ann


class DiskRlbenchDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, episodes_dir, frames):
        
        def angle_between_angles(a, b):
            diff = b - a
            return (diff + np.pi) % (2 * np.pi) - np.pi
        def to_relative_action(actions, robot_obs, max_pos=0.03, max_orn=0.2):
            rel_pos = actions[:3] - robot_obs[:3]
            rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
            rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
            rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = actions[-1:]
            return np.concatenate([rel_pos, rel_orn, gripper])
        r_euler = np.array(pybullet.getEulerFromQuaternion(frames[file_idx].gripper_pose[3:7]))
        a_euler = np.array(pybullet.getEulerFromQuaternion(frames[file_idx+1].gripper_pose[3:7]))
        if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8 and (r_euler[0]==0 or a_euler[0]==0 or r_euler[2]==0 or a_euler[2]==0): # 发生了万向节死锁
            assert False, 'Maybe gimbal lock has occurred.'
        # if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8: # 大于0.8就发生了万向节死锁
        #     sr_euler = np.array(scipyR.from_quat(frames[file_idx].gripper_pose[3:7]).as_euler('xyz', degrees=False))
        #     sa_euler = np.array(scipyR.from_quat(frames[file_idx+1].gripper_pose[3:7]).as_euler('xyz', degrees=False))
        #     if np.abs(angle_between_angles(a_euler, r_euler) - angle_between_angles(sa_euler, sr_euler)).max()<0.01: # 并没有发生万向节死锁
        #         pass
        #     elif np.abs(angle_between_angles(sa_euler, sr_euler)).max() < 0.8: # 大于0.8就发生了万向节死锁
        #         r_euler = sr_euler
        #         a_euler = sa_euler
        #     else:
        #         if np.abs(angle_between_angles(sa_euler, r_euler)).max() < 0.8:
        #             a_euler = sa_euler
        #         elif np.abs(angle_between_angles(a_euler, sr_euler)).max() < 0.8:
        #             r_euler = sr_euler
        #         else:
        #             assert False, 'Maybe gimbal lock has occurred.'
        robot_obs_euler = np.concatenate([frames[file_idx].gripper_pose[:3], r_euler, [1 if frames[file_idx].gripper_open>0.5 else -1]]) # YF: gripper_open=0/1, 注意要求的是-1和1
        actions_euler = np.concatenate([frames[file_idx+1].gripper_pose[:3], a_euler, [1 if frames[file_idx+1].gripper_open>0.5 else -1]])
        action = to_relative_action(actions_euler, robot_obs_euler[:6], max_pos=0.03, max_orn=0.2)
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        rgb = Image.open(episodes_dir/f"front_rgb/{file_idx}.png")
        rgb2 = Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png")
        # FOV from intrinsics
        def image_to_float_array(image, scale_factor): 
            image_array = np.array(image)
            assert 2 <= image_array.ndim <= 3, "Image must be either 2D or 3D array."
            
            if image_array.ndim == 3 and image_array.shape[2] == 3:
                # RGB image: Convert to 24-bit integer using bitwise operations for better performance
                float_array = (image_array[:, :, 0].astype(np.uint32) << 16) + \
                            (image_array[:, :, 1].astype(np.uint32) << 8) + \
                            image_array[:, :, 2].astype(np.uint32)
            else:
                # Grayscale image or single channel image
                float_array = image_array.astype(np.float32)
            
            scaled_array = float_array / scale_factor
            return scaled_array
        static_cam_config = {"width": rgb.width, "height": rgb.height, "fov": compute_fov_from_intrinsics(frames[file_idx].misc['front_camera_intrinsics'], rgb.height), "nearval": frames[file_idx].misc['front_camera_near'], "farval":  frames[file_idx].misc['front_camera_far']}
        gripper_cam_config = {"width": rgb2.width, "height": rgb2.height, "fov": compute_fov_from_intrinsics(frames[file_idx].misc['wrist_camera_intrinsics'], rgb2.height), "nearval": frames[file_idx].misc['wrist_camera_near'], "farval":  frames[file_idx].misc['wrist_camera_far']}
        depth_static = image_to_float_array(Image.open(episodes_dir/f"front_depth/{file_idx}.png"), 16777215)
        depth_gripper = image_to_float_array(Image.open(episodes_dir/f"wrist_depth/{file_idx}.png"), 16777215)
        depth_static = depth_to_meters(depth_static, static_cam_config['nearval'], static_cam_config['farval'], mode="linear")
        depth_gripper = depth_to_meters(depth_gripper, gripper_cam_config['nearval'], gripper_cam_config['farval'], mode="linear")
        static_extrinsic_matrix = frames[file_idx].misc['front_camera_extrinsics']
        gripper_extrinsic_matrix = frames[file_idx].misc['wrist_camera_extrinsics']
        static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix) * SIGN_FLIP_4X4 # 单位矩阵的逆和转置是一样的
        gripper_extrinsic_matrix = np.linalg.inv(gripper_extrinsic_matrix) * SIGN_FLIP_4X4 # 主要原因是deproject，对YZ加了一个符号
        # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.0], #  [ 0.016149   -0.3464728   0.84676421  1.        ] [0.47925746 0.38417864 1.57344687 1.        ]
            [0, 1, 0, 0.0], # 
            [0, 0, 1, 0.7], # 
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = AXIS_SWAP_R
        static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {
            "rgb_static": rgb,
            "rgb_gripper": rgb2,
            "depth_static": depth_static,
            "depth_gripper": depth_gripper,
            "rel_actions": action,
            "robot_obs": np.zeros(20),
            "scene_obs": np.zeros(20),
            "calib": {  "rgb_static": { "extrinsic_matrix": static_extrinsic_matrix,
                                        "intrinsic_matrix": frames[file_idx].misc['front_camera_intrinsics'],
                                        "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                        "rgb_gripper":{ "extrinsic_matrix": gripper_extrinsic_matrix,
                                        "intrinsic_matrix": frames[file_idx].misc['wrist_camera_intrinsics'],
                                        "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
            "cam_config":{  "static": static_cam_config,
                            "gripper":gripper_cam_config},
            "state_matrix": gripper_extrinsic_matrix  # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # demo_name, end_idx, video_len = self.episode_lookup[idx]
        # end_idx+=1
        # episodes_dir = self.abs_datasets_dir / demo_name
        # with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
        #     frames = pickle.load(f)
        # start_idx = max(end_idx - window_size, 0)
        demo_name, start_idx, video_len = self.episode_lookup[idx]
        episodes_dir = self.abs_datasets_dir / demo_name
        with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
            frames = pickle.load(f)
        end_idx = start_idx + window_size

        episodes = [self._get_episode(file_idx, episodes_dir, frames) for file_idx in range(start_idx, end_idx)]
        # episodes = self._get_episode(start_idx, end_idx, episodes_dir, frames)

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[demo_name][-1]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        tasks=[x for x in "place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap".split(" ")]
        for task_str in tasks:
            if self.only_single_task and "open_drawer" not in task_str: continue # 测试使用
            task_path = abs_datasets_dir / task_str / "all_variations" / "episodes"
            for num_demo in os.listdir(task_path):
                task_name = f"{task_str}/all_variations/episodes/{num_demo}"
                episodes_dir = abs_datasets_dir / task_name
                with open(episodes_dir/"variation_descriptions.pkl", 'rb') as f:
                    lang_str = pickle.load(f)
                # if self.only_single_task and "3" not in lang_str[0]: continue # 测试使用
                lang_ann[task_name] = lang_str
                with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
                    obs = pickle.load(f)
                step_num = len(obs)
                for j in range(step_num - self.min_window_size): # 由于有+1操作，因此不加载最后一帧
                # for j in range(0+1, step_num): # YF：看calvin
                    episode_lookup.append((task_name, j, step_num))

        return episode_lookup, lang_ann


class DiskColosseumDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx: int, episodes_dir, frames):
        
        def angle_between_angles(a, b):
            diff = b - a
            return (diff + np.pi) % (2 * np.pi) - np.pi
        def to_relative_action(actions, robot_obs, max_pos=0.03, max_orn=0.2):
            rel_pos = actions[:3] - robot_obs[:3]
            rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
            rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
            rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = actions[-1:]
            return np.concatenate([rel_pos, rel_orn, gripper])
        def quate2euler(quat, gimble_fix=False):
            quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
            if quat[-1] < 0: quat = -quat
            euler = pybullet.getEulerFromQuaternion(quat)
            return euler
        r_euler = np.array(quate2euler(frames[file_idx].gripper_pose[3:7]))
        a_euler = np.array(quate2euler(frames[file_idx+1].gripper_pose[3:7]))
        if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8 and (r_euler[0]==0 or a_euler[0]==0 or r_euler[2]==0 or a_euler[2]==0): # 发生了万向节死锁
            assert False, 'Maybe gimbal lock has occurred.'
        robot_obs_euler = np.concatenate([frames[file_idx].gripper_pose[:3], r_euler, [1 if frames[file_idx].gripper_open>0.5 else -1]]) # YF: gripper_open=0/1, 注意要求的是-1和1
        actions_euler = np.concatenate([frames[file_idx+1].gripper_pose[:3], a_euler, [1 if frames[file_idx+1].gripper_open>0.5 else -1]])
        action = to_relative_action(actions_euler, robot_obs_euler[:6], max_pos=0.03, max_orn=0.2)
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]
        rgb = Image.open(episodes_dir/f"front_rgb/{file_idx}.png")
        rgb2 = Image.open(episodes_dir/f"wrist_rgb/{file_idx}.png")
        def calculate_fov(intrinsic_matrix, image_width):
            fx = intrinsic_matrix[0, 0]
            fov = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
            return fov
        with open(episodes_dir / "front_depth" / f"{file_idx}.pkl",'rb') as f:
            depth_static = pickle.load(f)
        with open(episodes_dir / "wrist_depth" / f"{file_idx}.pkl",'rb') as f:
            depth_gripper = pickle.load(f)
        static_extrinsic_matrix = frames[file_idx].misc['front_camera_extrinsics']
        gripper_extrinsic_matrix = frames[file_idx].misc['wrist_camera_extrinsics']
        static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix) * SIGN_FLIP_4X4 # 单位矩阵的逆和转置是一样的
        gripper_extrinsic_matrix = np.linalg.inv(gripper_extrinsic_matrix) * SIGN_FLIP_4X4 # 主要原因是deproject，对YZ加了一个符号
        # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.0], #  [ 0.016149   -0.3464728   0.84676421  1.        ] [0.47925746 0.38417864 1.57344687 1.        ]
            [0, 1, 0, 0.0], # 
            [0, 0, 1, 0.7], # 
            [0, 0, 0, 1]
        ])
        static_extrinsic_matrix = np.dot(static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [0, 1, 0, 0],
            [-1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix=np.dot(static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.dot(gripper_extrinsic_matrix, R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {"rgb_static": rgb,
                "rgb_gripper": rgb2,
                "depth_static": depth_static,
                "depth_gripper": depth_gripper,
                "rel_actions": action,
                "robot_obs": np.zeros(20),
                "scene_obs": np.zeros(20),
                "calib":{   "rgb_static": { "extrinsic_matrix": static_extrinsic_matrix,
                                            "intrinsic_matrix": frames[file_idx].misc['front_camera_intrinsics'],
                                            "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                            "rgb_gripper":{ "extrinsic_matrix": gripper_extrinsic_matrix,
                                            "intrinsic_matrix": frames[file_idx].misc['wrist_camera_intrinsics'],
                                            "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
                "cam_config":{  "static": {"width": rgb.width, "height": rgb.height, "fov": calculate_fov(frames[file_idx].misc['front_camera_intrinsics'], rgb.height), "nearval": frames[file_idx].misc['front_camera_near'], "farval":  frames[file_idx].misc['front_camera_far']},
                                "gripper":{"width": rgb2.width, "height": rgb2.height, "fov": calculate_fov(frames[file_idx].misc['wrist_camera_intrinsics'], rgb2.height), "nearval": frames[file_idx].misc['wrist_camera_near'], "farval":  frames[file_idx].misc['wrist_camera_far']}},
                "state_matrix": gripper_extrinsic_matrix  # 注意此处是爪子矩阵
            }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx = 0
        # demo_name, end_idx, video_len = self.episode_lookup[idx]
        # end_idx+=1
        # episodes_dir = self.abs_datasets_dir / demo_name
        # with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
        #     frames = pickle.load(f)
        # start_idx = max(end_idx - window_size, 0)
        demo_name, start_idx, video_len = self.episode_lookup[idx]
        episodes_dir = self.abs_datasets_dir / demo_name
        with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
            frames = pickle.load(f)
        end_idx = start_idx + window_size
        # st=time.time()
        episodes = [self._get_episode(file_idx, episodes_dir, frames) for file_idx in range(start_idx, end_idx)]
        # print(f"loader: {time.time()-st}")
        episode = self._prepare_eposides(episodes)
        # print(f"loader: {time.time()-st}")
        if self.with_lang:
            episode["language"] = self.lang_ann[demo_name][-1]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        tasks = os.listdir(abs_datasets_dir)
        for task_str in tasks:
            if self.only_single_task and "open_drawer" not in task_str: continue # 测试使用
            task_path = abs_datasets_dir / task_str / "variation0" / "episodes"
            for num_demo in os.listdir(task_path):
                task_name = f"{task_str}/variation0/episodes/{num_demo}"
                episodes_dir = abs_datasets_dir / task_name
                with open(abs_datasets_dir/ task_str / "variation0" / "variation_descriptions.pkl", 'rb') as f:
                    lang_str = pickle.load(f)
                # if self.only_single_task and "3" not in lang_str[0]: continue # 测试使用
                lang_ann[task_name] = lang_str
                with open(episodes_dir/"low_dim_obs.pkl", 'rb') as f:
                    obs = pickle.load(f)
                step_num = len(obs)
                for j in range(step_num - self.min_window_size): # 由于有+1操作，因此不加载最后一帧
                # for j in range(0+1, step_num): # YF：看calvin
                    episode_lookup.append((task_name, j, step_num))

        return episode_lookup, lang_ann


class DiskMTDataset(BaseMultiDataset):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.static_cam_name = 'base_camera'
        self.gripper_cam_name = 'gripper_camera'
        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

    def _get_episode(self, file_idx, episodes_dir, episode_info) -> Path:
        def translate_intrinsic(left, top, intrinsic):
            intrinsic[0][2] -= left
            intrinsic[1][2] -= top
            return intrinsic
        def angle_between_angles(a, b):
            diff = b - a
            return (diff + np.pi) % (2 * np.pi) - np.pi
        def to_relative_action(actions, robot_obs, max_pos=0.03, max_orn=0.1):
            rel_pos = actions[:3] - robot_obs[:3]
            rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
            rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
            rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = actions[-1:]
            return np.concatenate([rel_pos, rel_orn, gripper])
        r_euler = np.array(pybullet.getEulerFromQuaternion(episode_info['ee_pose_root'][file_idx, [4,5,6,3]]))
        a_euler = np.array(pybullet.getEulerFromQuaternion(episode_info['ee_pose_root'][file_idx+1, [4,5,6,3]]))
        if np.abs(angle_between_angles(a_euler, r_euler)).max() > 0.8 and (r_euler[0]==0 or a_euler[0]==0 or r_euler[2]==0 or a_euler[2]==0): # 发生了万向节死锁
            assert False, 'Maybe gimbal lock has occurred.'
        robot_obs_euler = np.concatenate([episode_info['ee_pose_root'][file_idx,:3], r_euler, [1 if episode_info['gripper_target'][file_idx]>0.04 else -1]]) # YF: gripper_open=0/0.08, 注意要求的是-1和1
        actions_euler = np.concatenate([episode_info['ee_pose_root'][file_idx+1,:3], a_euler, [1 if episode_info['gripper_target'][file_idx+1]>0.04 else -1]])
        action = to_relative_action(actions_euler, robot_obs_euler[:6], max_pos=0.04, max_orn=0.12)
        action = np.array([-action[1], action[0], action[2], -action[4], action[3], action[5], action[6]])  # 对齐坐标系[-1, 0,2,-4,3,5,6]

        def pose_7d_to_matrix(pose):
            pose_mat = np.eye(4)
            pose_mat[:3, 3] = pose[:3]
            pose_mat[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            return pose_mat

        arm_pose_mat = pose_7d_to_matrix(episode_info['ee_pose_root'][file_idx])
        static_pose_mat = pose_7d_to_matrix(episode_info[self.static_cam_name].item()['pose'])
        if 'static' in self.static_cam_name:
            arm_base_world = pose_7d_to_matrix(episode_info['arm_base_pose_world'][file_idx])
            static_pose_mat = np.matmul(np.linalg.inv(arm_base_world), static_pose_mat)
        gripper_pose_mat = pose_7d_to_matrix(episode_info[self.gripper_cam_name].item()['pose'])
        calib = {'rgb_static': {'extrinsic_matrix': static_pose_mat,
                                'intrinsic_matrix': episode_info[self.static_cam_name].item()['intrinsics'],
                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                'rgb_gripper': {'extrinsic_matrix': np.matmul(arm_pose_mat, gripper_pose_mat),
                                'intrinsic_matrix': episode_info[self.gripper_cam_name].item()['intrinsics'],
                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}

        rgb_static = Image.open(os.path.join(episodes_dir, "%s/rgb/%04d.png" % (self.static_cam_name, file_idx)))
        rgb_gripper = Image.open(os.path.join(episodes_dir, "%s/rgb/%04d.png" % (self.gripper_cam_name, file_idx)))
        crop_area = (80, 0, 560, 480)
        rgb_static = rgb_static.crop(crop_area)
        rgb_gripper = rgb_gripper.crop(crop_area)
        calib['rgb_static']['intrinsic_matrix'] = translate_intrinsic(80, 0, calib['rgb_static']['intrinsic_matrix'])
        calib['rgb_gripper']['intrinsic_matrix'] = translate_intrinsic(80, 0, calib['rgb_gripper']['intrinsic_matrix'])
        cam_config = {'static':{'height': rgb_static.height,
                                'width': rgb_static.width,
                                'fov': np.degrees(episode_info[self.static_cam_name].item()['fovy'])}, #80
                     'gripper':{'height': rgb_gripper.height,
                                'width': rgb_gripper.width,
                                'fov': np.degrees(episode_info[self.gripper_cam_name].item()['fovy'])}}

        calib['rgb_gripper']['extrinsic_matrix'] = np.linalg.inv(calib['rgb_gripper']['extrinsic_matrix']) * SIGN_FLIP_4X4
        calib['rgb_static']['extrinsic_matrix'] = np.linalg.inv(calib['rgb_static']['extrinsic_matrix']) * SIGN_FLIP_4X4
        R = np.array([
            [0, 1, 0, 0.3],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        calib['rgb_gripper']['extrinsic_matrix']=np.dot(calib['rgb_gripper']['extrinsic_matrix'], R)
        calib['rgb_static']['extrinsic_matrix'] = np.dot(calib['rgb_static']['extrinsic_matrix'], R) # 往前为X轴；往左为Y轴；往上为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        return {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
            "rel_actions": action,
            "robot_obs": np.zeros(20),
            "scene_obs": np.zeros(20),
            "depth_static": np.array(Image.open(os.path.join(episodes_dir, "%s/depth/%04d.png" % (self.static_cam_name, file_idx))))[:, 80:-80] / 1000.,
            "depth_gripper": np.array(Image.open(os.path.join(episodes_dir, "%s/depth/%04d.png" % (self.gripper_cam_name, file_idx))))[:, 80:-80] / 1000.,
            "calib":calib,
            "cam_config": cam_config,
            "state_matrix": calib['rgb_gripper']['extrinsic_matrix'] # 注意此处是爪子矩阵
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        episode_id, start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        episodes_dir = os.path.join(self.abs_datasets_dir, "episode_%07d" % episode_id)
        episode_info = np.load(os.path.join(episodes_dir, 'episode_info.npz'),allow_pickle=True)
        episodes = [self._get_episode(file_idx, episodes_dir, episode_info) for file_idx in range(start_idx, end_idx)]

        episode = self._prepare_eposides(episodes)
        
        if self.with_lang:
            episode["language"] = self.lang_ann[episode_id]
            # if self.text_aug:
            #     task = self.lang_task[self.lang_lookup[idx]]
            #     enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
            #     episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()
        with open(os.path.join(abs_datasets_dir, 'data_info.json'), 'r') as f:
            info = json.load(f)
        
        episode_lookup, lang_ann = [], {}
        self._episode_idx = info['val_eps'] if self.validation else info['train_eps']
        for idx in self._episode_idx:
            if self.only_single_task and info['language_goal'][str(idx)] != 'Grasp the strawberry on the table': 
                continue 
            lang_ann[idx] = info['language_goal'][str(idx)]
            episode_length = info['ep_length'][str(idx)]
            for i in range(0, episode_length + 1 - (self.min_window_size+1)):
                episode_lookup.append((idx, i))
        return episode_lookup, lang_ann


class DiskChoresDataset(BaseMultiDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_ann
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)
        voxel_range = [[-2.0, 2.0], [-1.0, 3.0], [0.0, 1.0]]
        voxel_size = [0.0125*4, 0.0125*4, 0.0125*2]
        self.vfe_generator = OccupancyVFE(voxel_range, voxel_size) # 重新定义区域大小，导航需要更大的范围

    def _get_episode(self, start_idx, end_idx, traj, zip_file):
        
        action2idx = {  'm': 0,     'r': 1,     'l': 2,     'b': 3,     'end': 4, 'sub_done': 5,    'ls': 6,    'rs': 7, \
                        'p': 8,     'zm': 9,    'zp': 10,   'yp': 11,   'ym': 12, 'wp': 13,         'wm': 14, \
                        'yms': 15,  'zms': 16,  'zps': 17,  'yps': 18,  'd': 19}
        action = np.stack([[0]*6+[action2idx[traj["actions"][file_idx].decode('utf-8')]] for file_idx in range(start_idx, end_idx)]) # 注意假如后边与坐标对齐的时候，需要根据底下R矩阵来对应修改aciton

        crop_area = (86, 0, 310, 224)
        rgb_statics, rgb_grippers, depth_statics, depth_grippers = [], [], [], []
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_idx in range(start_idx, end_idx): 
                rgb_static = Image.open(load_from_zip(zip_ref, os.path.join(f'static_rgb_{file_idx}.jpg'))) # 224 396 3
                rgb_gripper = Image.open(load_from_zip(zip_ref, os.path.join(f'gripper_rgb_{file_idx}.jpg'))) # 224 396 3
                depth_static = np.load(load_from_zip(zip_ref, os.path.join(f"static_depth.npz")))['arr_0'][file_idx] # 224 396
                depth_gripper = np.load(load_from_zip(zip_ref, os.path.join(f"gripper_depth.npz")))['arr_0'][file_idx] # 224 396
                rgb_static = rgb_static.crop(crop_area)
                rgb_gripper = rgb_gripper.crop(crop_area)
                depth_static = depth_static[:, 86:-86]
                depth_gripper = depth_gripper[:, 86:-86]
                rgb_statics.append(rgb_static)
                rgb_grippers.append(rgb_gripper)
                depth_statics.append(depth_static)
                depth_grippers.append(depth_gripper)

        base_pose = traj["base_matrix"][start_idx:end_idx][()] # 4 4
        static_extrinsic_matrix = traj["calib/rgb_static/extrinsic_matrix"][start_idx:end_idx][()] # 4 4
        gripper_extrinsic_matrix = traj["calib/rgb_gripper/extrinsic_matrix"][start_idx:end_idx][()] # 4 4
        static_extrinsic_matrix = np.linalg.inv(static_extrinsic_matrix) * SIGN_FLIP_4X4   #注意这里没有求逆
        gripper_extrinsic_matrix = np.linalg.inv(gripper_extrinsic_matrix) * SIGN_FLIP_4X4 # 主要原因是deproject，对YZ加了一个符号 #注意这里没有求逆
        # 由于不同任务世界坐标系不同，这里使用本体坐标统一世界坐标系到本体上。
        static_extrinsic_matrix = np.einsum('mij,mjk->mik', static_extrinsic_matrix, base_pose)
        gripper_extrinsic_matrix = np.einsum('mij,mjk->mik', gripper_extrinsic_matrix, base_pose)
        # # 平移矩阵 T_translate
        T_translate = np.array([
            [1, 0, 0, 0.0],    # 
            [0, 1, 0, 1.0],     # 
            [0, 0, 1, 0.0],    #  
            [0, 0, 0, 1] 
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, T_translate)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, T_translate) # 为了与calvin点云坐标对齐
        R = np.array([
            [1, 0, 0, 0],
            [0,  0, -1, 0],
            [0,  1, 0, 0],
            [0,  0, 0, 1],
        ])
        static_extrinsic_matrix = np.einsum('mij,jk->mik', static_extrinsic_matrix, R)
        gripper_extrinsic_matrix = np.einsum('mij,jk->mik', gripper_extrinsic_matrix, R) # 往右X轴；往下为Y轴；往前为Z轴,需转为往右为X轴；往前为Y轴；往上为Z轴

        static_intrinsic_matrix = traj["calib/rgb_static/intrinsic_matrix"][start_idx:end_idx][()]
        gripper_intrinsic_matrix = traj["calib/rgb_gripper/intrinsic_matrix"][start_idx:end_idx][()]
        static_intrinsic_matrix[:, 0, 0] = static_intrinsic_matrix[:, 1, 1]
        static_intrinsic_matrix[:, 0, 2] = static_intrinsic_matrix[:, 1, 2]
        gripper_intrinsic_matrix[:, 0, 0] = gripper_intrinsic_matrix[:, 1, 1]
        gripper_intrinsic_matrix[:, 0, 2] = gripper_intrinsic_matrix[:, 1, 2]

        cam_config = {  "static": {"height": rgb_statics[0].height, "width": rgb_statics[0].width, "fov": traj["cam_config/static_fov"][start_idx][()]},
                            "gripper":{"height": rgb_grippers[0].height, "width": rgb_grippers[0].width, "fov": traj["cam_config/gripper_fov"][start_idx][()]},}
        return [{   "rgb_static": rgb_static_,
                    "rgb_gripper": rgb_gripper_,
                    "depth_static": depth_static_,
                    "depth_gripper": depth_gripper_,
                    "rel_actions": action_,
                    "robot_obs": np.zeros(20),
                    "scene_obs": np.zeros(20),
                    "calib": {  'rgb_static': { 'extrinsic_matrix': static_extrinsic_matrix_,         
                                                'intrinsic_matrix': static_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                                'rgb_gripper': {'extrinsic_matrix': gripper_extrinsic_matrix_,
                                                'intrinsic_matrix': gripper_intrinsic_matrix_,
                                                'distCoeffs_matrix': np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}},
                    "cam_config": cam_config,
                    "state_matrix": gripper_extrinsic_matrix_,  # 注意此处是爪子矩阵
        }for (rgb_static_, rgb_gripper_, depth_static_, depth_gripper_, action_, static_extrinsic_matrix_, static_intrinsic_matrix_, gripper_extrinsic_matrix_, gripper_intrinsic_matrix_) in zip(rgb_statics, rgb_grippers, depth_statics, depth_grippers, action, static_extrinsic_matrix, static_intrinsic_matrix, gripper_extrinsic_matrix, gripper_intrinsic_matrix)]


    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # idx=random.choice(list(range(12))); print("idx==0")
        dataset_file, episode_id, start_idx, episode_length, task = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        with h5py.File(dataset_file, "r") as f_out:
            house = f_out[str(task)]
            traj = house[str(episode_id)]
            zip_file = os.path.join(self.abs_datasets_dir, "train", str(task), f"{str(episode_id)}.zip")
            episodes = self._get_episode(start_idx, end_idx, traj, zip_file)

        episode = self._prepare_eposides(episodes)

        if self.with_lang:
            episode["language"] = self.lang_ann[f"{task}_{episode_id}"]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        lang_ann, episode_lookup = {}, []
        subset = "train" # zyf 默认为训练集！！！
        house_id_to_sub_house_id_json = os.path.join(abs_datasets_dir, f"house_id_to_sub_house_id_{subset}.json")
        with open(house_id_to_sub_house_id_json, "r") as f:
            house_id_to_sub_house_id = json.load(f)
        
        for task in list(house_id_to_sub_house_id.keys())[:100]: # 加载不同任务
            if self.only_single_task and "000000" != task: continue # 测试使用
            dataset_file = os.path.join(abs_datasets_dir, subset, task, "hdf5_sensors.hdf5")
            f_out = h5py.File(dataset_file, "r")
            house = f_out[str(task)]
            for traj_id in house_id_to_sub_house_id[task]: # 记录每一条轨迹的信息
                # 构建episode_lookup
                zip_file = os.path.join(abs_datasets_dir, subset, str(task), f"{str(traj_id)}.zip")
                if not os.path.exists(zip_file): continue
                if traj_id not in house: continue
                traj = house[str(traj_id)]
                lang_ann[f"{task}_{traj_id}"] = traj["language"][()].decode('utf-8')
                episode_length = traj["base_matrix"].shape[0]
                for start_frame_index in range(0, episode_length + 1 - self.min_window_size):
                    episode_lookup.append((dataset_file, traj_id, start_frame_index, episode_length, task))
            f_out.close()

        return episode_lookup, lang_ann


class DatasetBase(BaseMultiDataset):

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        delta_timestamps = {}
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(self.abs_datasets_dir)

        his_range = list(range(self.window_size))
        hist_objs_config = {"image":        his_range, 
                            "wrist_image":  his_range,
                            "state":        his_range,
                            "actions":      his_range,
                            "extrinsic_matrix": his_range,
                            "intrinsic_matrix": his_range,
                            "wrist_extrinsic_matrix": his_range,
                            "wrist_intrinsic_matrix": his_range,
                            "robot_obs": his_range,
                            "goals": his_range,
                            "fov": his_range,
                            "wrist_fov": his_range,
                            "depth": his_range,
                            "wrist_depth": his_range,
                            }
        for obj_key, obj_config in hist_objs_config.items():
            delta_timestamps.update(
                {obj_key: [t / dataset_meta.fps for t in obj_config]}
            )
        self.data_loader = LeRobotDataset(  # 数据读取配置
            repo_id=self.abs_datasets_dir,
            delta_timestamps=delta_timestamps,
        )

    def __len__(self) -> int:
        return len(self.data_loader)

    def _compute_relative_action(self, robot_obs: np.ndarray, goal_obs: np.ndarray, gripper_action: float) -> np.ndarray:
        """
        Compute relative action from robot observation and goal.
        
        Args:
            robot_obs: Robot observation array (at least 6 elements: xyz + euler angles)
            goal_obs: Goal observation array (at least 6 elements: xyz + euler angles)
            gripper_action: Gripper action value
            
        Returns:
            Relative action array with coordinate transformation applied.
            See: https://km.sankuai.com/collabpage/2537354641
        """
        # Convert euler angles to rotation matrices
        robot_obs_ort = np.array(pybullet.getMatrixFromQuaternion(
            pybullet.getQuaternionFromEuler(robot_obs[3:6])
        )).reshape(3, 3)
        actions_ort = np.array(pybullet.getMatrixFromQuaternion(
            pybullet.getQuaternionFromEuler(goal_obs[3:6])
        )).reshape(3, 3)
        
        # Coordinate system rotation matrix
        B = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        
        # Apply coordinate transformation
        robot_obs_ort = np.dot(B, robot_obs_ort)
        actions_ort = np.dot(B, actions_ort)
        
        # Compute relative rotation
        robot_obs_ort_inv = np.linalg.inv(robot_obs_ort)
        rel_actions_ort = np.dot(actions_ort, robot_obs_ort_inv)
        quat_error = robosuite_trans.mat2quat(rel_actions_ort)
        delta_ort = robosuite_trans.quat2axisangle(quat_error) / 0.05
        
        # Compute relative position
        delta_pos = (goal_obs[:3] - robot_obs[:3]) / 0.02
        
        # Combine position, orientation, and gripper action
        action = np.array([*delta_pos, *delta_ort, gripper_action])
        
        # Apply coordinate transformation to action
        return np.array([action[1], -action[0], action[2], action[3], action[4], action[5], action[6]])

    def _build_frame_episode(self, item: Dict, file_idx: int) -> Dict:
        """
        Build a single frame's episode dictionary from raw data item.
        
        Args:
            item: Raw data item from data loader
            file_idx: Frame index within the window
            
        Returns:
            Dictionary containing frame-level observations and metadata
        """
        episode = {}
        
        # Extract RGB images
        episode['rgb_static'] = Image.fromarray(
            (item['image'][file_idx].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        episode['rgb_gripper'] = Image.fromarray(
            (item['wrist_image'][file_idx].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        
        # Extract depth maps
        episode['depth_static'] = item['depth'][file_idx].numpy()
        episode['depth_gripper'] = item['wrist_depth'][file_idx].numpy()
        
        # Extract calibration matrices
        episode['calib'] = {
            "rgb_static": {
                "extrinsic_matrix": item['extrinsic_matrix'][file_idx].numpy(),
                "intrinsic_matrix": item['intrinsic_matrix'][file_idx].numpy(),
                "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            },
            "rgb_gripper": {
                "extrinsic_matrix": item['wrist_extrinsic_matrix'][file_idx].numpy(),
                "intrinsic_matrix": item['wrist_intrinsic_matrix'][file_idx].numpy(),
                "distCoeffs_matrix": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            }
        }
        
        # Extract camera configuration
        episode['cam_config'] = {
            "static": {
                "height": 200,
                "width": 200,
                "fov": item['fov'][file_idx].numpy()
            },
            "gripper": {
                "height": 84,
                "width": 84,
                "fov": item['wrist_fov'][file_idx].numpy()
            }
        }
        
        # Fixed gripper camera to gripper transform
        episode['gripper_cam2gripper'] = np.array([
            [-0.0008, 0.9999, 0., -0.0001],
            [0.9635, 0.0008, -0.2674, 0.0589],
            [-0.2674, 0.0000, -0.9635, -0.1616],
            [0., 0., 0., 1.]
        ])
        
        # Extract robot state
        episode['state_matrix'] = item['wrist_extrinsic_matrix'][file_idx].numpy()
        episode['robot_obs'] = item['robot_obs'][file_idx].numpy()
        episode['scene_obs'] = np.zeros(20)
        
        # Compute relative action
        robot_obs = item['robot_obs'][file_idx].numpy()
        goal_obs = item['goals'][file_idx].numpy()
        gripper_action = item['actions'][file_idx].numpy()[-1]
        episode['rel_actions'] = self._compute_relative_action(robot_obs, goal_obs, gripper_action)
        
        return episode

    def _process_frame_data(
        self, 
        ep: Dict, 
        colour_aug_random: int, 
        jitter_state: Optional[Tuple] = None
    ) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Process a single frame's data: generate point cloud, apply color augmentation, and prepare calibration.
        
        Args:
            ep: Frame episode dictionary
            colour_aug_random: Random value for color augmentation decision
            jitter_state: Optional state for consistent color jitter across frames
            
        Returns:
            Tuple of (rgb_dict, pcd_array, calib_dict)
        """
        calib = ep['calib']
        cam_config = ep['cam_config']
        static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
        gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
        
        # Create camera objects
        static_cam = cam(
            static_extrinsic_matrix,
            cam_config['static']['height'],
            cam_config['static']['width'],
            cam_config['static']['fov']
        )
        gripper_cam = cam(
            gripper_extrinsic_matrix,
            cam_config['gripper']['height'],
            cam_config['gripper']['width'],
            cam_config['gripper']['fov']
        )
        
        # Deproject depth to point clouds
        static_pcd = deproject(
            static_cam, ep['depth_static'], homogeneous=False, sanity_check=False
        ).transpose(1, 0)
        gripper_pcd = deproject(
            gripper_cam, ep['depth_gripper'], homogeneous=False, sanity_check=False
        ).transpose(1, 0)
        cloud = np.concatenate([static_pcd, gripper_pcd], axis=0)
        
        # Prepare RGB images
        rgb = {
            'rgb_static': ep['rgb_static'],
            'rgb_gripper': ep['rgb_gripper']
        }
        
        # Apply color augmentation if enabled
        if colour_aug_random > 2 and self.use_colour_aug:
            rgb['rgb_static'], rgb['rgb_gripper'], jitter_state = apply_color_jitter_pair(
                self.ColorJitter, rgb['rgb_static'], rgb['rgb_gripper'], jitter_state
            )
        
        # Generate point cloud with RGB
        static_rgb = np.reshape(np.array(rgb['rgb_static']), (-1, 3))
        gripper_rgb = np.reshape(np.array(rgb['rgb_gripper']), (-1, 3))
        pcd_rgb = np.concatenate([static_rgb, gripper_rgb], axis=0) / 255
        pcd = self.vfe_generator.generate(cloud[:, :3], pcd_rgb)
        
        # Prepare calibration matrices with sign flip
        calib['static_extrinsic_matrix'] = static_extrinsic_matrix * SIGN_FLIP_4X4
        calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
        calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']
        calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix * SIGN_FLIP_4X4
        calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
        calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']
        calib['state_matrix'] = ep['state_matrix'] * SIGN_FLIP_4X4
        calib['gripper_cam2gripper'] = ep['gripper_cam2gripper'] * SIGN_FLIP_4X4
        calib['static_fov'] = np.array([
            cam_config['static']['fov'],
            cam_config['static']['height'],
            cam_config['static']['width']
        ])
        calib['gripper_fov'] = np.array([
            cam_config['gripper']['fov'],
            cam_config['gripper']['height'],
            cam_config['gripper']['width']
        ])
        
        return rgb, pcd, calib

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load and process an episode from the dataset.
        
        Args:
            idx: Episode index
            window_size: Number of frames in the episode window
            
        Returns:
            Dictionary containing processed episode data
        """
        item = self.data_loader.__getitem__(idx)
        
        # Build frame-level episodes
        episodes = [self._build_frame_episode(item, file_idx) for file_idx in range(window_size)]
        
        # Get observation keys (excluding language and RGB which are handled separately)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        keys.remove("rgb_static")
        keys.remove("rgb_gripper")
        
        # Process all frames: generate point clouds, apply augmentation, prepare calibration
        calibs = []
        pcds = []
        rgbs = []
        colour_aug_random = random.randint(0, 10)
        jitter_state = None
        
        for ep in episodes:
            rgb, pcd, calib = self._process_frame_data(ep, colour_aug_random, jitter_state)
            rgbs.append(rgb)
            pcds.append(pcd)
            calibs.append(calib)
        
        # Merge all frames into final episode dictionary
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode.update({key: np.stack([calib[key] for calib in calibs]) for key in keys_calib})
        episode.update({key: [rgb[key] for rgb in rgbs] for key in RGB_KEYS})
        episode['pcd'] = np.stack(pcds)
        episode["language"] = item['task']
        
        return episode


    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        resample = True
        max_data_fetch_iteration = 300
        last_error = None

        for fetch_iteration in range(max_data_fetch_iteration):
            try:
                window_size = self.window_size
                return self._get_sequences(idx, window_size)
            except Exception as e:
                last_error = e
                print(f"[dataset] fetch_iter={fetch_iteration} idx={idx} error={e}")
                if not resample:
                    return None
                idx = random.randint(0, len(self) - 1)

        # Exhausted retries
        raise RuntimeError(f"Failed to fetch data after {max_data_fetch_iteration} attempts. Last error: {last_error}")

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        episode = self._load_episode(idx, window_size)
        if episode == 0:
            return 0
        
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )  # 保持不变，取episode['robot_obs']，并截取前15个
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)  # 执行self.transforms[rgb_obs_key]
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)  # 直接取depth
        seq_acts = process_actions(episode, self.observation_space, self.transforms)  # 直接取rel_actions
        info = get_state_info_dict(episode)
        seq_calib_obs = self.process_calib(episode)
        seq_pcd_obs = self.process_pcd(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **seq_calib_obs,
            **seq_pcd_obs,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        return seq_dict
        
class DiskMultiDataset(Dataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        data_types,
        datasets_dir,
        **kwargs: Any,
    ):
        # kwargs_cpy = copy.deepcopy(kwargs)
        data_path_list = kwargs.pop("data_path_list")
        state_matrixs_path = kwargs.pop("state_matrixs_path")
        self.datas_loder = []
        self.datas_len = [0] # 注意datas_len的长度比self.datas_loder多一个
        if not isinstance(data_types, list):
            data_types = [data_types] 
            datasets_dir = [datasets_dir]
            state_matrixs_path = [state_matrixs_path]
            data_path_list = [data_path_list]
        for ith, (data_type, dataset_dir) in enumerate(zip(data_types, datasets_dir)):
            
            if data_type == "calvin":
                data_loder = DiskCalvinDataset(
                    datasets_dir=Path(os.path.join(dataset_dir, "training")),
                    state_matrixs_path=state_matrixs_path[ith],
                    data_path_list=data_path_list[ith],
                    **kwargs)
            elif 'metaworld' in data_type:
                data_loder=DiskMetaWorldDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'libero' in data_type:
                data_loder=DiskLiberoDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'robocasa' in data_type:
                data_loder=DiskRobocasaDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'robomimic' in data_type:
                data_loder=DiskRoboMimicDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'maniskill2' in data_type:
                data_loder=DiskManiSkill2Dataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'rlbench' in data_type:
                data_loder=DiskRlbenchDataset(
                    datasets_dir=Path(os.path.join(dataset_dir)), # train
                    **kwargs)
            elif 'colosseum' in data_type:
                data_loder=DiskColosseumDataset(
                    datasets_dir=Path(os.path.join(dataset_dir)), # train
                    **kwargs)
            elif 'mt' in data_type:
                data_loder=DiskMTDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'chores' in data_type:
                data_loder=DiskChoresDataset(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            elif 'lerobot' in data_type:
                data_loder = DatasetBase(
                    datasets_dir=Path(dataset_dir),
                    **kwargs)
            else:
                assert False
            data_loder.data_name = data_type
            self.datas_loder.append(data_loder)
            self.datas_len.append(len(data_loder))
            print(f"loader {data_type} from {dataset_dir} with {len(data_loder)}")

        self.datas_len = np.cumsum(np.array(self.datas_len)) # 注意datas_len的长度比self.datas_loder多一个
        print(f"all loadered {self.datas_len[-1]} data")

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return self.datas_len[-1]

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        assert isinstance(idx, int)
        for i in range(len(self.datas_loder)):
            if idx >= self.datas_len[i] and idx <= self.datas_len[i+1]:
                # st = time.time()
                sample = self.datas_loder[i].__getitem__(int(idx-self.datas_len[i]))
                # if torch.distributed.get_rank() == 0:
                #     # self.datas_loder[i].tmp.append(time.time() - st)
                #     print(self.datas_loder[i].data_name, idx-self.datas_len[i], time.time() - st)
                return sample

    def collater(self, sample):
        # st = time.time()
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        static_extrinsic_matrix = torch.stack([s["calib_obs"]["static_extrinsic_matrix"] for s in sample])
        static_intrinsic_matrix = torch.stack([s["calib_obs"]["static_intrinsic_matrix"] for s in sample])
        static_distCoeffs_matrix = torch.stack([s["calib_obs"]["static_distCoeffs_matrix"] for s in sample])
        gripper_extrinsic_matrix = torch.stack([s["calib_obs"]["gripper_extrinsic_matrix"] for s in sample])
        gripper_intrinsic_matrix = torch.stack([s["calib_obs"]["gripper_intrinsic_matrix"] for s in sample])
        gripper_distCoeffs_matrix = torch.stack([s["calib_obs"]["gripper_distCoeffs_matrix"] for s in sample])
        state_matrix = torch.stack([s["calib_obs"]["state_matrix"] for s in sample])
        static_fovs = torch.stack([s["calib_obs"]["static_fov"] for s in sample])
        gripper_fovs = torch.stack([s["calib_obs"]["gripper_fov"] for s in sample])
        gripper_cam2gripper = torch.stack([s["calib_obs"]["gripper_cam2gripper"] for s in sample])

        # 使用多线程并行处理样本
        def process_sample(flag, sample, datas_loder):
            # st = time.time()
            if flag == 'text':
                stacked_language = [s["lang"] for s in sample]
                text_tensors, attention_mask = datas_loder[0].text_fn(stacked_language)
                # print(f"YF: {flag} {time.time()-st}")
                return flag, text_tensors, attention_mask
            elif flag == 'occ':
                pcd = torch.stack([s["pcd_obs"]["pcd"] for s in sample])
                # pcd = [s["pcd_obs"]["pcd"] for s in sample]
                # print(f"YF: {flag} {time.time()-st}")
                return flag, pcd
            elif flag == 'static':
                # image_tensors = torch.stack([datas_loder[0].image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
                image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
                if datas_loder[0].rgb_pad != -1:
                    bs, seq_len = image_tensors.shape[:2]
                    if datas_loder[0].traj_cons:
                        image_tensors = datas_loder[0].rgb_shift.forward_traj(image_tensors)
                    else:
                        image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                        image_tensors = datas_loder[0].rgb_shift(image_tensors)
                        image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
                # print(f"YF: {flag} {time.time()-st}")
                return flag, image_tensors
            elif flag == 'gripper':
                # gripper_tensors = torch.stack([datas_loder[0].image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
                gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])
                if datas_loder[0].gripper_pad != -1:
                    bs, seq_len = gripper_tensors.shape[:2]
                    if datas_loder[0].traj_cons:
                        gripper_tensors = datas_loder[0].gripper_shift.forward_traj(gripper_tensors)
                    else:
                        gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                        gripper_tensors = datas_loder[0].gripper_shift(gripper_tensors)
                        gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
                # print(f"YF: {flag} {time.time()-st}")
                return flag, gripper_tensors
        partial_process_sample = functools.partial(process_sample, sample=sample, datas_loder=self.datas_loder)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(partial_process_sample, ['text', 'occ', 'static', 'gripper']))
        result_dict = {flag: result for flag, *result in results}
        # print(f"YF: collater2: {time.time()-st}")
        text_tensors, attention_mask = result_dict['text']
        pcd = result_dict['occ'][0]
        image_tensors = result_dict['static'][0]
        gripper_tensors = result_dict['gripper'][0]
        # print(f"YF: collater3: {time.time()-st}")
        robot_obs = torch.zeros(1)
        if self.datas_loder[0].act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.datas_loder[0].window_size, self.datas_loder[0].act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.datas_loder[0].window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.datas_loder[0].act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.datas_loder[0].window_size, self.datas_loder[0].act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.datas_loder[0].window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.datas_loder[0].act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.datas_loder[0].act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.datas_loder[0].act_step-1)]
            state_tensors = state_tensors[:, :-(self.datas_loder[0].act_step-1)]
        
        idxes = torch.stack([torch.from_numpy(np.array(s["idx"])) for s in sample])
        # print(f"YF: collater: {time.time()-st}")
        return image_tensors, (text_tensors, attention_mask), action_tensors, gripper_tensors, state_tensors, robot_obs,(static_extrinsic_matrix,
                static_intrinsic_matrix,static_distCoeffs_matrix,gripper_extrinsic_matrix,gripper_intrinsic_matrix,gripper_distCoeffs_matrix,state_matrix,static_fovs,gripper_fovs,gripper_cam2gripper),pcd,idxes


def CalvinEvalSeq(
                 env,
                 dataset_path,
                 initial_state, eval_sequence,
                 val_annotations,
                 task_oracle,
                 transforms={},
                 EP_LEN = 360
                 ):
        if env is None:
            env = get_env(Path(dataset_path), show_gui=False)  # make_env(dataset_path)
        """
        Evaluates a sequence of language instructions.
        """
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        reset = False
        success_counter = 0
        is_reset = False
        for subtask_i, subtask in enumerate(eval_sequence):

            if subtask_i > 0: break  # 只测试task 0

            planned_actions = []
            if robot_obs is not None and scene_obs is not None and reset:
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                is_reset = True
            obs = env.get_obs()
            # get lang annotation for subtask
            lang_annotation = val_annotations[subtask][0]
            lang_annotation = lang_annotation.split('\n')[0]
            if '\u2019' in lang_annotation:
                lang_annotation.replace('\u2019', '\'')
            start_info = env.get_info()

            success = False # determine success or not

            for step in range(EP_LEN):
                
                ret = {
                    "success_before": success,
                    "lang": lang_annotation,
                    "subtask_i": subtask_i,
                    "eval_sequence": eval_sequence,
                    "success_counter": success_counter,
                    "step_cur": step,
                    "step_max": EP_LEN,
                    "is_reset": is_reset,
                    "rgb_static": transforms['rgb_static'](Image.fromarray(obs['rgb_obs']['rgb_static'])),
                    "rgb_gripper": transforms['rgb_gripper'](Image.fromarray(obs['rgb_obs']['rgb_gripper'])),
                    "rgb_static_ori": obs['rgb_obs']['rgb_static'],
                    "rgb_gripper_ori": obs['rgb_obs']['rgb_gripper'],
                    "robot_obs": obs['robot_obs'],
                    "done": False
                }
                action = yield ret
                if len(planned_actions) == 0:
                    if action.shape == (7,):
                        planned_actions.append(action)
                    else:
                        planned_actions.extend([action[i] for i in range(action.shape[0])])
                action = planned_actions.pop(0)
                obs, _, _, current_info = env.step(action)

                # check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    success = True
                    break
            if not success:
                break
            else:
                success_counter += 1
        
        ret = {"eval_sequence": eval_sequence,
                "success_counter": success_counter,
                "done": True
                }
        _ = yield ret
        
class CalvinSim(Dataset):
    def __init__(self,  
                 dataset_path,
                 calvin_conf_path,
                 calvin_seq_path,
                 transforms={},
                 NUM_SEQUENCES = 300,
                 ):
        super(CalvinSim, self).__init__()

        self.dataset_path = dataset_path

        conf_dir = Path(calvin_conf_path) # 
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.task_oracle = hydra.utils.instantiate(task_cfg)

        self.val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml") # 导入语言标注
        with open(calvin_seq_path, 'r') as f:
            self.eval_sequences = json.load(f)
        self.eval_sequences = self.eval_sequences[:NUM_SEQUENCES]
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.eval_sequences)
    
    def __getitem__(self, idx: int, fixed_seed=False):

        initial_state, eval_sequence = self.eval_sequences[idx]
        EP_LEN = 180
        return {"generator": CalvinEvalSeq,
                "dataset_path": self.dataset_path,
                "initial_state": initial_state,
                "eval_sequence": eval_sequence,
                "val_annotations": self.val_annotations,
                "task_oracle": self.task_oracle,
                "transforms": self.transforms,
                "EP_LEN": EP_LEN
                }
    
    def collater(self, sample):

        return sample

class CalvinDataset(Dataset):
    """Naive implementation of dataset to store
    calvin debug dataset, may be changed to WDS for the full dataset
    """

    def __init__(self, image_fn, text_fn, dataset_path, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.image_fn = image_fn
        self.text_fn = text_fn

        tag = "training" if is_train else "validation"
        self.file_prefix = f"{self.dataset_path}/{tag}"
        self.anns = np.load(
            f"{self.file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()
        self.tag = tag

    def __len__(self):
        return len(self.anns["info"]["indx"])

    def __getitem__(self, index):
        task = self.anns["language"]["task"][index]
        text = self.anns["language"]["ann"][index]
        st, ed = self.anns["info"]["indx"][index]
        # CJ: randomly sample a datapoint in the episode
        frame = random.randint(st, ed)
        frame = np.load(
            f"{self.file_prefix}/episode_{frame:07d}.npz"
        )  # , allow_pickle=True (lazy load)
        rgb_static = Image.fromarray(frame["rgb_static"])
        rgb_gripper = Image.fromarray(frame["rgb_gripper"])
        actions = np.array(frame["rel_actions"])
        
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        return rgb_static, text, actions

    def collater(self, sample):
        images = [s[0] for s in sample]
        texts = [s[1] for s in sample]
        actions = [s[2] for s in sample]

        image_tensors = self.image_fn(images)
        text_tensors = self.text_fn(texts)
        action_tensors = torch.FloatTensor(np.stack(actions))
        return image_tensors, text_tensors, action_tensors


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix(), allow_pickle=True)
    #return np.load(filename.as_posix())


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


def preprocess_text_calvin(sample, tokenizer, sample_mode=-1, window_size=12):
    tokenizer.padding_side = "right"
    # sample = [
    #     # (f"{s.strip()}{tokenizer.eos_token}")
    #     # for s in sample
    #     (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    # ]
    sample_modes = [0,1,2,3,4]
        #sample_modes = [3]
    sample_mode = random.choice(sample_modes) if sample_mode<0 else sample_mode 
    # YF：下边添加f,|endofchunk|>{tokenizer.eos_token}变为<|endofchunk|><|endoftext|>
    if sample_mode == 0: # 图片+action 
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<static_s><static0><static1><static2><static3><static4><static5><static6><static7><static_e><gripper_s><gripper0><gripper1><gripper2><gripper3><gripper4><gripper5><gripper6><gripper7><gripper_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 1: # action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 2: # occ+action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<obs_s><obs0><obs1><obs2><obs3><obs4><obs5><obs6><obs7><obs_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 3: # image+occ+action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<static_s><static0><static1><static2><static3><static4><static5><static6><static7><static_e><gripper_s><gripper0><gripper1><gripper2><gripper3><gripper4><gripper5><gripper6><gripper7><gripper_e><obs_s><obs0><obs1><obs2><obs3><obs4><obs5><obs6><obs7><obs_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    if sample_mode == 4:  # occ+image+action
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<obs_s><obs0><obs1><obs2><obs3><obs4><obs5><obs6><obs7><obs_e><static_s><static0><static1><static2><static3><static4><static5><static6><static7><static_e><gripper_s><gripper0><gripper1><gripper2><gripper3><gripper4><gripper5><gripper6><gripper7><gripper_e><action>"*window_size+f"<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    text = tokenizer(
        sample,
        max_length=1024,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"] # YF: input_ids为：50278, 29397,   598，其中50278为<image>， 29397为pick/slide/grasp等动词


def preprocess_interleaved(sample, tokenizer, clip_processor, sim_threshold):
    assert False # 可能有内存泄漏io.BytesIO
    info = json.loads(sample[0])
    tar_file_obj = io.BytesIO(sample[1])
    image_tar = tarfile.open(fileobj=tar_file_obj)
    sentences = info["text_list"]

    images, image_idxs = [], []
    for image_path, sim in zip(info["image_info"], info["similarity_matrix"]):
        # pick one image per sentence
        if info["image_info"][image_path]["matched_text_index"] in image_idxs:
            continue
        rawbytes = image_tar.extractfile(
            os.path.join(image_tar.getnames()[0], image_path)
        ).read()

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sim[info["image_info"][image_path]["matched_text_index"]] < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        image_idxs.append(info["image_info"][image_path]["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    # filter out images that are exact duplicates
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), MAX_NUM_IMAGES))
    images_tensors = images_tensors[keep_ixs]
    image_idxs = [image_idxs[ix] for ix in keep_ixs]

    # pad to 5 images
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros(
            (MAX_NUM_IMAGES - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # add in <image> and <eoc> tokens
    # eoc after sentence = "sentence loss"
    for ix in image_idxs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"

    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )

    if num_images == 0:
        raise ValueError("No images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_coco_dataset(args, image_processor, tokenizer, epoch=0):
    assert False # YF 代码未用
    coco_data_dir = "path/to/coco/train2014"
    coco_ann = "path/to/coco/annotations/captions_train2014.json"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    coco_dataset = CaptionDataset(coco_data_dir, coco_ann, preprocess_text_fn, image_processor)
    
    sampler = DistributedSampler(
        coco_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    
    dataloader = DataLoader(
        coco_dataset,
        batch_size=args.training.batch_size_vl,
        pin_memory=False,
        num_workers=args.training.workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=coco_dataset.collator,
        drop_last=True
    )
    
    return dataloader


def get_vqa_dataset(args, image_processor, tokenizer, epoch=0):
    assert False # YF 代码未用
    vqa_data_dir = "path/to/vqav2/train2014"
    vqa_questions = "path/to/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
    vqa_ann = "path/to/vqav2/v2_mscoco_train2014_annotations.json"
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    vqa_dataset = VQADataset(vqa_data_dir, vqa_questions, vqa_ann, preprocess_text_fn, image_processor)
    
    sampler = DistributedSampler(
        vqa_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    
    dataloader = DataLoader(
        vqa_dataset,
        batch_size=args.training.batch_size_vl,
        pin_memory=False,
        num_workers=args.training.workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=vqa_dataset.collator,
        drop_last=True
    )
    
    return dataloader


def get_multi_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.data.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    # preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_image_fn = functools.partial(lambda X: X)
    transforms = dict()
    transforms["rgb_static"] = image_processor
    transforms["rgb_gripper"] = image_processor
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, sample_mode=args.sample_mode, window_size=args.action_decoder.window_size)
    
    print(f"********************YF:sample_mode={args.sample_mode}***********************")
    if hasattr(args, 'data_tasks_groups'):
        if args.data.data_tasks_groups == 'None':  data_tasks_groups = None
        else: data_tasks_groups = args.data.data_tasks_groups
    else:
        data_tasks_groups = None
    calvin_dataset = DiskMultiDataset(
        data_types=args.data.data_type,
        datasets_dir=dataset_path,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.action_decoder.window_size if args.action_decoder.window_size==1 else args.action_decoder.window_size+1,  # 注意这里，假如这里改的时候，后边_build_file_indices_lang也需要改 +1是为了生成下一张图片
        rgb_pad=args.cameras.rgb_pad,
        gripper_pad=args.cameras.gripper_pad,
        traj_cons=args.action_decoder.traj_cons,
        text_aug=args.data.text_aug,
        dif_ws=args.action_decoder.dif_ws,
        min_window_size=args.action_decoder.min_window_size,
        max_window_size=args.action_decoder.max_window_size,
        act_step=args.action_decoder.multi_step_action,
        partial_data=args.data.partial_data,
        colour_aug=args.data.colour_aug,
        data_path_list = args.data.data_path_list,
        state_matrixs_path = args.data.state_matrixs_path,
        data_tasks_groups = data_tasks_groups,
        env_resample = args.data.env_resample,
        only_single_task = args.data.only_single_task,
        transforms=transforms,
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset) # 总样本数
    global_batch_size = args.training.batch_size_calvin * args.distributed.world_size  # 总batch size
    num_batches = round_fn(num_samples / global_batch_size)  # 总batch数
    num_workers = max(1, args.training.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.distributed.world_size,
        rank=args.distributed.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.training.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collater,
        drop_last=True
    )
    # dataloader = DataLoader(calvin_dataset, batch_size=args.training.batch_size_calvin)
    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)




def load_partial_traj_data():
    with open('partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data


