"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

import collections
import copy
import dataclasses
import json
import logging
import os
import pathlib
import sys
import functools
from collections import deque
import time
from pathlib import Path
from typing import Any, List, Tuple, Optional
import multiprocessing as mp
from omegaconf import OmegaConf
import hydra
from moviepy.editor import ImageSequenceClip
import pybullet as pb
import math
import imageio
import numpy as np
import torch
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from termcolor import colored
from tqdm import tqdm
import tyro

# # Add Calvin to path
# CALVIN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "calvin"
# sys.path.insert(0, str(CALVIN_ROOT))

from calvin_agent.evaluation.utils import (
    get_env_state_for_initial_condition,
    collect_plan,
    get_log_dir,
    print_and_save,
    count_success,
    create_tsne,
)
from collections import defaultdict
from calvin_env.envs.play_table_env import get_env


# Set OpenGL platform for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MUJOCO_GL'] = 'osmesa'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = 360  # Max steps per task
EPS = np.finfo(float).eps * 4.0

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8020
    # Multiple policy servers for load-balancing / parallel eval.
    # Format: "host1:port1,host2:port2". If empty, fall back to host/port.
    servers: str = ""
    only_once: bool = True

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/mnt/data/datasets/calvin/task_D_D"  # Path to Calvin dataset
    calvin_config_path: str = "/mnt/data/yanfeng/project/openpi/third_party/calvin/calvin_models/conf"
    eval_sequences_path: str = "/mnt/data/yanfeng/project/openpi/examples/calvin/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    seed: int = 0
    create_plan_tsne: bool = False
    window_size: int = 12
    
    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)

def get_gripper_camera_view_matrix(cam):
    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid,
        linkIndex=cam.gripper_cam_link,
        physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = pb.computeViewMatrix(
        camera_pos, camera_pos + cam_rot_y, -cam_rot_z
    )
    return view_matrix


def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q

def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""
    
    def __init__(self, host: str, port: int, window_size: int):
        self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        # self.tokenizer = tokenizer
        # self.replan = model.module.replan
        # self.decoder_type = model.module.decoder_type
        # self.cast_type = cast_dtype
        # self.use_diff = use_diff
        # self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, action_token = args.action_token,  multi_action_token=args.multi_action_token)
        # self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, sample_mode=args.sample_mode) # 注意此处是输出图片+OCC+action
        # self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        # self.action_hist_queue = []
        # self.feature_cache = None
        # self.dt_feat_cache = []
        # self.fusion_mode = self.model.module.fusion_mode
        # self.args = args
        
        # if use_diff:
        #     self.diffusion_model = None
        #     self.normalizer = None
        #     if isinstance(self.model, DistributedDataParallel):
        #         self.diffusion_model = self.model.module.diffusion_model
        #     else:
        #         self.diffusion_model = self.model.diffusion_model
        #     action_dim = self.diffusion_model.data_dim
        #     horizon = self.diffusion_model.horizon
        #     self.normalizer = self.diffusion_model.normalizer
        #     self.action_hist_queue = deque(maxlen=history_len-1)
        #     self.action_hist_queue.extend([np.zeros(action_dim) for _ in range(history_len-1)])

        #     if horizon-history_len+1:
        #         self.supp = None
        #     self.hist_len = history_len-1
        #     self.action_dim = action_dim
        #     self.horizon = horizon
        #     self.future_act_len = future_act_len
        
        # if self.model.module.pad_length != -1:
        # if self.model.module.pad_length == -1:
        #     history_len = self.model.module.window_size
        self.history_len = window_size
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.calib_queue = deque(maxlen=self.history_len)
        
    def reset(self): 
        """
        This is called
        """
        # if self.use_diff:
        #     self.action_hist_queue = deque(maxlen=self.hist_len)
        #     self.action_hist_queue.extend([np.zeros(self.action_dim) for _ in range(self.hist_len)])
        # if self.model.module.pad_length != -1:
        #     history_len = self.model.module.pad_length
        # else:
        #     history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.calib_queue = deque(maxlen=self.history_len)
        # self.feature_cache = None
        # self.dt_feat_cache = []
        
        # self.model.module.lang_encoder.lm_head.hidden_state = None
        # self.model.module.lang_encoder.lm_head.history_memory = []

        # if self.model.module.sep_lm_head:
        #     self.model.module.lm_head.hidden_state = None
        #     self.model.module.lm_head.history_memory = []
        
    def step(self, obs: dict, lang_annotation: str, env: Any) -> np.ndarray:
        
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """

        static_cam_env = env.cameras[0]
        gripper_cam_env = env.cameras[1]
        gripper_cam_env.viewMatrix = get_gripper_camera_view_matrix(gripper_cam_env)


        static_extrinsic = np.array(static_cam_env.viewMatrix).reshape((4, 4)).T
        gripper_extrinsic = np.array(gripper_cam_env.viewMatrix).reshape((4, 4)).T
        static_foc = static_cam_env.height / (2 * np.tan(np.deg2rad(static_cam_env.fov) / 2))
        gripper_foc = gripper_cam_env.height / (2 * np.tan(np.deg2rad(gripper_cam_env.fov) / 2))
        static_intrinsic = np.array([[static_foc , 0.0, static_cam_env.height/2], [0.0, static_foc , static_cam_env.height/2], [0.0, 0.0, 1.0]])
        gripper_intrinsic = np.array([[gripper_foc , 0.0, gripper_cam_env.height/2], [0.0, gripper_foc , gripper_cam_env.height/2], [0.0, 0.0, 1.0]])

        calib = {'rgb_static':{'extrinsic_matrix':static_extrinsic,
                                'intrinsic_matrix':static_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config':{"height": static_cam_env.height, "width": static_cam_env.width, "fov": static_cam_env.fov}},
                'rgb_gripper':{'extrinsic_matrix':gripper_extrinsic,
                                'intrinsic_matrix':gripper_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,]),
                                'cam_config':{"height": gripper_cam_env.height, "width": gripper_cam_env.width, "fov": gripper_cam_env.fov}}}
        static_extrinsic_matrix = calib['rgb_static']['extrinsic_matrix']
        gripper_extrinsic_matrix = calib['rgb_gripper']['extrinsic_matrix']
        # 平移矩阵 T_translate # 原始是0.3-0.8因此无需添加平移矩阵
        if 0:
            from robouniview.data.data_utils  import ColorJitter_ctm, OccupancyVFE, deproject, cam, RandomShiftsAug
            static_cam = cam(static_extrinsic_matrix, calib['rgb_static']['cam_config']['height'], calib['rgb_static']['cam_config']['width'], calib['rgb_static']['cam_config']['fov'])
            gripper_cam = cam(gripper_extrinsic_matrix,  calib['rgb_gripper']['cam_config']['height'],  calib['rgb_gripper']['cam_config']['width'],  calib['rgb_gripper']['cam_config']['fov'])

            rgb_static, depth_static = obs["rgb_obs"]['rgb_static'], obs["depth_obs"]['depth_static']
            rgb_gripper, depth_gripper = obs["rgb_obs"]['rgb_gripper'], obs["depth_obs"]['depth_gripper']

            static_pcd = deproject(
                static_cam, depth_static,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            gripper_pcd = deproject(
                gripper_cam, depth_gripper,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            rgb_static = rgb_static.reshape(-1, 3)/255.
            rgb_gripper = rgb_gripper.reshape(-1, 3)/255.

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(rgb_static)
            o3d.io.write_point_cloud("tmp.pcd", pcd)
            pcd.points = o3d.utility.Vector3dVector(gripper_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(rgb_gripper)
            o3d.io.write_point_cloud("tmp1.pcd", pcd)

        calib['static_extrinsic_matrix'] = static_extrinsic_matrix*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        calib['static_intrinsic_matrix'] = calib['rgb_static']['intrinsic_matrix']
        calib['static_distCoeffs_matrix'] = calib['rgb_static']['distCoeffs_matrix']

        calib['gripper_extrinsic_matrix'] = gripper_extrinsic_matrix*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])
        calib['gripper_intrinsic_matrix'] = calib['rgb_gripper']['intrinsic_matrix']
        calib['gripper_distCoeffs_matrix'] = calib['rgb_gripper']['distCoeffs_matrix']
        # gripper_cam2gripper = np.array([[-7.96325957e-04, 9.99999676e-01, 6.63332084e-09, -7.96272678e-05],
        #                                 [9.63557976e-01, 7.67306153e-04, -2.67498860e-01, 5.89059483e-02],
        #                                 [-2.67498777e-01, -2.13013621e-04, -9.63558266e-01, -1.61648083e-01],
        #                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) 
        gripper_cam2gripper = np.array([[-0.0008, 0.9999, 0., -0.0001],
                                        [0.9635, 0.0008, -0.2674, 0.0589],
                                        [-0.2674, 0.0000, -0.9635, -0.1616],
                                        [0., 0., 0., 1.]])
        gripper_cam2gripper = gripper_cam2gripper*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]])

        static_extrinsic_matrix = torch.from_numpy(calib['static_extrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        static_intrinsic_matrix = torch.from_numpy(calib['static_intrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        static_distCoeffs_matrix = torch.from_numpy(calib['static_distCoeffs_matrix']).unsqueeze(0).unsqueeze(0)
        gripper_extrinsic_matrix = torch.from_numpy(calib['gripper_extrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        gripper_intrinsic_matrix = torch.from_numpy(calib['gripper_intrinsic_matrix']).unsqueeze(0).unsqueeze(0)
        gripper_distCoeffs_matrix = torch.from_numpy(calib['gripper_distCoeffs_matrix']).unsqueeze(0).unsqueeze(0)
        static_fov = torch.from_numpy(np.array([calib['rgb_static']['cam_config']['fov'], calib['rgb_static']['cam_config']['height'], calib['rgb_static']['cam_config']['width']])).unsqueeze(0).unsqueeze(0)
        gripper_fov = torch.from_numpy(np.array([calib['rgb_gripper']['cam_config']['fov'], calib['rgb_gripper']['cam_config']['height'],  calib['rgb_gripper']['cam_config']['width']])).unsqueeze(0).unsqueeze(0)
        
        gripper_cam2gripper = torch.from_numpy(gripper_cam2gripper).unsqueeze(0).unsqueeze(0)
        calib = [static_extrinsic_matrix,static_intrinsic_matrix,static_distCoeffs_matrix,
                 gripper_extrinsic_matrix,gripper_intrinsic_matrix,gripper_distCoeffs_matrix,gripper_extrinsic_matrix,static_fov,gripper_fov,gripper_cam2gripper]

        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        # image = Image.fromarray(image)
        # image_x = self.image_process_fn([image])
        # # expand image dimension
        # image_x = image_x.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        
        # # fix window_size : ddp_model -> ... -> window_size
        # if self.model.module.sep_lm_head:
        #     window_size = self.model.module.lm_head.window_size
        #     self.model.module.lm_head.window_size = 1
        #     if self.model.module.pad_length != -1 and self.feature_cache is None:
        #         self.model.module.lm_head.window_size = self.model.module.pad_length
        # else:
        #     window_size = self.model.module.lang_encoder.lm_head.window_size
        #     self.model.module.lang_encoder.lm_head.window_size = 1
        #     if self.model.module.pad_length != -1 and self.feature_cache is None:
        #         self.model.module.lang_encoder.lm_head.window_size = self.model.module.pad_length
        # gripper = None
        # state = None

        # if self.model.module.use_gripper:
        #     gripper = obs["rgb_obs"]['rgb_gripper']
        #     gripper = Image.fromarray(gripper)
        #     gripper = self.image_process_fn([gripper])
        #     # expand image dimension
        #     gripper = gripper.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        gripper = obs["rgb_obs"]['rgb_gripper']

        state = obs['robot_obs']

        # if self.model.module.use_state or self.model.module.sep_lm_head:
        # if self.model.module.use_state or self.model.module.sep_lm_head:
        #     state = obs['robot_obs']
        #     state = torch.from_numpy(np.stack([state]))
        #     # if self.model.module.sep_lm_head:
        #     #     state = torch.cat([state[...,:6], state[...,[-1]]], dim=-1)
        #     # if self.fusion_mode == 'two_way':
        #     #     state = state.repeat(2, 1)
        #     state = state.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        #     state = state.to(torch.float32)
        if True:
        # with torch.no_grad():
        #     device = 'cuda'
        #     image_x = image_x.to(device)
        #     if gripper is not None:
        #         gripper = gripper.to(device)
        #     if state is not None:
        #         state = state.to(device)

            # if 'Temporal' in self.fusion_mode:
            #     self.model.module.pad_length = self.model.module.window_size

            # self.model.module.pad_length = -1   # YF:
            # if self.model.module.pad_length != -1:
            if len(self.img_queue) == 0:
                self.img_queue.append(image)
                for _ in range(self.history_len - 1):
                    self.img_queue.append(image)
            else:
                self.img_queue.append(image)
            if len(self.gripper_queue) == 0 and gripper is not None:
                self.gripper_queue.append(gripper)
                for _ in range(self.history_len - 1):
                    self.gripper_queue.append(gripper)
            else:
                self.gripper_queue.append(gripper)
            if len(self.state_queue) == 0 and state is not None:
                self.state_queue.append(state)
                for _ in range(self.history_len - 1):
                    self.state_queue.append(state)
            else:
                self.state_queue.append(state)
            if len(self.calib_queue) == 0:
                self.calib_queue.append(calib)
                for _ in range(self.history_len - 1):
                    self.calib_queue.append(calib)
            else:
                self.calib_queue.append(calib)

            if True: #'Temporal' in self.fusion_mode:
                image = np.stack(list(self.img_queue))
                gripper = np.stack(list(self.gripper_queue))
                state = np.stack(list(self.state_queue))

                # image_x = image.unsqueeze(2)
                # gripper = gripper.unsqueeze(2)
                calib = [np.hstack([tmp[i_calibe] for tmp in self.calib_queue]) for i_calibe in range(len(self.calib_queue[0]))]

            # self.model.module.pad_length = -1

            # expand text dimension
            # text_x, mask = self.text_process_fn([goal], window_size=len(self.img_queue))
            # text_x = text_x.to(device)
            # mask = mask.to(device)
            # text_x = lang_annotation
            
            # action_token_id = self.tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]
            # action_mask = mask.clone()
            # action_mask[text_x != action_token_id] = 0

            # # preimg0_token_id = self.tokenizer("<preimg0>", add_special_tokens=False)["input_ids"][-1]
            # # preimg1_token_id = self.tokenizer("<preimg1>", add_special_tokens=False)["input_ids"][-1]
            # # preimg2_token_id = self.tokenizer("<preimg2>", add_special_tokens=False)["input_ids"][-1]
            # # preimg3_token_id = self.tokenizer("<preimg3>", add_special_tokens=False)["input_ids"][-1]
            # # preimg4_token_id = self.tokenizer("<preimg4>", add_special_tokens=False)["input_ids"][-1]
            # # preimg5_token_id = self.tokenizer("<preimg5>", add_special_tokens=False)["input_ids"][-1]
            # # preimg6_token_id = self.tokenizer("<preimg6>", add_special_tokens=False)["input_ids"][-1]
            # # preimg7_token_id = self.tokenizer("<preimg7>", add_special_tokens=False)["input_ids"][-1]
            # # action_token_id = tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]
        
            # static0 = self.tokenizer("<static0>", add_special_tokens=False)["input_ids"][-1]
            # static1 = self.tokenizer("<static1>", add_special_tokens=False)["input_ids"][-1]
            # static2 = self.tokenizer("<static2>", add_special_tokens=False)["input_ids"][-1]
            # static3 = self.tokenizer("<static3>", add_special_tokens=False)["input_ids"][-1]
            # static4 = self.tokenizer("<static4>", add_special_tokens=False)["input_ids"][-1]
            # static5 = self.tokenizer("<static5>", add_special_tokens=False)["input_ids"][-1]
            # static6 = self.tokenizer("<static6>", add_special_tokens=False)["input_ids"][-1]
            # static7 = self.tokenizer("<static7>", add_special_tokens=False)["input_ids"][-1]
            
            # gripper0 = self.tokenizer("<gripper0>", add_special_tokens=False)["input_ids"][-1]
            # gripper1 = self.tokenizer("<gripper1>", add_special_tokens=False)["input_ids"][-1]
            # gripper2 = self.tokenizer("<gripper2>", add_special_tokens=False)["input_ids"][-1]
            # gripper3 = self.tokenizer("<gripper3>", add_special_tokens=False)["input_ids"][-1]
            # gripper4 = self.tokenizer("<gripper4>", add_special_tokens=False)["input_ids"][-1]
            # gripper5 = self.tokenizer("<gripper5>", add_special_tokens=False)["input_ids"][-1]
            # gripper6 = self.tokenizer("<gripper6>", add_special_tokens=False)["input_ids"][-1]
            # gripper7 = self.tokenizer("<gripper7>", add_special_tokens=False)["input_ids"][-1]
            
            # obs0 = self.tokenizer("<obs0>", add_special_tokens=False)["input_ids"][-1]
            # obs1 = self.tokenizer("<obs1>", add_special_tokens=False)["input_ids"][-1]
            # obs2 = self.tokenizer("<obs2>", add_special_tokens=False)["input_ids"][-1]
            # obs3 = self.tokenizer("<obs3>", add_special_tokens=False)["input_ids"][-1]
            # obs4 = self.tokenizer("<obs4>", add_special_tokens=False)["input_ids"][-1]
            # obs5 = self.tokenizer("<obs5>", add_special_tokens=False)["input_ids"][-1]
            # obs6 = self.tokenizer("<obs6>", add_special_tokens=False)["input_ids"][-1]
            # obs7 = self.tokenizer("<obs7>", add_special_tokens=False)["input_ids"][-1]

            # static_mask = mask.clone()
            # gripper_mask = mask.clone()
            # obs_mask = mask.clone()
            # static_mask[(text_x != static0) & (text_x != static1)& (text_x != static2)& (text_x != static3) \
            #     & (text_x != static4) & (text_x != static5)& (text_x != static6)& (text_x != static7) ] = 0
            
            # gripper_mask[(text_x != gripper0) & (text_x != gripper1)& (text_x != gripper2)& (text_x != gripper3) \
            #     & (text_x != gripper4) & (text_x != gripper5)& (text_x != gripper6)& (text_x != gripper7) ] = 0
            
            # obs_mask[(text_x != obs0) & (text_x != obs1)& (text_x != obs2)& (text_x != obs3) \
            #     & (text_x != obs4) & (text_x != obs5)& (text_x != obs6)& (text_x != obs7) ] = 0
            # static_mask=static_mask.bool()
            # gripper_mask=gripper_mask.bool()
            # obs_mask=obs_mask.bool()
            # mask=mask.bool()
            # action_mask=action_mask.bool()
            # if static_mask.sum() < 1 : static_mask = None
            # if gripper_mask.sum() < 1 : gripper_mask = None
            # if obs_mask.sum() < 1 :  obs_mask = None
            # preimg_mask = mask.clone()
            # preimg_mask[(text_x != preimg0_token_id) & (text_x != preimg1_token_id)& (text_x != preimg2_token_id)& (text_x != preimg3_token_id) \
            #     & (text_x != preimg4_token_id) & (text_x != preimg5_token_id)& (text_x != preimg6_token_id)& (text_x != preimg7_token_id) ] = 0


            # if len(self.mask_queue) == 0 and mask is not None:
            #     self.mask_queue.append(mask)
            #     for _ in range(self.model.module.pad_length - 1):
            #         self.mask_queue.append(mask)
            # if len(self.text_queue) == 0 and text_x is not None:
            #     self.text_queue.append(text_x)
            #     for _ in range(self.model.module.pad_length - 1):
            #         self.text_queue.append(text_x)
            
            # if self.model.module.pad_length != -1 and self.feature_cache is None:
            #     image_x = torch.cat(list(self.img_queue), dim=0)
            #     if gripper is not None:
            #         gripper = torch.cat(list(self.gripper_queue), dim=0)
            #     if state is not None:
            #         state = torch.cat(list(self.state_queue), dim=0)
            #     mask = torch.cat(list(self.mask_queue), dim=0)
            #     text_x = torch.cat(list(self.text_queue), dim=0)
            #     assert False 
            # if self.fusion_mode == 'vit_concat':
            #     image_x = torch.cat(list(self.img_queue), dim=0)
            #     if gripper is not None:
            #         gripper = torch.cat(list(self.gripper_queue), dim=0)
            #     if state is not None:
            #         state = torch.cat(list(self.state_queue), dim=0)
            #     pass
            #     assert False 

            # if self.use_diff:
            #     if self.fusion_mode == 'two_way':
            #         vision_x = torch.cat([image_x, gripper], dim=0)
            #         text_x = text_x.repeat(2, 1)
            #         mask = mask.repeat(2, 1)
            #         model_out = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
            #     else:
            #         model_out = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, state_tensor = state, return_feature=True)

            #     if not get_action:
            #         return None
            #     model_out = model_out.logits
            #     action_history = torch.tensor(np.stack(self.action_hist_queue, axis=0), dtype=torch.float, device=device).unsqueeze(0)
            #     action_history = self.normalizer.normalize(action_history)
            #     if self.supp is None:
            #         self.supp = torch.zeros(
            #             action_history.shape[0], self.horizon-self.hist_len, action_history.shape[-1], 
            #             dtype=action_history.dtype,
            #             device=action_history.device,
            #         )
            #     action_history = torch.concat([action_history, self.supp], dim=1)
            #     act_mask = torch.zeros_like(action_history, device=action_history.device, dtype=torch.bool)
            #     act_mask[:,:self.hist_len,...] = 1.
            #     pred_action_seq = self.diffusion_model.conditional_sample(cond_data=action_history, cond_mask=act_mask, global_cond=model_out)
            #     pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
            #     action = pred_action_seq[:,self.hist_len:,:]
            #     if self.future_act_len > 0:
            #         action = action[:,:self.future_act_len,:]
            #     action = action[0]
            #     action = action.cpu().detach().to(dtype=torch.float16).numpy()
            #     action[...,-1] = action[...,-1] > 0.5
            #     action[...,-1] = (action[...,-1] - 0.5) * 2  # scale to -1 or 1
            # else:
            if True:
                # if self.fusion_mode == 'two_way':
                #     vision_x = torch.cat([image_x, gripper], dim=0)
                #     text_x = text_x.repeat(2, 1)
                #     mask = mask.repeat(2, 1)
                #     action = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
                # else:
                if True:
                    element = {
                        "image": image,
                        "wrist_image": gripper,
                        "state": state,  # (15,)
                        "prompt": lang_annotation,
                        "calib": calib,
                    }
                    action = self.client.infer(element)["actions"]

                # if static_pred is not None:
                #     obs_preds_rgb_img = static_pred[-1]
                #     obs_preds_rgb_img = rearrange(obs_preds_rgb_img, "c h w  -> h w c")
                #     obs_preds_rgb_img = np.array(obs_preds_rgb_img.data.cpu())
                # else:
                #     obs_preds_rgb_img = None
                    
                # if gripper_pred is not None:
                #     obs_preds_gripper_img = gripper_pred[-1]
                #     obs_preds_gripper_img = rearrange(obs_preds_gripper_img, "c h w  -> h w c")
                #     obs_preds_gripper_img = np.array(obs_preds_gripper_img.data.cpu())
                # else:
                #     obs_preds_gripper_img = None

                # if self.model.module.pad_length != -1:
                #     if self.feature_cache is None:
                #         self.feature_cache = action.logits[-1]
                #     else:
                #         new_feat = torch.cat([self.feature_cache[1:], action.logits[-1]], dim=0)
                #         self.feature_cache = new_feat
                #         if not self.model.module.sep_lm_head:
                #             self.model.module.lang_encoder.lm_head.window_size = window_size
                #             lm_out = self.model.module.lang_encoder.lm_head(new_feat)
                #         else:
                #             self.model.module.lm_head.window_size = window_size
                #             lm_out = self.model.module.lm_head(new_feat)
                #         Output = namedtuple('Output', ['logits'])
                #         action = Output(lm_out)

                # if 'Temporal' in self.fusion_mode:
                if True:
                    pose = action[...,:6]
                    gripper = action[...,6:] > 0.5
                    # pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    # gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    # if self.args.multi_action_token:
                    if True:
                        pose = pose[:,-1,:]
                        gripper = gripper[:,-1,:]

                    action = np.concatenate([pose, gripper], axis=-1)
                    action = action[0] # select 第一个batch的
                # else:
                #     if self.model.module.act_step == 1:
                #         action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2).squeeze(0)[-1] # support multi step history
                #     else:
                #         pose = action.logits[0]
                #         gripper = action.logits[1] > 0.5
                #         pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                #         gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                #         action = torch.cat([pose, gripper], dim=-1)
                #         action = action[0] # select first step action
                    
                action[-1] = (action[-1] - 0.5) * 2  # scale to -1 or 1
                # action = action.cpu().detach().to(dtype=torch.float16).numpy()
        
        # if self.model.module.sep_lm_head:
        #     self.model.module.lm_head.window_size = window_size
        # else:
        #     self.model.module.lang_encoder.lm_head.window_size = window_size
        # if self.model.module.tcp_rel:
        #     state = obs['robot_obs']
        #     state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
        #     action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
        #     action = tcp_to_world_frame(action, state)
        #     action=action.squeeze().to(dtype=torch.float16).numpy()
        
        return action

        # # Preprocess images
        # rgb_static = obs['rgb_obs']['rgb_static']  # (200, 200, 3) uint8
        # rgb_gripper = obs['rgb_obs']['rgb_gripper']  # (84, 84, 3) uint8
        
        # # Resize and pad images
        # image = image_tools.convert_to_uint8(
        #     image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size)
        # )
        # wrist_image = image_tools.convert_to_uint8(
        #     image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        # )
        
        # # Prepare element for policy server
        # element = {
        #     "observation/image": image,
        #     "observation/wrist_image": wrist_image,
        #     "observation/state": obs['robot_obs'].astype(np.float32),  # (15,)
        #     "prompt": lang_annotation,
        # }
        
        # # Query model
        # model_output = self.client.infer(element)
        # action_chunk = model_output["actions"]  # (N, 7)
        
        # # Cache action chunk and return first action
        # self.action_plan.extend(action_chunk[:self.replan_steps])
        # return self.action_plan.popleft()


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    val_folder = Path(dataset_path) / "validation"
    
    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    from omegaconf import OmegaConf
    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Remove tactile sensor from camera list if it exists
    if hasattr(cfg.env, 'cameras') and 'tactile' in cfg.env.cameras:
        # Create a new camera dict without tactile
        new_cameras = OmegaConf.create({
            k: v for k, v in cfg.env.cameras.items() if k != 'tactile'
        })
        cfg.env.cameras = new_cameras
    
    # Initialize environment with modified config
    import hydra
    env = hydra.utils.instantiate(
        cfg.env, 
        show_gui=False, 
        use_vr=False, 
        use_scene_info=True
    )
    
    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def _parse_servers(servers: str, default_host: str, default_port: int) -> List[Tuple[str, int]]:
    if not servers:
        return [(default_host, int(default_port))]
    out: List[Tuple[str, int]] = []
    for item in servers.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            host, port_str = item.rsplit(":", 1)
            out.append((host.strip(), int(port_str)))
        else:
            out.append((item, int(default_port)))
    if not out:
        out = [(default_host, int(default_port))]
    return out


def _notify_eval_done(server: Tuple[str, int], avg_seq_len: float) -> None:
    """Best-effort notify policy server that evaluation has finished."""
    import urllib.request

    host, port = server
    # 0.0.0.0 is a bind-all address; not routable as a client target.
    if host == "0.0.0.0":
        host = "127.0.0.1"
    url = f"http://{host}:{port}/eval_done?avg_seq_len={avg_seq_len}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            _ = resp.read()
        logger.info(f"Sent eval_done signal to {url}")
    except Exception as e:
        logger.warning(f"Failed to send eval_done signal to {url}: {e}")


def _load_task_oracle_and_annotations(calvin_conf_path: str, diverse_inst: bool):
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    if diverse_inst:
        with open('/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return task_oracle, val_annotations


def _worker_eval(
    worker_id: int,
    server: Tuple[str, int],
    dataset_path: str,
    calvin_conf_path: str,
    eval_log_dir: str,
    debug: bool,
    reset: bool,
    diverse_inst: bool,
    create_plan_tsne: bool,
    seq_chunk: List[Tuple[int, Any, Any]],
    out_q: "mp.Queue",
    window_size: int,
):
    """Run a chunk of sequences on one policy server (in a separate process)."""
    try:
        host, port = server
        policy = CalvinPolicyClient(host, port, window_size)
        env = make_env(dataset_path)
        task_oracle, val_annotations = _load_task_oracle_and_annotations(calvin_conf_path, diverse_inst)
        plans = defaultdict(list)

        results: List[Tuple[int, int, Any]] = []
        for sequence_i, initial_state, eval_sequence in tqdm(seq_chunk, desc=f"Worker {worker_id} processing sequences", total=len(seq_chunk)):
            res = evaluate_sequence(
                env,
                policy,
                task_oracle,
                initial_state,
                eval_sequence,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                sequence_i,
                reset=reset,
                diverse_inst=diverse_inst,
            )
            results.append((sequence_i, res, eval_sequence))

        # Optional: create tsne per-worker (avoid cross-process merge complexity)
        if create_plan_tsne:
            worker_log_dir = os.path.join(eval_log_dir, f"tsne_worker_{worker_id}")
            os.makedirs(worker_log_dir, exist_ok=True)
            create_tsne(plans, worker_log_dir, 0)
        out_q.put((worker_id, results, None))
    except Exception as e:
        out_q.put((worker_id, None, repr(e)))


def evaluate_policy_multi_servers(
    *,
    servers: List[Tuple[str, int]],
    dataset_path: str,
    calvin_conf_path: str,
    eval_sequences_path: str,
    num_sequences: int,
    eval_log_dir: Optional[str],
    debug: bool,
    create_plan_tsne: bool,
    reset: bool,
    diverse_inst: bool,
    window_size: int,
):
    """Evaluate with multiple policy servers."""
    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        eval_sequences = json.load(f)
    if num_sequences and num_sequences > 0:
        eval_sequences = eval_sequences[:num_sequences]

    # Multi-process parallel evaluation (one worker per server).
    n_workers = len(servers)
    print(f"n_workers: {n_workers}")

    # Round-robin chunking keeps indices unique and balances slow sequences better than contiguous split.
    chunks: List[List[Tuple[int, Any, Any]]] = [[] for _ in range(n_workers)]
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        chunks[i % n_workers].append((i, initial_state, eval_sequence))

    out_q: "mp.Queue" = mp.Queue()
    procs: List[mp.Process] = []
    for worker_id in range(n_workers):
        p = mp.Process(
            target=_worker_eval,
            args=(
                worker_id,
                servers[worker_id],
                dataset_path,
                calvin_conf_path,
                eval_log_dir,
                debug,
                reset,
                diverse_inst,
                create_plan_tsne,
                chunks[worker_id],
                out_q,
                window_size,
            ),
        )
        p.start()
        procs.append(p)

    merged: dict = {}
    for _ in range(n_workers):
        worker_id, results, err = out_q.get()
        if err is not None:
            raise RuntimeError(f"Worker {worker_id} failed: {err}")
        assert results is not None
        for sequence_i, res, eval_sequence in results:
            merged[sequence_i] = (res, eval_sequence)

    for p in procs:
        p.join()

    results_sorted = [merged[i][0] for i in range(len(eval_sequences))]
    eval_sequences_sorted = [merged[i][1] for i in range(len(eval_sequences))]
    eval_sequences_for_print = list(enumerate(eval_sequences_sorted))
    print_and_save(results_sorted, eval_sequences_for_print, eval_log_dir, 0)
    return results_sorted


def evaluate_sequence(env, policy, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(env, policy, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, robot_obs=robot_obs, scene_obs=scene_obs, diverse_inst=diverse_inst)
        else:
            success = rollout(env, policy, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i,diverse_inst=diverse_inst)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, policy, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    policy.reset()
    start_info = env.get_info()

    if debug:
        img_queue = []

    for step in range(EP_LEN):

        action = policy.step(obs, lang_annotation, env)
        
        if 1:
            action = np.array([-action[1], action[0], action[2], action[3], action[4], action[5], action[6]]) # 见https://km.sankuai.com/collabpage/2537354641
            # 将旋转矩阵转换为四元数
            def rotation_matrix_to_quaternion(R):
                # 使用numpy计算四元数
                m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
                m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
                m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
                trace = m00 + m11 + m22
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    qw = 0.25 / s
                    qx = (m21 - m12) * s
                    qy = (m02 - m20) * s
                    qz = (m10 - m01) * s
                elif (m00 > m11) and (m00 > m22):
                    s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
                    qw = (m21 - m12) / s
                    qx = 0.25 * s
                    qy = (m01 + m10) / s
                    qz = (m02 + m20) / s
                elif m11 > m22:
                    s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
                    qw = (m02 - m20) / s
                    qx = (m01 + m10) / s
                    qy = 0.25 * s
                    qz = (m12 + m21) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
                    qw = (m10 - m01) / s
                    qx = (m02 + m20) / s
                    qy = (m12 + m21) / s
                    qz = 0.25 * s
                return [qx, qy, qz, qw]
            
            delta = action[3:6]
            current_orientation = np.array(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(env.robot.target_orn))).reshape(3,3) 
            delta *= 0.05
            quat_error = axisangle2quat(delta)
            rotation_mat_error = quat2mat(quat_error)
            B = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            current_orientation = np.dot(B, current_orientation) # 计算新的旋转矩阵A_new
            goal_orientation = np.dot(rotation_mat_error, current_orientation)
            goal_orientation = np.dot(np.linalg.inv(B), goal_orientation) # 计算新的旋转矩阵A_new

            goal_ort = np.array(pb.getEulerFromQuaternion(rotation_matrix_to_quaternion(goal_orientation)))
            def angle_between_angles(a, b, max_orn=0.05):
                diff = b - a
                rel_orn = (diff + np.pi) % (2 * np.pi) - np.pi
                rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
                return rel_orn
            action[3:6] = angle_between_angles(env.robot.target_orn, goal_ort, max_orn=0.05)
            action[:3] = np.clip(action[:3] * 0.02, -0.02, 0.02) / 0.02
        
        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


def main(args: Args):
    servers = _parse_servers(args.servers, args.host, args.port)

    from typing import Optional
    while True:
        err: Optional[Exception] = None
        try:
            results_sorted = evaluate_policy_multi_servers(
                servers=servers,
                dataset_path=args.dataset_path,
                calvin_conf_path=args.calvin_config_path,
                eval_sequences_path=args.eval_sequences_path,
                num_sequences=args.num_sequences,
                eval_log_dir=args.eval_log_dir,
                debug=args.debug,
                create_plan_tsne=args.create_plan_tsne,
                reset=args.reset,
                diverse_inst=args.diverse_inst,
                window_size=args.window_size,
            )
        except Exception as e:
            err = e
            raise
        finally:
            avg_seq_len = np.mean(results_sorted)
            # Deduplicate servers to avoid sending multiple eval_done signals to the same server
            unique_servers = list(set(servers))
            for server in unique_servers:
                _notify_eval_done(server, avg_seq_len)
                print(f"Sent eval_done signal to {server} with avg_seq_len {avg_seq_len}")

        if args.only_once:
            break
        time.sleep(60)
        print("Sleeping for 60 seconds")


if __name__ == "__main__":
    tyro.cli(main)


