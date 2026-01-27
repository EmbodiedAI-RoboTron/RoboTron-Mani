"""
Action head decoders for robot manipulation models.

This module provides various decoder architectures for predicting robot actions,
including LSTM, FC, GPT, and Diffusion-based decoders.
"""

# Standard library imports
import copy
import math
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, repeat
from open_flamingo.src.helpers import PerceiverResampler

# Local application imports
from robouniview.data import pytorch3d_transforms as pytorch3d_transforms
from robouniview.data.keyframe_data import normalise_quat, unnormalize_pos
from robouniview.models.normalizer import LinearNormalizer
from robouniview.models.trajectory_gpt2 import get_gpt_model
from robouniview.models.multihead_custom_attention import MultiheadCustomAttention


def lstm_decoder(
    in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float
) -> torch.nn.Module:
    return nn.LSTM(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )

class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPNohHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSigmoidHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPActionHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Create a linear layer for each action
        self.num_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        self.bin_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x[:, -1]  # pick up the last frame output
        x1 = self.num_head(x)
        x2 = self.bin_head(x).sigmoid()
        return x1, x2


class ActionDecoder(nn.Module):
    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_and_act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def clear_hidden_state(self) -> None:
        pass

class TokenFCDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        return_feature=False,
        multi_step_action=1
    ):
        super(TokenFCDecoder, self).__init__()
        self.return_feature = return_feature
        # if use_state:
        #     state_in_dim = 7
        #     state_out_dim = 128
        #     self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
        #     in_features += state_out_dim
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []

        self.use_diff = use_diff
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features*4, hidden_size),
        )
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features)
            self.gripper = MLPSigmoidHead(hidden_size, 1)
        self.hidden_state = None
        self.hidden_size = hidden_size * history_len
        
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        # self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)

    def forward(  # type: ignore
            self,
            input_feature: torch.Tensor,
            h_0: Optional[torch.Tensor] = None,
            state_tensor = None,
            action_mask = None,
    ):
        # if self.return_feature:
        #     org_feat = copy.deepcopy(input_feature) 
        #     org_feat = org_feat.view(self.window_size, *org_feat.shape[1:])
        # reshape

         
        input_feature = input_feature[action_mask == 1]
        input_feature = self.mlp(input_feature)
        #input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        # if self.use_diff:
        #     input_feature = input_feature.reshape(-1, self.window_size * input_feature.shape[1])
        #     return input_feature

        #input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        # if state_tensor is not None:
        #     state_tensor = self.fc_state(state_tensor)
        #     state_tensor = state_tensor.reshape(-1, self.window_size, state_tensor.shape[-1])
        #     input_feature = torch.cat([input_feature, state_tensor], dim=-1)

        actions = self.actions(input_feature)
        gripper = self.gripper(input_feature)

        # if self.return_feature:
        #     return actions, gripper, org_feat
        # else:
        return actions, gripper

class Multi_Action_Token_FCDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        return_feature=False,
        multi_step_action=1,
        nclass_gripper=1,
    ):
        super(Multi_Action_Token_FCDecoder, self).__init__()
        self.return_feature = return_feature
        # if use_state:
        #     state_in_dim = 7
        #     state_out_dim = 128
        #     self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
        #     in_features += state_out_dim
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []

        self.use_diff = use_diff
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features*4, hidden_size),
        )
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features)
            self.gripper = MLPSigmoidHead(hidden_size, nclass_gripper)
        self.hidden_state = None
        self.hidden_size = hidden_size * history_len
        
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        # self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)

    def forward(  # type: ignore
            self,
            input_feature: torch.Tensor,
            h_0: Optional[torch.Tensor] = None,
            state_tensor = None,
            action_mask = None,
    ):
        # if self.return_feature:
        #     org_feat = copy.deepcopy(input_feature) 
        #     org_feat = org_feat.view(self.window_size, *org_feat.shape[1:])
        # reshape

        b, t, d = input_feature.shape
        # action_feature = []    
        # for i in range(b):
        #     action_feature.append(input_feature[i][action_mask[i] == 1].unsqueeze(0))
        # #action_feature = torch.cat([vision_rgb, vision_gripper], dim=0) 
        # action_feature = torch.vstack(action_feature)
        mask_indices = action_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
        action_feature = input_feature[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
        action_feature = rearrange(action_feature, "(B T) D -> B T D", B=b) # 8表示8个token

        #input_feature = input_feature[action_mask == 1]
         
        action_feature = self.mlp(action_feature)
        #input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        # if self.use_diff:
        #     input_feature = input_feature.reshape(-1, self.window_size * input_feature.shape[1])
        #     return input_feature

        #input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        # if state_tensor is not None:
        #     state_tensor = self.fc_state(state_tensor)
        #     state_tensor = state_tensor.reshape(-1, self.window_size, state_tensor.shape[-1])
        #     input_feature = torch.cat([input_feature, state_tensor], dim=-1)

        actions = self.actions(action_feature)
        gripper = self.gripper(action_feature)

        # if self.return_feature:
        #     return actions, gripper, org_feat
        # else:
        return actions, gripper


class Multi_Action_Token_FCDecoder_for_keyframe(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 9,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        return_feature=False,
        multi_step_action=1,
        nclass_gripper=1,
    ):
        super(Multi_Action_Token_FCDecoder_for_keyframe, self).__init__()
        self.return_feature = return_feature
        # if use_state:
        #     state_in_dim = 7
        #     state_out_dim = 128
        #     self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
        #     in_features += state_out_dim
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.interpolation_length = 20 # 默认插入了20个值
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []

        self.use_diff = use_diff
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features*4, hidden_size),
        )
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*self.interpolation_length)
            self.gripper = MLPSigmoidHead(hidden_size, nclass_gripper*self.interpolation_length)
        self.hidden_state = None
        self.hidden_size = hidden_size * history_len
        
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        # self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)

    def forward(  # type: ignore
            self,
            input_feature: torch.Tensor,
            h_0: Optional[torch.Tensor] = None,
            state_tensor = None,
            action_mask = None,
    ):
        # if self.return_feature:
        #     org_feat = copy.deepcopy(input_feature) 
        #     org_feat = org_feat.view(self.window_size, *org_feat.shape[1:])
        # reshape

        b, t, d = input_feature.shape
        # action_feature = []    
        # for i in range(b):
        #     action_feature.append(input_feature[i][action_mask[i] == 1].unsqueeze(0))
        # #action_feature = torch.cat([vision_rgb, vision_gripper], dim=0) 
        # action_feature = torch.vstack(action_feature)
        mask_indices = action_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
        action_feature = input_feature[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
        action_feature = rearrange(action_feature, "(B T) D -> B T D", B=b) # 8表示8个token

        #input_feature = input_feature[action_mask == 1]
         
        action_feature = self.mlp(action_feature)
        #input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        # if self.use_diff:
        #     input_feature = input_feature.reshape(-1, self.window_size * input_feature.shape[1])
        #     return input_feature

        #input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        # if state_tensor is not None:
        #     state_tensor = self.fc_state(state_tensor)
        #     state_tensor = state_tensor.reshape(-1, self.window_size, state_tensor.shape[-1])
        #     input_feature = torch.cat([input_feature, state_tensor], dim=-1)

        actions = self.actions(action_feature)
        gripper = self.gripper(action_feature)

        actions = rearrange(actions, "B 1 (T D) -> B T D", T=self.interpolation_length) #
        gripper = rearrange(gripper, "B 1 (T D) -> B T D", T=self.interpolation_length) # 
        # if self.return_feature:
        #     return actions, gripper, org_feat
        # else:
        return actions, gripper


class FCDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        return_feature=False,
        multi_step_action=1
    ):
        super(FCDecoder, self).__init__()
        self.return_feature = return_feature
        # if use_state:
        #     state_in_dim = 7
        #     state_out_dim = 128
        #     self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
        #     in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []

        self.use_diff = use_diff
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features*4, hidden_size),
        )
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features)
            self.gripper = MLPSigmoidHead(hidden_size, 1)
        self.hidden_state = None
        self.hidden_size = hidden_size * history_len
        
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        # self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)

    def forward(  # type: ignore
            self,
            input_feature: torch.Tensor,
            h_0: Optional[torch.Tensor] = None,
            state_tensor = None,
    ):
        if self.return_feature:
            org_feat = copy.deepcopy(input_feature) 
            org_feat = org_feat.view(self.window_size, *org_feat.shape[1:])
        # reshape
        input_feature = self.mlp(input_feature)
        input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        if self.use_diff:
            input_feature = input_feature.reshape(-1, self.window_size * input_feature.shape[1])
            return input_feature

        #input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        # if state_tensor is not None:
        #     state_tensor = self.fc_state(state_tensor)
        #     state_tensor = state_tensor.reshape(-1, self.window_size, state_tensor.shape[-1])
        #     input_feature = torch.cat([input_feature, state_tensor], dim=-1)

        actions = self.actions(input_feature)
        gripper = self.gripper(input_feature)

        if self.return_feature:
            return actions, gripper, org_feat
        else:
            return actions, gripper


class DeterministicDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='max'
    ):
        super(DeterministicDecoder, self).__init__()
        self.fc_state = None
        self.use_state = use_state
        if use_state:
            print('Using state in decoder')
            state_in_dim = 7
            # state_out_dim = 256
            # in_features += state_out_dim
            # self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, state_out_dim), nn.ReLU())
            # self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, state_out_dim), nn.ReLU()) # one-hot gripper state
            # self.embed_state = torch.nn.Linear(2*state_out_dim, state_out_dim)

            self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, in_features), nn.ReLU())
            self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, in_features), nn.ReLU()) # one-hot gripper state
            self.embed_state = torch.nn.Linear(2*in_features, in_features)
        
        if fusion_mode == 'two_way':
            in_features *= 2
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        self.rnn = lstm_decoder
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def forward(  # type: ignore
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        state_tensor=None,
        return_feature=False
    ):
        
        
        # reshape
        if input_feature.dim() == 3:
            if self.fusion_mode == 'two_way':
                input_feature = input_feature.reshape(-1, self.window_size, *input_feature.shape[1:])
                
                bs = int(input_feature.shape[0] // 2)
                
                rgb_feat = input_feature[:bs].view(bs*self.window_size, *input_feature.shape[2:])
                rgb_feat = self.global_1d_pool(rgb_feat.permute(0, 2, 1)).squeeze(-1)
                
                gripper_feat = input_feature[bs:].view(bs*self.window_size, *input_feature.shape[2:])
                gripper_feat = self.global_1d_pool(gripper_feat.permute(0, 2, 1)).squeeze(-1)
                
                input_feature = torch.cat([rgb_feat, gripper_feat], dim=-1)
            else:
                input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        if self.return_feature:
            org_feat = copy.deepcopy(input_feature) 
            org_feat = org_feat.view(self.window_size, org_feat.shape[-1])

        if state_tensor is not None and self.use_state:
            arm_state = state_tensor[..., :6] # b,len,state_dim-1
            arm_state_embeddings = self.embed_arm_state(arm_state)
            arm_state_embeddings = arm_state_embeddings.view(-1, self.window_size, arm_state_embeddings.shape[-1]) # b,len,h
            gripper_state = ((state_tensor[..., -1]+1.0) / 2).long() # b,len,1
            gripper_state_embeddings = self.embed_gripper_state(gripper_state)
            gripper_state_embeddings = gripper_state_embeddings.view(-1, self.window_size, gripper_state_embeddings.shape[-1]) # b,len,h
            state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2) # b,len,2h
            state_embeddings = self.embed_state(state_embeddings) # b,len,h

            # input_feature = torch.cat([input_feature, state_embeddings], dim=-1)
            input_feature = input_feature + state_embeddings
        
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            # print('history len:',self.history_len)
            if input_feature.shape[1] == 1:
                self.history_memory.append(input_feature)
                if len(self.history_memory) <= self.history_len:
                    # print('cur hist_mem len: {}'.format(len(self.history_memory)))
                    x, h_n = self.rnn(input_feature, self.hidden_state)
                    self.hidden_state = h_n
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
                else:
                    # the hidden state need to be refreshed based on the history window
                    # print('hist_mem exceeded, refresh hidden state')
                    cur_len = len(self.history_memory)
                    for _ in range(cur_len - self.history_len):
                        self.history_memory.pop(0)
                    assert len(self.history_memory) == self.history_len
                    hist_feature = torch.cat(self.history_memory, dim=1)
                    self.hidden_state = None
                    x, h_n = self.rnn(hist_feature, self.hidden_state)
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
            else:
                # print('input feature lenght > 1', input_feature.shape)
                self.hidden_state = h_0
                x, h_n = self.rnn(input_feature, self.hidden_state)
                self.hidden_state = h_n
                if self.last_action:
                    x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
        else:
            raise NotImplementedError
        if self.use_diff:
            return self.rnn_out
        actions = self.actions(x)
        gripper = self.gripper(x)
        if self.return_feature:
            return actions, gripper, org_feat
        else:
            return actions, gripper

    def act(
        self,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions, self.hidden_state = self(
            input_feature, self.hidden_state
        )

        return pred_actions


class GPTDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size = None,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        last_action=False,
        use_diff=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='max',
        **kwargs
    ):
        super(GPTDecoder, self).__init__()
        
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        if hidden_size is None:
            hidden_size = in_features
        
        self.gpt = get_gpt_model(hidden_size, history_len)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        self.hidden_size = hidden_size
        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature: torch.Tensor):
        time_step=None
        attention_mask=None
        if input_feature.dim() == 3:
            input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1]) # bs, seq_len, feat_dim
        input_feature = self.fc(input_feature)
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            
            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step ,attention_mask)
                x = x[:, -1].unsqueeze(1)
                
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x= self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)
                
        else:
            x = self.gpt(input_feature, time_step, attention_mask)
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
        actions = self.actions(x)
        gripper = self.gripper(x)
        return actions, gripper
    
    def get_pattern_name(self):
        return 'gpt_{}_'.format(self.hidden_size, )

class GPTDecoderActPad(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        use_vision = False,
        history_len = None,
        out_features: int = 6,
        hidden_size = None,
        last_action=False,
        use_diff=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='sampler',
        global_latent=10,
        **kwargs
    ):
        super(GPTDecoderActPad, self).__init__()
        
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        if hidden_size is None:
            hidden_size = in_features
        
        self.gpt = get_gpt_model(hidden_size, history_len, use_pe=False)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        self.hidden_size = hidden_size
        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        self.global_latent = global_latent
        self.use_vision = use_vision
        if self.use_vision:
            self.vision_resampler = PerceiverResampler(dim=hidden_size)
        if pooling == 'sampler':
            self.global_1d_pool = PerceiverResampler(dim=hidden_size, depth=2, num_latents=global_latent)
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature: torch.Tensor, rgb=None):
        time_step=None
        attention_mask=None
        input_feature = self.global_1d_pool(input_feature.unsqueeze(1)).squeeze(1)
        input_feature = input_feature.view(-1, self.window_size, self.global_latent, input_feature.shape[-1]) # bs, seq_len, n_tok, feat_dim
        bs, seq_len, n_tok = input_feature.shape[:3]
        input_feature = self.fc(input_feature) # # bs, seq_len, n_tok, feat_dim
        attention_mask = torch.ones((bs, n_tok, seq_len), dtype=torch.long).to(input_feature.device)
        
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            
            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step ,attention_mask)
                x = x[:, -1].unsqueeze(1)
                
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x= self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)
                
        else:
            x = self.gpt(input_feature, time_step, attention_mask)
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
        actions = self.actions(x)
        gripper = nn.functional.sigmoid(self.gripper(x))
        return actions, gripper
    
    def get_pattern_name(self):
        return 'gpt_{}_'.format(self.hidden_size, )


class DiffusionDecoder(ActionDecoder):
    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        history_len = None,
        horizon = 32,
        input_dim: int = 7, # dim of vectors to be diffused
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        n_timesteps=150,
        clip_denoised=False,
        predict_epsilon=True,
        normalizer = LinearNormalizer()
    ):
        super(DiffusionDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.window_size = window_size
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        self.normalizer = normalizer
        self.data_dim = input_dim

        self.model = ConditionalUnet1D(
            input_dim,
            global_cond_dim=feature_dim,
            # global_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory
        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).clone()

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, local_cond=None, global_cond=None, returns=None):

        if returns is not None: 
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, local_cond, global_cond, returns, use_dropout=False)
            epsilon_uncond = self.model(x, t, local_cond, global_cond, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, t, local_cond, global_cond)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, local_cond=None, global_cond=None, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, local_cond=local_cond, global_cond=global_cond, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, verbose=False, return_diffusion=False, **kwargs
    ):
        device = self.betas.device

        batch_size = cond_data.shape[0]
        x = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device
        )

        if return_diffusion:
            diffusion = [x]

        x[cond_mask] = cond_data[cond_mask]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):

            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # 1. predict model output and replace sample
            x = self.p_sample(x, timesteps, local_cond, global_cond, returns)
            
            # 2. apply conditioning
            x[cond_mask] = cond_data[cond_mask]

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, action_seq_len=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        # horizon = action_seq_len or self.action_seq_len
        # batch_size = len(list(cond_data.values())[0])
        # shape = (batch_size, horizon, self.action_dim) # cond_data.shape
        return self.p_sample_loop(cond_data, cond_mask, local_cond, global_cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    def forward(
        self,
        x,
        t,
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        return self.model(x, t, local_cond, global_cond)

    def act(
        self,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions, self.hidden_state = self(
            input_feature, self.hidden_state
        )

        raise NotImplementedError


import math
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .multihead_custom_attention import MultiheadCustomAttention
from robouniview.data.keyframe_data import normalise_quat, unnormalize_pos
from robouniview.data import pytorch3d_transforms as pytorch3d_transforms

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code
    
class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, math.ceil(self.feature_dim/3), 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / math.ceil(self.feature_dim/3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code[..., :self.feature_dim, :]

class AdaLN(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
             nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (N, B, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x
    
class RelativeCrossAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0, use_adaln=False):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None, pad_mask=None):
        if diff_ts is not None:
            adaln_query = self.adaln(query, diff_ts)
        else:
            adaln_query = query
        attn_output, _ = self.multihead_attn(
            query=adaln_query,
            key=value,
            value=value,
            rotary_pe=None if query_pos is None else (query_pos, value_pos),
            key_padding_mask=pad_mask
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output

class FeedforwardLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0,
                 use_adaln=False):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, diff_ts=None):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output
    
class FFWRelativeSelfAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads, num_layers,
                 use_adaln=True):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, diff_ts=None,
                query_pos=None, context=None, context_pos=None):
        output = []
        for i in range(self.num_layers):
            query = self.attn_layers[i](
                query, query, diff_ts, query_pos, query_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output
    
class FFWRelativeCrossAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads, num_layers,
                 use_adaln=True):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None):
        output = []
        for i in range(self.num_layers):
            query = self.attn_layers[i](
                query, value, diff_ts, query_pos, value_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output
    
class DiffusionHead(nn.Module):
    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 rotation_parametrization='quat',
                 nhist=3):
        super().__init__()

        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion

        # Encoders
        self.traj_encoder = nn.Linear(9, embedding_dim) # 3+6
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )
        self.self_attn = FFWRelativeSelfAttentionModule(
            embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
        )

        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = FFWRelativeSelfAttentionModule(
            embedding_dim, num_attn_heads, 2, use_adaln=True
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)

    def forward(self, trajectory, timestep,
                context_feats, context, adaln_gripper_feats):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = rearrange(traj_feats, 'b l c -> l b c')
        context_feats = rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
        )
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = gripper_features
        rel_pos = rel_gripper_pos
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=None,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, num_gripper, instr_feats=None
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats=None
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return position, rotation, openess

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features


class DiffusionActorDecoder(ActionDecoder):
    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        history_len = None,
        horizon = 32,
        input_dim: int = 7, # dim of vectors to be diffused
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        n_timesteps=150,
        clip_denoised=False,
        predict_epsilon=True,
        normalizer = LinearNormalizer(),

        rotation_parametrization='6D',
        nhist=1, # 爪子位姿的长度
    ):
        super(DiffusionActorDecoder, self).__init__()

        # self.return_feature = return_feature
        self.in_features = feature_dim
        # self.out_features = out_features
        self.window_size = window_size
        # self.multi_step_action = multi_step_action
        self.history_len = history_len
        # self.use_diff = use_diff
        self.hidden_state = None
        self.history_memory = []

        self._rotation_parametrization = rotation_parametrization
        self.prediction_head = DiffusionHead(
            embedding_dim=feature_dim,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist
        )

        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        self.n_steps = n_timesteps

        # Current gripper learnable features
        num_attn_heads=8
        self.curr_gripper_embed = nn.Embedding(nhist, feature_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            feature_dim, num_attn_heads, num_layers=3, use_adaln=False
        )
        self.relative_pe_layer = RotaryPositionEncoding3D(feature_dim)

    #     return sample
    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context, # pos
            adaln_gripper_feats,
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            adaln_gripper_feats=adaln_gripper_feats,
        )
    
    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            pos = self.position_noise_scheduler.step(
                out[..., :3], t, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                out[..., 3:9], t, trajectory[..., 3:9]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        trajectory = torch.cat((trajectory, out[..., 9:]), -1)

        return trajectory

    def unconvert_rot(signal, _rotation_parametrization='6D', _quaternion_format = 'xyzw'):
        if _rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = pytorch3d_transforms.compute_rotation_matrix_from_ortho6d(rot)
                quat = pytorch3d_transforms.matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = pytorch3d_transforms.compute_rotation_matrix_from_ortho6d(rot)
                quat = pytorch3d_transforms.matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if _quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    def forward(
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        state_tensor = None,
        action_mask = None,
        gt_trajectory = None,  # GT,训练时需要在这个上边添加噪声
    ):

        interpolation_length = 20
        B, nhist, _, D = state_tensor.shape
        state_tensor = rearrange(state_tensor, "B T 1 D -> B T D")
        mask_indices = action_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
        action_feature = input_feature[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
        action_feature = rearrange(action_feature, "(B T) D -> B T D", B=B) # 8表示8个token
        action_pose = torch.zeros([action_feature.shape[0], action_feature.shape[1], 3], device=input_feature.device)

        adaln_gripper_feats, _ = self._encode_gripper(state_tensor,  self.curr_gripper_embed, action_feature, action_pose)
        fixed_inputs = (action_feature,
                        action_pose, # pos
                        adaln_gripper_feats,)

        if not self.training:
            # Condition on start-end pose
            cond_data = torch.zeros(
                (B, interpolation_length, 9),
                device=input_feature.device
            )
            cond_mask = torch.zeros_like(cond_data)
            cond_mask = cond_mask.bool()

            # Sample
            trajectory = self.conditional_sample(
                cond_data,
                cond_mask,
                fixed_inputs
            )

            # # Normalize quaternion
            # if self._rotation_parametrization != '6D':
            #     trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
            # # Back to quaternion
            # trajectory = self.unconvert_rot(trajectory)
            # # unnormalize position
            # trajectory[:, :, :3] = unnormalize_pos(trajectory[:, :, :3])
            # # Convert gripper status to probaility
            # if trajectory.shape[-1] > 7:
            #     trajectory[..., 7] = trajectory[..., 7].sigmoid()

            return trajectory[..., :9], trajectory[..., 9:]
        
        else:
            gt_openess = gt_trajectory[1]
            gt_trajectory = gt_trajectory[0]
        
            # Condition on start-end pose
            cond_data = torch.zeros_like(gt_trajectory)
            cond_mask = torch.zeros_like(cond_data)
            cond_mask = cond_mask.bool()

            # Sample noise
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

            # Sample a random timestep
            timesteps = torch.randint(
                0,
                self.position_noise_scheduler.config.num_train_timesteps,
                (len(noise),), device=noise.device
            ).long()

            # Add noise to the clean trajectories
            pos = self.position_noise_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3],
                timesteps
            )
            rot = self.rotation_noise_scheduler.add_noise(
                gt_trajectory[..., 3:9], noise[..., 3:9],
                timesteps
            )
            noisy_trajectory = torch.cat((pos, rot), -1)
            noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
            assert not cond_mask.any()

            # Predict the noise residual
            pred = self.policy_forward_pass(
                noisy_trajectory, timesteps, fixed_inputs
            )

            # Compute loss
            total_loss = 0
            for layer_pred in pred:
                trans = layer_pred[..., :3]
                rot = layer_pred[..., 3:9]
                loss = (
                    30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
                    + 10 * F.l1_loss(rot, noise[..., 3:9], reduction='mean')
                )
                if torch.numel(gt_openess) > 0:
                    openess = layer_pred[..., 9:]
                    loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
                total_loss = total_loss + loss
            return total_loss


    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos



if __name__ == "__main__":
    model = GPTDecoder(128, 24)
    in_feat = torch.randn((4*24, 12, 128))
    out = model(in_feat)
    print(out[0].shape, out[1].shape)
    pass