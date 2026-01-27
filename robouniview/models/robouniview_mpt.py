import torch
from einops import rearrange, repeat
from torch import nn
import copy
from open_flamingo.src.helpers import PerceiverResampler
from robouniview.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder
from robouniview.models.transformers.uvformer import DeformableTransformer
from robouniview.models.transformers.position_encoding import PositionEmbeddingSine, RotaryPositionEncoding3D
from collections import namedtuple
import yaml
import argparse
import copy
import yaml
import numpy as np
import cv2
import pybullet as p
from robouniview.models.transformers.petr import PETR
from robouniview.models.loss_func import (
    FocalLoss, Balanced_BCE_loss, CELoss, BinaryDiceLoss, CELossIgnoreSem, l1_loss)

from robouniview.models.occ_head import FastEncoderHead,Upsample2d_3d,Decoder_3d,Upsample2d_3d_tiny,Decoder_3d_tiny

class MPTFlamingo(nn.Module):
    def __init__(
        self,
        args,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        # this is the window size sampled from the episode
        window_size: int = 8,
        use_gripper=False,
        fusion_mode='',
        sep_resampler=False,
        use_state=False,
        use_diff=False,
        diff_horizon=32,
        last_action=False,
        n_timesteps=150,
        state_dim=15,
        use_hist=False,
        debug=False,
        predict_epsilon=True,
        pad_length=-1,
        multi_step_action=1,
        sep_lm_head=False,
        return_feature = False,
        llm='llama',
        pooling='max',
        residual=False,
        tcp_rel=False,
        replan=-1,
        decoder_type='lstm',
        hidden_size=None,
        fwd_pred=False,
        fwd_pred_hand=False,
        global_latent=10,
        no_image_patch=False,
        refresh=-1
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()
        self.args = args
        self.occ_loss =  args.loss.occ_loss
        self.train_action = args.training.train_action
        self.fusion_mode = fusion_mode
        self.vis_dim = vis_dim

        self.uvformer = DeformableTransformer(self.args, self.args.UVformer.transformer_config)
        if self.occ_loss:
            layers_config = {'in_channels': self.vis_dim, 'out_channels': 160, 'upsample': 2, 'head_module': 'regnet950MF'}

            self.occ_decoder = FastEncoderHead(layers_config)

            self.balanced_bce_loss = Balanced_BCE_loss(
                1,
                reduction="mean",
            )
        if hasattr(self.args, 'alignment_layer') and self.args.model.alignment_layer == 'Linear':
            self.alignment_layer = nn.Linear(self.vis_dim, self.vis_dim)
        
        if hasattr(self.args, 'alignment_layer') and self.args.model.alignment_layer == 'Resampler':
            self.alignment_layer = PerceiverResampler(dim=self.vis_dim)

        self.petr = PETR()

        self.occ_loss_weight = self.args.loss.occ_loss_weight

        self.Upsample2d_3d = Upsample2d_3d()
        self.occ_decoder = Decoder_3d()

        self.position_embedding = PositionEmbeddingSine(self.args.UVformer.transformer_config.hidden_dim/2, normalize=True)
        self.use_gripper = use_gripper
        self.use_state = use_state
        
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        
        self.window_size = window_size
        self.tcp_rel = tcp_rel
        self.act_step = multi_step_action
        print('window size: {}'.format(window_size))
        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.sep_resampler = sep_resampler
        self.use_hist = use_hist
        self.lang_encoder = lang_encoder
        self.pad_length = pad_length
        self.replan = replan
        if self.replan != -1:
            self.replan = min(int(replan * self.window_size), 180)
        self.refresh = refresh
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.residual = residual
        print(self.vis_dim, self.lang_dim)
        print(lang_encoder.config)
        if not debug:
            if 'llama' in llm:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    use_media_placement_augmentation=self.use_media_placement_augmentation,
                    residual=residual,
                )
            else:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    lang_hidden_size=self.lang_dim,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    gradient_checkpointing=False,
                )

        if sep_resampler:
            self.perceiver_gripper = PerceiverResampler(dim=self.vis_dim)
            self.perceiver_gripper.load_state_dict(copy.deepcopy(self.perceiver.state_dict()))
        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        if use_hist:
            self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
        # To-do: nn archiecture for actor
        self.llm = llm
        if llm=='llama':
            in_features = lang_encoder.lm_head.in_features
        else:
            in_features = self.lang_dim
        self.use_diff = use_diff
        self.decoder_type = decoder_type
        if decoder_type == 'lstm':
            lm_head = DeterministicDecoder(in_features, self.window_size, 
            use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling)
            self.lang_encoder.lm_head = lm_head
        elif decoder_type == 'fc':
            if use_hist:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            elif 'vit_concat' in fusion_mode:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            else:
                raise NotImplementedError
        elif decoder_type == 'diffusion':
            if use_diff:
                self.diffusion_model = DiffusionDecoder(
                    self.action_head.hidden_size, 
                    self.window_size,
                    input_dim=self.action_head.out_features+1,
                    n_timesteps=n_timesteps,
                    horizon=diff_horizon,
                    predict_epsilon=predict_epsilon,
                )
            else:
                raise NotImplementedError
        elif decoder_type=='gpt':
            lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling, hidden_size=hidden_size)
            self.lang_encoder.lm_head = self.action_head = lm_head
        else:
            raise NotImplementedError
        
        sep_lm_head = True
        self.sep_lm_head = sep_lm_head
        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()
        self.env = None

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        state_tensor = None,
        calib = None,
        pcd = None,
        return_feature = False,
        policy_mask=None
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        raw_rgb = vision_x.clone()
        raw_gripper = vision_gripper.clone()
        self.pcd = pcd
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.use_hist:
                self._encode_history_vision_post_fusion(vision_x, vision_gripper)
            else:
                self._encode_multi_vision_UVformer_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)

        if self.train_action:
            output = self.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask.bool(),
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True
            )

            output_hs = output.hidden_states[-1]
            output_hs = self.lm_head(output_hs, state_tensor=state_tensor, return_feature=return_feature)
            output.logits = output_hs
        else:
            output = []
        if self.occ_loss and self.pcd is not None:
            loss_occ = self.loss_occ()
        else:
            loss_occ = {}
        
        return output,loss_occ

    


    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_vision(self, vision_x: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

            
    def _encode_multi_vision_UVformer_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, calib, state_tensor=None):
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        feats = {}

        B, T, F, HxW, C = vision_rgb.shape
        vision_rgb = rearrange(vision_rgb, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
        vision_gripper = rearrange(vision_gripper, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
        
        _calib1 = {'rgb_static':{'extrinsic_matrix':rearrange(calib[0]," B T H W -> (B T) H W").cpu(),
                                'intrinsic_matrix':rearrange(calib[1]," B T H W -> (B T) H W").cpu(),
                                'distCoeffs_matrix':rearrange(calib[2]," B T H -> (B T) H").cpu()},
                    'rgb_gripper':{'extrinsic_matrix':rearrange(calib[3]," B T H W -> (B T) H W").cpu(),
                                    'intrinsic_matrix':rearrange(calib[4]," B T H W -> (B T) H W").cpu(),
                                    'distCoeffs_matrix':rearrange(calib[5]," B T H -> (B T) H").cpu()}}
        
        state_matrix = rearrange(calib[6]," B T H W -> (B T) H W").cpu()

        x = [[vision_rgb],[vision_gripper]]

        uv_feat = self.uvformer(x, _calib1) 
        occ_feat = uv_feat.clone()
        if self.occ_loss :
            occ_feat, occ_feat2 = self.Upsample2d_3d(occ_feat)#(B*T, C, Z, H, W)
            self.occ = self.occ_decoder(occ_feat)
            self.occ = rearrange(self.occ, "BT C Z H W -> BT H W Z C")
            
        pos = self.position_embedding(uv_feat)
      
        if hasattr(self.args, 'alignment_layer') and self.args.model.alignment_layer == 'Linear':
            uv_feat = rearrange(uv_feat, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
            pos = rearrange(pos, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
            uv_feat = uv_feat + pos
            uv_feat = self.alignment_layer(uv_feat)
        if hasattr(self.args, 'alignment_layer') and self.args.model.alignment_layer == 'Resampler':
       
            uv_feat = rearrange(uv_feat, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
            pos = rearrange(pos, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
            uv_feat = uv_feat + pos
            uv_feat = self.alignment_layer(uv_feat)
        feats['uv_feat'] = uv_feat

        rg_em = _calib1['rgb_gripper']['extrinsic_matrix'].clone()
        rs_em = _calib1['rgb_static']['extrinsic_matrix'].clone()
        for i, _rg_em in enumerate(rg_em):
            _calib1['rgb_gripper']['extrinsic_matrix'][i]  = rg_em[i] @ torch.linalg.inv(state_matrix[i])
        for i, _rs_em in enumerate(rs_em):
            _calib1['rgb_static']['extrinsic_matrix'][i]  = rs_em[i] @ torch.linalg.inv(state_matrix[i])
        _calib2 = {'rgb_gripper':_calib1['rgb_gripper']}
        
        pos = self.position_embedding(vision_gripper)
        pos_embed = self.petr(vision_gripper, _calib2, pos)
        uv_gripper_feat = vision_gripper + pos_embed
        uv_gripper_feat = rearrange(uv_gripper_feat, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
        uv_gripper_feat = self.perceiver(uv_gripper_feat)

        feats['uv_gripper_feat'] = uv_gripper_feat

        vision_x = []
        for feats_key in feats.keys():
            vision_x.append(feats[feats_key])
        
        vision_x = torch.concatenate(vision_x, dim=2)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x, feats
    
    
    def loss_occ(self):
        """
        Args:
            self.preds: shape of (bs, w, h, z, c)
            self.trues: shape of (bs, w, h, z, c)
        """

        self.occ_true = self.pcd
        self.occ_true = rearrange(self.occ_true, "B T H W Z C  -> (B T) H W Z C")

        c_classes = self.occ_true.shape[-1]
        grid_cls = ['occ','r','g','b']
        loss ={}
        for ind in range(c_classes):
            preds_ind = self.occ[:,:,:,:, ind]
            trues_ind = self.occ_true[:,:,:,:, ind] 
            if ind == 0: 
                loss_ind = self.balanced_bce_loss(preds_ind, trues_ind)
            else:
                loss_ind = l1_loss(preds_ind, trues_ind, self.occ_true[:,:,:,:, 0])
            loss[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * self.occ_loss_weight[ind]
            
        return loss
