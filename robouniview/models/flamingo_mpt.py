"""
MPT Flamingo model implementation.

This module provides the MPTFlamingo class which extends the base Flamingo
architecture with multi-modal fusion capabilities for robot manipulation.
"""

# Standard library imports
import copy
import time

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from open_flamingo.src.helpers import PerceiverResampler
from torch import nn
from torch.cuda.amp import autocast

# Local application imports
from robouniview.models.action_head import (
    DeterministicDecoder,
    DiffusionDecoder,
    FCDecoder,
    GPTDecoder,
    Multi_Action_Token_FCDecoder,
    TokenFCDecoder,
)
from robouniview.models.loss_func import Balanced_BCE_loss, l1_loss
from robouniview.models.occ_head import (
    Decoder_3d,
    Decoder_3d_4s,
    Upsample2d_3d,
    Upsample2d_3d_UVFormer,
)
from robouniview.models.transformers.petr import PETR, inverse_sigmoid
from robouniview.models.transformers.position_encoding import (
    PositionEmbeddingSine,
    get_2d_sincos_pos_embed,
)
from robouniview.models.transformers.uvformer import DeformableTransformer
from robouniview.models.vision_transformer import Block

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
        refresh=-1,
        nclass_gripper=1,
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
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size
        self.occ_loss =  args.loss.occ_loss
        self.occ_loss_weight = self.args.loss.occ_loss_weight
        self.train_action = args.training.train_action
        self.fusion_mode = fusion_mode
        self.vis_dim = vis_dim

        # decode image
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.lang_dim))
        self.decoder_pos_embed_static = nn.Parameter(torch.zeros(1, (112//14)**2, self.lang_dim), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed_static = get_2d_sincos_pos_embed(self.decoder_pos_embed_static.shape[-1], (112//14))
        self.decoder_pos_embed_static.data.copy_(torch.from_numpy(decoder_pos_embed_static).float().unsqueeze(0))
        self.decoder_pos_embed_gripper = nn.Parameter(torch.zeros(1, (112//14)**2,self.lang_dim), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed_gripper = get_2d_sincos_pos_embed(self.decoder_pos_embed_gripper.shape[-1], (112//14))
        self.decoder_pos_embed_gripper.data.copy_(torch.from_numpy(decoder_pos_embed_gripper).float().unsqueeze(0))
        self.decoder_embed = nn.Linear(self.lang_dim, self.lang_dim, bias=True)
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(self.lang_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(self.lang_dim)
        self.decoder_pred = nn.Linear(self.lang_dim, 14**2 * 3, bias=True) # decoder to patch
        
        # encoder 
        if "UVFormer" in self.fusion_mode:
            z_range = self.args.UVformer.transformer_config.ref_z_range
            assert int((z_range[1]-z_range[0]) / self.args.UVformer.transformer_config.grid_resolution[-1]) == 5
            self.Upsample2d_3d_UVFormer = Upsample2d_3d_UVFormer(in_channels=1024, out_channels=128, z=10) # UVFormer仅使用了两次上采样，从20*20*10-》80*80*40
            self.uvformer = DeformableTransformer(self.args, self.args.UVformer.transformer_config)
            self.occ_decoder_UVFormer = Decoder_3d_4s()
            if self.occ_loss:
                self.balanced_bce_loss = Balanced_BCE_loss(1, reduction="mean",)
            if hasattr(self.args.model, 'alignment_layer') and self.args.model.alignment_layer == 'Linear':
                self.alignment_layer = nn.Linear(self.vis_dim, self.vis_dim)
            elif hasattr(self.args.model, 'alignment_layer') and self.args.model.alignment_layer == 'Resampler':
                self.alignment_layer = PerceiverResampler(dim=self.vis_dim)
        else:
            self.rgb = nn.Embedding(1,1024)
            self.gripper = nn.Embedding(1,1024)
            
            # decoder occ
            self.decoder_pos_embed_obs = nn.Parameter(torch.zeros(1, (10)**2,self.lang_dim), requires_grad=False)  # (1, n_patch, h)
            decoder_pos_embed_obs = get_2d_sincos_pos_embed(self.decoder_pos_embed_obs.shape[-1], (10))
            self.decoder_pos_embed_obs.data.copy_(torch.from_numpy(decoder_pos_embed_obs).float().unsqueeze(0))
            self.decoder_embed_obs = nn.Linear(self.lang_dim, self.lang_dim, bias=True)
            self.decoder_blocks_obs = nn.ModuleList([
                Block(self.lang_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for i in range(decoder_depth)])
            self.occ_decoder = Decoder_3d()
            z_range = self.args.UVformer.transformer_config.ref_z_range
            assert int((z_range[1]-z_range[0]) / self.args.UVformer.transformer_config.grid_resolution[-1]) == 5
            self.Upsample2d_3d = Upsample2d_3d(in_channels=2048, out_channels=128, z=5) 
        # encoder 公用
        self.petr = PETR(**self.args.PETR) if hasattr(self.args, 'PETR') else PETR() # 当UVFormer的时候仅有夹爪位置
        # self.split = nn.Unfold(kernel_size=(14, 14), stride=(14, 14), padding=(0, 0))

        if hasattr(self.args, 'T_Embedding') and self.args.T_Embedding :  # 未使用
            self.T_Embedding_layer = nn.Sequential(
                nn.Linear( 1 , self.vis_dim*4),
                nn.ReLU(),
                nn.Linear(self.vis_dim*4, self.vis_dim),
            )


        self.position_embedding = PositionEmbeddingSine(self.args.UVformer.transformer_config.hidden_dim/2, normalize=True)
        # self.pos_rgb, self.pos_gripper = None, None
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
            elif self.args.action_decoder.multi_action_token :
                self.lang_encoder.lm_head = self.action_head = Multi_Action_Token_FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, nclass_gripper=nclass_gripper)
            elif self.args.action_decoder.action_token :
                self.lang_encoder.lm_head = self.action_head = TokenFCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            else:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
                #raise NotImplementedError
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

    def _decode_image_from_tokens(
        self,
        output_hs: torch.Tensor,
        mask: torch.Tensor,
        decoder_embed: nn.Module,
        decoder_blocks: nn.ModuleList,
        decoder_norm: nn.Module,
        decoder_pred: nn.Module,
        decoder_pos_embed: nn.Parameter,
        output_size: tuple = (112, 112),
        patch_size: int = 14,
        num_tokens: int = 8,
    ):
        """
        Decode image from masked tokens.
        
        Args:
            output_hs: Hidden states from language model (B, T, D)
            mask: Boolean mask indicating token positions (B, T)
            decoder_embed: Decoder embedding layer
            decoder_blocks: Decoder transformer blocks
            decoder_norm: Decoder normalization layer
            decoder_pred: Decoder prediction head
            decoder_pos_embed: Positional embedding for decoder
            output_size: Output image size (H, W)
            patch_size: Patch size for reconstruction
            num_tokens: Number of tokens to extract
            
        Returns:
            Reconstructed image tensor
        """
        b = output_hs.shape[0]
        mask_indices = mask.nonzero(as_tuple=True)
        feature = output_hs[mask_indices[0], mask_indices[1]]
        feature = rearrange(feature, "(B T N) D -> B T N D", B=b, N=num_tokens)
        B, T, N, D = feature.shape

        n_patches = (output_size[0] // patch_size) ** 2
        mask_tokens = self.mask_token.repeat(B, T, n_patches, 1)
        mask_tokens = mask_tokens + decoder_pos_embed.unsqueeze(0).repeat(B, T, 1, 1)
        
        feature = decoder_embed(feature)
        pred = torch.cat([feature, mask_tokens], dim=2)
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        
        for blk in decoder_blocks:
            pred = blk(pred)
        
        pred = decoder_norm(pred)
        pred = decoder_pred(pred)
        pred = pred.reshape(B, T, -1, pred.shape[-1])
        pred = pred[:, :, -n_patches:]
        pred = rearrange(pred, "B T N H -> (B T) H N")

        unsplit = nn.Fold(
            output_size=output_size,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        )
        pred = unsplit(pred)
        return pred

    def _prepare_calib_dict(self, calib: list, window_size: int):
        """
        Prepare calibration dictionary from calib tensors.
        
        Args:
            calib: List of calibration tensors [extrinsic_static, intrinsic_static, ...]
            window_size: Window size for temporal slicing
            
        Returns:
            Dictionary with calibration parameters for rgb_static and rgb_gripper
        """
        return {
            'rgb_static': {
                'extrinsic_matrix': rearrange(calib[0][:, :window_size], "B T H W -> (B T) H W").cpu(),
                'intrinsic_matrix': rearrange(calib[1][:, :window_size], "B T H W -> (B T) H W").cpu(),
                'distCoeffs_matrix': rearrange(calib[2][:, :window_size], "B T H -> (B T) H").cpu(),
                'fov': rearrange(calib[7][:, :window_size], "B T H -> (B T) H").cpu(),
            },
            'rgb_gripper': {
                'extrinsic_matrix': rearrange(calib[3][:, :window_size], "B T H W -> (B T) H W").cpu(),
                'intrinsic_matrix': rearrange(calib[4][:, :window_size], "B T H W -> (B T) H W").cpu(),
                'distCoeffs_matrix': rearrange(calib[5][:, :window_size], "B T H -> (B T) H").cpu(),
                'fov': rearrange(calib[8][:, :window_size], "B T H -> (B T) H").cpu(),
            }
        }

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
        action_mask = None,
        static_mask = None,
        gripper_mask = None,
        obs_mask = None,
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

        self.pcd = pcd
        assert (vision_x is not None) or use_cached_vision_x, ("Must provide either vision_x or use_cached_vision_x to True.")

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (vision_x is None), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()
        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.use_hist: self._encode_history_vision_post_fusion(vision_x, vision_gripper)
            if "UVFormer" in self.fusion_mode: self._encode_multi_vision_UVformer_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)
            else: self._encode_temporal_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)

        kwarg = {}
        output, static_pred, gripper_pred, obs_pred, aux_loss = [], None, None, None, {}
        if self.train_action:
            output = self.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask.bool(),
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
                **kwarg,
            )
            output_hs = output.hidden_states[-1]
            b, t, d = output_hs.shape
            
            # Decode static camera image
            if static_mask is not None:
                static_pred = self._decode_image_from_tokens(
                    output_hs, static_mask,
                    self.decoder_embed, self.decoder_blocks,
                    self.decoder_norm, self.decoder_pred,
                    self.decoder_pos_embed_static,
                    output_size=(112, 112), patch_size=14, num_tokens=8
                )

            # Decode gripper camera image
            if gripper_mask is not None:
                gripper_pred = self._decode_image_from_tokens(
                    output_hs, gripper_mask,
                    self.decoder_embed, self.decoder_blocks,
                    self.decoder_norm, self.decoder_pred,
                    self.decoder_pos_embed_gripper,
                    output_size=(112, 112), patch_size=14, num_tokens=8
                )

            # Decode occupancy from obs tokens
            if obs_mask is not None:
                b = output_hs.shape[0]
                mask_indices = obs_mask.nonzero(as_tuple=True)
                obs_feature = output_hs[mask_indices[0], mask_indices[1]]
                obs_feature = rearrange(obs_feature, "(B T N) D -> B T N D", B=b, N=8)
                B, T, N, D = obs_feature.shape

                mask_tokens = self.mask_token.repeat(B, T, 100, 1)  # 10x10 patches
                mask_tokens = mask_tokens + self.decoder_pos_embed_obs.unsqueeze(0).repeat(B, T, 1, 1)
                obs_feature = self.decoder_embed_obs(obs_feature)
                obs_pred_ = torch.cat([obs_feature, mask_tokens], dim=2)

                obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])
                for blk in self.decoder_blocks_obs:
                    obs_pred_ = blk(obs_pred_)
                obs_pred_ = obs_pred_[:, -100:]  # Extract 10x10 patches
                obs_pred_ = rearrange(obs_pred_, "B (H W) D -> B D H W", H=10, W=10)
                obs_pred_ = self.Upsample2d_3d(obs_pred_)
                obs_pred = self.occ_decoder(obs_pred_[0])
                obs_pred = rearrange(obs_pred, "BT C Z H W -> BT H W Z C")
            
            # Action prediction  
            if self.args.action_decoder.action_token or self.args.action_decoder.multi_action_token:
                output_hs = self.lm_head(output_hs, state_tensor=state_tensor, action_mask = action_mask)
            else:
                output_hs = self.lm_head(output_hs, state_tensor=state_tensor)
            output.logits = output_hs
            # Only fall back to UVFormer occ prediction when it was computed (pcd provided).
            if obs_pred is None and self.occ_loss and self.pcd is not None:
                obs_pred = self.occ
        
        if self.occ_loss and self.pcd is not None:
            aux_loss['loss_occ'] = self.loss_occ()
            
        return output, static_pred, gripper_pred, obs_pred, aux_loss


    def _encode_vision(self, vision_x: torch.Tensor, state_tensor=None):
        """
        Encode vision input through vision encoder with mixed precision.
        
        Args:
            vision_x: Vision input, shape (B, T_img, F, C, H, W)
                Currently only F=1 is supported (single-frame videos)
            state_tensor: Optional state tensor (unused in this method)
            
        Returns:
            Encoded vision features, shape (B, T, F, v, d)
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad(), autocast():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x


    def _encode_temporal_fusion(
        self,
        vision_rgb: torch.Tensor,
        vision_gripper: torch.Tensor,
        calib,
        state_tensor=None
    ):
        """
        Encode temporal multi-view fusion (without UVFormer).
        
        Args:
            vision_rgb: RGB vision input, shape (B, T, 1, 1, C, H, W)
            vision_gripper: Gripper vision input, shape (B, T, 1, 1, C, H, W)
            calib: Calibration tensors
            state_tensor: Optional state tensor
            
        Returns:
            vision_x: Fused vision features
            feats: Additional features dictionary
        """
        vision_rgb = vision_rgb.squeeze(2)
        vision_gripper = vision_gripper.squeeze(2)

        with torch.no_grad():
            vision_rgb = self._encode_vision(vision_rgb)
            vision_gripper = self._encode_vision(vision_gripper)
        B, T, F, v, d = vision_rgb.shape

        # Optional temporal embedding
        if hasattr(self.args, 'T_Embedding') and self.args.T_Embedding:
            for t in range(T):
                t_emb = self.T_Embedding_layer(
                    inverse_sigmoid(torch.tensor(t * 1.0 / T)).unsqueeze(0).to(vision_rgb.device)
                )
                vision_rgb[:, t, ...] += t_emb
                vision_gripper[:, t, ...] += t_emb

        vision_gripper = rearrange(vision_gripper, "B T F (H W) C -> (B T F) C H W", H=16, W=16)
        vision_rgb = rearrange(vision_rgb, "B T F (H W) C -> (B T F) C H W", H=16, W=16)

        # Apply calibration-based positional encoding
        if calib is not None:
            _calib_dict = self._prepare_calib_dict(calib, self.window_size)
            
            with torch.no_grad():
                pos_rgb = self.position_embedding(vision_rgb)
                pos_gripper = self.position_embedding(vision_gripper)

            pos_embed_rgb = self.petr(vision_rgb, _calib_dict['rgb_static'], pos_rgb, 200, 200, 0)
            pos_embed_gripper = self.petr(vision_gripper, _calib_dict['rgb_gripper'], pos_gripper, 84, 84, 0)
            vision_rgb = vision_rgb + pos_embed_rgb
            vision_gripper = vision_gripper + pos_embed_gripper

            vision_rgb = rearrange(vision_rgb, "(B T F) C BH BW -> B T F (BH BW) C", B=B, T=T)
            vision_gripper = rearrange(vision_gripper, "(B T F) C BH BW -> B T F (BH BW) C", B=B, T=T)

        # Add modality embeddings
        vision_rgb = vision_rgb + self.rgb.weight[0]
        vision_gripper = vision_gripper + self.gripper.weight[0]

        # Perceiver resampler
        vision_rgb = self.perceiver(vision_rgb)
        vision_gripper = self.perceiver(vision_gripper)

        # Reshape for multi-action-token or standard mode
        if not self.args.action_decoder.multi_action_token:
            vision_rgb = rearrange(vision_rgb, "B T N D -> B (T N) D").unsqueeze(1)
            vision_gripper = rearrange(vision_gripper, "B T N D -> B (T N) D").unsqueeze(1)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)

        # Condition language model
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x, {}
    
    def _encode_multi_vision_UVformer_fusion(
        self,
        vision_rgb: torch.Tensor,
        vision_gripper: torch.Tensor,
        calib,
        state_tensor=None
    ):
        """
        Encode multi-view vision with UVFormer fusion.
        
        Args:
            vision_rgb: RGB vision input, shape (B, T, 1, 1, C, H, W)
            vision_gripper: Gripper vision input, shape (B, T, 1, 1, C, H, W)
            calib: Calibration tensors
            state_tensor: Optional state tensor
            
        Returns:
            vision_x: Fused vision features
            feats: Additional features dictionary
        """
        vision_rgb = vision_rgb.squeeze(2)
        vision_gripper = vision_gripper.squeeze(2)
        dtype = vision_rgb.dtype
        
        with torch.no_grad():
            vision_rgb = self._encode_vision(vision_rgb).to(dtype)
            vision_gripper = self._encode_vision(vision_gripper).to(dtype)
        B, T, F, HxW, C = vision_rgb.shape

        vision_rgb = rearrange(vision_rgb, "B T F (H W) C -> (B T F) C H W", H=16, W=16)
        vision_gripper = rearrange(vision_gripper, "B T F (H W) C -> (B T F) C H W", H=16, W=16)

        # Prepare calibration dictionary
        _calib_dict = self._prepare_calib_dict(calib, self.window_size)
        
        # UVFormer multi-view fusion
        x = [[vision_rgb], [vision_gripper]]
        uv_feat = self.uvformer(x, _calib_dict)

        # Generate occupancy prediction if enabled
        if self.occ_loss and self.pcd is not None:
            occ_feat = uv_feat.clone()
            occ_feat, _ = self.Upsample2d_3d_UVFormer(occ_feat)
            self.occ = self.occ_decoder_UVFormer(occ_feat)
            self.occ = rearrange(self.occ, "BT C Z H W -> BT H W Z C")
        
        # Apply alignment layer to UVFormer features
        with torch.no_grad():
            pos_rgb = self.position_embedding(uv_feat)
            pos_gripper = self.position_embedding(vision_gripper)
        
        if hasattr(self.args.model, 'alignment_layer') and self.args.model.alignment_layer == 'Linear':
            uv_feat = rearrange(uv_feat, "(B T) C BH BW -> B T (BH BW) C", B=B, T=T)
            pos_rgb = rearrange(pos_rgb, "(B T) C BH BW -> B T (BH BW) C", B=B, T=T)
            uv_feat = uv_feat + pos_rgb
            uv_feat = self.alignment_layer(uv_feat)
        elif hasattr(self.args.model, 'alignment_layer') and self.args.model.alignment_layer == 'Resampler':
            uv_feat = rearrange(uv_feat, "(B T F) C BH BW -> B T F (BH BW) C", B=B, T=T)
            pos_rgb = rearrange(pos_rgb, "(B T F) C BH BW -> B T F (BH BW) C", B=B, T=T)
            uv_feat = uv_feat + pos_rgb
            uv_feat = self.alignment_layer(uv_feat)

        # Process gripper features with PETR + Perceiver
        state_matrix = rearrange(calib[6][:, :self.window_size], "B T H W -> (B T) H W").cpu()
        rg_em = _calib_dict['rgb_gripper']['extrinsic_matrix'].clone()
        for i, _rg_em in enumerate(rg_em):
            _calib_dict['rgb_gripper']['extrinsic_matrix'][i] = rg_em[i] @ torch.linalg.inv(state_matrix[i])
        
        pos_embed = self.petr(vision_gripper, _calib_dict['rgb_gripper'], pos_gripper)
        uv_gripper_feat = vision_gripper + pos_embed
        uv_gripper_feat = rearrange(uv_gripper_feat, "(B T F) C BH BW -> B T F (BH BW) C", B=B, T=T)
        uv_gripper_feat = self.perceiver(uv_gripper_feat)
        
        # Concatenate and condition language model
        vision_x = torch.concat([uv_feat, uv_gripper_feat], dim=2)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
        
        return vision_x, {}
        
    def loss_occ(self):
        """
        Compute occupancy loss for 3D voxel grid prediction.
        
        Computes separate losses for:
        - Channel 0 (occupancy): Balanced BCE loss
        - Channels 1-3 (RGB): L1 loss masked by occupancy
        
        Returns:
            Dictionary of losses for each channel
        """
        self.occ_true = rearrange(self.pcd, "B T H W Z C -> (B T) H W Z C")

        c_classes = self.occ_true.shape[-1]
        grid_cls = ['occ', 'r', 'g', 'b']
        loss = {}
        
        for ind in range(c_classes):
            preds_ind = self.occ[:, :, :, :, ind]
            trues_ind = self.occ_true[:, :, :, :, ind]
            
            if ind == 0:  # Occupancy channel: use balanced BCE
                loss_ind = self.balanced_bce_loss(preds_ind, trues_ind)
            else:  # RGB channels: use L1 masked by occupancy
                loss_ind = l1_loss(preds_ind, trues_ind, self.occ_true[:, :, :, :, 0])
            
            loss[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * self.occ_loss_weight[ind]
            
        return loss
