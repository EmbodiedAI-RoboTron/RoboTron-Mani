"""
Model factory for creating Flamingo models and transforms.

This module provides functions to create and configure Flamingo models
with various vision encoders and language models.
"""

# Standard library imports
from typing import Optional

# Third-party imports
import open_clip
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.factory import _infer_decoder_layers_attr_name
from open_flamingo.src.utils import extend_instance
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local application imports
from robouniview.models.flamingo_mpt import MPTFlamingo

# Model configuration dictionary
mpt_dict = {
    "mpt_3b": {
        "lang_encoder_path": "path_to/mpt-1b-redpajama-200b",
        "tokenizer_path": "path_to/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
    },
    "mpt_dolly_3b": {
        "lang_encoder_path": "/mnt/data/modelzoo/anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "tokenizer_path": "/mnt/data/modelzoo/anas-awadalla/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "/mnt/data/modelzoo/openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
    },
    "keyframe_3b": {
        "lang_encoder_path": "/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly",
        "tokenizer_path": "/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "/project/robotic/RoboFlamingo/checkpoints/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
    },
    "mpt_4b": {
        "lang_encoder_path": "path_to/RedPajama-INCITE-Instruct-3B-v1",
        "tokenizer_path": "path_to/RedPajama-INCITE-Instruct-3B-v1",
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-4B-vitl-rpj3b-langinstruct/checkpoint.pt"
    },
    "mpt_base_4b": {
        "lang_encoder_path": "path_to/RedPajama-INCITE-Base-3B-v1",
        "tokenizer_path": "path_to/RedPajama-INCITE-Base-3B-v1",
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-4B-vitl-rpj3b/checkpoint.pt"
    },
    "mpt_9b": {
        "lang_encoder_path": "path_to/mpt-7b",
        "tokenizer_path": "path_to/mpt-7b",
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
    },
    "llama_9b": {
        "lang_encoder_path": "path_to/llama-7b-hf-jxu124",
        "tokenizer_path": "path_to/llama-7b-hf-jxu124",
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-9B/checkpoint.pt"
    }
}


# ============================================================================
# Transform Functions
# ============================================================================

def get_transforms(
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
    tokenizer_path: str = "path_to/llama-7b-hf-jxu124",
    use_local_files: bool = False,
):
    """Get image processor and tokenizer for Flamingo.
    
    Args:
        clip_vision_encoder_path: Path to CLIP vision encoder
        clip_vision_encoder_pretrained: Pretrained dataset name
        tokenizer_path: Path to tokenizer
        use_local_files: Whether to use local files only
        
    Returns:
        Tuple of (image_processor, text_tokenizer)
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )

    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    return image_processor, text_tokenizer


# ============================================================================
# Helper Functions
# ============================================================================

def _add_special_tokens_to_tokenizer(tokenizer, args):
    """Add special tokens to tokenizer based on configuration.
    
    Args:
        tokenizer: Tokenizer instance
        args: Configuration arguments
    """
    if args.action_decoder.action_token:
        print("Adding action tokens to tokenizer")
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<|endofchunk|>", "<image>", "<action>",
                "<static0>", "<static1>", "<static2>", "<static3>",
                "<static4>", "<static5>", "<static6>", "<static7>",
                "<static_s>", "<static_e>"
            ]
        })
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<gripper0>", "<gripper1>", "<gripper2>", "<gripper3>",
                "<gripper4>", "<gripper5>", "<gripper6>", "<gripper7>",
                "<gripper_s>", "<gripper_e>"
            ]
        })
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<obs0>", "<obs1>", "<obs2>", "<obs3>",
                "<obs4>", "<obs5>", "<obs6>", "<obs7>",
                "<obs_s>", "<obs_e>"
            ]
        })
    else:
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|endofchunk|>", "<image>"]
        })
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})


def _setup_mpt_embeddings(lang_encoder, lang_encoder_path):
    """Setup embeddings for MPT models that don't have get_input_embeddings.
    
    Args:
        lang_encoder: Language encoder model
        lang_encoder_path: Path to language encoder
    """
    if "mpt-1b-redpajama-200b" in lang_encoder_path:
        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte
            
            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings
        
        extend_instance(lang_encoder, EmbeddingFnMixin)


def _get_model_class(llm_name: str):
    """Get appropriate model class based on LLM name.
    
    Args:
        llm_name: Name of the language model
        
    Returns:
        Model class
    """
    if 'llama' in llm_name:
        from robouniview.models.flamingo_bc import BCFlamingo
        return BCFlamingo
    elif 'mpt' in llm_name:
        return MPTFlamingo
    elif 'keyframe' in llm_name:
        return KeyframeFlamingo
    else:
        raise NotImplementedError(f"Model class not found for llm_name: {llm_name}")


def _setup_trainable_parameters(model, args):
    """Setup which parameters are trainable based on configuration.
    
    Args:
        model: Model instance
        args: Configuration arguments
    """
    # Freeze all parameters first
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze UVFormer components if needed
    if args.training.train_uvformer:
        if hasattr(model, 'uvformer'):
            model.uvformer.requires_grad_(True)
        if hasattr(model, 'uvformer_gripper'):
            model.uvformer_gripper.requires_grad_(True)
        if hasattr(model, 'occ_decoder_UVFormer'):
            model.occ_decoder_UVFormer.requires_grad_(True)
        if hasattr(model, 'Upsample2d_3d_UVFormer'):
            model.Upsample2d_3d_UVFormer.requires_grad_(True)
    
    if hasattr(args.training, 'train_uvformer_gripper') and args.training.train_uvformer_gripper:
        if hasattr(model, 'uvformer_gripper'):
            model.uvformer_gripper.requires_grad_(True)
    
    # Unfreeze common components
    if hasattr(model, 'linear'):
        model.linear.requires_grad_(True)
    if hasattr(model, 'alignment_layer'):
        model.alignment_layer.requires_grad_(True)
    if hasattr(model, 'alignment_layer_gripper'):
        model.alignment_layer_gripper.requires_grad_(True)
    if hasattr(model, 'petr'):
        model.petr.requires_grad_(True)

    # Unfreeze RGB/gripper encoders if not using UVFormer
    if "UVFormer" not in args.model.fusion_mode:
        model.rgb.requires_grad_(True)
        model.gripper.requires_grad_(True)
    
    # Unfreeze decoder components
    model.decoder_embed.requires_grad_(True)
    model.decoder_blocks.requires_grad_(True)
    model.decoder_norm.requires_grad_(True)
    model.decoder_pred.requires_grad_(True)
    
    if hasattr(model, 'decoder_embed_obs'):
        model.decoder_embed_obs.requires_grad_(True)
    if hasattr(model, 'decoder_blocks_obs'):
        model.decoder_blocks_obs.requires_grad_(True)
    if hasattr(model, 'occ_decoder'):
        model.occ_decoder.requires_grad_(True)
    if hasattr(model, 'Upsample2d_3d'):
        model.Upsample2d_3d.requires_grad_(True)

    if hasattr(model, 'T_Embedding_layer'):
        model.T_Embedding_layer.requires_grad_(True)
    
    if hasattr(args.training, 'unfreeze_old_decoder_blocks') and args.training.unfreeze_old_decoder_blocks:
        model.lang_encoder.requires_grad_(True)
    
    # Setup cross-attention layers
    if args.training.train_params == -1:
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        model.perceiver.requires_grad_(True)
    else:
        param_per_layer = 140
        layer_num = int(args.training.train_params / param_per_layer + 0.5)
        cnt = 0
        for ix in range(len(model.lang_encoder.gated_cross_attn_layers) - 1, -1, -1):
            if cnt >= layer_num:
                break
            if model.lang_encoder.gated_cross_attn_layers[ix] is not None:
                model.lang_encoder.gated_cross_attn_layers[ix].requires_grad_(True)
                cnt += 1
    
    if args.training.freeze_sampler:
        model.perceiver.requires_grad_(False)
    
    if not args.training.freeze_embed:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    
    model.lang_encoder.lm_head.requires_grad_(True)

    if model.sep_lm_head:
        model.lm_head.requires_grad_(True)
    
    if model.use_diff:
        model.diffusion_model.requires_grad_(True)
    
    if args.training.unfreeze_vit:
        # Note: Gradient updates handled in forward pass
        model.vision_encoder.requires_grad_(True)


# ============================================================================
# Main Factory Function
# ============================================================================

def create_model_and_transforms(
    args,
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    window_size: int = 32,
    freeze_embed: bool = False,
    train_params: int = -1,
    use_gripper: bool = False,
    use_state: bool = False,
    last_action: bool = False,
    fusion_mode: str = '',
    pad_length: int = -1,
    debug: bool = False,
    sep_resampler: bool = False,
    sep_lm_head: bool = False,
    unfreeze_vit: bool = False,
    return_feature: bool = False,
    multi_step_action: int = 1,
    llm_name: str = 'llama_9b',
    pooling: str = 'max',
    residual: bool = False,
    tcp_rel: bool = False,
    replan: int = -1,
    decoder_type: str = 'lstm',
    hidden_size: Optional[int] = None,
    freeze_sampler: bool = False,
    fwd_pred: bool = False,
    fwd_pred_hand: bool = False,
    no_image_patch: bool = False,
    global_latent: int = 1,
    refresh: int = -1,
    clip_cache_dir: Optional[str] = None,
    **flamingo_kwargs,
):
    """Initialize a Flamingo model from pretrained vision and language encoders.
    
    Args:
        args: Configuration arguments object
        clip_vision_encoder_path: Path to pretrained CLIP model
        clip_vision_encoder_pretrained: Name of pretraining dataset
        lang_encoder_path: Path to pretrained language encoder
        tokenizer_path: Path to pretrained tokenizer
        cross_attn_every_n_layers: How often to add cross-attention layer
        use_local_files: Whether to use local files only
        decoder_layers_attr_name: Name of decoder layers attribute
        window_size: Window size sampled from episode
        freeze_embed: Whether to freeze embeddings
        train_params: Number of trainable parameters (-1 for all)
        use_gripper: Whether to use gripper input
        use_state: Whether to use state input
        last_action: Whether to use last action only
        fusion_mode: Fusion mode string
        pad_length: Padding length
        debug: Debug mode flag
        sep_resampler: Whether to use separate resampler
        sep_lm_head: Whether to use separate LM head
        unfreeze_vit: Whether to unfreeze vision encoder
        return_feature: Whether to return features
        multi_step_action: Number of multi-step actions
        llm_name: Name of language model
        pooling: Pooling type
        residual: Whether to use residual connections
        tcp_rel: Whether to use TCP relative coordinates
        replan: Replanning parameter
        decoder_type: Type of decoder
        hidden_size: Hidden size for decoder
        freeze_sampler: Whether to freeze sampler
        fwd_pred: Whether to use forward prediction
        fwd_pred_hand: Whether to use forward prediction for hand
        no_image_patch: Whether to disable image patches
        global_latent: Global latent dimension
        refresh: Refresh parameter
        clip_cache_dir: Cache directory for CLIP
        **flamingo_kwargs: Additional Flamingo-specific arguments
        
    Returns:
        Tuple of (model, image_processor, text_tokenizer)
    """
    # Create vision encoder
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=clip_cache_dir,
    )
    vision_encoder.visual.output_tokens = True

    # Create tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=use_local_files
    )
    _add_special_tokens_to_tokenizer(text_tokenizer, args)

    # Create language encoder
    if debug:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path, ignore_keys=["config"], trust_remote_code=True
        )
        lang_encoder.init_weights(False)
    else:
        print(f"Loading language encoder from: {lang_encoder_path}")
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            local_files_only=use_local_files,
            trust_remote_code=True
        )

    # Setup MPT embeddings if needed
    _setup_mpt_embeddings(lang_encoder, lang_encoder_path)
    extend_instance(lang_encoder, FlamingoLMMixin)

    # Setup decoder layers
    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    # Get model class and create model
    Model_fn = _get_model_class(llm_name)
    
    vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"]
    eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
    media_token_id = text_tokenizer.encode("<image>")[-1]

    model = Model_fn(
        args,
        vision_encoder,
        lang_encoder,
        eoc_token_id,
        media_token_id,
        vis_dim=vis_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size,
        use_gripper=use_gripper,
        use_state=use_state,
        fusion_mode=fusion_mode,
        last_action=last_action,
        pad_length=pad_length,
        sep_resampler=sep_resampler,
        sep_lm_head=sep_lm_head,
        return_feature=return_feature,
        multi_step_action=multi_step_action,
        llm=llm_name,
        pooling=pooling,
        residual=residual,
        tcp_rel=tcp_rel,
        replan=replan,
        decoder_type=decoder_type,
        hidden_size=hidden_size,
        refresh=refresh,
        fwd_pred=fwd_pred,
        fwd_pred_hand=fwd_pred_hand,
        no_image_patch=no_image_patch,
        global_latent=global_latent,
        **flamingo_kwargs,
    )

    # Setup trainable parameters
    _setup_trainable_parameters(model, args)

    print(f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    return model, image_processor, text_tokenizer
