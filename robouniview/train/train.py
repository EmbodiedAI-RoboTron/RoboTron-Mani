"""
Main training script for RoboUniView robot manipulation models.

This script handles distributed training setup, model initialization, checkpoint
management, and training loop orchestration.
"""

# Standard library imports
import argparse
import glob
import math
import os
import random
from collections import OrderedDict
from typing import Optional

# Third-party imports
import numpy as np
import torch
import swanlab as wandb
import socket
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

# Local application imports
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from robouniview.data import get_data
from robouniview.models.factory import create_model_and_transforms, mpt_dict
from train_utils import (
    get_checkpoint,
    get_ckpt_name,
    get_ckpt_name_pattern,
    train_one_epoch_calvin,
)

# Environment setup (must be before other imports that use these)
os.environ['NCCL_BLOCKING_WAIT'] = '0'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/project/robotic/Metaworld/mujoco210'
os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12356')


# ============================================================================
# Configuration Management
# ============================================================================

class GlobalConfig:
    """
    Configuration class with BACKWARD COMPATIBLE nested and flat access.
    
    Supports both:
    - Nested: args.training.batch_size_calvin
    - Flat: args.training.batch_size_calvin (for backward compatibility)
    """
    
    def __init__(self, config: dict, _is_nested_obj: bool = False):
        """Initialize configuration from dictionary.
        
        Args:
            config: Configuration dictionary (can be nested)
            _is_nested_obj: Internal flag to indicate if this is a nested config object
        """
        self.config = config
        self._is_nested_obj = _is_nested_obj
        self._load_config()

    def _load_config(self):
        """Load config items as attributes, handling nested dictionaries with backward compatibility."""
        # First pass: create nested objects and set direct attributes
        for key, val in self.config.items():
            if isinstance(val, dict):
                # Convert nested dict to GlobalConfig object
                setattr(self, key, GlobalConfig(val, _is_nested_obj=True))
            else:
                setattr(self, key, val)
    
    def to_dict(self):
        """Convert config back to nested dictionary (for wandb logging, etc.)."""
        result = {}
        for key, val in self.__dict__.items():
            if key in ['config', '_is_nested_obj'] or key.startswith('_'):
                continue
            if isinstance(val, GlobalConfig):
                result[key] = val.to_dict()
            else:
                # Only include in result if it's from the original config
                # (not a flattened copy from nested config)
                if key in self.config:
                    result[key] = val
        return result
    
    def __contains__(self, key):
        """Support 'in' operator for checking if attribute exists."""
        return hasattr(self, key)
    
    def __getitem__(self, key):
        """Support dictionary-style access: config['key']."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in config")
    
    def __setitem__(self, key, value):
        """Support dictionary-style assignment: config['key'] = value."""
        setattr(self, key, value)
    
    def get(self, key, default=None):
        """Support dict.get() method."""
        return getattr(self, key, default)
    
    def keys(self):
        """Return keys for dictionary unpacking support."""
        return [k for k in self.__dict__.keys() if k not in ['config', '_is_nested_obj'] and not k.startswith('_')]
    
    def values(self):
        """Return values for dictionary unpacking support."""
        return [getattr(self, k) for k in self.keys()]
    
    def items(self):
        """Return items for dictionary unpacking support."""
        return [(k, getattr(self, k)) for k in self.keys()]
    
    def __iter__(self):
        """Support iteration over keys (required for ** unpacking)."""
        return iter(self.keys())
    
    def __repr__(self):
        return f"GlobalConfig({self.to_dict()})"


def init_wandb(args, resuming: bool, log_code: bool = False, enabled: bool = True):
    """
    Initialize wandb/swanlab for experiment tracking.
    Note: SwanLab does not support resume parameter like WandB,
    so we create a new run even when resuming training.
    """
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Get experiment name from environment or config
    experiment_name = os.getenv("SWANLAB_PROJECT", args.run_name)
    if resuming:
        experiment_name = f"{experiment_name}_resumed"
        print(f"Resuming training with new SwanLab experiment: {experiment_name}")
    
    # Initialize run (SwanLab compatible API)
    if args.distributed.rank == 0 and args.distributed.local_rank == 0:
        run = wandb.init(
            experiment_name=experiment_name,  # SwanLab uses 'experiment_name'
            project="RoboTron_Mani",
            config=vars(args),
        )
    
    # Log source code if requested (may not be supported by SwanLab)
    if log_code and hasattr(wandb, 'run') and hasattr(wandb.run, 'log_code'):
        try:
            wandb.run.log_code(os.path.join(os.path.dirname(__file__), '..'))
        except Exception as e:
            print(f"Failed to log code: {e}")
    
    return


def load_global_config_yaml_only(config_path: str) -> GlobalConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        GlobalConfig instance
    """
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)
    return GlobalConfig(config)


# ============================================================================
# Utility Functions
# ============================================================================

def random_seed(seed: int = 42, rank: int = 0) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Base seed value
        rank: Process rank for distributed training
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.1,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a cosine learning rate schedule with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate (as fraction of max)
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch index for resuming
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress) * (1 - min_lr) + min_lr))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ============================================================================
# Setup Functions
# ============================================================================

def setup_environment(args) -> None:
    """Setup environment variables and configuration.
    
    Args:
        args: Training arguments
    """
    if args.logging.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    if args.action_decoder.tcp_rel:
        args.experimental.clip_state = True
    
    if args.action_decoder.eval_hist_size == -1:
        args.action_decoder.eval_hist_size = args.action_decoder.window_size
        if args.action_decoder.head_type == "diffusion":
            args.action_decoder.eval_hist_size = args.action_decoder.n_obs_steps
    
    if args.logging.save_checkpoints_to_wandb and not args.logging.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")


def setup_model(args, device_id: int):
    """Initialize and setup model.
    
    Args:
        args: Training arguments
        device_id: Device ID for model placement
        
    Returns:
        Tuple of (model, image_processor, tokenizer)
    """
    # Get model paths from config
    args.model.lm_path = mpt_dict[args.model.llm_name]["lang_encoder_path"]
    args.model.tokenizer_path = mpt_dict[args.model.llm_name]["tokenizer_path"]
    args.model.cross_attn_every_n_layers = mpt_dict[args.model.llm_name]["cross_attn_every_n_layers"]
    args.model.openflamingo_checkpoint = mpt_dict[args.model.llm_name]["openflamingo_checkpoint"]
    
    # Create model
    model, image_processor, tokenizer = create_model_and_transforms(
        args,
        args.model.vision_encoder_path,
        args.model.vision_encoder_pretrained,
        args.model.lm_path,
        args.model.tokenizer_path if args.model.tokenizer_path else args.model.lm_path,
        cross_attn_every_n_layers=args.model.cross_attn_every_n_layers,
        use_gripper=args.model.use_gripper,
        use_state=args.model.use_state,
        use_hist=args.action_decoder.use_hist,
        fusion_mode=args.model.fusion_mode,
        use_local_files=args.logging.offline,
        use_media_placement_augmentation=args.model.use_media_placement_augmentation,
        window_size=args.action_decoder.eval_hist_size,
        freeze_embed=args.training.freeze_embed,
        train_params=args.training.train_params,
        sep_resampler=args.model.sep_resampler,
        last_action=args.action_decoder.last_action,
        use_diff=(args.action_decoder.head_type == "diffusion"),
        n_timesteps=args.action_decoder.n_timesteps,
        diff_horizon=args.action_decoder.diff_horizon,
        predict_epsilon=args.action_decoder.predict_epsilon,
        sep_lm_head=args.experimental.sep_lm_head,
        unfreeze_vit=args.training.unfreeze_vit,
        multi_step_action=args.action_decoder.multi_step_action,
        llm_name=args.model.llm_name,
        pooling=args.action_decoder.pooling,
        residual=args.model.residual,
        tcp_rel=args.action_decoder.tcp_rel,
        decoder_type=args.action_decoder.decoder_type,
        hidden_size=args.model.hidden_size,
        freeze_sampler=args.training.freeze_sampler,
        fwd_pred=args.experimental.fwd_pred,
        fwd_pred_hand=args.experimental.fwd_pred_hand,
        no_image_patch=args.experimental.no_image_patch,
        global_latent=args.experimental.global_latent,
        clip_cache_dir=args.model.clip_cache_dir,
        nclass_gripper=args.nclass_gripper if hasattr(args, 'nclass_gripper') else 1,
    )
    
    # Load pretrained checkpoint
    if not args.debug and not args.model.no_pretrain:
        checkpoint_path = args.model.openflamingo_checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        if model.lang_encoder.transformer.wte.weight.shape == checkpoint['lang_encoder.transformer.wte.weight'].shape:
            model.load_state_dict(checkpoint, strict=False)
        else:
            # Handle vocabulary size mismatch
            random_wte = model.lang_encoder.transformer.wte.weight.data
            random_wte[:checkpoint['lang_encoder.transformer.wte.weight'].shape[0], :] = \
                checkpoint['lang_encoder.transformer.wte.weight']
            checkpoint['lang_encoder.transformer.wte.weight'] = random_wte
            model.load_state_dict(checkpoint, strict=False)
        
        if args.model.residual:
            model.lang_encoder.clone_parameters()
    
    print(f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Set precision
    if args.training.precision in ("bf16", "amp_bfloat16", "amp_bf16"):
        model = model.bfloat16()
    elif args.training.precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    # Move to device and wrap with DDP
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    ddp_model._set_static_graph()
    
    return ddp_model, image_processor, tokenizer


def setup_optimizer_and_scheduler(args, model, calvin_dataset):
    """Setup optimizer and learning rate scheduler.
    
    Args:
        args: Training arguments
        model: Model to optimize
        calvin_dataset: Dataset for calculating training steps
        
    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    def get_grouped_params(model):
        """Group parameters for different weight decay."""
        params_with_wd, params_without_wd = [], []
        
        def apply_decay(x):
            return (
                ("gated_cross_attn_layer" in x
                 and "ff_gate" not in x
                 and "attn_gate" not in x
                 and "norm" not in x
                 and "bias" not in x)
                or ("uvformer" in x)
                or ("Upsample2d_3d" in x)
                or ("alignment_layer" in x)
                or ("occ_decoder" in x)
            )
        
        for n, p in model.named_parameters():
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)
        
        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.training.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    
    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.training.learning_rate)
    
    total_training_steps = (len(calvin_dataset.dataset) // (args.training.batch_size_calvin * args.distributed.world_size)) * args.training.num_epochs
    
    if args.distributed.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    
    # Setup learning rate scheduler
    if args.training.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.training.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.training.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.training.warmup_steps,
            num_training_steps=total_training_steps,
            min_lr=0.001,
        )
    elif args.training.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.training.warmup_steps
        )
    
    return optimizer, lr_scheduler


def load_checkpoint(args, model, optimizer=None, lr_scheduler=None):
    """Load checkpoint for resuming or initialization.
    
    Args:
        args: Training arguments
        model: Model to load checkpoint into
        optimizer: Optional optimizer to load state
        lr_scheduler: Optional scheduler to load state
        
    Returns:
        Resume epoch number
    """
    resume_from_epoch = 0
    
    # Load from specified checkpoint (for initialization)
    if args.model.load_from_checkpoint is not None:
        if args.distributed.rank == 0:
            print(f"Loading checkpoint from {args.model.load_from_checkpoint}")
        checkpoint = torch.load(args.model.load_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Resume from checkpoint (for continuing training)
    if args.model.resume_from_checkpoint is not None:
        if args.distributed.rank == 0:
            print(f"Resuming from checkpoint {args.model.resume_from_checkpoint}")
        checkpoint = torch.load(args.model.resume_from_checkpoint, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        if not getattr(args, 'from_scratch', False):
            resume_from_epoch = checkpoint["epoch"] + 1
        
        if not args.data.real_data and optimizer is not None and lr_scheduler is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            except Exception as e:
                if args.distributed.rank == 0:
                    print(f"Warning: Could not load optimizer/scheduler state: {e}")
    
    if args.data.real_data:
        resume_from_epoch = 0
    
    return resume_from_epoch


def find_resume_checkpoint(args) -> Optional[str]:
    """Find checkpoint to resume from if not specified.
    
    Args:
        args: Training arguments
        
    Returns:
        Checkpoint path or None
    """
    if args.model.resume_from_checkpoint is not None:
        return args.model.resume_from_checkpoint
    
    if os.path.exists(args.run_name):
        ckpt_name = get_ckpt_name_pattern(args)
        checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
        checkpoint_list = [
            _ for _ in checkpoint_list
            if "__sep" not in _ and 'iter' not in _ and 'weights' not in _
        ]
        
        if len(checkpoint_list) == 0:
            if args.distributed.rank == 0:
                print(f"Found no checkpoints for run {args.run_name}.")
            return None
        else:
            checkpoint_path = sorted(
                checkpoint_list,
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            if args.distributed.rank == 0:
                print(f"Found checkpoint {checkpoint_path} for run {args.run_name}.")
            return checkpoint_path
    
    return None


def _save_epoch_checkpoint(args, model, optimizer, lr_scheduler, epoch):
    """Save checkpoint after an epoch.
    
    Args:
        args: Training arguments
        model: Model to save
        optimizer: Optimizer state
        lr_scheduler: Learning rate scheduler state
        epoch: Current epoch number
    """
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": get_checkpoint(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    ckpt_name = get_ckpt_name(args, epoch)
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    print(f"Saving checkpoint to {ckpt_path}")
    torch.save(checkpoint_dict, ckpt_path)
    
    if args.logging.delete_previous_checkpoint and epoch > 0:
        os.remove(ckpt_path)


def create_evaluate_func(args, image_processor, tokenizer):
    """Create evaluation function.
    
    Args:
        args: Training arguments
        image_processor: Image processor
        tokenizer: Tokenizer
        
    Returns:
        Evaluation function
    """
    def evaluate_func(model):
        """
        Evaluation entry used during training.

        If `args.evaluation.eval_via_policy_server` is True:
        - training starts one websocket policy server per rank
        - external env/evaluator runs elsewhere and calls:
            GET http://<rank0_host>:<rank0_port>/eval_done
          when evaluation finishes (rank0 will unblock)
        - then training stops servers and resumes
        """
        eval_log_dir = args.save_dir
        model.eval()

        try:
            from robouniview.eval.policy_server import WebsocketPolicyServer

            # Serve the underlying module to avoid unnecessary DDP overhead.
            policy = model.module if hasattr(model, "module") else model

            # Determine device from parameters and attach `.device` for policy_server compatibility.
            try:
                device = next(policy.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            setattr(policy, "device", device)

            base_port = int(getattr(args.evaluation, "eval_server_base_port", 8020))
            port = base_port + int(getattr(args.distributed, "rank", 0))
            host = getattr(args.evaluation, "eval_server_host", "127.0.0.1")

            # If host is 0.0.0.0 (bind all), advertise a concrete IP for clients.
            if host == "0.0.0.0":
                host = socket.gethostbyname(socket.gethostname())

            server = WebsocketPolicyServer(
                policy=policy,
                host="0.0.0.0",  # bind
                port=port,
                metadata={},
                image_processor=image_processor,
                tokenizer=tokenizer,
                sample_mode=args.sample_mode,
            )
            print(f"[rank {args.distributed.rank}] starting policy server on 0.0.0.0:{port} (advertise {host}:{port})")
            server.serve_forever()
            avg_seq_len = server.result
        except Exception as e:
            print(f"Error in evaluate_func: {e}")
            avg_seq_len = 0.0
        finally:
            model.train()
        return avg_seq_len
    
    return evaluate_func


def get_train_function(args):
    """Get appropriate training function based on configuration.
    
    Args:
        args: Training arguments
        
    Returns:
        Training function
    """
    # Try to import other training functions if they exist
    try:
        from train_utils import (
            train_one_epoch_calvin_diff,
            train_one_epoch_calvin_two_way,
            train_one_epoch_move,
            train_one_epoch_keyframe,
        )
        
        if args.action_decoder.head_type == "diffusion":
            return train_one_epoch_calvin_diff
    except ImportError:
        # Fall back to default if other functions don't exist
        pass
    
    return train_one_epoch_calvin


# ============================================================================
# Main Training Function
# ============================================================================

@record
def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train RoboUniView model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--report_to_wandb",
        type=bool,
        default=False,
        help="Whether to report to wandb",
    )
    cli_args = parser.parse_args()
    
    # Load configuration
    args = load_global_config_yaml_only(cli_args.config)
    
    # Override save_dir if specified
    if cli_args.save_dir is not None:
        args.save_dir = cli_args.save_dir
    args.run_name = args.save_dir
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    attributes_and_values = vars(args)
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(attributes_and_values['config'], f, default_flow_style=None, sort_keys=False)
    
    # Setup environment
    setup_environment(args)
    
    # Initialize distributed training
    args.distributed.local_rank, args.distributed.rank, args.distributed.world_size = world_info_from_env()
    device_id = init_distributed_device(args.distributed)
    print(f"device_id: {device_id}")
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}")
    
    # Set random seed
    random_seed(args.seed, args.distributed.rank)
    print(f"Start running training on rank {args.distributed.rank}.")
    
    # Initialize wandb
    init_wandb(args, resuming=args.model.resume_from_checkpoint is not None, log_code=True, enabled=cli_args.report_to_wandb)
    
    # Setup model
    ddp_model, image_processor, tokenizer = setup_model(args, device_id)
    
    # Load dataset
    dataset_type = 'keyframe' if 'keyframe' in args.model.llm_name else ''
    calvin_dataset = get_data(args, image_processor, tokenizer, dataset_type)
    
    # Setup diffusion normalizer if needed
    if args.action_decoder.head_type == "diffusion" and not args.debug:
        normalizer = ddp_model.module.diffusion_model.normalizer
        all_actions = np.vstack([
            calvin_dataset.dataset.__getitem__((i, 1), True)["actions"]
            for i in range(0, 10000)
        ])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')
    
    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(args, ddp_model, calvin_dataset)
    
    # Find and load checkpoint
    resume_checkpoint = find_resume_checkpoint(args)
    if resume_checkpoint:
        args.model.resume_from_checkpoint = resume_checkpoint
    
    resume_from_epoch = load_checkpoint(args, ddp_model, optimizer, lr_scheduler)
    
    # Create evaluation function
    evaluate_func = create_evaluate_func(args, image_processor, tokenizer)
    
    # Run evaluation first if requested
    if hasattr(args.evaluation, 'eval_first') and args.evaluation.eval_first:
        evaluate_func(ddp_model)
    
    # Training loop
    train_fn = get_train_function(args)

    # Track best evaluation metric (avg_seq_len) and checkpoint
    best_avg_seq_len = float("-inf")
    best_ckpt_path = os.path.join(args.save_dir, "best_avg_seq_len.pt")
    
    for epoch in range(resume_from_epoch, args.training.num_epochs):
        ddp_model.train()
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader
        
        # Train one epoch
        train_fn(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            evaluate_func=evaluate_func,
        )

        # Save checkpoint per-epoch only when no eval metric is used
        if args.distributed.rank == 0 and not args.evaluation.eval:
            _save_epoch_checkpoint(args, ddp_model, optimizer, lr_scheduler, epoch)

        # Run evaluation
        if args.evaluation.eval and epoch % args.evaluation.eval_epoch_interval == 0:
            avg_seq_len = evaluate_func(ddp_model)
            if args.distributed.rank == 0:
                print(f"Epoch {epoch} evaluation average sequence length: {avg_seq_len}")
                wandb.log({"avg_seq_len": avg_seq_len}, step=epoch)
                # Save best model by avg_seq_len
                if avg_seq_len > best_avg_seq_len:
                    best_avg_seq_len = avg_seq_len
                    torch.save(get_checkpoint(ddp_model), best_ckpt_path)
                    print(f"Saved best model to {best_ckpt_path} with avg_seq_len {avg_seq_len}, epoch {epoch}")
    # Save final checkpoint
    if args.distributed.rank == 0:
        ckpt_name = get_ckpt_name(args)
        final_path = os.path.join(args.save_dir, ckpt_name)
        torch.save(get_checkpoint(ddp_model), final_path)
        print(f"Saved final checkpoint to {final_path}")
    
    if args.distributed.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
