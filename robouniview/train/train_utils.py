"""
Training utilities for robot manipulation models.

This module provides training functions, checkpoint management, and utility
classes for training various robot manipulation models.
"""

# Standard library imports
import os
import time
from collections import defaultdict
from contextlib import suppress
from typing import Optional
import swanlab as wandb
# Third-party imports
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

# Local application imports
from robouniview.models.loss_func import Balanced_BCE_loss, MSE_Loss
from robouniview.utils import world_to_tcp_frame


# ============================================================================
# Utility Functions
# ============================================================================

def get_cast_dtype(precision: str) -> Optional[torch.dtype]:
    """Get the cast dtype based on precision string.
    
    Args:
        precision: Precision string ("bf16", "amp_bf16", "fp16", etc.)
        
    Returns:
        torch.dtype or None
    """
    if precision in ("bf16", "amp_bf16"):
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    return None


def get_autocast(precision: str):
    """Get autocast context manager based on precision.
    
    Args:
        precision: Precision string ("amp", "amp_bfloat16", "amp_bf16", etc.)
        
    Returns:
        Autocast context manager or suppress context manager
    """
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision in ("amp_bfloat16", "amp_bf16"):
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def _build_ckpt_name_base(args) -> str:
    """Build the base checkpoint name from args.
    
    Args:
        args: Training arguments object
        
    Returns:
        Base checkpoint name string
    """
    if args.model.use_gripper:
        ckpt_name = f'checkpoint_gripper_{args.model.fusion_mode}_hist_{args.action_decoder.hist_window}_{"" if not args.model.sep_resampler else "sep_"}'
    else:
        ckpt_name = f'checkpoint_no_gripper_hist_{args.action_decoder.hist_window}_{"" if not args.model.sep_resampler else "sep_"}'
    
    # Build suffix components
    if args.data.real_data:
        ckpt_name += 'real_'
    if args.training.train_params != -1:
        ckpt_name += f'train_{args.training.train_params}_'
    if args.model.no_pretrain:
        ckpt_name += 'no_pretrain_'
    if args.experimental.fwd_pred:
        ckpt_name += 'pred_rgb_'
    if args.experimental.fwd_pred_hand:
        ckpt_name += 'pred_hand_'
    if args.training.freeze_sampler:
        ckpt_name += 'freeze_sam_'
    if args.model.use_state:
        ckpt_name += 'state_'
    if args.cameras.rgb_pad != -1 or args.cameras.gripper_pad != -1:
        ckpt_name += f'aug_{args.cameras.rgb_pad}_{args.cameras.gripper_pad}_'
    if args.action_decoder.use_hist:
        ckpt_name += 'fc_'
    if args.action_decoder.head_type == "diffusion":
        ckpt_name += 'diff_'
    if args.action_decoder.traj_cons:
        ckpt_name += 'traj_cons_'
    if args.experimental.sep_lm_head:
        ckpt_name += 'lm_head_'
    if args.action_decoder.dif_ws:
        ckpt_name += f'difws_{args.data.min_window_size}_{args.data.max_window_size}_'
    elif args.action_decoder.window_size != 8:
        ckpt_name += f'ws_{args.action_decoder.window_size}_'
    if args.training.unfreeze_vit:
        ckpt_name += 'unfreeze_vit_'
    if args.model.llm_name != 'llama':
        ckpt_name += f'{args.model.llm_name}_'
    if args.action_decoder.pooling != 'max':
        ckpt_name += f'{args.action_decoder.pooling}_'
    if args.data.text_aug:
        ckpt_name += 'text_aug_'
    if args.model.residual:
        ckpt_name += 'res_'
    if args.training.freeze_embed:
        ckpt_name += 'freeze_emb_'
    if args.action_decoder.tcp_rel:
        ckpt_name += 'tcp_'
    if args.action_decoder.multi_step_action != 1:
        ckpt_name += f'{args.action_decoder.multi_step_action}_fur_step_'
    if args.action_decoder.decoder_type != 'lstm':
        ckpt_name += f'{args.action_decoder.decoder_type}_{args.model.hidden_size}_'
    if args.training.lr_scheduler != 'constant':
        ckpt_name += f'{args.training.lr_scheduler}_'
    
    return ckpt_name


def get_ckpt_name(args, epoch: int = -1) -> str:
    """Generate checkpoint filename from args and epoch.
    
    Args:
        args: Training arguments object
        epoch: Epoch number (-1 for final weights)
        
    Returns:
        Checkpoint filename string
    """
    ckpt_name = _build_ckpt_name_base(args)
    
    if epoch != -1:
        if epoch > 1000:
            ckpt_name += f'{epoch}_iter.pth'
        else:
            ckpt_name += f'{epoch}.pth'
    else:
        ckpt_name += 'final_weights.pth'
    
    return ckpt_name


def get_ckpt_name_pattern(args) -> str:
    """Generate checkpoint filename pattern for matching.
    
    Args:
        args: Training arguments object
        
    Returns:
        Checkpoint filename pattern string with wildcard
    """
    ckpt_name = _build_ckpt_name_base(args)
    ckpt_name += '*.pth'
    return ckpt_name



def get_checkpoint(model) -> dict:
    """Get model checkpoint state dict, excluding frozen parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        State dictionary with only trainable parameters (except normalizer)
    """
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict


# ============================================================================
# Helper Classes
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value.
    
    This class is useful for tracking metrics during training, such as
    loss values, step times, and data loading times.
    """

    def __init__(self):
        """Initialize the AverageMeter."""
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics with a new value.
        
        Args:
            val: New value to add
            n: Number of samples (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================================
# Training Helper Functions
# ============================================================================

def _prepare_token_ids(tokenizer, device_id: int) -> tuple:
    """Prepare token IDs for training.
    
    Args:
        tokenizer: Tokenizer instance
        device_id: Device ID for tensor placement
        
    Returns:
        Tuple of (media_token_id, endofchunk_token_id, action_token_id,
                 static_token_ids, gripper_token_ids, obs_token_ids)
    """
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    action_token_id = tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]
    
    static_token_ids = []
    gripper_token_ids = []
    obs_token_ids = []
    for i_th in range(8):
        static_token_ids.append(tokenizer(f"<static{i_th}>", add_special_tokens=False)["input_ids"][-1])
        gripper_token_ids.append(tokenizer(f"<gripper{i_th}>", add_special_tokens=False)["input_ids"][-1])
        obs_token_ids.append(tokenizer(f"<obs{i_th}>", add_special_tokens=False)["input_ids"][-1])
    
    static_token_ids = torch.tensor(static_token_ids).to(device_id, non_blocking=True)
    gripper_token_ids = torch.tensor(gripper_token_ids).to(device_id, non_blocking=True)
    obs_token_ids = torch.tensor(obs_token_ids).to(device_id, non_blocking=True)
    
    return media_token_id, endofchunk_token_id, action_token_id, static_token_ids, gripper_token_ids, obs_token_ids


def _prepare_batch_data(batch_calvin, args, device_id, cast_dtype, action_token_id, 
                        static_token_ids, gripper_token_ids, obs_token_ids):
    """Prepare batch data for model forward pass.
    
    Args:
        batch_calvin: Batch data from dataloader
        args: Training arguments
        device_id: Device ID
        cast_dtype: Cast dtype for tensors
        action_token_id: Action token ID
        static_token_ids: Static token IDs tensor
        gripper_token_ids: Gripper token IDs tensor
        obs_token_ids: Observation token IDs tensor
        
    Returns:
        Dictionary containing prepared batch data
    """
    images = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)
    gripper = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2)
    
    # Prepare input_ids and attention_mask
    if 'Temporal' not in args.model.fusion_mode:
        input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1).bool()
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)
        action_mask = static_mask = gripper_mask = obs_mask = None
    else:
        input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).bool()
        action_mask = torch.any(input_ids[..., None] == action_token_id, dim=2)
        static_mask = torch.any(input_ids[..., None] == static_token_ids, dim=2)
        gripper_mask = torch.any(input_ids[..., None] == gripper_token_ids, dim=2)
        obs_mask = torch.any(input_ids[..., None] == obs_token_ids, dim=2)
        if not static_mask.any(): static_mask = None
        if not gripper_mask.any(): gripper_mask = None
        if not obs_mask.any(): obs_mask = None
    
    # Prepare state and labels
    state_tensor = batch_calvin[4]
    robot_obs = batch_calvin[5]
    if args.experimental.clip_state:
        state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
    
    labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
    if args.action_decoder.tcp_rel:
        if args.action_decoder.multi_step_action == 1:
            labels = world_to_tcp_frame(labels, state_tensor)
        else:
            bs, seq_len = labels.shape[:2]
            labels = world_to_tcp_frame(labels, robot_obs)
            labels = labels.view(bs, seq_len, args.action_decoder.multi_step_action, -1)
    
    state_tensor = state_tensor.unsqueeze(2).unsqueeze(2).flatten(0, 1)
    
    # Prepare calibration and point cloud data
    calib = batch_calvin[6]
    pcd = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True)
    
    # Process labels
    if args.action_decoder.use_hist:
        labels = labels[:, [-1]]
    if args.model.fusion_mode == 'vit_concat':
        labels = labels[:, -1]
    labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]
    
    return {
        'images': images,
        'gripper': gripper,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'state_tensor': state_tensor,
        'labels': labels,
        'calib': calib,
        'pcd': pcd,
        'action_mask': action_mask,
        'static_mask': static_mask,
        'gripper_mask': gripper_mask,
        'obs_mask': obs_mask,
    }


def _compute_losses(output, labels, args, device_id, static_mask, gripper_mask, obs_mask,
                   static_pred, gripper_pred, obs_pred, pcd, images, gripper, aux_loss):
    """Compute all losses for training.
    
    Args:
        output: Model output
        labels: Ground truth labels
        args: Training arguments
        device_id: Device ID
        static_mask: Static mask
        gripper_mask: Gripper mask
        obs_mask: Observation mask
        static_pred: Static predictions
        gripper_pred: Gripper predictions
        obs_pred: Observation predictions
        pcd: Point cloud data
        images: Image data
        gripper: Gripper image data
        aux_loss: Auxiliary losses from model
        
    Returns:
        Dictionary containing all computed losses
    """
    loss_rgb = torch.tensor(0.0, device=device_id)
    loss_gripper_rgb = torch.tensor(0.0, device=device_id)
    
    # RGB reconstruction losses
    if static_mask is not None:
        assert len(images[0]) == args.action_decoder.window_size + 1
        raw_rgb = rearrange(images[:, 1:], "B T F G D H W -> (B T F G) D H W")
        raw_rgb = F.interpolate(raw_rgb, (112, 112), mode='bilinear')
        rgbmask = torch.ones_like(raw_rgb)
        loss_rgb = MSE_Loss(static_pred, raw_rgb, rgbmask) * 0.1
    
    if gripper_mask is not None:
        raw_gripper = rearrange(gripper[:, 1:], "B T F G D H W -> (B T F G) D H W")
        raw_gripper = F.interpolate(raw_gripper, (112, 112), mode='bilinear')
        grippermask = torch.ones_like(raw_gripper)
        loss_gripper_rgb = MSE_Loss(gripper_pred, raw_gripper, grippermask) * 0.1
    
    # Observation losses
    loss_obs = defaultdict(float)
    balanced_bce_loss = Balanced_BCE_loss(1, reduction="mean")
    
    if obs_mask is not None:
        pcd_rearranged = rearrange(pcd[:, :args.action_decoder.window_size], "B T H W Z C -> (B T) H W Z C")
        c_classes = pcd_rearranged.shape[-1]
        grid_cls = ['occ', 'r', 'g', 'b']
        
        for ind in range(c_classes):
            preds_ind = obs_pred[:, :, :, :, ind]
            trues_ind = pcd_rearranged[:, :, :, :, ind]
            if ind == 0:
                loss_ind = balanced_bce_loss(preds_ind, trues_ind)
            else:
                loss_ind = MSE_Loss(preds_ind, trues_ind, pcd_rearranged[:, :, :, :, 0])
            loss_obs[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * args.loss.occ_loss_weight[ind]
    
    if args.loss.occ_loss and 'loss_occ' in aux_loss:
        loss_occ = aux_loss['loss_occ']
        for ind, k in enumerate(loss_occ.keys()):
            loss_obs[f"grid_cls_{k}_loss"] += loss_occ[k] * args.loss.occ_loss_weight[ind]
    
    # Action losses
    loss_calvin_num = torch.tensor(0.0, device=device_id)
    loss_calvin_bin = torch.tensor(0.0, device=device_id)
    
    if args.action_decoder.multi_action_token:
        num_actions, bin_actions = output.logits[0], output.logits[1]
        if 'Temporal' in args.model.fusion_mode:
            loss_calvin_num = F.huber_loss(num_actions, labels[0][:, :args.action_decoder.window_size, :])
            loss_calvin_bin = F.binary_cross_entropy(bin_actions, labels[1][:, :args.action_decoder.window_size, :])
    
    if args.data.real_data:
        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
    else:
        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
    
    return {
        'loss_calvin': loss_calvin,
        'loss_calvin_num': loss_calvin_num,
        'loss_calvin_bin': loss_calvin_bin,
        'loss_rgb': loss_rgb,
        'loss_gripper_rgb': loss_gripper_rgb,
        'loss_obs': loss_obs,
    }


def _save_checkpoint(args, model, optimizer, lr_scheduler, epoch, global_step, checkpoint_name):
    """Save training checkpoint.
    
    Args:
        args: Training arguments
        model: Model to save
        optimizer: Optimizer state
        lr_scheduler: Learning rate scheduler state
        epoch: Current epoch
        global_step: Current global step
        checkpoint_name: Name for the checkpoint file
    """
    if args.distributed.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": get_checkpoint(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }
        
        ckpt_path = os.path.join(args.run_name, checkpoint_name)
        print(f"Saving checkpoint to {ckpt_path}")
        torch.save(checkpoint_dict, ckpt_path)
        
        if args.logging.delete_previous_checkpoint and epoch > 0:
            os.remove(ckpt_path)


# ============================================================================
# Main Training Functions
# ============================================================================

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    evaluate_func=None
):
    """Train one epoch on Calvin dataset.
    
    Args:
        args: Training arguments
        model: Model to train
        epoch: Current epoch number
        calvin_loader: Data loader for Calvin dataset
        tokenizer: Tokenizer instance
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        device_id: Device ID
        evaluate_func: Optional evaluation function
    """
    num_batches_per_epoch = calvin_loader.num_batches
    total_training_steps = num_batches_per_epoch * args.training.num_epochs
    
    autocast = get_autocast(args.training.precision)
    cast_dtype = get_cast_dtype(args.training.precision)
    
    # Prepare token IDs
    media_token_id, endofchunk_token_id, action_token_id, static_token_ids, gripper_token_ids, obs_token_ids = \
        _prepare_token_ids(tokenizer, device_id)
    
    model.train()
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    # Setup progress bar
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.distributed.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.training.num_epochs}")
    
    mv_avg_loss = []
    calvin_avg_loss = []
    occ_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # Prepare batch data
        batch_data = _prepare_batch_data(
            batch_calvin, args, device_id, cast_dtype,
            action_token_id, static_token_ids, gripper_token_ids, obs_token_ids
        )
        
        # Forward pass
        with autocast():
            output, static_pred, gripper_pred, obs_pred, aux_loss = model(
                vision_x=batch_data['images'][:, :args.action_decoder.window_size],
                lang_x=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                vision_gripper=batch_data['gripper'][:, :args.action_decoder.window_size],
                state_tensor=batch_data['state_tensor'] if (args.model.use_state or args.experimental.sep_lm_head) else None,
                calib=batch_data['calib'],
                pcd=batch_data['pcd'][:, :args.action_decoder.window_size],
                action_mask=batch_data['action_mask'],
                static_mask=batch_data['static_mask'],
                gripper_mask=batch_data['gripper_mask'],
                obs_mask=batch_data['obs_mask']
            )
        
        # Compute losses
        losses = _compute_losses(
            output, batch_data['labels'], args, device_id,
            batch_data['static_mask'], batch_data['gripper_mask'], batch_data['obs_mask'],
            static_pred, gripper_pred, obs_pred, batch_data['pcd'],
            batch_data['images'], batch_data['gripper'], aux_loss
        )
        
        # Combine losses
        divided_loss_calvin = losses['loss_calvin'] / args.training.gradient_accumulation_steps
        loss = divided_loss_calvin * args.loss.loss_multiplier_calvin * args.loss.loss_weight['action']
        
        for k in losses['loss_obs'].keys():
            loss += losses['loss_obs'][k] * args.loss.loss_weight['occ']
        
        loss = loss + losses['loss_rgb'] + losses['loss_gripper_rgb']
        
        # Track losses
        mv_avg_loss.append(loss.item())
        calvin_avg_loss.append(losses['loss_calvin'].item())
        if 'grid_cls_occ_loss' in losses['loss_obs']:
            occ_avg_loss.append(losses['loss_obs']['grid_cls_occ_loss'].item())
        
        # Backward pass
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        if (((num_steps + 1) % args.training.gradient_accumulation_steps) == 0) or (num_steps == num_batches_per_epoch - 1):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            step_time_m.update(time.time() - end)
            end = time.time()
        # Update progress bar
        loss_dic = {
            "avg_action_loss": sum(calvin_avg_loss[-min(100, len(calvin_avg_loss)):]) / min(100, len(calvin_avg_loss)),
            "action_loss": losses['loss_calvin'].item(),
            "Lnum": losses['loss_calvin_num'].item(),
            "Lbin": losses['loss_calvin_bin'].item(),
            "avg_occ_loss": sum(occ_avg_loss[-min(100, len(occ_avg_loss)):]) / (min(100, len(occ_avg_loss)) + 0.00001),
            "lr": optimizer.param_groups[0]["lr"],
            "loss_rgb": losses['loss_rgb'].item(),
            "loss_gripper_rgb": losses['loss_gripper_rgb'].item(),
        }
        for k in losses['loss_obs'].keys():
            loss_dic[k] = losses['loss_obs'][k].item()
        t.set_postfix(loss_dic)

        # Console logging
        if ((num_steps + 1) % args.logging.logging_steps == 0) and args.distributed.rank == 0:
            if args.action_decoder.multi_action_token:
                num_actions = output.logits[0]
                final_pos_l2 = ((num_actions[..., :3] - batch_data['labels'][0][:, :args.action_decoder.window_size, :3]) ** 2).sum(-1)[..., -1].sqrt().detach()
                pos_l2_final = final_pos_l2.mean()
                pos_l2_final05 = (final_pos_l2 < 0.05).to(torch.float).mean()
                l1 = (num_actions[..., 3:] - batch_data['labels'][0][:, :args.action_decoder.window_size, 3:]).abs().sum(-1)[..., -1].detach()
                rot_l1 = l1.to(torch.float).mean()
                rot_l105 = (l1 < 0.05).to(torch.float).mean()
                rot_l1025 = (l1 < 0.025).to(torch.float).mean()
                bin_actions = output.logits[1]
                gripper_acc = ((bin_actions > 0.5).to(torch.int) == (batch_data['labels'][1][:, :args.action_decoder.window_size, :] > 0.5).to(torch.int)).to(torch.float).mean().detach()
                print(f"pos_l2_final:{pos_l2_final:.3f}, pos_l2_final05:{pos_l2_final05:.3f}, "
                      f"rot_l1:{rot_l1:.3f}, rot_l105:{rot_l105:.3f}, rot_l1025:{rot_l1025:.3f}, "
                      f"gripper_acc:{gripper_acc:.3f}")
            
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.training.num_epochs} complete. "
                  f"Loss: (all){losses['loss_calvin'].item():.3f} (mse){losses['loss_calvin_num'].item():.3f} "
                  f"(bce){losses['loss_calvin_bin'].item():.3f}")
        
            wandb.log(loss_dic, step=global_step)

        # if num_steps > 10:
        #     return True
