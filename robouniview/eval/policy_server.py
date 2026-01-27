""" Main training script """
print('inininininii')
import argparse
import glob
import os
import sys
import socket
import urllib
from collections import OrderedDict
import copy
import random
import logging
from PIL import Image
from torch.distributed.elastic.multiprocessing.errors import record

import numpy as np
import torch
import wandb
import torch.distributed as dist
from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter, defaultdict, namedtuple
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP


# from robouniview.data.multidata import get_data, get_env
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from robouniview.models.factory import create_model_and_transforms, mpt_dict
logger = logging.getLogger(__name__)
#from lff_lightning.machine_learning.global_config import GlobalConfig
import yaml

import asyncio
import http
import logging
import time
import traceback
import functools
from robouniview.data.multi_cam_data import preprocess_image, preprocess_text_calvin
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

# os.environ["MASTER_PORT"] = str(8080)
# os.environ["MASTER_ADDR"] = "0.0.0.0"

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        host: str = "0.0.0.0",
        port: int = 8020,
        image_processor = None,
        tokenizer = None,
        metadata: dict = {},
        sample_mode: int = 1,
        max_instances: int = None,
        model_memory_gb: float = None,  # None = auto-detect
        total_memory_gb: float = None,
        enable_batching: bool = True,
        max_batch_size: int = 4,
        batch_timeout_ms: float = 100.0,
    ) -> None:
        # Store original policy for cloning
        self._original_policy = policy
        self.tokenizer = tokenizer
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer, sample_mode=sample_mode) # 注意此处是输出图片+OCC+action
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self._host = host
        self._port = port
        self._metadata = metadata
        self.result = 0.0
        
        # Model pool configuration
        if total_memory_gb is None and torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Auto-detect model memory usage if not provided
        if model_memory_gb is None or model_memory_gb <= 0:
            model_memory_gb = self._measure_model_memory(policy)
            logger.info(f"Auto-detected model memory usage: {model_memory_gb:.2f} GB")
        
        if max_instances is None:
            # Calculate max instances based on available memory (leave 10% buffer)
            max_instances = max(1, int((total_memory_gb * 0.9) / model_memory_gb)) if total_memory_gb else 1
        self._max_instances = max_instances
        self._model_memory_gb = model_memory_gb
        
        # Batching configuration
        self._enable_batching = enable_batching
        self._max_batch_size = max_batch_size
        self._batch_timeout_ms = batch_timeout_ms
        
        # Model and inference control
        self._policy = policy
        self._inference_lock = None  # For non-batched mode or batch processing
        
        # Dynamic batching: queue for pending requests
        if self._enable_batching:
            self._request_queue = None  # Will be created in async context
            self._batch_processor_task = None
            logger.info(f"Initialized policy server with DYNAMIC BATCHING: "
                       f"max_batch_size={self._max_batch_size}, "
                       f"batch_timeout={self._batch_timeout_ms}ms, "
                       f"model_memory_gb={self._model_memory_gb:.2f} GB")
        else:
            self._request_queue = None
            logger.info(f"Initialized policy server with SERIAL inference (batching disabled), "
                       f"model_memory_gb={self._model_memory_gb:.2f} GB")

        self.action_token_id = self.tokenizer("<action>", add_special_tokens=False)["input_ids"][-1]

        self.static0 = self.tokenizer("<static0>", add_special_tokens=False)["input_ids"][-1]
        self.static1 = self.tokenizer("<static1>", add_special_tokens=False)["input_ids"][-1]
        self.static2 = self.tokenizer("<static2>", add_special_tokens=False)["input_ids"][-1]
        self.static3 = self.tokenizer("<static3>", add_special_tokens=False)["input_ids"][-1]
        self.static4 = self.tokenizer("<static4>", add_special_tokens=False)["input_ids"][-1]
        self.static5 = self.tokenizer("<static5>", add_special_tokens=False)["input_ids"][-1]
        self.static6 = self.tokenizer("<static6>", add_special_tokens=False)["input_ids"][-1]
        self.static7 = self.tokenizer("<static7>", add_special_tokens=False)["input_ids"][-1]
        
        self.gripper0 = self.tokenizer("<gripper0>", add_special_tokens=False)["input_ids"][-1]
        self.gripper1 = self.tokenizer("<gripper1>", add_special_tokens=False)["input_ids"][-1]
        self.gripper2 = self.tokenizer("<gripper2>", add_special_tokens=False)["input_ids"][-1]
        self.gripper3 = self.tokenizer("<gripper3>", add_special_tokens=False)["input_ids"][-1]
        self.gripper4 = self.tokenizer("<gripper4>", add_special_tokens=False)["input_ids"][-1]
        self.gripper5 = self.tokenizer("<gripper5>", add_special_tokens=False)["input_ids"][-1]
        self.gripper6 = self.tokenizer("<gripper6>", add_special_tokens=False)["input_ids"][-1]
        self.gripper7 = self.tokenizer("<gripper7>", add_special_tokens=False)["input_ids"][-1]
        
        self.obs0 = self.tokenizer("<obs0>", add_special_tokens=False)["input_ids"][-1]
        self.obs1 = self.tokenizer("<obs1>", add_special_tokens=False)["input_ids"][-1]
        self.obs2 = self.tokenizer("<obs2>", add_special_tokens=False)["input_ids"][-1]
        self.obs3 = self.tokenizer("<obs3>", add_special_tokens=False)["input_ids"][-1]
        self.obs4 = self.tokenizer("<obs4>", add_special_tokens=False)["input_ids"][-1]
        self.obs5 = self.tokenizer("<obs5>", add_special_tokens=False)["input_ids"][-1]
        self.obs6 = self.tokenizer("<obs6>", add_special_tokens=False)["input_ids"][-1]
        self.obs7 = self.tokenizer("<obs7>", add_special_tokens=False)["input_ids"][-1]

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def _measure_model_memory(self, model: torch.nn.Module) -> float:
        """Measure the actual GPU memory usage of a model.
        
        Args:
            model: The model to measure
            
        Returns:
            Memory usage in GB
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot measure model memory. Using default 20.0 GB")
            return 20.0
        
        device = next(model.parameters()).device
        if device.type != 'cuda':
            logger.warning(f"Model is on {device}, cannot measure GPU memory. Using default 20.0 GB")
            return 20.0
        
        # Clear cache to get accurate measurement
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Measure baseline memory (before model)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        
        # The model is already loaded, so we measure the difference
        # We can estimate by checking parameter memory + some overhead
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Add overhead (typically 20-30% for PyTorch overhead)
        estimated_memory = (param_memory + buffer_memory) * 1.25  # 25% overhead
        
        # Alternative: measure actual allocated memory if model is the only thing on GPU
        # This is more accurate but requires the model to be isolated
        current_allocated = torch.cuda.memory_allocated(device)
        current_reserved = torch.cuda.memory_reserved(device)
        
        # Use the larger of estimated or actual (if model seems to be the main consumer)
        if current_allocated > estimated_memory:
            model_memory_bytes = current_allocated - baseline_allocated if baseline_allocated < current_allocated else current_allocated
        else:
            model_memory_bytes = estimated_memory
        
        model_memory_gb = model_memory_bytes / (1024**3)
        
        logger.info(f"Model memory measurement: {model_memory_gb:.2f} GB "
                   f"(params: {param_memory/(1024**3):.2f} GB, "
                   f"buffers: {buffer_memory/(1024**3):.2f} GB, "
                   f"allocated: {current_allocated/(1024**3):.2f} GB)")
        
        return max(model_memory_gb, 1.0)  # At least 1 GB

    async def _acquire_inference_lock(self):
        """Acquire lock for inference (ensures serial execution for single model)."""
        if self._inference_lock is None:
            self._inference_lock = asyncio.Lock()
        await self._inference_lock.acquire()
    
    async def _release_inference_lock(self):
        """Release inference lock."""
        if self._inference_lock is not None:
            self._inference_lock.release()

    async def _batch_processor(self):
        """Background task that collects requests and processes them in batches."""
        while True:
            try:
                batch_requests = []
                batch_futures = []
                
                # Wait for first request (blocking)
                first_item = await self._request_queue.get()
                batch_requests.append(first_item[0])
                batch_futures.append(first_item[1])
                
                # Collect more requests (non-blocking, with timeout)
                deadline = time.monotonic() + (self._batch_timeout_ms / 1000.0)
                while len(batch_requests) < self._max_batch_size:
                    timeout = max(0, deadline - time.monotonic())
                    if timeout <= 0:
                        break
                    
                    try:
                        item = await asyncio.wait_for(self._request_queue.get(), timeout=timeout)
                        batch_requests.append(item[0])
                        batch_futures.append(item[1])
                    except asyncio.TimeoutError:
                        break
                
                # print(f"batch size: {len(batch_requests)}")
                # Process batch
                try:
                    results = await self._process_batch(batch_requests)
                    # Distribute results
                    for future, result in zip(batch_futures, results):
                        future.set_result(result)
                except Exception as e:
                    # Propagate error to all requests in batch
                    for future in batch_futures:
                        future.set_exception(e)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)

    async def _process_batch(self, batch_requests):
        """Process a batch of requests together.
        
        Args:
            batch_requests: List of preprocessed request dicts
            
        Returns:
            List of results (one per request)
        """
        batch_size = len(batch_requests)
        device = self._policy.device

        # Stack vision/state
        images_batch = torch.cat([req['image_x'] for req in batch_requests], dim=0)  # (B, ...)
        grippers_batch = torch.cat([req['gripper'] for req in batch_requests], dim=0)
        states_batch = torch.cat([req['state'] for req in batch_requests], dim=0)

        # Text processing after batching for consistent padding
        prompts = [req['prompt'] for req in batch_requests]
        text_x, mask = self.text_process_fn(prompts)
        text_x, mask = text_x.to(device), mask.to(device)

        # Vectorized mask construction (aligns with training path)
        mask = mask.bool()
        action_masks_batch = (text_x == self.action_token_id)

        static_token_ids = torch.tensor(
            [self.static0, self.static1, self.static2, self.static3,
             self.static4, self.static5, self.static6, self.static7],
            device=device,
            dtype=text_x.dtype,
        )
        gripper_token_ids = torch.tensor(
            [self.gripper0, self.gripper1, self.gripper2, self.gripper3,
             self.gripper4, self.gripper5, self.gripper6, self.gripper7],
            device=device,
            dtype=text_x.dtype,
        )
        obs_token_ids = torch.tensor(
            [self.obs0, self.obs1, self.obs2, self.obs3,
             self.obs4, self.obs5, self.obs6, self.obs7],
            device=device,
            dtype=text_x.dtype,
        )

        static_masks_batch = torch.any(text_x[..., None] == static_token_ids, dim=2)
        gripper_masks_batch = torch.any(text_x[..., None] == gripper_token_ids, dim=2)
        obs_masks_batch = torch.any(text_x[..., None] == obs_token_ids, dim=2)

        # Drop empty optional masks
        if not static_masks_batch.any():
            static_masks_batch = None
        if not gripper_masks_batch.any():
            gripper_masks_batch = None
        if not obs_masks_batch.any():
            obs_masks_batch = None

        texts_batch = text_x
        masks_batch = mask
        
        # Combine calib (list of lists)
        calib_batch = [torch.cat([batch_requests[bith]['calib'][i] for bith in range(len(batch_requests))], dim=0) for i in range(len(batch_requests[0]['calib']))]
        # Inference
        with torch.no_grad():
            action, static_pred, gripper_pred, obs_pred, _ = self._policy(
                vision_x=images_batch,
                lang_x=texts_batch,
                attention_mask=masks_batch,
                vision_gripper=grippers_batch,
                state_tensor=states_batch,
                calib=calib_batch,
                action_mask=action_masks_batch,
                static_mask=static_masks_batch,
                gripper_mask=gripper_masks_batch,
                obs_mask=obs_masks_batch,
                return_feature=False,
            )
        
        # Split results
        results = []
        for i in range(batch_size):
            result = {
                "actions": torch.cat([t[i:i+1] for t in action.logits], dim=-1).cpu().detach().numpy(),
            }
            results.append(result)
        
        return results

    def serve_forever(self) -> None:
        """Run the server until it is closed. Handles CancelledError gracefully."""
        try:
            asyncio.run(self.run())
        except asyncio.CancelledError:
            # Server was closed, which is expected behavior - ignore the cancellation
            print("Server closed (CancelledError is expected when shutting down)")
        except KeyboardInterrupt:
            # User interrupted - also expected
            print("Server interrupted by user")

    async def run(self):
        # Start batch processor if batching is enabled
        if self._enable_batching:
            if self._request_queue is None:
                self._request_queue = asyncio.Queue()
            self._batch_processor_task = asyncio.create_task(self._batch_processor())
        
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=self._process_request,
            ) as server:
                self._ws_server = server  # Store server reference for shutdown
                await server.serve_forever()
        finally:
            # Stop batch processor
            if self._batch_processor_task:
                self._batch_processor_task.cancel()
                try:
                    await self._batch_processor_task
                except asyncio.CancelledError:
                    pass

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())
                    
                    device = self._policy.device
                    
                    images = [Image.fromarray(image) for image in obs['image']]
                    gripper = [Image.fromarray(image) for image in obs['wrist_image']]
                    image_x = self.image_process_fn(images).to(device)
                    gripper = self.image_process_fn(gripper).to(device)
                    state = torch.from_numpy(obs['state']).to(device)
                    calib = [torch.from_numpy(tmp) for tmp in obs['calib']]
                    image_x = image_x.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                    gripper = gripper.unsqueeze(0).unsqueeze(2).unsqueeze(2)

                    request_payload = {
                        "prompt": obs['prompt'],
                        "image_x": image_x,
                        "gripper": gripper,
                        "state": state,
                        "calib": calib,
                    }

                    if self._enable_batching:
                        # Dynamic batching path: enqueue request and wait for batch result
                        if self._request_queue is None:
                            self._request_queue = asyncio.Queue()
                        loop = asyncio.get_running_loop()
                        future = loop.create_future()
                        await self._request_queue.put((request_payload, future))

                        infer_time = time.monotonic()
                        result = await future  # result produced by _batch_processor/_process_batch
                        infer_time = time.monotonic() - infer_time
                        action = dict(result)
                    else:
                        # Fallback to single-request inference
                        infer_time = time.monotonic()

                        # Text processing per request (no batching)
                        text_x, mask = self.text_process_fn([obs['prompt']])
                        text_x, mask = text_x.to(device), mask.to(device)
                        action_mask = mask.clone()
                        action_mask[text_x != self.action_token_id] = 0

                        static_mask = mask.clone()
                        gripper_mask = mask.clone()
                        obs_mask = mask.clone()
                        static_mask[(text_x != self.static0) & (text_x != self.static1)& (text_x != self.static2)& (text_x != self.static3) \
                            & (text_x != self.static4) & (text_x != self.static5)& (text_x != self.static6)& (text_x != self.static7) ] = 0
                        
                        gripper_mask[(text_x != self.gripper0) & (text_x != self.gripper1)& (text_x != self.gripper2)& (text_x != self.gripper3) \
                            & (text_x != self.gripper4) & (text_x != self.gripper5)& (text_x != self.gripper6)& (text_x != self.gripper7) ] = 0
                        
                        obs_mask[(text_x != self.obs0) & (text_x != self.obs1)& (text_x != self.obs2)& (text_x != self.obs3) \
                            & (text_x != self.obs4) & (text_x != self.obs5)& (text_x != self.obs6)& (text_x != self.obs7) ] = 0
                        static_mask=static_mask.bool()
                        gripper_mask=gripper_mask.bool()
                        obs_mask=obs_mask.bool()
                        mask=mask.bool()
                        action_mask=action_mask.bool()
                        if static_mask.sum() < 1 : static_mask = None
                        if gripper_mask.sum() < 1 : gripper_mask = None
                        if obs_mask.sum() < 1 :  obs_mask = None

                        await self._acquire_inference_lock()
                        try:
                            action_out, static_pred, gripper_pred, obs_pred, _ = self._policy(
                                vision_x=image_x,
                                lang_x=text_x,
                                attention_mask=mask,
                                vision_gripper=gripper,
                                state_tensor=state,
                                calib=calib,
                                action_mask=action_mask,
                                static_mask=static_mask,
                                gripper_mask=gripper_mask,
                                obs_mask=obs_mask,
                                # Feature return is expensive and not needed for action serving.
                                return_feature=False,
                            )
                        finally:
                            # Always release inference lock
                            await self._release_inference_lock()
                        
                        infer_time = time.monotonic() - infer_time
                        action = {"actions": torch.cat(action_out.logits, dim=-1).cpu().detach().numpy()}

                    # Attach basic timing info (includes queue wait when batching)
                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            pass  # Inference slot is always released in the try block


    def _process_request(self, connection: _server.ServerConnection, request: _server.Request) -> _server.Response:
        """HTTP request handler for health checks and eval_done signal."""
        print(f"request.path: {request.path}")

        # Normalize path + query (request.path may include query string)
        parsed = urllib.parse.urlparse(request.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")

        if path == "/eval_done":
            avg_seq_len = None
            # Prefer websockets Request query_params if available
            if hasattr(request, "query_params"):
                avg_seq_len = request.query_params.get("avg_seq_len")
            if avg_seq_len is None and "avg_seq_len" in query:
                avg_seq_len = query["avg_seq_len"][0]

            try:
                avg_seq_len = float(avg_seq_len)
            except Exception:
                avg_seq_len = None

            if avg_seq_len is not None:
                self.result = avg_seq_len
                print(f"eval_done - shutting down server with avg_seq_len {avg_seq_len}")
            else:
                print("eval_done - missing avg_seq_len, shutting down without metric")

            # Schedule shutdown after a short delay to ensure response is sent
            ws_server = getattr(self, "_ws_server", None)
            if ws_server is not None:
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule shutdown with a small delay to ensure HTTP response is sent first
                    async def _delayed_shutdown():
                        await asyncio.sleep(0.1)  # Small delay to ensure response is sent
                        ws_server.close()
                    loop.create_task(_delayed_shutdown())
                except RuntimeError:
                    # Fallback: if no loop is running, just close synchronously
                    ws_server.close()
            return connection.respond(http.HTTPStatus.OK, "DONE\n")

        # Continue with the normal websocket upgrade.
        return None


class GlobalConfig:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self._load_config()

    def _load_config(self):
        for key, val in self.config.items():
            setattr(self, key, val)


def load_global_config_yaml_only(config_path: str) -> GlobalConfig:
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)
    global_config = GlobalConfig(config)
    return global_config

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate_from_checkpoint",
        type=str,
        help="path to checkpoint to evaluate , this should contain model",
        default=None,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="port to serve the policy",
        default=8020,
    )

    _args = parser.parse_args()
    yaml_path = _args.evaluate_from_checkpoint.replace(os.path.basename(_args.evaluate_from_checkpoint),'config.yaml')
    args = load_global_config_yaml_only(yaml_path)
    args.evaluate_from_checkpoint = _args.evaluate_from_checkpoint

    if args.action_decoder.head_type == "diffusion":
        args.pad_length = args.action_decoder.n_obs_steps
    if args.action_decoder.eval_hist_size == -1:
        args.action_decoder.eval_hist_size = args.action_decoder.window_size
        if args.action_decoder.head_type == "diffusion":
            args.action_decoder.eval_hist_size = args.action_decoder.n_obs_steps
    if args.logging.save_checkpoints_to_wandb and not args.logging.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if 'sep' in args.evaluate_from_checkpoint:
        args.model.sep_resampler = True
    if 'lm_head' in args.evaluate_from_checkpoint:
        args.experimental.sep_lm_head = True
    if 'res_' in args.evaluate_from_checkpoint:
        args.model.residual = True
    if 'tcp' in args.evaluate_from_checkpoint:
        args.action_decoder.tcp_rel = True
    if 'fur' in args.evaluate_from_checkpoint.split('_'):
        name_attrs = args.evaluate_from_checkpoint.split('_')
        args.action_decoder.multi_step_action = int(name_attrs[name_attrs.index('fur')-1])
    if 'difws' in args.evaluate_from_checkpoint:
        args.action_decoder.dif_ws = True
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('difws')
        min_ws = int(name_attrs[ix+1])
        max_ws = int(name_attrs[ix+2])
        args.action_decoder.min_window_size = min_ws
        args.action_decoder.max_window_size = max_ws
        args.action_decoder.window_size = max_ws
    if 'latent' in args.evaluate_from_checkpoint:
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('latent')
        args.experimental.global_latent = int(name_attrs[ix+1])
    if 'no_image_patch' in args.evaluate_from_checkpoint:
        args.experimental.no_image_patch = True
    if 'gpt' in args.evaluate_from_checkpoint:
        args.action_decoder.decoder_type = 'gpt'
        name_attrs = args.evaluate_from_checkpoint.split('_')
        hidden_size = int(name_attrs[name_attrs.index('gpt')+1])
        args.model.hidden_size = hidden_size
    for name in ['mpt_3b', 'mpt_4b', 'mpt_9b', 'mpt_dolly_3b', 'mpt_base_4b']:
        if name in args.evaluate_from_checkpoint:
            args.model.llm_name = name
            break
    
    args.model.lm_path = mpt_dict[args.model.llm_name]["lang_encoder_path"]
    args.model.tokenizer_path = mpt_dict[args.model.llm_name]["tokenizer_path"]
    args.model.cross_attn_every_n_layers = mpt_dict[args.model.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.model.llm_name]["openflamingo_checkpoint"]
    
    if args.logging.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.distributed.local_rank, args.distributed.rank, args.distributed.world_size = world_info_from_env()

    device_id = init_distributed_device(args.distributed)
    print("device_id: ", device_id)
    print("world_size: ", torch.distributed.get_world_size())
    random_seed(args.seed, args.distributed.rank)

    model, image_processor, tokenizer = create_model_and_transforms(
        args,
        args.model.vision_encoder_path,
        args.model.vision_encoder_pretrained,
        args.model.lm_path,
        args.model.tokenizer_path if args.model.tokenizer_path else args.model.lm_path,
        cross_attn_every_n_layers=args.model.cross_attn_every_n_layers,
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
        fusion_mode=args.model.fusion_mode,
        use_gripper=args.model.use_gripper,
        use_state=args.model.use_state,
        use_hist=args.action_decoder.use_hist,
        pad_length=args.action_decoder.pad_length,
        debug=args.training.debug,
        multi_step_action=args.action_decoder.multi_step_action,
        llm_name=args.model.llm_name,
        sep_lm_head=args.experimental.sep_lm_head,
        return_feature=True,
        residual=args.model.residual,
        tcp_rel=args.action_decoder.tcp_rel,
        replan=args.action_decoder.replan,
        decoder_type=args.action_decoder.decoder_type,
        hidden_size=args.model.hidden_size,
        freeze_sampler=args.training.freeze_sampler,
        fwd_pred=args.experimental.fwd_pred,
        fwd_pred_hand=args.experimental.fwd_pred_hand,
        no_image_patch=args.experimental.no_image_patch,
        global_latent=args.experimental.global_latent,
        # refresh=args.refresh,
        clip_cache_dir=args.model.clip_cache_dir,
        nclass_gripper=args.action_decoder.nclass_gripper if hasattr(args.action_decoder, 'nclass_gripper') else 1,
    )
    if args.experimental.sep_lm_head:
        model.lm_head.requires_grad_(True)
    else:
        model.lang_encoder.lm_head.requires_grad_(True)

    if args.distributed.rank == 0 and args.logging.report_to_wandb:
        wandb.init(
            project=args.logging.wandb_project,
            entity=args.logging.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.distributed.rank % torch.cuda.device_count()
    if args.training.precision == "bf16" or args.training.precision == "amp_bfloat16" or args.training.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.training.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    model = model.to(device_id)
    model.eval()

    ddp_model = DDP(model, device_ids=[device_id])
    if args.model.residual:
        model.lang_encoder.clone_parameters()
    # if args.evaluate_from_checkpoint is specified, load checkpoint
    assert args.evaluate_from_checkpoint is not None, "Please specify a checkpoint to evaluate."
    if args.distributed.rank == 0:
        print(f"Loading robot-flamingo checkpoint from {args.evaluate_from_checkpoint}")
    checkpoint = torch.load(args.evaluate_from_checkpoint, map_location="cpu")
    def filter_ckpt(checkpoint, flags=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                load_p = True
                for flag in flags:
                    if flag in key:
                        load_p = True
                
                if load_p:
                    if 'bevformer' in key:
                        key = key.replace('bevformer','uvformer')

                    if 'bev2vision' in key:
                        key = key.replace('bev2vision','alignment_layer')
                    new_state_dict[key] = value
            return new_state_dict
    flags = []

    checkpoint = filter_ckpt(checkpoint,flags)

    ddp_model.load_state_dict(checkpoint, False)

    #ddp_model.load_state_dict(checkpoint["model_state_dict"], False)  # 只保存了求梯度的部分

    ddp_model.eval()
    eval_log_dir = None
    # if args.debug:
    eval_log_dir = '{}'.format(args.evaluate_from_checkpoint.split('.')[0])

    # eval_one_epoch_ddp(
    #     args=args,
    #     model=ddp_model,
    #     image_processor=image_processor,
    #     tokenizer=tokenizer,
    #     dataset_path=args.data.calvin_dataset,
    #     future_act_len=args.action_decoder.future_act_len,
    #     eval_log_dir=eval_log_dir,
    #     reset=args.evaluation.reset,
    #     diverse_inst=args.evaluation.diverse_inst,
    #     debug=True,
    #     isTraining=False,
    #     llm_name = args.model.llm_name,
    # )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(        policy=ddp_model,
        host="0.0.0.0",
        port=_args.port,
        metadata={},
        image_processor=image_processor,
        tokenizer=tokenizer,
        sample_mode=args.sample_mode,
    )
    server.serve_forever()
    print(f"Server result: {server.result}")
    return server.result


if __name__ == "__main__":
    main()

