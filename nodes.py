"""Main node definitions for WAN 2.2 BlockSwap functionality.

This module defines the main ComfyUI node class that users interact with.
It provides a clean interface to the block swapping functionality while
handling all the complex callback registration and parameter management.
"""

import torch
import comfy.model_management as mm
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher

from .config import BlockSwapConfig
from .callbacks import lazy_load_callback, cleanup_callback


class WanVideo22BlockSwap:
    """
    Block swapping for WAN 2.1/2.2 models with LAZY LOADING.

    Offloads transformer blocks to CPU DURING model loading to prevent
    VRAM spikes. Blocks are loaded directly to their target device
    instead of loading everything to GPU first.

    GGUF Compatible: Automatically detects and handles GGUF quantized models.

    This is the main ComfyUI node class that users will see in the UI.
    It provides all the configuration options and handles the callback
    registration for lazy loading and cleanup operations.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """Get input types for the node."""
        return BlockSwapConfig.get_input_types()

    RETURN_TYPES: tuple = ("MODEL",)
    RETURN_NAMES: tuple = ("model",)
    CATEGORY: str = "ComfyUI_Wan22Blockswap"
    FUNCTION: str = "apply_block_swap"
    DESCRIPTION: str = (
        "Apply LAZY LOADING block swapping to WAN 2.1/2.2 models. "
        "Blocks are offloaded DURING loading to prevent VRAM spikes. "
        "Apply block swapping to WAN 2.1/2.2 models to reduce VRAM usage. "
        "Swaps last N transformer blocks to CPU memory. "
        "Compatible with all WAN model variants (1.3B, 5B, 14B, LongCat). "
        "Supports optional VACE model block swapping for multi-modal tasks. "
        "GGUF compatible: Gracefully handles quantized models with best-effort swapping."
    )

    def apply_block_swap(
        self,
        model: ModelPatcher,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        use_non_blocking: bool = False,
        vace_blocks_to_swap: int = 0,
        prefetch_blocks: int = 0,
        block_swap_debug: bool = False,
    ) -> tuple:
        """
        Apply block swapping configuration to ComfyUI native WAN model.

        This function registers a callback that executes when the model is
        loaded, swapping the specified number of transformer blocks to the
        offload device (CPU) to reduce VRAM usage. The swap uses a clever
        strategy: keeps early blocks on GPU (where most computation happens)
        and moves later blocks to CPU (where activations can be staged).

        Pre-routed Detection:
        If the model was loaded via WAN22BlockSwapLoader, blocks are already
        routed to their target devices. This function detects that case and
        skips redundant operations, only registering cleanup callbacks.

        Args:
            model (ModelPatcher): The ComfyUI model patcher instance.
            blocks_to_swap (int): Number of transformer blocks to swap.
                Range depends on model variant:
                    - 1.3B/5B models: 0-30
                    - 14B model: 0-40
                    - LongCat: 0-48
            offload_txt_emb (bool): Whether to offload text embeddings to CPU.
            offload_img_emb (bool): Whether to offload image embeddings (I2V).
            use_non_blocking (bool): Use non-blocking transfers for speed.
            vace_blocks_to_swap (int): VACE blocks to swap (0=auto detection).
            prefetch_blocks (int): Blocks to prefetch ahead for pipeline.
            block_swap_debug (bool): Enable performance monitoring.

        Returns:
            tuple: Modified model patcher with block swap callback registered.
        """
        # Check if model was loaded with pre-routing from WAN22BlockSwapLoader
        model_options = getattr(model, "model_options", {}) or {}
        blockswap_info = model_options.get("wan22_blockswap_info", {})
        
        if blockswap_info.get("pre_routed", False):
            # Model already has blocks pre-routed during load
            pre_routed_blocks = blockswap_info.get("blocks_to_swap", 0)
            total_blocks = blockswap_info.get("total_blocks", 30)
            
            if pre_routed_blocks >= blocks_to_swap:
                # No additional swapping needed - blocks already on correct devices
                if block_swap_debug:
                    print(f"[BlockSwap] Model pre-routed with {pre_routed_blocks} swap blocks")
                    print(f"[BlockSwap] Requested {blocks_to_swap} blocks - skipping redundant swap")
                    print(f"[BlockSwap] Registering cleanup callback only")
                
                # Clone and register only cleanup callback
                model_copy = model.clone()
                model_copy.add_callback(CallbacksMP.ON_CLEANUP, cleanup_callback)
                
                # Store info that we detected pre-routing
                if not hasattr(model_copy, "model_options"):
                    model_copy.model_options = {}
                model_copy.model_options["wan22_blockswap_detected_preroute"] = True
                
                return (model_copy,)
            else:
                # Need to swap additional blocks beyond what was pre-routed
                additional_blocks = blocks_to_swap - pre_routed_blocks
                if block_swap_debug:
                    print(f"[BlockSwap] Model pre-routed with {pre_routed_blocks} swap blocks")
                    print(f"[BlockSwap] Requested {blocks_to_swap} - swapping {additional_blocks} additional blocks")
                
                # Adjust blocks_to_swap for the callback
                # The callback should only move blocks that aren't already on CPU
                blocks_to_swap = additional_blocks

        def lazy_load_callback_wrapper(
            model_patcher: ModelPatcher,
            device_to,
            lowvram_model_memory,
            force_patch_weights,
            full_load,
        ) -> None:
            """Wrapper for lazy load callback with proper parameter passing."""
            lazy_load_callback(
                model_patcher=model_patcher,
                device_to=device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                full_load=full_load,
                blocks_to_swap=blocks_to_swap,
                offload_txt_emb=offload_txt_emb,
                offload_img_emb=offload_img_emb,
                use_non_blocking=use_non_blocking,
                vace_blocks_to_swap=vace_blocks_to_swap,
                prefetch_blocks=prefetch_blocks,
                block_swap_debug=block_swap_debug,
            )

        # Clone model and register lazy loading callback
        model_copy = model.clone()

        # Register both ON_LOAD and ON_CLEANUP callbacks
        model_copy.add_callback(CallbacksMP.ON_LOAD, lazy_load_callback_wrapper)
        model_copy.add_callback(CallbacksMP.ON_CLEANUP, cleanup_callback)

        if block_swap_debug:
            print("[BlockSwap] Both ON_LOAD and ON_CLEANUP callbacks registered")

        return (model_copy,)


class WANBlockSwapWrapper:
    """
    Wrapper node that adds block swapping to ANY model's forward pass.
    
    This node wraps a model's forward() method to enable block swapping
    during inference, making it compatible with ANY KSampler including
    WanVideoLooper and other custom sampling nodes.
    
    Works by:
    1. Moving specified blocks to CPU after model loads
    2. Wrapping forward() to temporarily move blocks to GPU during inference
    3. Moving blocks back to CPU after each forward pass
    
    Compatible with ALL samplers - no custom sampler needed!
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 48,
                    "step": 1,
                    "display": "slider",
                }),
                "offload_txt_emb": ("BOOLEAN", {"default": False}),
                "offload_img_emb": ("BOOLEAN", {"default": False}),
                "use_non_blocking": ("BOOLEAN", {"default": True}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES: tuple = ("MODEL",)
    RETURN_NAMES: tuple = ("model",)
    CATEGORY: str = "ComfyUI_Wan22Blockswap"
    FUNCTION: str = "wrap_model"
    DESCRIPTION: str = (
        "Wraps model's forward pass with block swapping logic. "
        "Compatible with ANY KSampler including WanVideoLooper. "
        "Offloads transformer blocks to CPU and loads them on-demand during inference."
    )
    
    def wrap_model(
        self,
        model: ModelPatcher,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        use_non_blocking: bool,
        enable_debug: bool,
    ) -> tuple:
        """
        Wrap the model's forward pass with block swapping logic.
        
        Args:
            model: ComfyUI model patcher
            blocks_to_swap: Number of blocks from end to swap to CPU
            offload_txt_emb: Offload text embeddings to CPU
            offload_img_emb: Offload image embeddings to CPU
            use_non_blocking: Use non-blocking GPU transfers
            enable_debug: Print debug information
            
        Returns:
            Wrapped model with block swapping enabled
        """
        if blocks_to_swap == 0:
            if enable_debug:
                print("[BlockSwapWrapper] blocks_to_swap=0, returning model unchanged")
            return (model,)
        
        # Validate model structure
        if not hasattr(model, 'model') or not hasattr(model.model, 'diffusion_model'):
            raise ValueError("[BlockSwapWrapper] Not a valid diffusion model - missing model.diffusion_model")
        
        transformer = model.model.diffusion_model
        
        if not hasattr(transformer, 'blocks'):
            raise ValueError("[BlockSwapWrapper] Model doesn't have 'blocks' attribute - not a WAN model?")
        
        total_blocks = len(transformer.blocks)
        if blocks_to_swap > total_blocks:
            if enable_debug:
                print(f"[BlockSwapWrapper] WARNING: blocks_to_swap ({blocks_to_swap}) > total_blocks ({total_blocks})")
                print(f"[BlockSwapWrapper] Clamping to {total_blocks}")
            blocks_to_swap = total_blocks
        
        swap_start_idx = total_blocks - blocks_to_swap
        
        if enable_debug:
            print(f"[BlockSwapWrapper] Model has {total_blocks} blocks")
            print(f"[BlockSwapWrapper] Will swap blocks {swap_start_idx}-{total_blocks-1} to CPU")
        
        # Clone model to avoid modifying original
        model_copy = model.clone()
        
        # Function to setup block swapping (called after model is loaded to GPU)
        def setup_blockswap_wrapper(
            patcher: ModelPatcher,
            device_to,
            lowvram_model_memory,
            force_patch_weights,
            full_load,
        ):
            """Setup block swapping after model is loaded."""
            nonlocal blocks_to_swap, swap_start_idx, offload_txt_emb, offload_img_emb
            
            transformer = patcher.model.diffusion_model
            
            # Check if already wrapped
            if hasattr(transformer, '_blockswap_wrapped'):
                if enable_debug:
                    print("[BlockSwapWrapper] Model already wrapped, skipping")
                return
            
            if enable_debug:
                print(f"[BlockSwapWrapper] Setting up block swap on device: {device_to}")
            
            # Move blocks to CPU
            for i in range(swap_start_idx, len(transformer.blocks)):
                transformer.blocks[i].to('cpu', non_blocking=use_non_blocking)
                if enable_debug:
                    print(f"[BlockSwapWrapper] Moved block {i} to CPU")
            
            # Optionally offload embeddings
            if offload_txt_emb and hasattr(transformer, 'txt_in'):
                transformer.txt_in.to('cpu')
                if enable_debug:
                    print("[BlockSwapWrapper] Offloaded text embeddings to CPU")
            
            if offload_img_emb and hasattr(transformer, 'img_in'):
                transformer.img_in.to('cpu')
                if enable_debug:
                    print("[BlockSwapWrapper] Offloaded image embeddings to CPU")
            
            # Wrap forward method
            if not hasattr(transformer, '_original_forward'):
                transformer._original_forward = transformer.forward
                
                def forward_with_blockswap(*args, **kwargs):
                    """Forward pass with block swapping."""
                    # Determine target device
                    target_device = device_to if device_to is not None else 'cuda'
                    
                    if enable_debug:
                        print(f"[BlockSwapWrapper] Forward pass - moving {blocks_to_swap} blocks to {target_device}")
                    
                    # Move swapped blocks to GPU
                    for i in range(swap_start_idx, len(transformer.blocks)):
                        transformer.blocks[i].to(target_device, non_blocking=use_non_blocking)
                    
                    # Move embeddings if needed
                    if offload_txt_emb and hasattr(transformer, 'txt_in'):
                        transformer.txt_in.to(target_device, non_blocking=use_non_blocking)
                    if offload_img_emb and hasattr(transformer, 'img_in'):
                        transformer.img_in.to(target_device, non_blocking=use_non_blocking)
                    
                    # Wait for transfers if non-blocking
                    if use_non_blocking:
                        torch.cuda.synchronize()
                    
                    try:
                        # Execute original forward
                        result = transformer._original_forward(*args, **kwargs)
                    finally:
                        # Move blocks back to CPU
                        for i in range(swap_start_idx, len(transformer.blocks)):
                            transformer.blocks[i].to('cpu', non_blocking=use_non_blocking)
                        
                        # Move embeddings back
                        if offload_txt_emb and hasattr(transformer, 'txt_in'):
                            transformer.txt_in.to('cpu', non_blocking=use_non_blocking)
                        if offload_img_emb and hasattr(transformer, 'img_in'):
                            transformer.img_in.to('cpu', non_blocking=use_non_blocking)
                        
                        if enable_debug:
                            print(f"[BlockSwapWrapper] Forward pass complete - blocks back on CPU")
                    
                    return result
                
                transformer.forward = forward_with_blockswap
                transformer._blockswap_wrapped = True
                
                if enable_debug:
                    print("[BlockSwapWrapper] Forward method wrapped successfully")
        
        # Cleanup function to restore original forward
        def cleanup_blockswap_wrapper(patcher: ModelPatcher):
            """Cleanup: restore original forward and move blocks back to CPU."""
            transformer = patcher.model.diffusion_model
            
            if hasattr(transformer, '_original_forward'):
                transformer.forward = transformer._original_forward
                delattr(transformer, '_original_forward')
                if enable_debug:
                    print("[BlockSwapWrapper] Restored original forward method")
            
            if hasattr(transformer, '_blockswap_wrapped'):
                delattr(transformer, '_blockswap_wrapped')
            
            # Ensure blocks are on CPU
            for i in range(swap_start_idx, len(transformer.blocks)):
                if transformer.blocks[i].device.type != 'cpu':
                    transformer.blocks[i].to('cpu')
            
            if enable_debug:
                print("[BlockSwapWrapper] Cleanup complete")
        
        # Register callbacks
        model_copy.add_callback(CallbacksMP.ON_LOAD, setup_blockswap_wrapper)
        model_copy.add_callback(CallbacksMP.ON_CLEANUP, cleanup_blockswap_wrapper)
        
        if enable_debug:
            print("[BlockSwapWrapper] Callbacks registered")
        
        return (model_copy,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "wan22BlockSwap": WanVideo22BlockSwap,
    "WANBlockSwapWrapper": WANBlockSwapWrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wan22BlockSwap": "WAN 2.2 BlockSwap (Lazy Load + GGUF Safe)",
    "WANBlockSwapWrapper": "WAN BlockSwap Wrapper (Universal)",
}
