import torch
import torch.distributed as dist

class CPUOffloadEngine:
    """
    Manages the asynchronous streaming of gradients from GPU to CPU,
    and updated weights from CPU back to GPU.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
        # CPU accumulator for gradients
        self.cpu_grads = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Use pinned memory (page-locked) for fast PCIe transfers
                self.cpu_grads[name] = torch.zeros_like(p.data, device='cpu', dtype=torch.float32).pin_memory()
                
        # CUDA stream for non-blocking transfers
        self.transfer_stream = torch.cuda.Stream()

    def offload_layer_gradients(self, layer_idx: int):
        """
        Called immediately after a layer's backward pass.
        Streams gradients to CPU and frees them from GPU VRAM.
        """
        layer = self.model.layers[layer_idx]
        
        with torch.cuda.stream(self.transfer_stream):
            for name, p in layer.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # Construct full name
                    full_name = f"layers.{layer_idx}.{name}"
                    
                    # Async copy to pinned CPU memory
                    self.cpu_grads[full_name].copy_(p.grad, non_blocking=True)
                    
                    # Free GPU memory immediately! (Crucial for VRAM reduction)
                    p.grad = None
                    
    def offload_remaining_gradients(self):
        """Offload gradients for embeddings, MTP, and final norm."""
        with torch.cuda.stream(self.transfer_stream):
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if name not in self.cpu_grads:
                        self.cpu_grads[name] = torch.zeros_like(p.data, device='cpu', dtype=torch.float32).pin_memory()
                    self.cpu_grads[name].copy_(p.grad, non_blocking=True)
                    p.grad = None

    def sync_transfers(self):
        """Wait for all GPU -> CPU transfers to complete."""
        self.transfer_stream.synchronize()

    def update_and_stream_weights_to_gpu(self, lr_scale: float = 1.0, fp8_training: bool = True):
        """
        1. Run optimizer step on CPU.
        2. Convert updated master weights to FP8.
        3. Stream back to GPU.
        """
        # Step optimizer on CPU
        self.optimizer.step(self.cpu_grads, lr_scale=lr_scale)
        
        with torch.cuda.stream(self.transfer_stream):
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    master_w = self.optimizer.get_master_weight(name)
                    
                    if fp8_training and 'norm' not in name:
                        # Convert to FP8 before streaming to GPU
                        # (Using standard PyTorch FP8 cast for now; in prod, this is scaled)
                        fp8_w = master_w.to(torch.float8_e4m3fn)
                        p.data.copy_(fp8_w, non_blocking=True)
                    else:
                        # Norms and non-FP8 layers stay in BF16
                        p.data.copy_(master_w, non_blocking=True)
                        
        self.transfer_stream.synchronize()
