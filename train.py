"""
Autoresearch pretraining script. Single-GPU, single-file.
Optimized for RTX 5070 Ti (Blackwell) and consumer NVIDIA GPUs.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py [--no-compile] [--smoke-test]
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Runtime Detection
# ---------------------------------------------------------------------------

# Peak BF16 TFLOPS by GPU name (for MFU calculation)
GPU_PEAK_FLOPS = {
    # Blackwell
    "NVIDIA GeForce RTX 5090":          419.2e12,
    "NVIDIA GeForce RTX 5080":          266.5e12,
    "NVIDIA GeForce RTX 5070 Ti":       190.0e12,
    "NVIDIA GeForce RTX 5070":          126.8e12,
    # Ada Lovelace
    "NVIDIA GeForce RTX 4090":          330.3e12,
    "NVIDIA GeForce RTX 4080 SUPER":    244.8e12,
    "NVIDIA GeForce RTX 4080":          244.8e12,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 184.1e12,
    "NVIDIA GeForce RTX 4070 Ti":       184.1e12,
    "NVIDIA GeForce RTX 4070 SUPER":    140.0e12,
    "NVIDIA GeForce RTX 4070":          116.7e12,
    "NVIDIA GeForce RTX 4060 Ti":        85.0e12,
    "NVIDIA GeForce RTX 4060":           60.0e12,
    # Ampere
    "NVIDIA GeForce RTX 3090 Ti":       160.0e12,
    "NVIDIA GeForce RTX 3090":          142.6e12,
    "NVIDIA GeForce RTX 3080 Ti":       136.5e12,
    "NVIDIA GeForce RTX 3080":          119.5e12,
    "NVIDIA GeForce RTX 3070 Ti":        87.0e12,
    "NVIDIA GeForce RTX 3070":           81.3e12,
    "NVIDIA GeForce RTX 3060 Ti":        65.0e12,
    "NVIDIA GeForce RTX 3060":           48.0e12,
    # Turing
    "NVIDIA GeForce RTX 2080 Ti":        53.8e12,
    "NVIDIA GeForce RTX 2080 SUPER":     44.4e12,
    "NVIDIA GeForce RTX 2080":           40.3e12,
    "NVIDIA GeForce RTX 2070 SUPER":     36.4e12,
    "NVIDIA GeForce RTX 2070":           29.1e12,
    "NVIDIA GeForce RTX 2060 SUPER":     28.9e12,
    "NVIDIA GeForce RTX 2060":           25.9e12,
    # Datacenter
    "NVIDIA H100":                       989.5e12,
    "NVIDIA A100 80GB PCIe":             312.0e12,
    "NVIDIA A100-SXM4-80GB":             312.0e12,
    "NVIDIA A100-SXM4-40GB":             312.0e12,
}


@dataclass
class RuntimeConfig:
    gpu_name: str
    compute_capability: tuple
    vram_total_mb: int
    peak_flops: float  # 0.0 if unknown
    amp_dtype: torch.dtype


def detect_runtime():
    """Detect GPU capabilities and return runtime configuration."""
    assert torch.cuda.is_available(), "CUDA is required"
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = props.total_memory // (1024 * 1024)

    # Look up peak FLOPS (exact match, then partial)
    peak_flops = GPU_PEAK_FLOPS.get(gpu_name, 0.0)
    if peak_flops == 0.0:
        for name, flops in GPU_PEAK_FLOPS.items():
            if name in gpu_name or gpu_name in name:
                peak_flops = flops
                break

    # bf16 native on Ampere+ (CC 8.0+); fall back to fp16 for Turing
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = RuntimeConfig(
        gpu_name=gpu_name,
        compute_capability=cap,
        vram_total_mb=vram_mb,
        peak_flops=peak_flops,
        amp_dtype=amp_dtype,
    )
    print(f"GPU: {gpu_name} (SM {cap[0]}.{cap[1]}, {vram_mb} MB VRAM)")
    print(f"AMP dtype: {amp_dtype}")
    if peak_flops > 0:
        print(f"Peak BF16 FLOPS: {peak_flops/1e12:.1f} TFLOPS")
    else:
        print("Peak FLOPS: unknown (MFU will show n/a)")
    return config


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    use_activation_checkpointing: bool = False
    compute_dtype: torch.dtype = torch.bfloat16


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self._use_gqa = self.n_kv_head < self.n_head
        self._mask_cache = {}

    def _get_sdpa_mask(self, seq_len, window, device):
        """Build and cache a causal + sliding window boolean mask for SDPA.
        Returns a (T, T) bool tensor where True = attend, False = masked."""
        cache_key = (seq_len, window, device.type, device.index)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col <= row  # causal
        if 0 < window < seq_len:
            mask = mask & (col >= (row - window))
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Transpose to (B, H, T, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        window = window_size[0] if isinstance(window_size, tuple) else window_size
        if window >= T:
            # Full context — use is_causal for optimal kernel selection
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self._use_gqa)
        else:
            # Sliding window — explicit mask
            mask = self._get_sdpa_mask(T, window, q.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self._use_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings (exact seq_len, no 10x overallocation)
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        embed_dtype = self.config.compute_dtype
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to compute dtype
        self.transformer.wte.to(dtype=embed_dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=embed_dtype)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(self.config.compute_dtype)
        sin = sin.to(self.config.compute_dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR proportional to 1/sqrt(dmodel) (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Chunk Muon groups to reduce peak memory from torch.stack
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            for ci in range(0, len(group_params), MUON_GROUP_CHUNK):
                chunk = group_params[ci:ci + MUON_GROUP_CHUNK]
                param_groups.append(dict(
                    kind='muon', params=chunk, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if self.config.use_activation_checkpointing:
                x = torch_checkpoint(block, x, ve, cos_sin, self.window_sizes[i], use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Muon orthogonalization dtype — set at runtime based on GPU capabilities
MUON_COMPUTE_DTYPE = torch.bfloat16


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad.to(exp_avg.dtype), 1 - beta1_t)
    exp_avg_sq.lerp_(grad.to(exp_avg_sq.dtype).square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_((exp_avg / denom).to(p.dtype), alpha=-step_size)


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.to(MUON_COMPUTE_DTYPE)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# Runtime-configured implementations (plain or torch.compiled)
_adamw_step_impl = adamw_step_fused
_muon_step_impl = muon_step_fused


def configure_compile(use_compile):
    """Set up optimizer step implementations — compiled or plain."""
    global _adamw_step_impl, _muon_step_impl
    if use_compile:
        _adamw_step_impl = torch.compile(adamw_step_fused, dynamic=False, fullgraph=True)
        _muon_step_impl = torch.compile(muon_step_fused, dynamic=False, fullgraph=True)
    else:
        _adamw_step_impl = adamw_step_fused
        _muon_step_impl = muon_step_fused


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                # Use fp32 moments when params are fp16 to avoid underflow
                moment_dtype = torch.float32 if p.dtype == torch.float16 else p.dtype
                state['exp_avg'] = torch.zeros_like(p, dtype=moment_dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, dtype=moment_dtype)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            _adamw_step_impl(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        _muon_step_impl(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 16   # per-device batch size (safe for 16GB VRAM)
EVAL_BATCH_SIZE = 8      # separate eval batch size (lower to avoid OOM during eval)

# Optimizer chunking
MUON_GROUP_CHUNK = 8     # max params per Muon group (reduces peak from torch.stack)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Module-level compile flag — set by main() before training
USE_COMPILE = True


def build_model_config(depth, vocab_size, runtime):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        compute_dtype=runtime.amp_dtype,
    )


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


def run_training(runtime, tokenizer, config, device_batch_size, time_budget):
    """Run a single training attempt. Returns result dict or raises on OOM/failure."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=runtime.amp_dtype)

    t_start = time.time()

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")
    print(f"Model config: {asdict(config)}")

    # Create model
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    if USE_COMPILE:
        model = torch.compile(model, dynamic=False)

    train_loader = make_dataloader(tokenizer, device_batch_size, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    warmup_steps = 10
    print(f"Time budget: {time_budget}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Activation checkpointing: {config.use_activation_checkpointing}")

    # Training loop
    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    train_loss = torch.tensor(0.0, device=device)
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        # Progress and schedules
        progress = min(total_training_time / time_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()

        # Fast fail: abort if loss is exploding or NaN
        if math.isnan(train_loss_f) or train_loss_f > 100:
            raise RuntimeError("FAIL: training loss exploded or NaN")

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > warmup_steps:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        if runtime.peak_flops > 0:
            mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / runtime.peak_flops
            mfu_str = f"{mfu:.1f}%"
        else:
            mfu_str = "n/a"
        remaining = max(0, time_budget - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu_str} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management (Python's GC causes ~500ms stalls)
        if step == 0:
            gc.collect()
            if hasattr(gc, 'freeze'):
                gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > warmup_steps and total_training_time >= time_budget:
            break

    print()  # newline after \r training log

    total_tokens = step * TOTAL_BATCH_SIZE

    return {
        'model': model,
        'training_seconds': total_training_time,
        'total_tokens': total_tokens,
        'step': step,
        'num_params': num_params,
        'num_flops_per_token': num_flops_per_token,
        't_start': t_start,
        't_start_training': t_start_training,
        'warmup_steps': warmup_steps,
    }


def _restore_gc():
    """Restore GC to normal state after a failed attempt."""
    if hasattr(gc, 'unfreeze'):
        gc.unfreeze()
    gc.enable()
    gc.collect()


def main():
    global USE_COMPILE, MUON_COMPUTE_DTYPE

    parser = argparse.ArgumentParser(description="Autoresearch pretraining (Blackwell-optimized)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation (10s, ~3 steps)")
    args = parser.parse_args()

    USE_COMPILE = not args.no_compile
    runtime = detect_runtime()

    # Set Muon compute dtype based on GPU
    MUON_COMPUTE_DTYPE = runtime.amp_dtype

    configure_compile(USE_COMPILE)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    config = build_model_config(DEPTH, vocab_size, runtime)
    time_budget = 10 if args.smoke_test else TIME_BUDGET

    # Training with OOM cascade
    candidates = [
        (DEVICE_BATCH_SIZE, False),
        (max(DEVICE_BATCH_SIZE // 2, 2), True),
        (max(DEVICE_BATCH_SIZE // 4, 2), True),
    ]
    result = None
    for batch_size, use_ckpt in candidates:
        try:
            config.use_activation_checkpointing = use_ckpt
            print(f"\n--- Training attempt: batch_size={batch_size}, checkpointing={use_ckpt} ---")
            result = run_training(runtime, tokenizer, config, batch_size, time_budget)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at batch_size={batch_size}. Retrying with smaller batch.")
            _restore_gc()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "FAIL" in str(e):
                print(f"\n{e}")
                exit(1)
            raise

    if result is None:
        print("FAIL: OOM at all batch sizes")
        exit(1)

    model = result['model']
    t_start = result['t_start']

    # Save pre-eval checkpoint (defensive, in case eval OOMs)
    pre_eval_path = "checkpoint_pre_eval.pt"
    try:
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(raw_model.state_dict(), pre_eval_path)
    except Exception:
        pass  # non-critical

    # Evaluation with OOM cascade
    model.eval()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=runtime.amp_dtype)
    val_bpb = None
    for eval_bs in [EVAL_BATCH_SIZE, 4, 2, 1]:
        try:
            torch.cuda.empty_cache()
            with autocast_ctx:
                val_bpb = evaluate_bpb(model, tokenizer, eval_bs)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"Eval OOM at batch_size={eval_bs}; trying smaller.")
            torch.cuda.empty_cache()

    if val_bpb is None:
        print("FAIL: eval OOM at all batch sizes")
        exit(1)

    # Clean up pre-eval checkpoint
    try:
        os.remove(pre_eval_path)
    except OSError:
        pass

    # Final summary
    t_end = time.time()
    total_training_time = result['training_seconds']
    step = result['step']
    warmup_steps = result['warmup_steps']
    num_params = result['num_params']
    num_flops_per_token = result['num_flops_per_token']
    total_tokens = result['total_tokens']

    if runtime.peak_flops > 0 and total_training_time > 0:
        steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - warmup_steps) / total_training_time / runtime.peak_flops
        mfu_str = f"{steady_state_mfu:.2f}"
    else:
        mfu_str = "n/a"

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {mfu_str}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()
