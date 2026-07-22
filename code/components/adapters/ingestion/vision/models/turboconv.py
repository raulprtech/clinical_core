"""Fake-quantized Conv3d baselines for the STU-Net TurboConv kill test.

TurboConv applies a deterministic randomized Walsh-Hadamard transform to each
convolution input channel vector and the compensating transform to its weight
tensor. In full precision this is functionally equivalent to the original
convolution. PTQ and TurboConv then use the same symmetric bit width and the
same calibration protocol.

This module evaluates numerical viability. The fake-quantized tensors execute
with regular PyTorch kernels; latency is therefore not an INT-kernel benchmark.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def deterministic_signs(length: int, name: str, seed: int = 2026) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little"))
    return torch.randint(0, 2, (length,), generator=generator).float().mul_(2).sub_(1)


def fwht(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform along a power-of-two axis."""
    size = tensor.shape[dim]
    if not is_power_of_two(size):
        raise ValueError(f"FWHT dimension must be a power of two, got {size}")
    output = tensor.movedim(dim, -1)
    original_shape = output.shape
    flat = output.reshape(-1, size)
    block = 1
    while block < size:
        view = flat.reshape(-1, size // (2 * block), 2, block)
        left = view[:, :, 0]
        right = view[:, :, 1]
        flat = torch.cat((left + right, left - right), dim=-1).reshape(-1, size)
        block *= 2
    flat = flat / math.sqrt(size)
    return flat.reshape(original_shape).movedim(-1, dim)


def rotate_channels(
    tensor: torch.Tensor,
    signs: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    shape = [1] * tensor.ndim
    shape[dim] = signs.numel()
    signed = tensor * signs.to(device=tensor.device, dtype=tensor.dtype).reshape(shape)
    return fwht(signed, dim=dim)


def inverse_rotate_channels(
    tensor: torch.Tensor,
    signs: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    transformed = fwht(tensor, dim=dim)
    shape = [1] * tensor.ndim
    shape[dim] = signs.numel()
    return transformed * signs.to(
        device=tensor.device, dtype=tensor.dtype
    ).reshape(shape)


def fake_quant_symmetric(
    tensor: torch.Tensor,
    bits: int,
    scale: torch.Tensor,
) -> torch.Tensor:
    if bits < 2:
        raise ValueError("Symmetric quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    safe_scale = torch.clamp(scale.to(tensor.device, tensor.dtype), min=torch.finfo(tensor.dtype).eps)
    integer = torch.clamp(torch.round(tensor / safe_scale), -qmax, qmax)
    return integer * safe_scale


def weight_scale_per_output(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    max_abs = weight.detach().abs().amax(dim=tuple(range(1, weight.ndim)), keepdim=True)
    return torch.clamp(max_abs / qmax, min=torch.finfo(weight.dtype).eps)


@dataclass(frozen=True)
class QuantizationSpec:
    variant: str
    weight_bits: int
    activation_bits: int
    rotation_seed: int = 2026

    def __post_init__(self) -> None:
        if self.variant not in {"ptq", "turboconv"}:
            raise ValueError(f"Unknown quantization variant: {self.variant}")


class FakeQuantConv3d(nn.Module):
    """Drop-in Conv3d with calibrated activation and weight fake quantization."""

    def __init__(
        self,
        source: nn.Conv3d,
        name: str,
        spec: QuantizationSpec,
        activation_scale: float,
        quantize: bool = True,
    ):
        super().__init__()
        if source.groups != 1:
            raise ValueError("TurboConv pilot supports Conv3d groups=1 only")
        self.name = name
        self.spec = spec
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.quantize = quantize
        signs = deterministic_signs(source.in_channels, name, spec.rotation_seed)
        self.register_buffer("signs", signs)
        self.register_buffer(
            "activation_scale", torch.tensor(float(activation_scale), dtype=torch.float32)
        )

        weight = source.weight.detach().clone()
        if spec.variant == "turboconv" and source.in_channels > 1:
            if not is_power_of_two(source.in_channels):
                raise ValueError(
                    f"TurboConv requires power-of-two input channels; {name} has "
                    f"{source.in_channels}"
                )
            # x_rot = H D x, hence W_rot = W D H = (H D W^T)^T.
            weight = rotate_channels(weight, signs, dim=1)
        if quantize:
            scale = weight_scale_per_output(weight, spec.weight_bits)
            weight = fake_quant_symmetric(weight, spec.weight_bits, scale)
        self.register_buffer("weight", weight)
        if source.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = inputs
        if self.spec.variant == "turboconv" and self.in_channels > 1:
            transformed = rotate_channels(transformed, self.signs, dim=1)
        if self.quantize:
            transformed = fake_quant_symmetric(
                transformed,
                self.spec.activation_bits,
                self.activation_scale,
            )
        return F.conv3d(
            transformed,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def effective_weight(self) -> torch.Tensor:
        if self.spec.variant == "turboconv" and self.in_channels > 1:
            return inverse_rotate_channels(self.weight, self.signs, dim=1)
        return self.weight


def replace_conv3d_modules(
    model: nn.Module,
    spec: QuantizationSpec,
    activation_scales: Mapping[str, float],
    quantize: bool = True,
) -> dict[str, FakeQuantConv3d]:
    replaced: dict[str, FakeQuantConv3d] = {}

    def recurse(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Conv3d):
                if full_name not in activation_scales:
                    raise KeyError(f"Missing activation scale for {full_name}")
                replacement = FakeQuantConv3d(
                    child,
                    full_name,
                    spec,
                    activation_scales[full_name],
                    quantize=quantize,
                )
                setattr(parent, child_name, replacement)
                replaced[full_name] = replacement
            else:
                recurse(child, full_name)

    recurse(model)
    return replaced


def weight_error_summary(
    original_weights: Mapping[str, torch.Tensor],
    quantized_modules: Mapping[str, FakeQuantConv3d],
) -> dict[str, float]:
    squared_error = 0.0
    squared_reference = 0.0
    max_abs_error = 0.0
    elements = 0
    for name, module in quantized_modules.items():
        reference = original_weights[name].detach().float().cpu()
        effective = module.effective_weight().detach().float().cpu()
        difference = effective - reference
        squared_error += float(torch.sum(difference.square()))
        squared_reference += float(torch.sum(reference.square()))
        max_abs_error = max(max_abs_error, float(difference.abs().max()))
        elements += reference.numel()
    return {
        "weight_mse": squared_error / max(1, elements),
        "weight_relative_l2": math.sqrt(squared_error / max(squared_reference, 1e-30)),
        "weight_max_abs_error": max_abs_error,
        "weight_elements": float(elements),
    }
