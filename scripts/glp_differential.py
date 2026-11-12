#!/usr/bin/env python3
"""Measure G0 fixed-point GLP projection error against float32 arithmetic.

This is an independent numerical differential, not a vLLM kernel execution.
Only NumPy and the Python standard library are required.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        end = self.pos + count
        if end > len(self.data):
            raise ValueError("truncated GGUF")
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8")

    def skip(self, value_type):
        if value_type == 8:
            self.string()
        elif value_type == 9:
            element_type = self.u32()
            if element_type == 9:
                raise ValueError("nested GGUF arrays")
            for _ in range(self.u64()):
                self.skip(element_type)
        else:
            self.take(SIZES[value_type])


def read_glp(path):
    data = Path(path).read_bytes()
    r = Reader(data)
    if r.take(4) != b"GGUF" or r.u32() != 3:
        raise ValueError("not GGUF v3")
    tensor_count = r.u64()
    metadata_count = r.u64()
    metadata = {}
    for _ in range(metadata_count):
        key = r.string()
        value_type = r.u32()
        if key in ("glp.mode", "glp.hook_point"):
            if value_type != 8:
                raise ValueError(f"{key}: wrong type")
            metadata[key] = r.string()
        elif key in ("glp.spec_version", "glp.rank", "general.alignment"):
            if value_type != 4:
                raise ValueError(f"{key}: wrong type")
            metadata[key] = r.u32()
        elif key == "glp.alpha_default":
            if value_type != 6:
                raise ValueError("alpha: wrong type")
            metadata[key] = struct.unpack("<f", r.take(4))[0]
        else:
            r.skip(value_type)
    if metadata.get("glp.mode") != "project":
        raise ValueError("only project GLP is supported")
    if metadata.get("glp.spec_version") != 1:
        raise ValueError("only GLP spec version 1 is supported")
    if metadata.get("glp.hook_point") != "residual_stream_post_layer":
        raise ValueError("only post-layer GLP is supported")
    if metadata.get("glp.rank", 1) != 1 or "glp.alpha_default" not in metadata:
        raise ValueError("rank-1 GLP with explicit alpha is required")

    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        dimensions = r.u32()
        shape = [r.u64() for _ in range(dimensions)]
        tensor_type = r.u32()
        offset = r.u64()
        if name.startswith("direction."):
            if dimensions != 1 or tensor_type != 0 or not name[10:].isdigit():
                raise ValueError(f"unsupported direction: {name}")
            layer = int(name[10:])
            if layer == 0:
                raise ValueError("direction.0")
            tensors.append((layer, shape[0], offset))
    if not tensors:
        raise ValueError("no directions")
    alignment = metadata.get("general.alignment", 32)
    if alignment == 0 or alignment & (alignment - 1):
        raise ValueError("invalid alignment")
    data_start = (r.pos + alignment - 1) & ~(alignment - 1)
    layers = {}
    for layer, width, offset in tensors:
        if layer in layers:
            raise ValueError("duplicate layer")
        start = data_start + offset
        values = np.frombuffer(data, dtype="<f4", count=width, offset=start).copy()
        if not np.isfinite(values).all():
            raise ValueError("nonfinite direction")
        layers[layer] = values
    widths = {len(value) for value in layers.values()}
    if len(widths) != 1:
        raise ValueError("inconsistent direction widths")
    return data, float(metadata["glp.alpha_default"]), layers


def fixed_projection(h, direction, alpha, direction_shift, alpha_shift):
    quantized_direction = np.rint(direction * (1 << direction_shift)).astype(np.int64)
    quantized_alpha = int(np.rint(alpha * (1 << alpha_shift)))
    projection = h.astype(np.int64) @ quantized_direction
    scaled = projection * quantized_alpha
    shift = 2 * direction_shift + alpha_shift
    term = np.floor_divide(
        scaled[:, None] * quantized_direction + (1 << (shift - 1)), 1 << shift
    )
    return h - term


def float32_projection(h, direction, alpha):
    activations = h.astype(np.float32)
    coefficient = activations @ direction
    projection = coefficient[:, None] * direction
    return activations - np.float32(alpha) * projection


def evaluate(args):
    raw, alpha, layers = read_glp(args.artifact)
    rng = np.random.default_rng(args.seed)
    selected = [args.layer] if args.layer is not None else sorted(layers)
    total_abs = 0.0
    total_square = 0.0
    total_reference_square = 0.0
    max_abs = 0.0
    worst = None
    for layer in selected:
        direction = layers[layer]
        h = rng.integers(-args.scale, args.scale, size=(args.samples, len(direction)), dtype=np.int64)
        reference = float32_projection(h, direction, alpha).astype(np.float64)
        fixed = fixed_projection(h, direction, alpha, args.direction_shift, args.alpha_shift)
        error = fixed.astype(np.float64) - reference
        layer_abs = float(np.abs(error).sum())
        layer_square = float((error * error).sum())
        layer_reference_square = float((reference * reference).sum())
        layer_max = float(np.abs(error).max())
        total_abs += layer_abs
        total_square += layer_square
        total_reference_square += layer_reference_square
        max_abs = max(max_abs, layer_max)
        relative = layer_square**0.5 / max(layer_reference_square**0.5, 1e-300)
        if worst is None or relative > worst["relative_l2"]:
            worst = {"layer": layer, "relative_l2": relative, "max_abs": layer_max}
    count = args.samples * sum(len(layers[layer]) for layer in selected)
    return {
        "artifact": str(args.artifact),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "alpha": alpha,
        "layers": len(selected),
        "samples_per_layer": args.samples,
        "activation_range": [-args.scale, args.scale - 1],
        "direction_shift": args.direction_shift,
        "alpha_shift": args.alpha_shift,
        "reference": "float32 NumPy projection; no vLLM kernel executed",
        "values_compared": count,
        "max_abs_error": max_abs,
        "mean_abs_error": total_abs / count,
        "rmse": (total_square / count) ** 0.5,
        "relative_l2": total_square**0.5 / max(total_reference_square**0.5, 1e-300),
        "worst_layer": worst,
    }


def self_test():
    h = np.array([[8, 4]], dtype=np.int64)
    direction = np.array([0.5, 0.0], dtype=np.float32)
    fixed = fixed_projection(h, direction, 1.0, 1, 1)
    reference = float32_projection(h, direction, 1.0)
    assert fixed.tolist() == [[6, 4]]
    assert reference.tolist() == [[6.0, 4.0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--scale", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-shift", type=int, default=12)
    parser.add_argument("--alpha-shift", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    if args.artifact is None or args.samples <= 0 or args.scale <= 0:
        parser.error("artifact is required; samples and scale must be positive")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
