#!/usr/bin/env python3
"""Compare steered integer and float32 models on upstream's small_float_pair.

Runs the actual FlashProve integer model and its Hugging Face float32
counterpart. No vLLM server or GPU is required.
"""

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FLASHPROVE = ROOT / "spikes" / "flashprove"
PACKAGE = FLASHPROVE / "integrations" / "vllm" / "integer_model"
sys.path.insert(0, str(PACKAGE))

import integer_model  # noqa: E402
from integer_model_vllm import float_pair  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


def prompts(number):
    return [[((37 + 2 * j) * i + 11 + 5 * j) % 512 for i in range(16)] for j in range(number)]


def directions(parameters, seed):
    generator = torch.Generator().manual_seed(seed)
    found = {}
    for layer in range(parameters.L):
        direction = torch.randn(parameters.d, generator=generator)
        found[layer] = (direction / direction.norm()).float()
    return found


def fixed_projection(x, direction, alpha, direction_shift, alpha_shift):
    quantized = torch.round(direction * (1 << direction_shift)).long()
    quantized_alpha = int(torch.round(torch.tensor(alpha * (1 << alpha_shift))))
    projection = torch.einsum("...h,h->...", x, quantized)
    scaled = projection * quantized_alpha
    rounding_shift = 2 * direction_shift + alpha_shift
    term = torch.floor_divide(
        scaled[..., None] * quantized + (1 << (rounding_shift - 1)), 1 << rounding_shift
    )
    return x - term


def float_projection(x, direction, alpha):
    coefficient = torch.einsum("...h,h->...", x, direction)
    projection = coefficient[..., None] * direction
    return x - alpha * projection


class SteeredIntegerModel(integer_model.IntegerModel):
    def __init__(self, parameters, weights, device, steering, alpha, shift, alpha_shift):
        super().__init__(parameters, weights, device)
        self.steering = steering
        self.steering_alpha = alpha
        self.steering_shift = shift
        self.steering_alpha_shift = alpha_shift

    def _feed_forward(self, layer, x, checks, note):
        x = super()._feed_forward(layer, x, checks, note)
        return fixed_projection(
            x, self.steering[layer], self.steering_alpha, self.steering_shift, self.steering_alpha_shift
        )


def integer_logits(model, tokens):
    found = {}
    sequence = model.new_sequence()
    model.extend(sequence, tokens, record=lambda name, layer, tensor: found.__setitem__(name, tensor.double()))
    model.free(sequence)
    return found["lam"]


def float_logits(model, tokens):
    with torch.no_grad():
        return model(torch.tensor([tokens])).logits[0].double()


def attach_float_steering(model, steering, alpha):
    def hook(direction):
        def apply(_module, _args, output):
            if torch.is_tensor(output):
                return float_projection(output, direction, alpha)
            hidden = float_projection(output[0], direction, alpha)
            return (hidden, *output[1:])
        return apply

    handles = []
    for layer, direction in steering.items():
        handles.append(model.model.layers[layer].register_forward_hook(hook(direction)))
    return handles


def compare(integer, floats, sequences):
    errors = []
    same = 0
    for tokens in sequences:
        integer_values = integer_logits(integer, tokens)
        float_values = float_logits(floats, tokens)
        errors += ((float_values - integer_values).pow(2).mean(dim=1).sqrt() / integer_values.std(dim=1)).tolist()
        same += int((float_values.argmax(dim=1) == integer_values.argmax(dim=1)).sum())
    return {
        "positions": len(errors),
        "same_choice": same,
        "different_choice": len(errors) - same,
        "agreement_pct": 100.0 * same / len(errors),
        "median_normalized_logit_rmse": statistics.median(errors),
    }


def compare_cached(floats, sequences, baseline_values):
    errors = []
    same = 0
    for tokens, baseline in zip(sequences, baseline_values):
        steered = float_logits(floats, tokens)
        errors += ((steered - baseline).pow(2).mean(dim=1).sqrt() / baseline.std(dim=1)).tolist()
        same += int((steered.argmax(dim=1) == baseline.argmax(dim=1)).sum())
    return {
        "positions": len(errors),
        "same_choice": same,
        "different_choice": len(errors) - same,
        "agreement_pct": 100.0 * same / len(errors),
        "median_normalized_logit_rmse": statistics.median(errors),
    }


def evaluate(shift, alpha_shift, alpha, direction_seed, prompt_count):
    with tempfile.TemporaryDirectory(prefix="glp-token-differential-") as temporary:
        pair = Path(temporary) / "pair"
        pair.mkdir()
        float_pair.make_float_pair(pair, FLASHPROVE)
        integer = integer_model.load_from_folder(pair / "model", "cpu")
        floats = AutoModelForCausalLM.from_pretrained(pair / "float_model", attn_implementation="eager").eval()
        sequences = [(prompt * 4)[: integer.parameters.T] for prompt in prompts(prompt_count)]
        baseline = compare(integer, floats, sequences)
        baseline_floats = [float_logits(floats, tokens) for tokens in sequences]

        steering = directions(integer.parameters, direction_seed)
        steered_integer = SteeredIntegerModel(
            integer.parameters, integer.weights, "cpu", steering, alpha, shift, alpha_shift
        )
        handles = attach_float_steering(floats, steering, alpha)
        try:
            steered = compare(steered_integer, floats, sequences)
            steering_effect = compare_cached(floats, sequences, baseline_floats)
        finally:
            for handle in handles:
                handle.remove()
    return {
        "model": "small_float_pair",
        "reference": "FlashProve IntegerModel versus Hugging Face float32 Llama4ForCausalLM",
        "direction_seed": direction_seed,
        "alpha": alpha,
        "direction_shift": shift,
        "alpha_shift": alpha_shift,
        "prompts": prompt_count,
        "baseline": baseline,
        "steered": steered,
        "steering_effect_on_float": steering_effect,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", type=int, default=14)
    parser.add_argument("--alpha-shift", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--direction-seed", type=int, default=0)
    parser.add_argument("--prompts", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.shift <= 14 or not 1 <= args.alpha_shift <= 31 or args.prompts <= 0:
        parser.error("shift must be in 1..14, alpha-shift in 1..31, and prompts positive")
    result = evaluate(args.shift, args.alpha_shift, args.alpha, args.direction_seed, args.prompts)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
