# GLP fixed-point differential

## Scope

This measurement compares G0's exact integer projection with the equivalent
float32 arithmetic. It is not a vLLM execution and does not establish token,
behavioral, BF16, or serving-kernel equivalence. Its purpose is to quantify the
error introduced by direction/alpha quantization and integer projection
rounding before any serving claim is made.

Harness: `scripts/glp_differential.py`. It uses NumPy and the Python standard
library; no Torch installation is required. The float32 reference follows the
same operation order as weightless's `SteeringCore.apply`: a direction dot
product, an outer product, scalar-alpha multiplication, and subtraction.

## Input

- Artifact: `GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf`
- SHA-256: `1feeb25691442ad2ffb0e02fca7bd7d553dd79696af778b982a88ab8cdd9ef31`
- 77 rank-1 directions, alpha 1.0
- 256 deterministic rows per layer, seed 0
- Each row has 4,096 integer activations sampled uniformly from `[-2048, 2047]`
- 121,110,528 output values compared per configuration

Run:

```bash
python3 scripts/glp_differential.py --self-test
python3 scripts/glp_differential.py /path/to/GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf \
  --samples 256 --scale 2048 --direction-shift 14 --alpha-shift 14
```

## Results

| Direction/alpha shift | Max absolute error | Mean absolute error | RMSE | Relative L2 |
|---:|---:|---:|---:|---:|
| 8 | 149.392578 | 1.385206 | 1.903520 | 0.161004% |
| 12 | 7.836304 | 0.261741 | 0.310422 | 0.026256% |
| 14 | 2.239746 | 0.248811 | 0.288632 | 0.024413% |

The worst relative-L2 layer was layer 10 at shift 8 and layer 1 at shifts 12
and 14. Increasing precision primarily removes quantization-driven outliers.
The remaining mean/RMSE floor is consistent with integer output rounding, not
something that more direction precision eliminates.

## Token-level differential: `small_float_pair`

`scripts/glp_token_differential.py` runs the upstream FlashProve
`small_float_pair`: the exact integer model and its float32
`Llama4ForCausalLM` counterpart from Hugging Face Transformers. Eight random
unit directions (seed 0) steer every layer. Eight prompts exercise 512 token
positions. The null gate at alpha 0 reproduces the unsteered 501/512 agreement
exactly. No vLLM server is used.

Baseline: 501/512 matching top choices, median normalized logit RMSE
0.00768478. At alpha 0.001, the fixed/float comparison improves to 507/512 and
the direct float steering effect changes no choices. At alpha 0.25, the
comparison is 506/512 and steering changes 33/512 float choices.

At alpha 1.0, steering itself changes 130/512 float-model top choices, with
median normalized RMSE 0.163054. The fixed-versus-float results are:

| Direction/alpha shift | Matching choices | Different choices | Median normalized logit RMSE |
|---:|---:|---:|---:|
| 8 | 504/512 | 8 | 0.00924253 |
| 12 | 505/512 | 7 | 0.00799324 |
| 14 | 509/512 | 3 | 0.00771275 |

The 14-bit result is closest to baseline on this fixture. Agreement above the
unsteered baseline is not evidence of semantic improvement: integer and float
computations can move in either direction relative to each other. These 512
synthetic positions do not establish a universal token-agreement bound.

Run on a host with the vLLM image available:

```bash
docker run --rm --entrypoint python3 -v "$PWD:/workspace" \
  vllm/vllm-openai:v0.28.0 \
  /workspace/scripts/glp_token_differential.py \
  --shift 14 --alpha-shift 14 --prompts 8 --alpha 1.0
```

## BF16 token differential

The same harness also runs the float checkpoint as BF16. At alpha 0, the null
gate is exact: 493/512 matching choices, 19 differences, median normalized
logit RMSE 0.01643848, and zero direct steering effect. At alpha 1.0 and shift
14, the fixed/BF16 comparison is 492/512 with median normalized RMSE
0.01667815. Steering changes 134/512 BF16 top choices, with direct-effect RMSE
0.16336276. Thus the stronger 509/512 float32 result must not be presented as a
BF16 serving-dtype result.

## Decision and remaining gap

Use **14-bit direction and alpha shifts** for G0 unless a model-specific range
analysis selects another value. Two-limb direction encoding supports this
setting, and projection arithmetic remains checked for overflow.

These results do not prove that a steered floating-point serving engine emits
the same tokens as the integer model in production. The token-level fixture is
small and synthetic, and the numerical projection matrix uses random large
activations rather than captured serving activations. Larger real-checkpoint
behavior and the actual vLLM serving lane remain separate measurements before
any production equivalence claim.
