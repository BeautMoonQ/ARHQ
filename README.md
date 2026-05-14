# ARHQ: Activation Residual Hessian Quantization

This repository provides the reference implementation for **ARHQ**, a low-rank-assisted quantization method designed to reduce the propagation of activation quantization error through linear layers.

ARHQ decomposes each linear weight into a quantized residual branch and a full-precision low-rank branch:

```text
W = W_res + L,      L = B A^T
Y_hat = Q_x(X) Q_w(W_res)^T + X L^T
```

Unlike a standard low-rank reconstruction objective, ARHQ chooses `L` according to the activation quantization residual:

```text
E_x = X - Q_x(X)
min_rank(L)<=r || E_x (W - L)^T ||_F^2
```

With `G_x = E_x^T E_x / N`, this becomes a weighted low-rank decomposition under the activation residual Hessian metric. The resulting LoRA branch is kept in floating point, while the residual branch is evaluated with simulated nvfp4 quantization. The implementation uses 1D block scaling for activations and 16x16 2D block scaling for weights, matching the default NVFP4 weight-scaling layout used by Transformer Engine.

## Repository Scope

The codebase is organized as a minimal research implementation:

```text
arhq/
  calibration.py      # Build calibration tensors from calibration data
  decompose.py        # Extract ARHQ LoRA factors and residual weights
  eval_quantized.py   # Simulated nvfp4 + LoRA inference
  lowrank.py          # ARHQ objective, decomposition, and SNR utilities
  quant.py            # nvfp4 quantization simulation
  transforms.py       # Optional smoothing transforms
  data.py             # Lightweight evaluation data loader
  eval_utils.py       # Generation result helpers

scripts/
  01_build_calibration.sh
  02_extract_lora.sh
  03_eval_nvfp4_lora.sh
  04_eval_single.sh

```

The implementation also includes a reproduced SVDQuant-style baseline for comparison experiments. ARHQ remains the main method in this repository.

## Workflow

### 1. Prepare Calibration Data

Place the calibration data source under:

```text
data/calib_data/
```

Then collect layer-wise activation and weight tensors:

```bash
bash scripts/01_build_calibration.sh cuda:0 0-35 all 0-127 30000
```

The extracted calibration tensors are saved to:

```text
data/calib_tensor/
```

### 2. Extract ARHQ LoRA Factors

Run ARHQ decomposition with rank 128:

```bash
bash scripts/02_extract_lora.sh cuda:0 0-35 128 all
```

The decomposition artifacts are saved under:

```text
results/layer_results/
```

Each layer projection stores the LoRA factors, the residual weight, and optional smoothing metadata. The same entry point can also run the reproduced SVDQuant baseline for side-by-side SNR comparison.

### 3. Simulate nvfp4 + LoRA Inference

Run evaluation with the extracted ARHQ factors. In our current experiments, we evaluate ARHQ on ZebraLogic:

```bash
bash scripts/03_eval_nvfp4_lora.sh arhq smoothing 128 ZebraLogic cuda:0 all
```

For a single prompt:

```bash
QUESTION="What is 2+2? Put the final answer in \\boxed{}." \
bash scripts/04_eval_single.sh arhq smoothing 128 cuda:0 all
```

## Python Entry Points

The shell scripts are thin wrappers around the Python modules:

```bash
python -m arhq.calibration \
  --model_path /path/to/model \
  --result_dir data/calib_data \
  --output_dir data/calib_tensor \
  --layers 0-35 \
  --module_set all
```

```bash
python -m arhq.decompose \
  --calib_dir data/calib_tensor \
  --output_dir results/layer_results \
  --layers 0-35 \
  --module_set all \
  --rank 128 \
  --configs arhq:raw,arhq:smoothing,svdquant:smoothing
```

```bash
python -m arhq.eval_quantized \
  --model_path /path/to/model \
  --decomp_dir results/layer_results \
  --method arhq \
  --setting smoothing \
  --rank 128 \
  --module_set all \
  --datasets ZebraLogic
```

## Notes

`method=arhq` is the proposed method. `method=r_only` is kept only as a backward-compatible alias for older experiment artifacts.

`method=svdquant` is a reproduced comparison baseline.

The current inference path is a simulation of nvfp4 quantization rather than a fused hardware kernel implementation.
