# Soft-MSM

Reference code for the research paper **"Soft-MSM: Differentiable Elastic Time Series Alignment with Context-Aware Transition Costs"**.

This repository provides differentiable implementations of Soft-MSM and Soft-DTW for PyTorch, JAX, and TensorFlow, with tests against the aeon/Numba reference implementations. The goal is to make elastic time-series alignment usable inside gradient-based learning pipelines while preserving numerical equivalence to the reference dynamic programs.

## Overview

Soft-MSM is a differentiable relaxation of the Move-Split-Merge (MSM) distance. It replaces hard minimum decisions in the dynamic program with smooth soft-min operations and uses context-aware transition costs for split and merge moves.

The repository includes:

- Soft-MSM loss, alignment matrix, and gradient helpers.
- Soft-DTW loss, alignment matrix, and gradient helpers for comparison.
- PyTorch, JAX, and TensorFlow implementations.
- CPU, CUDA, and Apple MPS-aware PyTorch support.
- Equivalence tests against aeon's soft elastic-distance reference code.
- Experiment utilities for averaging and forecasting workflows.

## Support Matrix

| Backend | Soft-MSM | Soft-DTW | Autograd | Notes |
| --- | --- | --- | --- | --- |
| PyTorch | Yes | Yes | Yes | CPU, CUDA, and MPS tested where available. |
| JAX | Yes | Yes | Yes | Enable `jax_enable_x64` for closest aeon parity. |
| TensorFlow | Yes | Yes | Yes | Uses `tf.GradientTape`. |
| aeon/Numba | Reference | Reference | Helper exports | Used for equivalence checks. |

Soft-DTW supports multivariate inputs by summing squared local distances over channels. Soft-MSM currently follows aeon's univariate MSM convention: inputs may be shaped `(B, C, T)`, but MSM uses channel 0 only, and gradients are non-zero only for channel 0.

## Installation

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv venv
source ./venv/bin/activate
pip install -e .
```

Install backend extras as needed:

```bash
pip install -e ".[pytorch]"
pip install -e ".[jax]"
pip install -e ".[tensorflow]"
pip install -e ".[all]"
```

For development:

```bash
pip install -e ".[all,dev]"
pre-commit install
```

## PyTorch Example

```python
import torch

from soft_msm.torch import SoftMSMLoss, soft_msm_alignment_matrix, soft_msm_grad_x

x = torch.randn(8, 1, 64, requires_grad=True)
y = torch.randn(8, 1, 72)

loss_fn = SoftMSMLoss(c=1.0, gamma=0.1, reduction="mean")
loss = loss_fn(x, y)
loss.backward()

E, costs = soft_msm_alignment_matrix(x, y, c=1.0, gamma=0.1)
dx, costs = soft_msm_grad_x(x, y, c=1.0, gamma=0.1)
```

Soft-DTW uses the same style of API:

```python
from soft_msm.torch import SoftDTWLoss, soft_dtw_alignment_matrix, soft_dtw_grad_x

loss = SoftDTWLoss(gamma=0.1)(x, y)
E, costs = soft_dtw_alignment_matrix(x, y, gamma=0.1)
dx, costs = soft_dtw_grad_x(x, y, gamma=0.1)
```

## JAX Example

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from soft_msm.jax import soft_msm_loss, soft_msm_alignment_matrix, soft_msm_grad_x

x = jnp.ones((8, 1, 64))
y = jnp.ones((8, 1, 72))

loss = soft_msm_loss(x, y, c=1.0, gamma=0.1, reduction="mean")
grad = jax.grad(lambda x_: soft_msm_loss(x_, y, c=1.0, gamma=0.1))(x)

E, costs = soft_msm_alignment_matrix(x, y, c=1.0, gamma=0.1)
dx, costs = soft_msm_grad_x(x, y, c=1.0, gamma=0.1)
```

## TensorFlow Example

```python
import tensorflow as tf

from soft_msm.tensorflow import SoftMSMLoss, soft_msm_alignment_matrix, soft_msm_grad_x

x = tf.Variable(tf.random.normal((8, 1, 64)))
y = tf.random.normal((8, 1, 72))

loss_fn = SoftMSMLoss(c=1.0, gamma=0.1, reduction="mean")

with tf.GradientTape() as tape:
    loss = loss_fn(x, y)

grad = tape.gradient(loss, x)
E, costs = soft_msm_alignment_matrix(x, y, c=1.0, gamma=0.1)
dx, costs = soft_msm_grad_x(x, y, c=1.0, gamma=0.1)
```

## API Summary

PyTorch:

- `soft_msm.torch.SoftMSMLoss`
- `soft_msm.torch.soft_msm_alignment_matrix`
- `soft_msm.torch.soft_msm_grad_x`
- `soft_msm.torch.SoftDTWLoss`
- `soft_msm.torch.soft_dtw_alignment_matrix`
- `soft_msm.torch.soft_dtw_grad_x`

JAX:

- `soft_msm.jax.soft_msm_loss`
- `soft_msm.jax.soft_msm_alignment_matrix`
- `soft_msm.jax.soft_msm_grad_x`
- `soft_msm.jax.soft_dtw_loss`
- `soft_msm.jax.soft_dtw_alignment_matrix`
- `soft_msm.jax.soft_dtw_grad_x`

TensorFlow:

- `soft_msm.tensorflow.SoftMSMLoss`
- `soft_msm.tensorflow.soft_msm_alignment_matrix`
- `soft_msm.tensorflow.soft_msm_grad_x`
- `soft_msm.tensorflow.SoftDTWLoss`
- `soft_msm.tensorflow.soft_dtw_alignment_matrix`
- `soft_msm.tensorflow.soft_dtw_grad_x`

## Equivalence And Testing

The test suite checks distance values, expected alignment matrices, and gradients against aeon reference implementations across gamma and cost grids.

Run all tests:

```bash
source ./venv/bin/activate
pytest soft_msm
```

Run pre-commit checks:

```bash
pre-commit run --all-files
```

Current expected result in the development environment:

- `pytest soft_msm`: 608 tests passing.
- `pre-commit run --all-files`: passing.

## Notes On Precision And Devices

- PyTorch CPU/CUDA paths preserve float64 parity when float64 inputs are provided.
- PyTorch MPS does not support float64 tensors, so scalar helper outputs are returned in the input dtype on MPS while values are cross-checked against CPU float64 references where needed.
- JAX defaults may not enable float64; set `jax.config.update("jax_enable_x64", True)` before constructing arrays when exact reference parity is required.
- TensorFlow alignment and gradient helpers internally use float64 casts for reference-level numerical agreement.

## Repository Layout

```text
soft_msm/
  torch/         PyTorch implementations and tests
  jax/           JAX implementations and tests
  tensorflow/    TensorFlow implementations and tests
  numba/         aeon/Numba reference re-exports
  experiments/   experiment runners and utilities
  evaluation/    result summarisation helpers
```

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{softmsm,
  title = {Soft-MSM: Differentiable Elastic Time Series Alignment with Context-Aware Transition Costs},
  author = {Holder, Christopher},
  year = {2026},
  note = {Manuscript in preparation}
}
```

Update the BibTeX entry with the final publication venue and metadata when available.
