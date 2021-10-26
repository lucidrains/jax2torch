## jax2torch

Use Jax functions in Pytorch with DLPack, as outlined <a href="https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9">in a gist</a> by <a href="https://github.com/mattjj">@mattjj</a>. The repository was made for the purposes of making the <a href="https://github.com/spetti/SMURF">differentiable alignment</a> work here interoperable with Pytorch.

## Install

```bash
$ pip install jax2torch
```

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GBEEnpuCvLS1bhb_xGCO5Y40rFiQrh6G?usp=sharing) Quick test

```python
import jax
import torch
from jax2torch import jax2torch

# Jax function

@jax.jit
def jax_pow(x, y = 2):
  return x ** y

# convert to Torch function

torch_pow = jax2torch(jax_pow)

# run it on Torch data!

x = torch.tensor([1., 2., 3.])
y = torch_pow(x, y = 3)
print(y)  # tensor([1., 8., 27.])

# And differentiate!

x = torch.tensor([2., 3.], requires_grad = True)
y = torch.sum(torch_pow(x, y = 3))
y.backward()
print(x.grad) # tensor([12., 27.])
```
