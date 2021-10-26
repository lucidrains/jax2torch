## jax2torch

Use Jax functions in Pytorch with DLPack, as outlined <a href="https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9">in a gist</a> by <a href="https://github.com/mattjj">@mattjj</a>. Right now only supports one tensor input (with optional non-tensor input arguments) to one tensor output, for the purposes of <a href="https://github.com/spetti/SMURF">differentiable alignment</a>.

## Install

```bash
$ pip install jax2torch
```

## Usage

```python
import jax
import torch
from jax2torch import jax2torch

# Jax function

@jax.jit
def jax_pow(x, y=2):
  return x ** y

# convert to Torch function

torch_pow = jax2torch(jax_pow)

# run it on Torch data!

x = torch.tensor([1., 2., 3.])
y = torch_pow(x, 3)
print(y)  # tensor([1., 8., 27.])

# And differentiate!

x = torch.tensor([2., 3.], requires_grad=True)
y = torch.sum(torch_pow(x, 3))
y.backward()
print(x.grad) # tensor([12., 27.])
```
