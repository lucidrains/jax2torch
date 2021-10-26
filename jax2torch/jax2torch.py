# https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9

import torch
from torch.utils import dlpack as torch_dlpack

import jax
from jax import dlpack as jax_dlpack
import jax.numpy as jnp
from jax.tree_util import tree_map

from inspect import signature
from functools import wraps

def j2t(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

def t2j(x_torch):
    x_torch = x_torch.contiguous()
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax

def tree_t2j(x_torch):
    return tree_map(lambda t: t2j(t) if isinstance(t, torch.Tensor) else t, x_torch)

def tree_j2t(x_jax):
    return tree_map(lambda t: j2t(t) if isinstance(t, jnp.ndarray) else t, x_jax)

def jax2torch(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        class JaxFun(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                ctx.num_args = len(args)
                args = tree_t2j(args)
                y_, ctx.fun_vjp = jax.vjp(fn, *args)
                return tree_j2t(y_)

            @staticmethod
            def backward(ctx, *grad_args):
                if len(grad_args) > 1:
                    grad_args = tree_t2j(grad_args)
                else:
                    grad_args = t2j(grad_args[0])
                grads, *_ = ctx.fun_vjp(grad_args)
                ret = tree_j2t(grads), *((None,) * (ctx.num_args - 1))
                return ret

        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return JaxFun.apply(*bound.arguments.values())
    return inner
