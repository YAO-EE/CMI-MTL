import torch
from functools import partial
from typing import Callable

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.cuda.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd
else:
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd  # type: ignore[attr-defined]

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
