# meta_module_extended.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple, Any, Optional

def to_var(x: torch.Tensor, requires_grad: bool = True, device: Union[str, torch.device] = 'cuda:0') -> torch.Tensor:
    """
    A utility function to ensure a tensor is on the correct device and has the
    correct requires_grad status. It properly creates a new leaf node if needed.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
    target_device = torch.device(device)
    x = x.to(target_device)
    if x.requires_grad != requires_grad:
        return x.clone().detach().requires_grad_(requires_grad)
    return x


class MetaModule(nn.Module):
    """
    The base class for all our custom modules. It inherits from nn.Module
    and provides helper methods compatible with meta-learning libraries like `higher`.
    The key is that any module inheriting from this class can be made "functional"
    by such libraries.
    """
    def meta_parameters(self, recurse: bool = True):
        """A convenience method to yield all parameters of the module."""
        for _, param in self.named_meta_parameters(recurse=recurse):
            yield param

    def named_meta_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Yields all named parameters. This method now correctly mirrors the signature
        of nn.Module.named_parameters, making it compatible with PyTorch's internal mechanisms.
        """
        # We simply delegate to the standard named_parameters method, which works correctly
        # now that all sub-layers register their parameters properly as nn.Parameter.
        for name, param in self.named_parameters(prefix=prefix, recurse=recurse):
            yield name, param

    def update_params(self, lr_inner: float, source_params=None, detach=False):
        """A utility method to manually update module parameters, used in some meta-learning algorithms."""
        if source_params is not None:
            # Update from a given list of gradients
            for param, grad in zip(self.meta_parameters(), source_params):
                if grad is not None:
                    param.data.sub_(lr_inner * grad.detach())
        else:
            # Update using the attached .grad attribute
            for param in self.meta_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    param.data.sub_(lr_inner * grad)
                elif detach:
                    param.detach_()

# --- Meta-Learning Compatible Layers ---
# Each layer below is a "meta" version of a standard PyTorch layer.
# The core principle is to inherit from MetaModule and correctly register
# weights and biases as nn.Parameter.

class MetaLinear(MetaModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        # Correctly register weight as a learnable parameter with Kaiming initialization.
        weight_data = torch.empty((out_features, in_features), **factory_kwargs)
        nn.init.kaiming_uniform_(weight_data, a=math.sqrt(5))
        self.weight = nn.Parameter(weight_data)

        if bias:
            # Correctly register bias as a learnable parameter with standard initialization.
            bias_data = torch.empty(out_features, **factory_kwargs)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias_data, -bound, bound)
            self.bias = nn.Parameter(bias_data)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use the efficient functional implementation.
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return f"MetaLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class MetaConv1d(MetaModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, str] = 0, dilation: Union[int, Tuple[int]] = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str ='zeros', device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Storing config for __repr__ and forward pass
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride, self.padding, self.dilation = stride, padding, dilation

        weight_data = torch.empty((out_channels, in_channels // groups, *self.kernel_size), **factory_kwargs)
        nn.init.kaiming_uniform_(weight_data, a=math.sqrt(5))
        self.weight = nn.Parameter(weight_data)

        if bias:
            bias_data = torch.empty(out_channels, **factory_kwargs)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias_data, -bound, bound)
            self.bias = nn.Parameter(bias_data)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        return (f"MetaConv1d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride})")


class MetaLayerNorm(MetaModule):
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def __repr__(self):
        return (f"MetaLayerNorm(normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine})")


class MetaEmbedding(MetaModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        weight_data = torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        nn.init.normal_(weight_data)
        if padding_idx is not None:
            with torch.no_grad():
                weight_data[padding_idx].fill_(0)
        self.weight = nn.Parameter(weight_data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight, self.padding_idx)

    def __repr__(self):
        return f"MetaEmbedding({self.num_embeddings}, {self.embedding_dim}, padding_idx={self.padding_idx})"