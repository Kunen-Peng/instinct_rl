"""Reusable cost-critic modules for constrained RL."""

from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

from .actor_critic import get_activation


def _append_mlp_layers(layers, input_dim, hidden_dims, output_dim, activation_name, output_activation):
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(get_activation(activation_name))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())


def _init_linear_layers(module, output_gain):
    linear_layers = [submodule for submodule in module.modules() if isinstance(submodule, nn.Linear)]
    if not linear_layers:
        return

    for linear in linear_layers[:-1]:
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.constant_(linear.bias, 0.0)

    output_layer = linear_layers[-1]
    nn.init.orthogonal_(output_layer.weight, gain=output_gain)
    nn.init.constant_(output_layer.bias, 0.0)


def resolve_multi_head_dims(
    default_hidden_dims: List[int],
    backbone_hidden_dims: Optional[List[int]],
    head_hidden_dims: Optional[List[int]],
) -> Tuple[List[int], List[int]]:
    if backbone_hidden_dims is not None or head_hidden_dims is not None:
        return list(backbone_hidden_dims or []), list(head_hidden_dims or [])

    if len(default_hidden_dims) <= 1:
        return [], list(default_hidden_dims)

    return list(default_hidden_dims[:-1]), [default_hidden_dims[-1]]


class VectorCostCritic(nn.Module):
    """Single MLP cost critic that outputs all cost values jointly."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_costs: int,
        activation_name: str,
        output_activation: Optional[Type[nn.Module]] = None,
        output_gain: float = 0.01,
    ):
        super().__init__()
        layers = []
        _append_mlp_layers(
            layers,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_costs,
            activation_name=activation_name,
            output_activation=output_activation,
        )
        self.model = nn.Sequential(*layers)
        _init_linear_layers(self.model, output_gain=output_gain)

    def forward(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.model(critic_observations)


class MultiHeadCostCritic(nn.Module):
    """Shared cost backbone followed by one independent head per constraint."""

    def __init__(
        self,
        input_dim: int,
        num_costs: int,
        activation_name: str,
        backbone_hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        output_activation: Optional[Type[nn.Module]] = None,
        output_gain: float = 0.01,
    ):
        super().__init__()
        backbone_hidden_dims = list(backbone_hidden_dims or [])
        head_hidden_dims = list(head_hidden_dims or [])

        if backbone_hidden_dims:
            backbone_layers = []
            prev_dim = input_dim
            for hidden_dim in backbone_hidden_dims:
                backbone_layers.append(nn.Linear(prev_dim, hidden_dim))
                backbone_layers.append(get_activation(activation_name))
                prev_dim = hidden_dim
            self.backbone = nn.Sequential(*backbone_layers)
            _init_linear_layers(self.backbone, output_gain=np.sqrt(2))
            head_input_dim = backbone_hidden_dims[-1]
        else:
            self.backbone = nn.Identity()
            head_input_dim = input_dim

        self.heads = nn.ModuleList()
        for _ in range(num_costs):
            head_layers = []
            _append_mlp_layers(
                head_layers,
                input_dim=head_input_dim,
                hidden_dims=head_hidden_dims,
                output_dim=1,
                activation_name=activation_name,
                output_activation=output_activation,
            )
            head = nn.Sequential(*head_layers)
            _init_linear_layers(head, output_gain=output_gain)
            self.heads.append(head)

    def forward(self, critic_observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(critic_observations)
        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=-1)
