"""
Definitions for custom optimizers.

The SGDG and AdamG optimizers (and related utilities) are copied from the following
repository: https://github.com/MinhyungCho/riemannian-batch-normalization-pytorch
"""

from typing import Iterable, Union, Dict, Callable

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, _RequiredParameter, required

from meta.networks import MLPNetwork, ConvNetwork, ResNetwork
from meta.networks.utils import ResNetBasicBlock


class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by SGD-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        grassmann=False,
        omega=0,
        grad_clip=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            grassmann=grassmann,
            omega=0,
            grad_clip=grad_clip,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            grassmann = group["grassmann"]

            if grassmann:
                grad_clip = group["grad_clip"]
                omega = group["omega"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    unity, _ = unit(p.data.view(p.size()[0], -1))
                    g = p.grad.data.view(p.size()[0], -1)

                    if omega != 0:
                        # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                        # dL/dY=2(YY'Y-Y)
                        g.add_(
                            2 * omega,
                            torch.mm(torch.mm(unity, unity.t()), unity) - unity,
                        )

                    h = gproj(unity, g)

                    if grad_clip is not None:
                        h_hat = clip_by_norm(h, grad_clip)
                    else:
                        h_hat = h

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(h_hat.size())
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state[
                                "momentum_buffer"
                            ].cuda()

                    mom = param_state["momentum_buffer"]
                    mom_new = momentum * mom - group["lr"] * h_hat

                    p.data.copy_(gexp(unity, mom_new).view(p.size()))
                    mom.copy_(gpt(unity, mom_new))

            else:
                # This routine is from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                weight_decay = group["weight_decay"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group["lr"], d_p)

        return loss


class AdamG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by Adam-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use Adam-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        beta2 (float, optional): the exponential decay rate for the second moment estimates (defulat: 0.99)
        epsilon (float, optional): a small constant for numerical stability (default: 1e-8)
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        grassmann=False,
        beta2=0.99,
        epsilon=1e-8,
        omega=0,
        grad_clip=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            grassmann=grassmann,
            beta2=beta2,
            epsilon=epsilon,
            omega=0,
            grad_clip=grad_clip,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            grassmann = group["grassmann"]

            if grassmann:
                beta1 = group["momentum"]
                beta2 = group["beta2"]
                epsilon = group["epsilon"]
                grad_clip = group["grad_clip"]
                omega = group["omega"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    unity, _ = unit(p.data.view(p.size()[0], -1))
                    g = p.grad.data.view(p.size()[0], -1)

                    if omega != 0:
                        # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                        # dL/dY=2(YY'Y-Y)
                        g.add_(
                            2 * omega,
                            torch.mm(torch.mm(unity, unity.t()), unity) - unity,
                        )

                    h = gproj(unity, g)

                    if grad_clip is not None:
                        h_hat = clip_by_norm(h, grad_clip)
                    else:
                        h_hat = h

                    param_state = self.state[p]
                    if "m_buffer" not in param_state:
                        size = p.size()
                        param_state["m_buffer"] = torch.zeros(
                            [size[0], int(np.prod(size[1:]))]
                        )
                        param_state["v_buffer"] = torch.zeros([size[0], 1])
                        if p.is_cuda:
                            param_state["m_buffer"] = param_state["m_buffer"].cuda()
                            param_state["v_buffer"] = param_state["v_buffer"].cuda()

                        param_state["beta1_power"] = beta1
                        param_state["beta2_power"] = beta2

                    m = param_state["m_buffer"]
                    v = param_state["v_buffer"]
                    beta1_power = param_state["beta1_power"]
                    beta2_power = param_state["beta2_power"]

                    mnew = beta1 * m + (1.0 - beta1) * h_hat
                    vnew = beta2 * v + (1.0 - beta2) * xTy(h_hat, h_hat)

                    alpha = np.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
                    deltas = mnew / vnew.add(epsilon).sqrt()
                    deltas.mul_(-alpha * group["lr"])

                    p.data.copy_(gexp(unity, deltas).view(p.size()))
                    m.copy_(gpt2(unity, mnew, deltas))
                    v.copy_(vnew)

                    param_state["beta1_power"] *= beta1
                    param_state["beta2_power"] *= beta2
            else:
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group["lr"], d_p)

        return loss


def norm(v):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=1, keepdim=True)


def unit(v, eps=1e-8):
    vnorm = norm(v)
    return v / vnorm.add(eps), vnorm


def xTy(x, y):
    assert len(x.size()) == 2 and len(y.size()) == 2, "xTy"
    return torch.sum(x * y, dim=1, keepdim=True)


def clip_by_norm(v, clip_norm):
    v_norm = norm(v)
    if v.is_cuda:
        scale = torch.ones(v_norm.size()).cuda()
    else:
        scale = torch.ones(v_norm.size())
    mask = v_norm > clip_norm
    scale[mask] = clip_norm / v_norm[mask]

    return v * scale


def gproj(y, g, normalize=False):
    if normalize:
        y, _ = unit(y)

    yTg = xTy(y, g)
    return g - (yTg * y)


def gexp(y, h, normalize=False):
    if normalize:
        y, _ = unit(y)
        h = gproj(y, h)

    u, hnorm = unit(h)
    return y * hnorm.cos() + u * hnorm.sin()


# parallel translation of tangent vector h1 toward h2
# both h1 and h2 are targent vector on y
def gpt2(y, h1, h2, normalize=False):
    if normalize:
        h1 = gproj(y, h1)
        h2 = gproj(y, h2)

    # h2 = u * sigma  svd of h2
    [u, unorm] = unit(h2)
    uTh1 = xTy(u, h1)
    return h1 - uTh1 * (unorm.sin() * y + (1 - unorm.cos()) * u)


# parallel translation if h1=h2
def gpt(y, h, normalize=False):
    if normalize:
        h = gproj(y, h)

    [u, unorm] = unit(h)
    return (u * unorm.cos() - y * unorm.sin()) * unorm


class PSISGD(Optimizer):
    """
    Optimizer to execute the PSI-SGD algorithm from: https://arxiv.org/abs/2101.02916
    """

    def __init__(
        self,
        params: Union[Iterable[nn.Parameter], Iterable[Dict]],
        lr: Union[float, _RequiredParameter] = required,
        momentum: float = 0,
        PSI: bool = False,
        dampening: float = 0,
    ):
        """
        Init function for PSISGD.

        Arguments
        ---------
            params : Iterable
                Iterable of parameters to optimize or dicts defining parameter groups.
                When PSI=True, the parameters of each layer should be given in a
                separate param group, so that the weight and bias of a single layer are
                the only parameters in their group.
            lr : float
                Learning rate.
            momentum : float
                Momentum factor. Default: 0.
            PSI : bool
                Whether to use PSI-SGD or regular SGD. Default: False.
            dampening : float
                Dampening constant for momentum. Default: 0.
        """
        defaults = dict(lr=lr, momentum=momentum, PSI=PSI, dampening=dampening)
        super(PSISGD, self).__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments
        ---------
            closure : Callable
                A closure that reevaluates the model and returns the loss.
        """

        # Evaluate loss, if necessary.
        loss = None
        if closure is not None:
            loss = closure()

        # Update each parameter group.
        for group in self.param_groups:
            momentum = group["momentum"]
            PSI = group["PSI"]

            if PSI:

                # Check that parameter group consists of only a weight (fully connected
                # or convolutional) and a bias.
                params = list(group["params"])
                assert len(params) == 2
                assert len(params[0].shape) in [2, 4]
                assert len(params[1].shape) == 1
                assert params[0].shape[0] == params[1].shape[0]

                # Compute norms of parameter vector for each neuron.
                reshaped_params = []
                if len(params[0].shape) == 2:
                    reshaped_params.append(params[0])
                else:
                    reshaped_params.append(params[0].view(params[0].shape[0], -1))
                reshaped_params.append(params[1].unsqueeze(-1))
                neuron_params = torch.cat(reshaped_params, dim=-1)
                neuron_norms = torch.sum(neuron_params ** 2, dim=-1)
                num_neurons = neuron_norms.shape[0]

                # Update each parameter with the PSI update.
                for p in params:
                    if p.grad is None:
                        continue

                    # Compute single-step update.
                    if len(p.shape) == 2:
                        reshaped_norms = neuron_norms.view(num_neurons, 1)
                    elif len(p.shape) == 4:
                        reshaped_norms = neuron_norms.view(num_neurons, 1, 1, 1)
                    elif len(p.shape) == 1:
                        reshaped_norms = neuron_norms
                    else:
                        raise NotImplementedError
                    d_p = p.grad.data * reshaped_norms

                    # Retrieve or initialize momentum.
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(d_p.size())
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state[
                                "momentum_buffer"
                            ].cuda()

                    # Update momentum and p.
                    param_state["momentum_buffer"] = (
                        momentum * param_state["momentum_buffer"] - group["lr"] * d_p
                    )
                    p.data.add(param_state["momentum_buffer"])

            else:

                # This routine is from
                # https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                dampening = group["dampening"]
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    d_p = p.grad.data
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        d_p = buf

                    p.data.add_(-group["lr"], d_p)

        return loss


def get_grassmann_optimizer(
    network: nn.Module,
    opt_type: str,
    lr: float,
    momentum: float = None,
    grad_clip: float = None,
) -> Union[SGDG, AdamG]:
    """
    Returns an optimizer on the Grassmann manifold (either SGDG or AdamG) for the
    parameters in `network`.
    """

    # Check for a valid optimizer type.
    if opt_type not in ["sgdg", "adamg"]:
        raise ValueError(f"Unrecognized Grassmann optimizer: '{opt_type}'.")

    # Check that network uses batch normalization.
    if not hasattr(network, "batch_norm") or not network.batch_norm:
        raise ValueError(
            "Riemannian optimizers should only be used for networks that "
            " use batch normalization."
        )

    # The Riemannian optimizer is only used on parameters in layers which
    # precede a batch normalization layer. Collecting these parameters is only
    # supported for certain architectures.
    pre_bn_param_names = []
    pre_bn_params = []
    if isinstance(network, ConvNetwork):

        for i, layer in enumerate(network.conv):
            assert isinstance(layer[0], nn.Conv2d)
            assert isinstance(layer[1], nn.BatchNorm2d)
            for name, p in layer[0].named_parameters():
                full_name = f"conv.{i}.0.{name}"
                pre_bn_param_names.append(full_name)
                pre_bn_params.append(p)
        for i, layer in enumerate(network.fc[:-1]):
            assert isinstance(layer[0], nn.Linear)
            assert isinstance(layer[1], nn.BatchNorm1d)
            for name, p in layer[0].named_parameters():
                full_name = f"fc.{i}.0.{name}"
                pre_bn_param_names.append(full_name)
                pre_bn_params.append(p)

    elif isinstance(network, MLPNetwork):

        for i, layer in enumerate(network.layers):
            if i == len(network.layers) - 1:
                continue
            assert isinstance(layer[0], nn.Linear)
            assert isinstance(layer[1], nn.BatchNorm1d)
            for name, p in layer[0].named_parameters():
                full_name = f"layers.{i}.0.{name}"
                pre_bn_param_names.append(full_name)
                pre_bn_params.append(p)

    elif isinstance(network, ResNetwork):

        for i, layer in enumerate(network.backbone):
            assert isinstance(layer, ResNetBasicBlock)
            assert layer.batch_norm
            conv_layers = {"conv1": layer.conv1, "conv2": layer.conv2}
            for conv_name, conv in conv_layers.items():
                for name, p in conv.named_parameters():
                    full_name = f"backbone.{i}.{conv_name}.{name}"
                    pre_bn_param_names.append(full_name)
                    pre_bn_params.append(p)

            if layer.downsample is not None:
                assert isinstance(layer.downsample[0], nn.Conv2d)
                assert isinstance(layer.downsample[1], nn.BatchNorm2d)
                for name, p in layer.downsample[0].named_parameters():
                    full_name = f"backbone.{i}.downsample.0.{name}"
                    pre_bn_param_names.append(full_name)
                    pre_bn_params.append(p)

    else:
        raise NotImplementedError
    pre_bn_params = [p for p in pre_bn_params if p.requires_grad]
    other_params = []
    for name, p in network.named_parameters():
        if name not in pre_bn_param_names and p.requires_grad:
            other_params.append(p)

    # Scale all of the pre-BatchNorm parameters to unit norm.
    for i, p in enumerate(pre_bn_params):
        unitp, _ = unit(p.data.view(p.shape[0], -1))
        pre_bn_params[i].data.copy_(unitp.view(p.size()))

    # Initialize optimizer.
    if opt_type == "sgdg":
        optimizer_cls = SGDG
    elif opt_type == "adamg":
        optimizer_cls = AdamG
    else:
        raise NotImplementedError
    return optimizer_cls(
        [
            {
                "params": pre_bn_params,
                "lr": lr,
                "momentum": momentum,
                "grassmann": True,
                "grad_clip": grad_clip,
            },
            {
                "params": other_params,
                "lr": lr,
                "momentum": momentum,
                "grassmann": False,
            },
        ]
    )


def get_PSI_optimizer(network: nn.Module, lr: float, momentum: float) -> PSISGD:
    """ Returns a PSI optimizer for the parameters in `network`. """

    # Check that network uses batch normalization.
    if not hasattr(network, "batch_norm") or not network.batch_norm:
        raise ValueError(
            "PSI optimizer should only be used for networks that use batch"
            " normalization."
        )

    # The PSI optimizer is only used on parameters in layers which precede a
    # batch normalization layer. Collecting these parameters is only supported
    # for certain architectures. In addition, each layer's parameters (weight
    # and bias) must be in their own separate param group.
    param_groups = []
    pre_bn_param_names = []
    if isinstance(network, MLPNetwork):

        for i, layer in enumerate(network.layers):

            # Batch normalization isn't applied in last layer, so no need to use
            # PSI optimizer.
            if i == len(network.layers) - 1:
                continue

            # Check that layer structure matches our expectations.
            assert isinstance(layer[0], nn.Linear)
            assert isinstance(layer[1], nn.BatchNorm1d)

            # Collect parameter names.
            for name, p in layer[0].named_parameters():
                full_name = f"layers.{i}.0.{name}"
                pre_bn_param_names.append(full_name)

            # Add parameter group for layer.
            param_groups.append(
                {
                    "params": [p for p in layer[0].parameters() if p.requires_grad],
                    "lr": lr,
                    "momentum": momentum,
                    "PSI": True,
                }
            )

    elif isinstance(network, ConvNetwork):

        for i, layer in enumerate(network.conv):

            # Check that layer structure matches our expectations.
            if not (
                isinstance(layer[0], nn.Conv2d) and isinstance(layer[1], nn.BatchNorm2d)
            ):
                raise ValueError(
                    f"Conv layer {i} of network with type {type(network)} doesn't match"
                    " expected architecture. The given architecture is currently"
                    " unsupported by the PSI optimizer."
                )

            # Collect parameter names.
            for name, p in layer[0].named_parameters():
                full_name = f"conv.{i}.0.{name}"
                pre_bn_param_names.append(full_name)

            # Add parameter group for layer.
            param_groups.append(
                {
                    "params": [p for p in layer[0].parameters() if p.requires_grad],
                    "lr": lr,
                    "momentum": momentum,
                    "PSI": True,
                }
            )

        for i, layer in enumerate(network.fc):

            # Batch normalization isn't applied in last layer, so no need to use
            # PSI optimizer.
            if i == len(network.fc) - 1:
                continue

            # Check that layer structure matches our expectations.
            if not (
                isinstance(layer[0], nn.Linear) and isinstance(layer[1], nn.BatchNorm1d)
            ):
                raise ValueError(
                    f"FC layer {i} of network with type {type(network)} doesn't match "
                    "expected architecture. The given architecture is currently "
                    "unsupported by the PSI optimizer."
                )

            # Collect parameter names.
            for name, p in layer[0].named_parameters():
                full_name = f"fc.{i}.0.{name}"
                pre_bn_param_names.append(full_name)

            # Add parameter group for layer.
            param_groups.append(
                {
                    "params": [p for p in layer[0].parameters() if p.requires_grad],
                    "lr": lr,
                    "momentum": momentum,
                    "PSI": True,
                }
            )

    elif isinstance(network, ResNetwork):

        for i, layer in enumerate(network.backbone):

            # Check that layer structure matches our expectations.
            assert isinstance(layer, ResNetBasicBlock)
            if layer.downsample is not None:
                assert isinstance(layer.downsample[0], nn.Conv2d)
                assert isinstance(layer.downsample[1], nn.BatchNorm2d)

            # Collect parameter names and add a parameter group for each convolutional
            # layer.
            conv_layers = {"conv1": layer.conv1, "conv2": layer.conv2}
            for conv_name, conv in conv_layers.items():
                for name, p in conv.named_parameters():
                    full_name = f"backbone.{i}.{conv_name}.{name}"
                    pre_bn_param_names.append(full_name)

                param_groups.append(
                    {
                        "params": [p for p in conv.parameters() if p.requires_grad],
                        "lr": lr,
                        "momentum": momentum,
                        "PSI": True,
                    }
                )

            if layer.downsample is not None:
                for name, p in layer.downsample[0].named_parameters():
                    full_name = f"backbone.{i}.downsample.0.{name}"
                    pre_bn_param_names.append(full_name)

                param_groups.append(
                    {
                        "params": [
                            p
                            for p in layer.downsample[0].parameters()
                            if p.requires_grad
                        ],
                        "lr": lr,
                        "momentum": momentum,
                        "PSI": True,
                    }
                )

    else:
        raise NotImplementedError

    # Add parameter group for parameters optimized with regular SGD.
    other_params = []
    for name, p in network.named_parameters():
        if name not in pre_bn_param_names and p.requires_grad:
            other_params.append(p)
    param_groups.append(
        {"params": other_params, "lr": lr, "momentum": momentum, "PSI": False}
    )

    # Initialize optimizer.
    return PSISGD(param_groups)
