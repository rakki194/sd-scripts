import torch
from torch.optim import Optimizer
from typing import Callable, Optional, Tuple, List, Union

class SAVEUS(Optimizer):
    r"""
    Implements the SAVEUS optimization algorithm, incorporating ADOPT's advanced techniques.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional):
            Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional):
            Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.9999)).
        eps (float, optional):
            Term added to the denominator to improve numerical stability (default: 1e-6).
        amp_fac (float, optional):
            Amplification factor for the first moment filter (default: 2.0).
        weight_decay (float, optional):
            Weight decay (L2 penalty) (default: 0.1).
        centralization (float, optional):
            Center model gradient (default: 0.5).
        normalization (float, optional):
            Alpha for normalized gradient interpolation (default: 0.5).
        normalize_channels (bool, optional):
            Whether to perform channel-wise normalization (default: True).
        clip_lambda (Optional[Callable[[int], float]], optional):
            Lambda function to compute gradient clipping threshold based on the step (default: lambda step: step**0.25).
        decouple (bool, optional):
            Whether to decouple weight decay from the gradient update (default: False).
        maximize (bool, optional):
            Whether to maximize the objective function (default: False).
        capturable (bool, optional):
            Whether the optimizer is capturable for CUDA graph support (default: False).
        differentiable (bool, optional):
            Whether the optimizer supports differentiable optimization (default: False).
        fused (Optional[bool], optional):
            Whether to use fused kernels for optimization (default: None).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.9999),
        eps=1e-8,
        amp_fac=2.0,
        weight_decay=0.1,
        centralization=0.5,
        normalization=0.5,
        normalize_channels=True,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        decouple: bool = False,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, torch.Tensor):
            if fused and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and fused=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be a single element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.amp_fac = amp_fac
        self.clip_lambda = clip_lambda
        self.decouple = decouple

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            normalization=normalization,
            normalize_channels=normalize_channels,
            decouple=decouple,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super(SAVEUS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("decouple", False)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)

    def normalize_gradient(
        self,
        x: torch.Tensor,
        use_channels: bool = False,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        r"""Normalize gradient with standard deviation.
        
        Args:
            x (torch.Tensor): Gradient tensor.
            use_channels (bool): Whether to perform channel-wise normalization.
            alpha (float): Interpolation factor between original and normalized gradient.
            epsilon (float): Small value to prevent division by zero.
        """
        size: int = x.dim()
        if size > 1 and use_channels:
            s = x.std(dim=tuple(range(1, size)), keepdim=True).add_(epsilon)
            x.lerp_(x.div_(s), weight=alpha)
        elif torch.numel(x) > 2:
            s = x.std().add_(epsilon)
            x.lerp_(x.div_(s), weight=alpha)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss: The loss evaluated by the closure, if provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("SAVEUS does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data)
                    state["ema_squared"] = torch.zeros_like(p.data)

                ema, ema_squared = state["ema"], state["ema_squared"]
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                normalization = group["normalization"]
                normalize_channels = group["normalize_channels"]
                decouple = group["decouple"]
                maximize = group["maximize"]
                state["step"] += 1

                # Center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # Normalize the gradient
                if normalization != 0:
                    self.normalize_gradient(
                        grad, use_channels=normalize_channels, alpha=normalization
                    )

                # Bias correction
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** 0.5
                step_size = lr / bias_correction

                # Decay the first and second moment running averages
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad.add_(ema, alpha=self.amp_fac)
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    if decouple:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                # Gradient clipping
                if self.clip_lambda is not None:
                    clip = self.clip_lambda(state["step"])
                    grad.clamp_(-clip, clip)

                p.data.addcdiv_(grad, denom, value=-step_size)

        return loss
