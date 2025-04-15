import torch
from torch.optim import Optimizer
from typing import Callable, Optional, Tuple
import random


# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Apply stochastic rounding by manipulating bits in the mantissa.
    Based on Lode's implementation.

    Args:
        target: Target tensor to store result in BF16 format
        source: Source tensor in FP32 format
    """
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class SPARKLES(Optimizer):
    r"""
    Implements the SPARKLES optimization algorithm: Stochastic Parameter Adjustment with Randomized Kick for Learning Enhancement Strategy.

    Based on SAVEUS, but incorporates the stochastic strategy from SADAM research to help escape local minima and saddle points.

    The optimizer combines several advanced optimization techniques:
    1. Gradient Centralization: Removes the mean of gradients for each layer:
       g_t = g_t - mean(g_t)

    2. Adaptive Gradient Normalization: Normalizes gradients using their standard deviation:
       g_t = (1 - α) * g_t + α * (g_t / std(g_t))
       where α is the normalization parameter

    3. Momentum with Amplification:
       - First moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
       - Amplified gradient: g_t = g_t + amp_fac * m_t
       where β₁ is the first moment decay rate

    4. Adaptive Step Sizes:
       - Second moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
       - Bias correction: m̂_t = m_t / (1 - β₁ᵗ)
                         v̂_t = v_t / (1 - β₂ᵗ)
       - Step size: η_t = lr / (1 - β₁ᵗ)
       where β₂ is the second moment decay rate

    5. Stochastic Update Strategy:
       - When gradients become small between iterations (||g_t - g_{t-1}|| < threshold),
         apply stochastic operator R to the updates to escape potential local minima
       - The stochastic operator randomly rearranges the gradients without changing their magnitude
       - Various permutation strategies are available: global (default), magnitude-aware, local neighborhood, and adaptive

    6. Stochastic BF16 Rounding (Enabled by default):
       - Applies controlled stochastic noise during bfloat16 conversion
       - Helps improve training dynamics by adding beneficial noise at the precision level
       - Uses bit-manipulation for true stochastic rounding

    Complete Update Rule:
    1. If decouple_weight_decay:
       θ_t = θ_{t-1} * (1 - η_t * λ) - η_t * g_t / √(v̂_t + ε)
    2. Otherwise:
       θ_t = θ_{t-1} - η_t * (g_t + λ * θ_{t-1}) / √(v̂_t + ε)

    Where:
    - θ_t: Parameters at step t
    - η_t: Learning rate with bias correction
    - g_t: Gradient (after centralization, normalization, and amplification)
    - v̂_t: Bias-corrected second moment estimate
    - λ: Weight decay coefficient
    - ε: Small constant for numerical stability

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional):
            Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional):
            Coefficients for computing running averages of gradient (β₁) and its square (β₂) (default: (0.9, 0.999)).
        eps (float, optional):
            Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional):
            Weight decay (L2 penalty) (default: 0).
        centralization (float, optional):
            Strength of gradient centralization (default: 0.5).
        normalization (float, optional):
            Interpolation factor for normalized gradients (default: 0.5).
        normalize_channels (bool, optional):
            Whether to normalize gradients channel-wise (default: True).
        amp_fac (float, optional):
            Amplification factor for the momentum term (default: 2.0).
        clip_lambda (Optional[Callable[[int], float]], optional):
            Function computing gradient clipping threshold from step number (default: step**0.25).
        decouple_weight_decay (bool, optional):
            Whether to apply weight decay directly to weights (default: False).
        clip_gradients (bool, optional):
            Whether to enable gradient clipping (default: False).
        stochastic_threshold (float, optional):
            Threshold for applying stochastic updates (default: 1e-6).
        use_stochastic_rounding (bool, optional):
            Whether to apply stochastic rounding when converting to BF16 (default: True).
        stochastic_rounding_prob (float, optional):
            Probability of applying noise in stochastic rounding (default: 0.5).
        stochastic_rounding_magnitude (float, optional):
            Magnitude of noise to apply in stochastic rounding (default: None, uses epsilon of BF16).
        use_bit_manipulation (bool, optional):
            Whether to use bit manipulation for stochastic rounding (default: True).
        permutation_strategy (str, optional):
            Which permutation strategy to use ('global', 'magnitude', 'local', 'adaptive') (default: 'global').
        magnitude_bands (int, optional):
            Number of magnitude bands for magnitude-aware shuffling (default: 5).
        local_neighborhood_size (float, optional):
            Size of local neighborhood as fraction of tensor size (default: 0.1).
        adaptive_scale_factor (float, optional):
            Scaling factor for adaptive permutation based on gradient variance (default: 1.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        centralization: float = 0.5,
        normalization: float = 0.5,
        normalize_channels: bool = True,
        amp_fac: float = 2.0,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        decouple_weight_decay: bool = False,
        clip_gradients: bool = False,
        stochastic_threshold: float = 1e-6,
        use_stochastic_rounding: bool = True,
        stochastic_rounding_prob: float = 0.5,
        stochastic_rounding_magnitude: Optional[float] = None,
        use_bit_manipulation: bool = True,
        permutation_strategy: str = "global",
        magnitude_bands: int = 5,
        local_neighborhood_size: float = 0.1,
        adaptive_scale_factor: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            normalization=normalization,
            normalize_channels=normalize_channels,
            amp_fac=amp_fac,
            clip_lambda=clip_lambda,
            decouple_weight_decay=decouple_weight_decay,
            clip_gradients=clip_gradients,
            stochastic_threshold=stochastic_threshold,
            use_stochastic_rounding=use_stochastic_rounding,
            stochastic_rounding_prob=stochastic_rounding_prob,
            stochastic_rounding_magnitude=stochastic_rounding_magnitude,
            use_bit_manipulation=use_bit_manipulation,
            permutation_strategy=permutation_strategy,
            magnitude_bands=magnitude_bands,
            local_neighborhood_size=local_neighborhood_size,
            adaptive_scale_factor=adaptive_scale_factor,
        )
        super(SPARKLES, self).__init__(params, defaults)

    def normalize_gradient(
        self,
        x: torch.Tensor,
        use_channels: bool = False,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        r"""Normalize gradient with standard deviation.
        :param x: torch.Tensor. Gradient.
        :param use_channels: bool. Channel-wise normalization.
        :param alpha: float. Interpolation weight between original and normalized gradient.
        :param epsilon: float. Small value to prevent division by zero.
        """
        size: int = x.dim()
        if size > 1 and use_channels:
            s = x.std(dim=tuple(range(1, size)), keepdim=True).add_(epsilon)
            x.lerp_(x.div_(s), weight=alpha)
        elif torch.numel(x) > 2:
            s = x.std().add_(epsilon)
            x.lerp_(x.div_(s), weight=alpha)

    def apply_global_permutation(self, x: torch.Tensor) -> None:
        """Apply global permutation - shuffles all elements randomly.

        :param x: torch.Tensor. Tensor to permute.
        """
        if x.dim() > 1:
            # For ND tensors (N>1), shuffle along first dimension
            perm_idx = torch.randperm(x.size(0), device=x.device)
            x.copy_(x[perm_idx])
        else:
            # For 1D tensors, permute all elements
            perm_idx = torch.randperm(x.numel(), device=x.device)
            x.copy_(x[perm_idx])

    def apply_magnitude_aware_permutation(
        self, x: torch.Tensor, bands: int = 5
    ) -> None:
        """Apply magnitude-aware permutation - only shuffle elements within similar magnitude bands.

        :param x: torch.Tensor. Tensor to permute.
        :param bands: int. Number of magnitude bands to create.
        """
        if x.numel() <= 1:
            return

        # Flatten tensor for easier handling
        original_shape = x.shape
        flat_x = x.view(-1)
        abs_x = torch.abs(flat_x)

        # Get min and max values for magnitude bands
        min_val = abs_x.min().item()
        max_val = abs_x.max().item()

        # Add small epsilon to prevent division by zero
        if min_val == max_val:
            min_val = min_val - 1e-8
            max_val = max_val + 1e-8

        # Create logarithmic magnitude bands
        if min_val <= 0:
            min_val = 1e-8

        log_min = torch.log10(torch.tensor(min_val))
        log_max = torch.log10(torch.tensor(max_val))
        band_width = (log_max - log_min) / bands

        # Create bands and permute within each band
        for i in range(bands):
            lower_bound = torch.pow(10, log_min + i * band_width)
            upper_bound = torch.pow(10, log_min + (i + 1) * band_width)

            # Find elements in this magnitude band
            if i == bands - 1:  # Include max value in the last band
                mask = (abs_x >= lower_bound) & (abs_x <= upper_bound)
            else:
                mask = (abs_x >= lower_bound) & (abs_x < upper_bound)

            # Skip if no elements in this band
            if not torch.any(mask):
                continue

            # Get indices and values in this band
            indices = torch.nonzero(mask).view(-1)
            if indices.numel() <= 1:
                continue

            # Permute elements within the band
            perm_indices = indices[torch.randperm(indices.numel(), device=x.device)]
            flat_x[indices] = flat_x[perm_indices]

        # Reshape back to original shape
        x.copy_(flat_x.view(original_shape))

    def apply_local_neighborhood_permutation(
        self, x: torch.Tensor, neighborhood_size: float = 0.1
    ) -> None:
        """Apply local neighborhood permutation - shuffle elements only within a local region.

        :param x: torch.Tensor. Tensor to permute.
        :param neighborhood_size: float. Size of local neighborhood as fraction of tensor size.
        """
        if x.numel() <= 1:
            return

        # Flatten tensor for easier handling
        original_shape = x.shape
        flat_x = x.view(-1)
        n = flat_x.numel()

        # Calculate actual neighborhood size
        k = max(2, int(n * neighborhood_size))

        # Create a permuted version by sliding a window
        result = flat_x.clone()

        # Process in chunks to allow for local permutations
        for i in range(0, n, k):
            # Get local chunk (handle edge case at the end)
            end_idx = min(i + k, n)
            chunk = flat_x[i:end_idx]

            # Only permute if we have multiple elements
            if chunk.numel() > 1:
                perm_idx = torch.randperm(chunk.numel(), device=x.device)
                result[i:end_idx] = chunk[perm_idx]

        # Reshape back to original shape
        x.copy_(result.view(original_shape))

    def apply_adaptive_permutation(
        self, x: torch.Tensor, scale_factor: float = 1.0
    ) -> None:
        """Apply adaptive permutation strength based on tensor variance.

        :param x: torch.Tensor. Tensor to permute.
        :param scale_factor: float. Scaling factor for permutation strength.
        """
        if x.numel() <= 1:
            return

        # Flatten tensor for easier handling
        original_shape = x.shape
        flat_x = x.view(-1)
        n = flat_x.numel()

        # Compute variance to determine permutation strength
        var = torch.var(flat_x).item()

        # Normalize variance to a reasonable range [0, 1]
        # Using sigmoid-like normalization
        norm_var = 1.0 / (1.0 + torch.exp(-scale_factor * var))

        # Determine how much of the tensor to permute based on variance
        # High variance -> less permutation, Low variance -> more permutation
        permutation_ratio = 1.0 - norm_var
        k = max(2, int(n * permutation_ratio))

        # No need to permute if permutation ratio is tiny
        if k <= 1:
            return

        # Select k random elements to permute
        indices = torch.randperm(n, device=x.device)[:k]
        values = flat_x[indices]

        # Permute just these k elements
        perm_indices = torch.randperm(k, device=x.device)
        flat_x[indices] = values[perm_indices]

        # Reshape back to original shape
        x.copy_(flat_x.view(original_shape))

    def apply_stochastic_operator(
        self,
        x: torch.Tensor,
        strategy: str = "global",
        magnitude_bands: int = 5,
        local_neighborhood_size: float = 0.1,
        adaptive_scale_factor: float = 1.0,
    ) -> None:
        r"""Apply stochastic operator R to tensor x.
        This implementation provides multiple permutation strategies.

        :param x: torch.Tensor. Tensor to apply stochastic operator to.
        :param strategy: str. One of 'global', 'magnitude', 'local', 'adaptive'
        :param magnitude_bands: int. Number of magnitude bands for magnitude-aware shuffling.
        :param local_neighborhood_size: float. Size of local neighborhood as fraction of tensor size.
        :param adaptive_scale_factor: float. Scaling factor for adaptive permutation.
        """
        if strategy == "global":
            self.apply_global_permutation(x)
        elif strategy == "magnitude":
            self.apply_magnitude_aware_permutation(x, bands=magnitude_bands)
        elif strategy == "local":
            self.apply_local_neighborhood_permutation(
                x, neighborhood_size=local_neighborhood_size
            )
        elif strategy == "adaptive":
            self.apply_adaptive_permutation(x, scale_factor=adaptive_scale_factor)
        else:
            raise ValueError(
                f"Unknown permutation strategy: {strategy}. "
                f"Must be one of 'global', 'magnitude', 'local', 'adaptive'."
            )

    def apply_stochastic_bf16_rounding(
        self,
        x: torch.Tensor,
        probability: float = 0.5,
        magnitude: Optional[float] = None,
        use_bit_manipulation: bool = True,
    ) -> torch.Tensor:
        """Apply stochastic rounding during conversion to bfloat16.

        This adds controlled noise to the conversion process to improve training dynamics.
        The noise is applied with probability and magnitude controlled by parameters.

        Args:
            x: Input tensor to convert
            probability: Probability of applying noise to each element
            magnitude: Magnitude of noise to apply (defaults to BF16 epsilon)
            use_bit_manipulation: Whether to use bit manipulation for stochastic rounding

        Returns:
            Tensor converted to BF16 with stochastic rounding applied
        """
        if use_bit_manipulation:
            # Use bit manipulation method (Lode's approach)
            if x.dtype == torch.bfloat16:
                # Already BF16, convert to float32 temporarily for bit manipulation
                x_fp32 = x.to(torch.float32)
                result = torch.empty_like(x)
                copy_stochastic_(result, x_fp32)
                return result
            else:
                # Create new tensor in BF16 format
                result = torch.empty_like(x, dtype=torch.bfloat16)
                copy_stochastic_(result, x)
                return result
        else:
            # Use the original noise-based method
            # Convert to BF16 first
            x_bf16 = x.to(torch.bfloat16)

            # Get epsilon for BF16 if magnitude not specified
            if magnitude is None:
                magnitude = torch.finfo(torch.bfloat16).eps

            # Create random mask based on probability
            rand_mask = torch.rand_like(x, device=x.device) < probability

            # Create noise tensor: positive or negative epsilon based on another random mask
            sign_mask = torch.rand_like(x, device=x.device) < 0.5
            noise = torch.ones_like(x, device=x.device)
            noise = torch.where(sign_mask, noise, -noise)
            noise = noise * magnitude

            # Only apply noise where rand_mask is True
            noise = torch.where(rand_mask, noise, torch.zeros_like(x, device=x.device))

            # Add noise to BF16 tensor
            return x_bf16 + noise.to(torch.bfloat16)

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            normalization = group["normalization"]
            normalize_channels = group["normalize_channels"]
            amp_fac = group["amp_fac"]
            clip_lambda = group["clip_lambda"]
            decouple_weight_decay = group["decouple_weight_decay"]
            clip_gradients = group["clip_gradients"]
            stochastic_threshold = group["stochastic_threshold"]
            use_stochastic_rounding = group["use_stochastic_rounding"]
            stochastic_rounding_prob = group["stochastic_rounding_prob"]
            stochastic_rounding_magnitude = group["stochastic_rounding_magnitude"]
            use_bit_manipulation = group["use_bit_manipulation"]
            permutation_strategy = group["permutation_strategy"]
            magnitude_bands = group["magnitude_bands"]
            local_neighborhood_size = group["local_neighborhood_size"]
            adaptive_scale_factor = group["adaptive_scale_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("SPARKLES does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data)
                    state["ema_squared"] = torch.zeros_like(p.data)
                    state["prev_grad"] = torch.zeros_like(p.data)
                    state["stochastic_applied"] = False

                ema, ema_squared = state["ema"], state["ema_squared"]
                prev_grad = state["prev_grad"]
                beta1, beta2 = betas
                state["step"] += 1

                # Use float32 for calculations, will convert back to BF16 later
                grad_fp32 = grad.to(torch.float32)
                p_fp32 = (
                    p.data.to(torch.float32)
                    if p.data.dtype == torch.bfloat16
                    else p.data.clone()
                )
                ema_fp32 = (
                    ema.to(torch.float32)
                    if ema.dtype == torch.bfloat16
                    else ema.clone()
                )
                ema_squared_fp32 = (
                    ema_squared.to(torch.float32)
                    if ema_squared.dtype == torch.bfloat16
                    else ema_squared.clone()
                )
                prev_grad_fp32 = (
                    prev_grad.to(torch.float32)
                    if prev_grad.dtype == torch.bfloat16
                    else prev_grad.clone()
                )

                # Center the gradient
                if centralization != 0:
                    grad_fp32.sub_(
                        grad_fp32.mean(
                            dim=tuple(range(1, grad_fp32.dim())), keepdim=True
                        ).mul_(centralization)
                    )

                # Normalize the gradient
                if normalization != 0:
                    self.normalize_gradient(
                        grad_fp32, use_channels=normalize_channels, alpha=normalization
                    )

                # Bias correction
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** 0.5
                step_size = lr / bias_correction

                # Update EMA of gradient
                ema_fp32.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                # Amplify gradient with EMA
                grad_fp32.add_(ema_fp32, alpha=amp_fac)
                # Update EMA of squared gradient
                ema_squared_fp32.mul_(beta2).addcmul_(
                    grad_fp32, grad_fp32, value=1 - beta2
                )

                # Compute denominator
                denom = ema_squared_fp32.sqrt().div_(bias_correction_sqrt).add_(eps)

                # Prepare update
                if decouple_weight_decay and weight_decay != 0:
                    p_fp32.mul_(1 - step_size * weight_decay)
                    update = grad_fp32.div(denom).mul(step_size)
                else:
                    if weight_decay != 0:
                        grad_fp32.add_(p_fp32, alpha=weight_decay)
                    update = grad_fp32.div(denom).mul(step_size)

                # Check if gradient hasn't changed much (possible saddle point or local minimum)
                if state["step"] > 1:
                    grad_diff_norm = torch.norm(grad_fp32 - prev_grad_fp32)
                    if grad_diff_norm < stochastic_threshold:
                        # Apply stochastic operator to update vector with selected strategy
                        self.apply_stochastic_operator(
                            update,
                            strategy=permutation_strategy,
                            magnitude_bands=magnitude_bands,
                            local_neighborhood_size=local_neighborhood_size,
                            adaptive_scale_factor=adaptive_scale_factor,
                        )
                        state["stochastic_applied"] = True
                    else:
                        state["stochastic_applied"] = False

                # Apply gradient clipping if enabled
                if clip_gradients and clip_lambda is not None:
                    clip = clip_lambda(state["step"])
                    update.clamp_(-clip, clip)

                # Update parameters with stochastic BF16 rounding
                p_fp32.sub_(update)

                # Apply stochastic BF16 rounding if enabled
                if p.dtype == torch.bfloat16:
                    if use_stochastic_rounding:
                        # Apply stochastic rounding when converting back to BF16
                        (
                            copy_stochastic_(p.data, p_fp32)
                            if use_bit_manipulation
                            else p.data.copy_(
                                self.apply_stochastic_bf16_rounding(
                                    p_fp32,
                                    probability=stochastic_rounding_prob,
                                    magnitude=stochastic_rounding_magnitude,
                                    use_bit_manipulation=False,
                                )
                            )
                        )
                    else:
                        # Standard conversion to BF16
                        p.data.copy_(p_fp32.to(torch.bfloat16))
                else:
                    # For non-BF16 parameters, just update directly
                    p.data.copy_(p_fp32)

                # Store current gradient for next iteration (with stochastic rounding if BF16)
                if prev_grad.dtype == torch.bfloat16:
                    if use_stochastic_rounding:
                        (
                            copy_stochastic_(prev_grad, grad_fp32)
                            if use_bit_manipulation
                            else prev_grad.copy_(
                                self.apply_stochastic_bf16_rounding(
                                    grad_fp32,
                                    probability=stochastic_rounding_prob,
                                    magnitude=stochastic_rounding_magnitude,
                                    use_bit_manipulation=False,
                                )
                            )
                        )
                    else:
                        prev_grad.copy_(grad_fp32.to(torch.bfloat16))
                else:
                    prev_grad.copy_(grad_fp32)

                # Update state tensors with stochastic rounding if BF16
                if ema.dtype == torch.bfloat16:
                    if use_stochastic_rounding:
                        (
                            copy_stochastic_(ema, ema_fp32)
                            if use_bit_manipulation
                            else ema.copy_(
                                self.apply_stochastic_bf16_rounding(
                                    ema_fp32,
                                    probability=stochastic_rounding_prob,
                                    magnitude=stochastic_rounding_magnitude,
                                    use_bit_manipulation=False,
                                )
                            )
                        )
                    else:
                        ema.copy_(ema_fp32.to(torch.bfloat16))
                else:
                    ema.copy_(ema_fp32)

                if ema_squared.dtype == torch.bfloat16:
                    if use_stochastic_rounding:
                        (
                            copy_stochastic_(ema_squared, ema_squared_fp32)
                            if use_bit_manipulation
                            else ema_squared.copy_(
                                self.apply_stochastic_bf16_rounding(
                                    ema_squared_fp32,
                                    probability=stochastic_rounding_prob,
                                    magnitude=stochastic_rounding_magnitude,
                                    use_bit_manipulation=False,
                                )
                            )
                        )
                    else:
                        ema_squared.copy_(ema_squared_fp32.to(torch.bfloat16))
                else:
                    ema_squared.copy_(ema_squared_fp32)

        return loss
