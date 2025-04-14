import torch
import functools
import logging
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Configuration state
_bf16_enabled = False


def is_bf16_enabled() -> bool:
    """Check if full bf16 mode is enabled."""
    return _bf16_enabled


def enable_bf16() -> None:
    """Enable full bf16 mode."""
    global _bf16_enabled
    _bf16_enabled = True
    logger.info("Full BF16 mode enabled - models will be converted to bf16 precision")


def convert_model_to_bf16(model: T) -> T:
    """Convert model to bf16 precision if full bf16 mode is enabled."""
    if is_bf16_enabled():
        logger.info(f"Converting model to bf16 precision: {type(model).__name__}")
        return cast(T, model.to(torch.bfloat16))
    return model


def with_bf16_support(func: Callable) -> Callable:
    """Decorator to add bf16 support to model loading functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        return convert_model_to_bf16(model)

    return wrapper
