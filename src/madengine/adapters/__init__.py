"""Black-box integrations for AIImageBuilder and AIWorkloads."""

from madengine.adapters.base import Adapter, AdapterResult
from madengine.adapters.aiimagebuilder import AIImageBuilderAdapter
from madengine.adapters.aiworkloads import AIWorkloadsAdapter

__all__ = [
    "Adapter",
    "AdapterResult",
    "AIImageBuilderAdapter",
    "AIWorkloadsAdapter",
]
