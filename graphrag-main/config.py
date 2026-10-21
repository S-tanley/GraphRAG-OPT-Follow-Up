"""
Global configuration: fast-mode presets and reproducibility seed.
"""
import os
import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Fast-mode presets
# ---------------------------------------------------------------------------
QUICK_MODE_CONFIGS = {
    "smoke": {
        "samples": 16,
        "batch_size": 8,
        "description": "Ultra-fast smoke test (~30 seconds)",
    },
    "quick": {
        "samples": 32,
        "batch_size": 8,
        "description": "Quick development test (~1-2 minutes)",
    },
    "medium": {
        "samples": 128,
        "batch_size": 16,
        "description": "Medium validation test (~5-8 minutes)",
    },
    "full": {
        "samples": 512,
        "batch_size": 64,
        "description": "Full benchmark test (~15+ minutes)",
    },
    "neurips": {
        "samples": 64,
        "batch_size": 16,
        # Same generation params as the RoG paper: num_beams=4, max_new_tokens=50
        # Reports standard NeurIPS KGQA metrics: Hits@1 + F1
        "max_new_tokens": 50,
        "num_beams": 4,
        "description": "NeurIPS-standard fast check (64 samples, beam=4, Hits@1 + F1)",
    },
    "original": {
        "samples": 256,
        "batch_size": 64,
        # Matches graphrag_rog_v3.py exactly
        "max_new_tokens": 50,
        "num_beams": 4,
        "description": "Original GraphRAG baseline (256 shuffled samples, matches graphrag_rog_v3.py)",
    },
}


def set_reproducible_seed(seed: int = 42) -> None:
    """Set seeds for reproducible results across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
