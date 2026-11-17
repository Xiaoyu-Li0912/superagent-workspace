# Modified for AIAA3102 course integration; see MLEvolve/LOCAL_CHANGES.md.
"""Environment/package prompt."""

import random


def get_prompt_environment(agent=None):
    """Installed packages description."""
    if agent is not None and getattr(agent, "integration", None) is not None:
        return agent.integration.environment()
    pkgs = [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "lightGBM",
        "torch",
        "torchvision",
        "torch-geometric",
        "bayesian-optimization",
        "timm",
        "transformers",
        "sentence-transformers",
        "opencv-python",
        "Pillow",
    ]
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    return {
        "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
    }
