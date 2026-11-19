### SEARCH
# Global epoch budget for the main model.  Originally referenced by MODEL_CFG
# but never defined before the dict literal was evaluated -> NameError at
# import time.  Defined here so the config (and every module-level consumer)
# has a concrete value available.
MAX_EPOCHS = 120

MODEL_CFG = dict(
    emb_dim=8,
    n_cross=2,
    hidden=(256, 128),
    n_experts=4,
    expert_hidden=64,
    dropout=0.30,
    lr=2e-3,
    weight_decay=1e-4,
    total_epochs=MAX_EPOCHS,
    label_smoothing=0.05,
)
### REPLACE
MAX_EPOCHS = 120

MODEL_CFG = dict(
    emb_dim=8,
    n_cross=2,
    hidden=(256, 128),
    n_experts=4,
    expert_hidden=64,
    dropout=0.30,
    lr=2e-3,
    weight_decay=1e-4,
    total_epochs=MAX_EPOCHS,
    label_smoothing=0.05,
)

MLP_CFG = dict(
    emb_dim=8,
    n_freq=4,
    hidden=160,
    n_blocks=2,
    dropout=0.15,
)
MLP_SEEDS = 2
MLP_REFIT = 3
MLP_EPOCHS = 45
MLP_PATIENCE = 7