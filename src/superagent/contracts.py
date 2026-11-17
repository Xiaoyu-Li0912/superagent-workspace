from dataclasses import dataclass, asdict
from typing import Literal


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    status: Literal["valid", "failed"]
    metric: float | None
    elapsed_seconds: float
    diagnostics: str
    artifact_dir: str

    def to_dict(self):
        return asdict(self)


CANDIDATE_CONTRACT = """
COURSE CONTRACT OVERRIDES ANY KAGGLE OR UPSTREAM INSTRUCTIONS:
The FINAL assembled solution.py must implement argparse subcommands:
  python solution.py train --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
Train input: /data/features.csv, /data/labels.csv (one column 'label'), /data/schema.json.
Predict input: /data/features.csv, /data/schema.json; NO labels. Never use validation
or test labels for training or preprocessing. Training gets ONLY training rows.
Save ALL learned preprocessing, model, and label mapping under --model. Prediction
runs in a NEW PROCESS with --model read-only and must load, not refit, the pipeline.
features.csv has no target and row order is authoritative. Output exactly columns
row_id,prediction; row_id is 0..N-1 in input order; prediction is an ORIGINAL label.
Use the installed packages reported by the host environment and the standard library.
Execution resources/packages are described by the host environment, not a course
prohibition on GPU/deep learning. No internet or pip installs in candidate sandboxes.
For images, features contains sandbox-relative image paths.
Use schema modality/feature names and training data, not a dataset name lookup.
The supervisor computes accuracy independently on held-out predictions and provides
aggregate feedback. Internal holdout/CV on TRAINING rows is allowed, but do not
invent supervisor validation scores or use held-out validation labels.
Do NOT create Kaggle submissions, read ./input, or access unknown files.
The final script needs both subcommands protected by if __name__ == '__main__'.
Persist a complete usable model within the host's configured resource limits.
During staged generation, output only the requested module; during diff generation
or code review, use the requested SEARCH/REPLACE format. This contract constrains
the final assembled script, not the intermediate agent response format.
"""
