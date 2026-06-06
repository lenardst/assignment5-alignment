"""Modal entrypoint for the prompting-baselines eval that does not require the
`wandb` secret (which the stock `cs336_alignment.modal_utils.run_command`
function attaches).

Run:
    uv run modal run -d scripts/run_eval_prompting_modal.py
    uv run modal volume get a5-experiments-lenardst prompting_baselines ./experiments
"""

from __future__ import annotations

import subprocess
import sys

import modal

from cs336_alignment.modal_utils import (
    GPU,
    RUN_TIMEOUT_SECONDS,
    SUNET_ID,
    app,
    image,
    quote_command,
)


VOLUME_MOUNT = "/results"
EVAL_CMD = [
    sys.executable,
    "-u",
    "scripts/eval_prompting_baselines.py",
    "--model",
    "allenai/OLMo-2-0425-1B",
    "--output-dir",
    f"{VOLUME_MOUNT}/prompting_baselines",
]


EXPERIMENTS_VOLUME = modal.Volume.from_name(
    f"a5-experiments-{SUNET_ID}", create_if_missing=True
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: EXPERIMENTS_VOLUME},
)
def run_eval_persisted() -> str:
    command_str = quote_command(EVAL_CMD)
    print(command_str, flush=True)
    subprocess.run(EVAL_CMD, check=True)
    EXPERIMENTS_VOLUME.commit()
    return command_str


@app.local_entrypoint()
def main() -> None:
    print(
        f"Launching prompting-baselines eval on {GPU}; "
        f"results -> volume a5-experiments-{SUNET_ID}:prompting_baselines/",
        flush=True,
    )
    run_eval_persisted.remote()
    print(
        "Done. Pull results with:\n"
        f"  modal volume get a5-experiments-{SUNET_ID} "
        f"prompting_baselines ./experiments",
        flush=True,
    )
