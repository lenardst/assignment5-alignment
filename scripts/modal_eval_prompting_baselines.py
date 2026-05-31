"""Run scripts/eval_prompting_baselines.py on Modal.

Two entrypoints:

1. `quick` -- uses the standard `submit_commands` plumbing from
   `cs336_alignment.modal_utils`. Results stay inside the container and only
   come back through stdout / Modal logs. Good for sanity-checks and for the
   summary JSON which gets printed at the end of the run.

2. `persist` -- defines a sibling Modal function that mounts a named Volume
   at `/root/experiments`, so the per-example JSONL records survive after
   the container exits. Pull them back to the local machine with
       modal volume get a5-experiments-<sunet> prompting_baselines ./experiments
   You need this one for the part (a) manual audit (looking at >=10 cat2
   and cat3 examples per prompt).

Usage:
    # quickest path -- writes summary to logs, JSONL is ephemeral
    modal run -d scripts/modal_eval_prompting_baselines.py::quick

    # recommended -- JSONL is persisted to a modal Volume
    modal run -d scripts/modal_eval_prompting_baselines.py::persist
    modal volume get a5-experiments-lenardst prompting_baselines ./experiments
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
    submit_commands,
)


RESULTS_MOUNT = "/root/results"

EVAL_CMD = [
    sys.executable,
    "-u",
    "scripts/eval_prompting_baselines.py",
    "--model",
    "allenai/OLMo-2-0425-1B",
    "--output-dir",
    f"{RESULTS_MOUNT}/prompting_baselines",
]


# ---------------------------------------------------------------------------
# Variant 1: stock submit_commands pipeline (results ephemeral).
# ---------------------------------------------------------------------------
@app.local_entrypoint(name="quick")
def quick() -> None:
    submit_commands([EVAL_CMD])


# ---------------------------------------------------------------------------
# Variant 2: same image, but the function mounts a persistent volume so the
# JSONL records survive past the container's lifetime.
# ---------------------------------------------------------------------------
EXPERIMENTS_VOLUME = modal.Volume.from_name(
    f"a5-experiments-{SUNET_ID}", create_if_missing=True
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    volumes={RESULTS_MOUNT: EXPERIMENTS_VOLUME},
)
def run_eval_persisted() -> str:
    command_str = quote_command(EVAL_CMD)
    print(command_str, flush=True)
    subprocess.run(EVAL_CMD, check=True)
    # Force a commit so the JSONL/summary files are visible to `modal volume get`.
    EXPERIMENTS_VOLUME.commit()
    return command_str


@app.local_entrypoint(name="persist")
def persist() -> None:
    print(
        f"Launching prompting-baselines eval on {GPU}; "
        f"results -> volume a5-experiments-{SUNET_ID}:prompting_baselines/",
        flush=True,
    )
    # spawn() rather than remote() so the call doesn't get cancelled if we
    # detach (modal run -d). The function will commit the volume on exit.
    run_eval_persisted.spawn()
    print(
        "Submitted. Once finished, pull results with:\n"
        f"  MODAL_ENVIRONMENT=cs336-{SUNET_ID} \\\n"
        f"  modal volume get a5-experiments-{SUNET_ID} "
        f"prompting_baselines ./experiments",
        flush=True,
    )
