from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(quick: bool):
    suffix = ["--quick"] if quick else []
    commands = [
        [sys.executable, "scripts/run_benchmark.py", *suffix],
        [sys.executable, "scripts/run_ngram_multiseed.py", *suffix],
        [sys.executable, "scripts/run_qlbs_variants.py", *suffix],
        [sys.executable, "scripts/run_mixing_time.py", *suffix],
        [sys.executable, "scripts/qlbs_learned_adversary.py", *suffix],
        [sys.executable, "scripts/generate_figures.py"],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    metadata = {
        "mode": "quick_smoke_test" if quick else "full",
        "reportable_as_paper_results": not quick,
        "command": "python reproduce_all.py --quick" if quick else "python reproduce_all.py",
    }
    output = ROOT / "results" / "generated" / "run_metadata.json"
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Reproduction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smoke test with reduced episodes/epochs.")
    main(parser.parse_args().quick)
