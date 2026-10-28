# Copyright 2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
verify_result.py - Deterministic verifier for monai-agent-lung-nodule evidence packs.
See ../SKILL.md for the gate definitions this script implements.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MAX_DIAMETER_MM = 100.0
MAX_RUNTIME_MS = 5000


def gate_schema(result: Dict[str, Any]) -> Tuple[bool, str]:
    required = {"study_id", "status", "nodules_detected", "findings", "execution_metadata"}
    missing = required - result.keys()
    if missing:
        return False, f"missing top-level keys: {sorted(missing)}"
    return True, "ok"


def gate_sanity(result: Dict[str, Any]) -> Tuple[bool, str]:
    findings = result.get("findings", [])
    if result.get("nodules_detected") != len(findings):
        return False, "nodules_detected does not match len(findings)"
    for f in findings:
        if f.get("AI_Generated") is not True:
            return False, f"finding {f.get('nodule_id')} missing AI_Generated: true"
        if not f.get("Verification_Status"):
            return False, f"finding {f.get('nodule_id')} missing Verification_Status"
    return True, "ok"


def gate_physical_bounds(result: Dict[str, Any]) -> Tuple[bool, str]:
    for f in result.get("findings", []):
        dia = f.get("max_diameter_mm")
        if dia is None or not (0 <= dia <= MAX_DIAMETER_MM):
            return False, f"finding {f.get('nodule_id')} max_diameter_mm out of bounds: {dia}"
        vol = f.get("volume_mm3", f.get("volume_voxels"))
        if vol is None or vol < 0:
            return False, f"finding {f.get('nodule_id')} negative or missing volume"
    return True, "ok"


def gate_runtime(result: Dict[str, Any]) -> Tuple[bool, str]:
    meta = result.get("execution_metadata", {})
    ms = meta.get("inference_time_ms")
    if ms is None:
        return False, "execution_metadata.inference_time_ms missing"
    if not (0 <= ms <= MAX_RUNTIME_MS):
        return False, f"inference_time_ms out of declared bound: {ms}"
    return True, "ok"


def gate_reproducibility(rerun_cmd: str, result_path: Path) -> Tuple[bool, str]:
    if not rerun_cmd:
        return True, "skipped (no --rerun-cmd provided)"
    first = json.loads(result_path.read_text())["findings"]
    subprocess.run(rerun_cmd, shell=True, check=True, capture_output=True)
    second = json.loads(result_path.read_text())["findings"]
    if first != second:
        return False, "repeat invocation produced different findings in mock mode"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a monai-agent-lung-nodule result_json evidence pack")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--rerun-cmd", default="", help="Command to re-invoke the skill for the reproducibility gate")
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text())

    gates: List[Tuple[str, Tuple[bool, str]]] = [
        ("schema", gate_schema(result)),
        ("sanity", gate_sanity(result)),
        ("physical_bounds", gate_physical_bounds(result)),
        ("runtime", gate_runtime(result)),
        ("reproducibility", gate_reproducibility(args.rerun_cmd, args.result_json)),
    ]

    all_passed = True
    for name, (passed, message) in gates:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {message}")
        all_passed = all_passed and passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
