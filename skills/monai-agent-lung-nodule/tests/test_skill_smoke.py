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
Smoke tests for monai-agent-lung-nodule. Runs in --mock mode only, so no GPU
is required -- matching the project's current CI, which has no GPU runner
available. Real-mode assertions are skipped when torch/monai are absent.
"""

import json
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
RUN_SCRIPT = SKILL_DIR / "scripts" / "run_agent_lung_nodule.py"
VERIFY_SCRIPT = SKILL_DIR / "verifiers" / "agent_output_quality_v1" / "scripts" / "verify_result.py"


def _run_mock(tmp_path, study_id="TEST_STUDY_001"):
    out_dir = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--study-id", study_id, "--mock", "--output-dir", str(out_dir)],
        check=True,
        capture_output=True,
    )
    return out_dir / "result_json.json"


def test_mock_mode_produces_valid_schema(tmp_path):
    result_path = _run_mock(tmp_path)
    result = json.loads(result_path.read_text())
    assert result["execution_metadata"]["mode"] == "simulated_monai"
    assert result["nodules_detected"] == len(result["findings"])
    for f in result["findings"]:
        assert f["AI_Generated"] is True
        assert f["Verification_Status"] == "Pending"


def test_verifier_passes_on_mock_output(tmp_path):
    result_path = _run_mock(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), str(result_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[FAIL]" not in proc.stdout


def test_mock_mode_is_reproducible(tmp_path):
    result_path = _run_mock(tmp_path, study_id="REPRO_STUDY")
    # Double quotes (not !r/repr's single quotes) are required here: this command runs
    # via shell=True and must be valid for both POSIX shells and Windows cmd.exe, which
    # does not treat single quotes as quoting.
    rerun_cmd = (
        f'"{sys.executable}" "{RUN_SCRIPT}" --study-id REPRO_STUDY --mock '  # noqa: B907
        f'--output-dir "{result_path.parent}"'  # noqa: B907
    )
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), str(result_path), "--rerun-cmd", rerun_cmd],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
