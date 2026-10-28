#!/usr/bin/env python3
"""Dry `monai-deploy package` inspection for cchmc-tcv-fast.

Drives the REAL holoscan_cli 4.2.0 packager code path (the same code
`monai-deploy package` runs) all the way up to the Dockerfile write, then
STOPS BEFORE IT. No docker is invoked, no image is built. It asserts the
staged build context the single-pass bake will consume carries the whole
contract:

  1 manifests_exist — map/app.json + map/pkg.json written by the
                                  real _create_app_manifest /
                                  _create_package_manifest and parse as JSON
  2 pkg_json_shared_memory — pkg.json resources carry sharedMemory "4Gi"
                                  verbatim + the other four resources intact
  3 merged_requirements_pins — staged pip/requirements.txt (what the image
                                  pip-installs) carries the validated pins
  4 staged_app_proto — staged app/proto carries BOTH committed pb2
                                  stubs + codegen.py + the .proto
                                  (the `COPY ./app` payload; no protoc in-image)
  5 codegen_check_in_context — codegen.py --check passes FROM the staged
                                  copy (end-to-end on the exact bytes the
                                  image would COPY)
  6 dockerfile_copy_app — generated Dockerfile: COPY ./app /opt/holoscan/app
  7 dockerfile_copy_requirements — generated Dockerfile: COPY ./pip/requirements.txt
                                    /tmp/requirements.txt
  8 dockerfile_entrypoint — generated Dockerfile: ENTRYPOINT
                                  ["/var/holoscan/tools"]
  9 tools_argv_forwarding — tools.sh forwards ALL docker-run args
                                  verbatim (eval ${command} "$@") — proves
                                  `docker run ... serve` reaches the argv
                                  router unmodified; NO entrypoint surgery
 10 app_json_facts — app.json command/sdk/sdkVersion/readiness/
                                  liveness + the environment block recorded as
                                  FACT lines (record, not assert; only
                                  `command` is gated). The environment block
                                  must carry ONLY the standard HOLOSCAN_* vars
                                  (proves the packager does not merge user
                                  env, hence the in-code EMIT_NPY container
                                  default).

STOP POINT: the last packager call made is
PythonAppBuilder._write_dockerfile. The packager's build would next
run _build_internal — the docker buildx build (a ~12 GB image). We never
call the builder's build entry point nor its _build_internal entry point,
and never import docker here; a separate packaging step performs the real build.

Run:
    cd examples/apps/cchmc-tcv-fast
    SDK_WHEEL_DIR=/path/to/wheel-dir \
        python scripts/dry_package_inspect.py
    (build the SDK wheel first, from the repo root:
     pip wheel . --no-deps -w /path/to/wheel-dir)

Staging dir: $DRY_STAGE_DIR (default: a temp dir). If it already exists and
is non-empty the script fails fast and asks the operator to rm -rf it
(never silently delete). The staging dir is KEPT after a run — inspect it.

Exit 0 only if all checks pass; prints
DRY-PACKAGE: ALL INSPECTION CHECKS PASS — VERDICT: SAFE on success.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
STAGE = Path(os.environ.get("DRY_STAGE_DIR") or tempfile.mkdtemp(prefix="dry-package-stage-"))
# The SDK wheel (built from the repo root: pip wheel . --no-deps -w <dir>).
# Set SDK_WHEEL to the wheel file, or SDK_WHEEL_DIR to its directory.
SDK_WHEEL_DIR = os.environ.get("SDK_WHEEL_DIR", "")
SDK_WHEEL = os.environ.get(
    "SDK_WHEEL",
    (
        sorted(glob.glob(os.path.join(SDK_WHEEL_DIR, "monai_deploy_app_sdk-*.whl")))[0]
        if SDK_WHEEL_DIR and glob.glob(os.path.join(SDK_WHEEL_DIR, "monai_deploy_app_sdk-*.whl"))
        else ""
    ),
)
CODEGEN_TIMEOUT_S = 300

# grpcio-tools is a DEV-ONLY codegen tool: it is intentionally absent from
# my_app/requirements.txt and the runtime image (the committed stubs ride via
# COPY ./app). Pinned to the runtime grpcio so the check regenerates with the
# same toolchain family the image's gencode was produced by.
GRPCTOOLS_PIN = "grpcio-tools==1.83.1"


def _grpc_tools_env():
    """Return an env making grpc_tools importable by sys.executable.

    If the running interpreter already imports grpc_tools, returns None (no
    change). Otherwise installs GRPCTOOLS_PIN into a private, reusable
    directory under the system temp area (pip's wheel cache keeps repeat
    runs fast) and returns an env with PYTHONPATH pointing at it — so the
    staged codegen check is always able to run, even in a bare interpreter.
    Never touches my_app/requirements.txt or any image requirements.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import grpc_tools"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode == 0:
        return None
    target = Path(tempfile.gettempdir()) / "dry-package-grpctools"
    if not (target / "grpc_tools").is_dir():
        print(f"DRY-PACKAGE: providing {GRPCTOOLS_PIN} into {target} ...", flush=True)
        inst = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--target",
                str(target),
                GRPCTOOLS_PIN,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if inst.returncode != 0 or not (target / "grpc_tools").is_dir():
            tail = (inst.stderr or inst.stdout or "")[-1500:]
            raise RuntimeError(f"pip install {GRPCTOOLS_PIN} failed:\n{tail}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# cchmc-tcv-fast pins that MUST survive into the staged in-image
# requirements (what the image pip-installs).
REQUIRED_PINS = (
    "nnunetv2==2.8.1",
    "torch==2.13.0+cu130",
    "monai==1.5.2",
    "grpcio==1.83.1",
    "protobuf>=6.33,<7",
)

CHECKS = [
    "manifests_exist",
    "pkg_json_shared_memory",
    "merged_requirements_pins",
    "staged_app_stubs",
    "codegen_check_in_context",
    "dockerfile_copy_app",
    "dockerfile_copy_requirements",
    "dockerfile_entrypoint",
    "tools_argv_forwarding",
    "app_json_facts",
]

RESULTS = {}


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = "PASS" if ok else "FAIL"
    print(f"DRY-PACKAGE: {name} — {'PASS' if ok else 'FAIL'} — {detail}")


def main() -> int:
    if not SDK_WHEEL or not os.path.isfile(SDK_WHEEL):
        sys.stderr.write(
            "ERROR: SDK wheel not found. Set SDK_WHEEL to the wheel file, or\n"
            "SDK_WHEEL_DIR to its directory (build it from the repo root with\n"
            "pip wheel . --no-deps -w <dir>).\n"
        )
        return 1
    print(f"DRY-PACKAGE: SDK-WHEEL — {SDK_WHEEL}")

    # --- staging dir preflight (fail fast, never silently delete) ---
    if STAGE.exists() and any(STAGE.iterdir()):
        sys.stderr.write(
            f"ERROR: staging dir {STAGE} exists and is non-empty.\n"
            "Inspect it, then rm -rf it yourself, and re-run.\n"
            "This script never deletes its own staging output.\n"
        )
        return 1
    STAGE.mkdir(parents=True, exist_ok=True)

    # --- drive the real packager code path up to the stop point (no docker) ---
    # parse_args derives the program name from sys.argv[0] for SDK detection
    # (detect_sdk would reject this script's own name) — set it FIRST.
    sys.argv = ["monai-deploy"]
    from holoscan_cli.__main__ import parse_args
    from holoscan_cli.packager.arguments import PackagingArguments
    from holoscan_cli.packager.container_builder import PythonAppBuilder
    from holoscan_cli.packager.packager import (
        _create_app_manifest,
        _create_package_manifest,
    )

    args = parse_args(
        [
            "monai-deploy",
            "package",
            f"{APP_DIR}/my_app",
            "-c",
            f"{APP_DIR}/my_app/app.yaml",
            "-m",
            f"{APP_DIR}/models",
            "--platform",
            "x86_64",
            "--cuda",
            "13",
            "-t",
            "cchmc-tcv-fast:latest",
            "--monai-deploy-sdk-file",
            SDK_WHEEL,
            "-l",
            "WARN",
        ]
    )

    pa = PackagingArguments(args, str(STAGE))
    _create_app_manifest(pa.application_manifest, str(STAGE))
    _create_package_manifest(pa.package_manifest, str(STAGE))
    builder = PythonAppBuilder(pa.build_parameters, str(STAGE))
    builder._copy_supporting_files(pa.platforms[0])
    builder._write_dockerfile(pa.platforms[0])
    # STOP POINT: the next real-packager step would be the docker image build
    # (BuilderBase.build -> _build_internal -> buildx). We deliberately never
    # reach it — no docker invocation anywhere in the dry inspection.

    try:
        stage_listing = sorted(os.listdir(STAGE))
    except OSError:
        stage_listing = []
    print(f"DRY-PACKAGE: STAGE-LISTING — {stage_listing}")

    # --- 1: manifests exist + parse ---
    try:
        app_json = json.loads((STAGE / "map" / "app.json").read_text())
        pkg_json = json.loads((STAGE / "map" / "pkg.json").read_text())
        check("manifests_exist", True, "map/app.json + map/pkg.json parse as JSON")
    except (OSError, json.JSONDecodeError) as e:
        check("manifests_exist", False, f"manifests missing or unparseable: {e}")
        return _finish()

    # --- 2: pkg.json sharedMemory + the other four resources intact ---
    res = pkg_json.get("resources", {})
    expected = {"cpu": 4, "gpu": 1, "memory": "8Gi", "gpuMemory": "20Gi", "sharedMemory": "4Gi"}
    bad = {k: res.get(k) for k, v in expected.items() if res.get(k) != v}
    check(
        "pkg_json_shared_memory",
        not bad,
        f"resources={json.dumps(res)}" + (f" MISMATCH={json.dumps(bad)}" if bad else ""),
    )

    # --- 3: merged pip requirements pins (what the image pip-installs) ---
    try:
        req_text = (STAGE / "pip" / "requirements.txt").read_text()
    except OSError as e:
        req_text = ""
        check("merged_requirements_pins", False, f"staged pip/requirements.txt unreadable: {e}")
    else:
        missing = [p for p in REQUIRED_PINS if p not in req_text]
        check(
            "merged_requirements_pins",
            not missing,
            (
                "all five pins present (nnunetv2/torch/monai/grpcio/protobuf)"
                if not missing
                else f"missing pins: {missing}"
            ),
        )

    # --- 4: staged app/proto carries both pb2 stubs + codegen + proto ---
    stub_files = [
        "map_service_pb2.py",
        "map_service_pb2_grpc.py",
        "codegen.py",
        "map_service.proto",
    ]
    missing = [f for f in stub_files if not (STAGE / "app" / "proto" / f).is_file()]
    check(
        "staged_app_stubs",
        not missing,
        (
            "all four proto contract files staged under app/proto"
            if not missing
            else f"missing from staged app/proto: {missing}"
        ),
    )

    # --- 5: codegen --check from the staged copy (end-to-end) ---
    # The staged check re-runs protoc through the RUNNING interpreter
    # (sys.executable), so grpc_tools must be importable there.
    # grpcio-tools is dev-only by design (never in my_app/requirements.txt
    # or the runtime image); _grpc_tools_env() PROVIDES it on demand
    # (private pip --target install + PYTHONPATH) when the interpreter lacks
    # it, so the check self-heals in a bare environment.
    codegen_env = None
    env_error = None
    try:
        codegen_env = _grpc_tools_env()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        env_error = str(e)
    if env_error is not None:
        check(
            "codegen_check_in_context",
            False,
            f"grpcio-tools unavailable and could not be provided automatically: {env_error}",
        )
    else:
        try:
            cp = subprocess.run(
                [sys.executable, f"{STAGE}/app/proto/codegen.py", "--check"],
                cwd=str(STAGE / "app"),
                env=codegen_env,
                capture_output=True,
                text=True,
                timeout=CODEGEN_TIMEOUT_S,
            )
            clean = cp.returncode == 0 and "PBCHECK-CLEAN" in (cp.stdout + cp.stderr)
            check(
                "codegen_check_in_context",
                clean,
                f"rc={cp.returncode} PBCHECK-CLEAN={'yes' if 'PBCHECK-CLEAN' in cp.stdout + cp.stderr else 'no'}",
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            check("codegen_check_in_context", False, f"codegen --check did not complete: {e}")

    # --- 6-8: generated Dockerfile facts ---
    try:
        dockerfile = (STAGE / "Dockerfile").read_text()
    except OSError as e:
        dockerfile = ""
        for name in ("dockerfile_copy_app", "dockerfile_copy_requirements", "dockerfile_entrypoint"):
            check(name, False, f"Dockerfile unreadable: {e}")
    else:
        want = {
            "dockerfile_copy_app": "COPY ./app /opt/holoscan/app",
            "dockerfile_copy_requirements": "COPY ./pip/requirements.txt /tmp/requirements.txt",
            "dockerfile_entrypoint": 'ENTRYPOINT ["/var/holoscan/tools"]',
        }
        for name, needle in want.items():
            ok = needle in dockerfile
            check(name, ok, f"line present: {needle}" if ok else f"line NOT found: {needle}")

    # --- 9: tools.sh forwards all docker-run args verbatim ---
    try:
        tools = (STAGE / "tools").read_text()
        ok = 'eval ${command} "$@"' in tools
        check(
            "tools_argv_forwarding",
            ok,
            (
                'tools.sh: eval ${command} "$@" — ALL docker-run args forwarded '
                "(docker run ... serve reaches the argv router; no entrypoint surgery)"
                if ok
                else 'tools.sh missing `eval ${command} "$@"` argv forwarding'
            ),
        )
    except OSError as e:
        check("tools_argv_forwarding", False, f"staged tools script unreadable: {e}")

    # --- 10: app.json facts (record; only command is gated) ---
    facts = {
        "command": app_json.get("command"),
        "sdk": app_json.get("sdk"),
        "sdkVersion": app_json.get("sdkVersion"),
        "readiness": app_json.get("readiness"),
        "liveness": app_json.get("liveness"),
    }
    for k, v in facts.items():
        print(f"DRY-PACKAGE: FACT-APPJSON-{k} — {json.dumps(v)}")
    env_block = app_json.get("environment", {})
    print(f"DRY-PACKAGE: FACT-APPJSON-environment — {json.dumps(env_block, sort_keys=True)}")
    # Record (not gate): the environment block must carry ONLY standard
    # HOLOSCAN_* vars. If a user var leaked in, that would contradict the
    # research §3 finding (no user-env merge) — flag it loudly, still PASS.
    non_std = {k: v for k, v in env_block.items() if not k.startswith("HOLOSCAN_") and k not in ("PYTHONUNBUFFERED",)}
    if non_std:
        print(
            f"DRY-PACKAGE: FACT-APPJSON-env-NONSTD — {json.dumps(non_std, sort_keys=True)} "
            f"(unexpected: packager appears to merge user env — re-check §3)"
        )
    cmd = facts["command"] or ""
    ok = (isinstance(cmd, str) and "/opt/holoscan/app" in cmd) or (
        isinstance(cmd, list) and any("/opt/holoscan/app" in c for c in cmd)
    )
    check(
        "app_json_facts",
        bool(ok),
        f"command gates on /opt/holoscan/app (command={json.dumps(cmd)}; "
        "readiness/liveness null for sdk=monai-deploy — in-app CheckHealth on "
        "port 8765 is the readiness signal; no grpc_health_probe)",
    )

    return _finish()


def _finish() -> int:
    n_pass = sum(1 for v in RESULTS.values() if v == "PASS")
    for name in CHECKS:
        if RESULTS.get(name) == "PASS":
            print(f"{name} PASS")
    if n_pass == len(CHECKS):
        print("DRY-PACKAGE: ALL INSPECTION CHECKS PASS — VERDICT: SAFE")
        return 0
    failed = [n for n in CHECKS if RESULTS.get(n) != "PASS"]
    print(f"DRY-PACKAGE: {n_pass}/{len(CHECKS)} checks pass — VERDICT: FAIL ({failed})")
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 1
    # os._exit skips atexit/flush — flush stdio explicitly or all piped
    # output (tee'd evidence log) is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
