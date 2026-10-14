#!/usr/bin/env python3
"""Regenerate + self-check the committed map_service gRPC stubs.

The generated stubs (`map_service_pb2.py`, `map_service_pb2_grpc.py`) are
COMMITTED in this directory: the MAP packager flat-copies `my_app/` into the
image and the image has no protoc and no build hooks, so the stubs ride via
`COPY ./app` (`--check` runs in the build context during packaging).

`grpcio-tools` is a DEV-ONLY tool — it is intentionally NOT in
`my_app/requirements.txt` (the image resolves grpcio/protobuf via
tritonclient's bounds: grpcio 1.83.1, protobuf 6.33.x).

Generation uses `python -m grpc_tools.protoc` (the invoking interpreter's
venv tooling). The generated import style is kept FLAT (`import map_service_pb2
as ...` inside the grpc stub); consuming code adds this directory to
`sys.path` before import (flat-import pattern).

Version-marker normalization:
    The image runtime is pinned to protobuf >=6.30,<7 (tritonclient hard bound),
    so the committed pb2 must carry a protobuf 6.x gencode version — the runtime
    fatally rejects gencode with a newer major. Newer grpcio-tools bundles a
    newer protoc (e.g. 1.83.1 bundles libprotoc 35.1 -> 7.x marker), so both
    modes here rewrite the two version markers in the pb2
    (`# Protobuf Python Version:` comment and the
    `ValidateProtobufRuntimeVersion` major/minor/patch arguments) to the
    image-target `6.33.0` before writing/comparing. For this plain proto3
    contract the rest of the gencode is byte-identical across protoc 33.6 and
    35.1 (verified by diff 2026-09-10), so the normalized file is exactly
    protoc-33-era gencode and imports cleanly on the image's protobuf 6.33.6.

Usage:
    python codegen.py # regenerate both stubs IN PLACE (next to the proto)
    python codegen.py --check # regenerate into a temp dir, byte-compare against
                                 # the committed stubs; print PBCHECK-CLEAN and
                                 # exit 0 if identical, else print the unified
                                 # diff and exit 1 (CI/build self-check)
"""

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile

PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(PROTO_DIR, "map_service.proto")
STUBS = ("map_service_pb2.py", "map_service_pb2_grpc.py")

# Image-target gencode version (tritonclient bounds protobuf to >=6.30,<7;
# the verified MAP image runs 6.33.6).
TARGET_PB_VERSION = (6, 33, 0)


def _run_protoc(out_dir: str) -> None:
    """Run the venv's grpc_tools.protoc into out_dir (flat layout)."""
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I",
        PROTO_DIR,
        os.path.join(PROTO_DIR, "map_service.proto"),
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
    ]
    subprocess.run(cmd, check=True, cwd=PROTO_DIR)


def _normalize_pb2(text: str) -> str:
    """Pin the pb2's version markers to TARGET_PB_VERSION.

    Only touches the `# Protobuf Python Version:` header comment and the
    ValidateProtobufRuntimeVersion major/minor/patch integer lines — the 3
    bare-integer lines after the Domain argument in the current gencode
    layout. The Domain argument, the suffix line, and the proto-file
    location line are left untouched.
    """
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# Protobuf Python Version:"):
            out.append("# Protobuf Python Version: %d.%d.%d\n" % TARGET_PB_VERSION)
            i += 1
        elif "ValidateProtobufRuntimeVersion(" in line:
            out.append(line)
            i += 1
            # Skip the Domain argument line, then pin the 3 version ints.
            if not re.search(r"Domain\.\w+,?\s*$", lines[i].strip()):
                raise RuntimeError(
                    "unexpected gencode layout after ValidateProtobufRuntimeVersion: " + repr(lines[i].strip())
                )
            out.append(lines[i])
            i += 1
            for val in TARGET_PB_VERSION:
                # Each version component is on its own line: ` <int>,`
                if not re.fullmatch(r"\d+,?", lines[i].strip()):
                    raise RuntimeError("unexpected gencode layout at version marker: " + repr(lines[i].strip()))
                out.append(" %d,\n" % val)
                i += 1
        else:
            out.append(line)
            i += 1
    return "".join(out)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def regenerate(out_dir: str) -> dict:
    _run_protoc(out_dir)
    result = {}
    for name in STUBS:
        text = _read(os.path.join(out_dir, name))
        if name.endswith("_pb2.py"):
            text = _normalize_pb2(text)
        result[name] = text
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate or self-check the committed map_service stubs.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="byte-compare regenerated stubs against the committed ones; "
        "exit 0 + PBCHECK-CLEAN on match, unified diff + exit 1 on mismatch. "
        "Requires the pinned protoc/protobuf toolchain (my_app/requirements.txt); "
        "a different protoc version may legitimately emit different stubs — "
        "regenerate with the pinned toolchain before checking.",
    )
    args = parser.parse_args()

    if not args.check:
        generated = regenerate(PROTO_DIR)
        for name, text in generated.items():
            _write(os.path.join(PROTO_DIR, name), text)
            print(f"regenerated {os.path.join(PROTO_DIR, name)}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        generated = regenerate(tmp)
        clean = True
        for name, text in generated.items():
            committed = _read(os.path.join(PROTO_DIR, name))
            if text != committed:
                clean = False
                diff = difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    fromfile=f"{name} (committed)",
                    tofile=f"{name} (regenerated)",
                )
                sys.stdout.writelines(diff)
    if clean:
        print("PBCHECK-CLEAN")
        return 0
    print("PBCHECK-DIRTY")
    return 1


if __name__ == "__main__":
    sys.exit(main())
