# AGENTS.md

## Environment

All testing, development, and dependency resolution for this project must use the uv-managed virtual environment at:

```
/tmp/monai-env/.venv
```

### Activating the environment

```bash
source activate-env.sh
```

### Running commands

Always prefix commands with the venv's Python/pip or activate the environment first:

```bash
/tmp/monai-env/.venv/bin/python <script>
/tmp/monai-env/.venv/bin/pip install <package>
```

### Key installed packages (editable)

- **nnunetv2** — vendored at `./nnUNet/` (no `.git`, editable install)
- **monai-deploy-app-sdk** — this project root (editable install)

### Important

- Do **not** create new virtual environments or use the system Python for testing.
- Do **not** install `nnunetv2` from PyPI — the vendored editable copy must be used.
- When running tests or example apps, ensure the environment is activated so the correct dependencies (monai, torch, holoscan, etc.) are available.

## Git Commits

After every major code change, commit the work immediately so it's easy to roll back:

```bash
git add <changed files>
git commit -m "<concise description>"
```

- Messages should be **short and imperative** (e.g., `fix: resolve tensor shape mismatch`, `feat: add nnunet checkpoint loader`).
- Commit early and often — prefer small, atomic commits over large bundles.
