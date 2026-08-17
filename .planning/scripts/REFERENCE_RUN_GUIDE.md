# Reference MAP Run Guide — `cchmc_nnunet_fifteen_ckpt_app`

How to run the **reference** nnUNet MAP (the baseline `cchmc_nnunet_fifteen_ckpt_app`)
pythonically to regenerate the ground-truth SC/SEG/SR outputs. Use this any time you
need a freshly regenerated reference to validate the new `cchmc-nnunet-fast` app against
(Phase 1 pixel-exact gate).

**Last verified:** 2026-08-17 — fresh run reproduced the historical `testdata/airway_output`
(see Phase 0 VERIFICATION for the byte-level comparison).

## Prerequisites

- venv: `/tmp/monai-env/.venv` (has holoscan-cu13, monai 1.3.0, itk, nnunetv2 2.8.1 editable, etc.)
- Driver: 610.57.04 (CUDA 13.3), A100 — required for `torch` CUDA availability.
- Models present at `examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`
  (`MRI_NICU-Airway_TRAINv2`: 3d_fullres, 3d_lowres, 3d_cascade_fullres; **2d absent**).

> ⚠ venv is scratch (`/tmp`). After any `monai` reinstall, re-apply the two NumPy-2.0
> `ndarray.ptp()` patches in venv monai 1.3.0 (`data/utils.py`, `transforms/spatial/functional.py`).

## Steps

1. **cd into the reference app root** (the `.env` and `my_app/` package live here —
   run from here so the local package wins over the `cchmc-nnunet-fast` `my_app` collision):

   ```bash
   cd examples/apps/cchmc_nnunet_fifteen_ckpt_app
   ```

2. **Activate the venv:**

   ```bash
   source /tmp/monai-env/.venv/bin/activate
   ```

3. **Load the environment variables** (already committed in `./.env`; adjust paths if needed):

   ```bash
   source .env
   # HOLOSCAN_INPUT_PATH=/users/srv-mde/projects/monai-deploy-app-sdk/testdata/airway_input
   # HOLOSCAN_MODEL_PATH=models
   # HOLOSCAN_OUTPUT_PATH=/users/srv-mde/projects/monai-deploy-app-sdk/testdata/current_output
   ```

4. **(Optional) clear the output dir** before running:

   ```bash
   rm -rf "$HOLOSCAN_OUTPUT_PATH"
   ```

5. **Run the bundle pythonically** (the exact command from the app `README.md`):

   ```bash
   python my_app -i "$HOLOSCAN_INPUT_PATH" -o "$HOLOSCAN_OUTPUT_PATH" -m "$HOLOSCAN_MODEL_PATH"
   ```

   Expected: exit 0; SC/SEG/SR written to `$HOLOSCAN_OUTPUT_PATH` (~124 s on A100).

## Compare against historical ground truth

The historical ground truth is `testdata/airway_output/{SC,SEG,SR}`. A fresh run writes
to `testdata/current_output/{SC,SEG,SR}` (note: SOP UIDs differ between runs — compare
segment **pixel data + geometry**, not file names/UIDs).

Quick SEG parity check (1-bit binary segmentation):

```bash
/tmp/monai-env/.venv/bin/python - <<'PY'
import glob, numpy as np
from pydicom import dcmread
def seg(p):
    ds = dcmread(glob.glob(p+"/*.dcm")[0], force=True)
    return ds, np.frombuffer(bytes(ds.PixelData), dtype=np.uint8)
hds,hb = seg("testdata/airway_output/SEG")
c,cb   = seg("testdata/current_output/SEG")
hbv = int((np.unpackbits(hb, bitorder='little')>0).sum())
cbv = int((np.unpackbits(cb, bitorder='little')>0).sum())
same = int((hb==cb).sum())
print("HIST voxels:",hvb," CUR voxels:",cbv,
      " byte-identical %: %.4f"%(100*same/len(hb)))
PY
```

Pass bar (Phase 0 baseline): **≥99.9% byte-identical** and near-identical segment voxel
counts (2026-08-17 result: 99.902% identical, 2430 vs 2447 voxels — differences confined
to the airway band). The `cchmc-nnunet-fast` app in Phase 1 must meet the same bar *and*
satisfy the `≥5 CT studies` corpus (TEST-01) once supplied.

## Notes / hazards

- **Do not** `python -m my_app` from the repo root or from `cchmc-nnunet-fast` — the
  `cchmc-nnunet-fast` editable install registers a meta-path finder that maps package
  `my_app` to the *new* skeleton. Always run from the reference app root.
- **Docker build + container test are deferred** until after Phase 3 optimizations are in
  place (decided 2026-08-17). Pythonic runs above are the validation path until then.
