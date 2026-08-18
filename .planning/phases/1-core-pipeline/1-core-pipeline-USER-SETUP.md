# Phase 1 (Core Pipeline): User Setup

**Generated:** 2026-08-18
**Phase:** 1-core-pipeline
**Status:** Complete (verified 2026-08-18)

## Setup Items

| Status | Item | Notes |
|--------|------|-------|
| [x] | Fresh reference output at `testdata/current_output` | Already regenerated during Phase 0 closure (2026-08-17); `SC/`, `SEG/`, `SR/` present. 99.902% byte-identical to historical `testdata/airway_output` — accepted as the pixel-exact gate target. Regenerate any time via `.planning/scripts/REFERENCE_RUN_GUIDE.md`. |

## Verification

```bash
# Reference output present
ls testdata/current_output/{SC,SEG,SR}
```

No further human action required for Plan 01.
