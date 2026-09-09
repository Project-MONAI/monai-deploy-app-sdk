#!/usr/bin/env python
"""Phase 10.1 static label <-> description registration suite (plan 10-01,
Task 3; OUT-03 — the bug class the pixel gate is blind to).

House pattern (mirrors scripts/test_merge_remap.py): standalone executable,
NOT pytest; FAILURES counter; per-check [PASS]/[FAIL]; exits 0 iff every
invariant holds. Headless by construction: CPU only, no GPU, no DICOM
corpus, no model files — the ground truth is the byte-identical oracle
copy of ai_segment_descriptions.py itself plus the frozen EXPECTED_LABELS
literal (extracted from that copy and cross-verified 117/117 against the
pinned 44238 oracle SEG dcm's SegmentLabel sequence).

Attribute facts (plan-check 2026-08-27, verified live): SegmentDescription
objects expose NO public segment_label / algorithm_name / algorithm_version
attributes — only the private _segment_label, _segmented_property_category,
_segmented_property_type, and _algorithm_identification (a highdicom
AlgorithmIdentificationSequence with .name/.version).

Name spaces: sd._segment_label is the DISPLAY name ("Spleen", "Right
Kidney") — what the SEG writer writes into the DICOM SegmentLabel tag.
volume_labels keys are DOTTED identifiers ("spleen", "r.kidney") used by
the metrics/SR side. The two are DIFFERENT name spaces; this test NEVER
compares a display name to a key.

Invariants:
  1. len(ai_segment_descriptions) == 117
  2. len(volume_labels) == 118; volume_labels["background"] == 0;
     set(volume_labels.values()) == set(range(118))
  3. (core) tuple(sd._segment_label for sd in ai_segment_descriptions) ==
     the frozen 117-entry EXPECTED_LABELS literal. List position i
     (1-based) is unified label i — the contract the SEG writer consumes
     (to_segment_description numbers the list 1-based in list order). A
     re-sorted, shortened, or renamed list fails here.
  4. Per-description consistency: for every sd —
     _algorithm_identification.name == "CT TotalSegmentator Segmentation",
     .version == "2.14.0", _segmented_property_category is not None,
     _segmented_property_type is not None, _segment_label non-empty and
     len <= 64 (DICOM LO).
  5. Part-range coverage: the 5 intervals (offset, offset + cap] built
     from config.MODEL_PARTS offsets (0/24/50/68/91) and
     EXPECTED_MAX_LOCAL_LABEL caps (24/26/18/23/26) are pairwise disjoint
     and their union == set(1..117) — guards the offset/cap table the
     117-label SEG space is built from.
  6. Uniqueness: len({sd._segment_label ...}) == 117

Exit codes:
  0 — every invariant passes
  1 — a registration invariant failed (a real divergence to escalate, NOT
      a test to relax)
  2 — environment missing (import chain unavailable — e.g. SDK-less env);
      distinct code so CI separates env from regression

Run:
  cd examples/apps/totalsegmentator-fast
  /tmp/monai-env/.venv/bin/python scripts/test_label_registration.py
"""

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

try:
    from my_app.ai_segment_descriptions import ai_segment_descriptions, volume_labels  # noqa: E402
    from my_app.config import EXPECTED_MAX_LOCAL_LABEL, MODEL_PARTS  # noqa: E402
except ImportError as e:
    print(f"ENV-MISSING: {e!r} (SDK/pydicom unavailable in this environment)")
    sys.exit(2)

# Frozen 117-entry display-name literal (extracted from the byte-identical
# oracle copy of ai_segment_descriptions.py on 2026-08-27; cross-verified
# 117/117 against the pinned 44238 oracle SEG dcm's SegmentLabel sequence —
# [0]="Spleen", [116]="Costal Cartilages"). The ONLY ground truth a
# headless test can use. Do not re-derive from volume_labels keys — there
# is no mechanical transform ("r.kidney" -> "Right Kidney",
# "c1.vrtbrae" -> "C1 Vertebra").
EXPECTED_LABELS = [
    "Spleen",
    "Right Kidney",
    "Left Kidney",
    "Gallbladder",
    "Liver",
    "Stomach",
    "Pancreas",
    "Right Adrenal Gland",
    "Left Adrenal Gland",
    "Left Lung Upper Lobe",
    "Left Lung Lower Lobe",
    "Right Lung Upper Lobe",
    "Right Lung Middle Lobe",
    "Right Lung Lower Lobe",
    "Esophagus",
    "Trachea",
    "Thyroid",
    "Small Bowel",
    "Duodenum",
    "Colon",
    "Urinary Bladder",
    "Prostate",
    "Left Kidney Cyst",
    "Right Kidney Cyst",
    "Sacrum",
    "S1 Vertebra",
    "L5 Vertebra",
    "L4 Vertebra",
    "L3 Vertebra",
    "L2 Vertebra",
    "L1 Vertebra",
    "T12 Vertebra",
    "T11 Vertebra",
    "T10 Vertebra",
    "T9 Vertebra",
    "T8 Vertebra",
    "T7 Vertebra",
    "T6 Vertebra",
    "T5 Vertebra",
    "T4 Vertebra",
    "T3 Vertebra",
    "T2 Vertebra",
    "T1 Vertebra",
    "C7 Vertebra",
    "C6 Vertebra",
    "C5 Vertebra",
    "C4 Vertebra",
    "C3 Vertebra",
    "C2 Vertebra",
    "C1 Vertebra",
    "Heart",
    "Aorta",
    "Pulmonary Vein",
    "Brachiocephalic Trunk",
    "Right Subclavian Artery",
    "Left Subclavian Artery",
    "Right Common Carotid Artery",
    "Left Common Carotid Artery",
    "Left Brachiocephalic Vein",
    "Right Brachiocephalic Vein",
    "Left Atrial Appendage",
    "Superior Vena Cava",
    "Inferior Vena Cava",
    "Portal Vein and Splenic Vein",
    "Left Iliac Artery",
    "Right Iliac Artery",
    "Left Iliac Vein",
    "Right Iliac Vein",
    "Left Humerus",
    "Right Humerus",
    "Left Scapula",
    "Right Scapula",
    "Left Clavicle",
    "Right Clavicle",
    "Left Femur",
    "Right Femur",
    "Left Hip",
    "Right Hip",
    "Spinal Cord",
    "Left Gluteus Maximus",
    "Right Gluteus Maximus",
    "Left Gluteus Medius",
    "Right Gluteus Medius",
    "Left Gluteus Minimus",
    "Right Gluteus Minimus",
    "Left Autochthon",
    "Right Autochthon",
    "Left Iliopsoas",
    "Right Iliopsoas",
    "Brain",
    "Skull",
    "Left Rib 1",
    "Left Rib 2",
    "Left Rib 3",
    "Left Rib 4",
    "Left Rib 5",
    "Left Rib 6",
    "Left Rib 7",
    "Left Rib 8",
    "Left Rib 9",
    "Left Rib 10",
    "Left Rib 11",
    "Left Rib 12",
    "Right Rib 1",
    "Right Rib 2",
    "Right Rib 3",
    "Right Rib 4",
    "Right Rib 5",
    "Right Rib 6",
    "Right Rib 7",
    "Right Rib 8",
    "Right Rib 9",
    "Right Rib 10",
    "Right Rib 11",
    "Right Rib 12",
    "Sternum",
    "Costal Cartilages",
]

FAILURES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILURES
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES += 1
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not ok else ""))


def main() -> int:
    # ------------------------------------------------------- invariant 1
    check(
        "1 len(ai_segment_descriptions) == 117",
        len(ai_segment_descriptions) == 117,
        f"got {len(ai_segment_descriptions)}",
    )

    # ------------------------------------------------------- invariant 2
    ok = (
        len(volume_labels) == 118
        and volume_labels.get("background") == 0
        and set(volume_labels.values()) == set(range(118))
    )
    check(
        "2 volume_labels: 118 entries, background=0, values == 0..117",
        ok,
        f"len={len(volume_labels)} bg={volume_labels.get('background')!r} "
        f"missing={sorted(set(range(118)) - set(volume_labels.values()))[:5]}",
    )

    # ------------------------------------------------------- invariant 3
    got = tuple(sd._segment_label for sd in ai_segment_descriptions)
    if got == tuple(EXPECTED_LABELS):
        check(
            "3 registration order: _segment_label[i] == EXPECTED_LABELS[i] (117/117)",
            True,
        )
    else:
        diffs = [(i, g, e) for i, (g, e) in enumerate(zip(got, EXPECTED_LABELS)) if g != e]
        check(
            "3 registration order: _segment_label[i] == EXPECTED_LABELS[i]",
            False,
            f"len got={len(got)} expected={len(EXPECTED_LABELS)}; " f"first diffs={diffs[:5]}",
        )

    # ------------------------------------------------------- invariant 4
    bad = []
    for i, sd in enumerate(ai_segment_descriptions):
        ident = sd._algorithm_identification
        label = sd._segment_label
        if (
            ident is None
            or ident.name != "CT TotalSegmentator Segmentation"
            or ident.version != "2.14.0"
            or sd._segmented_property_category is None
            or sd._segmented_property_type is None
            or not label
            or len(label) > 64  # DICOM LO
        ):
            bad.append(i)
    check(
        "4 per-description: algorithm name/version + property cat/type + label (LO<=64)",
        not bad,
        f"bad indices={bad[:10]}",
    )

    # ------------------------------------------------------- invariant 5
    intervals = []
    for part in MODEL_PARTS:
        off = part["label_offset"]
        cap = EXPECTED_MAX_LOCAL_LABEL[part["name"]]
        intervals.append(set(range(off + 1, off + cap + 1)))
    ok = True
    detail = ""
    for a in range(len(intervals)):
        for b in range(a + 1, len(intervals)):
            inter = intervals[a] & intervals[b]
            if inter:
                ok = False
                detail = f"overlap {MODEL_PARTS[a]['name']}x{MODEL_PARTS[b]['name']}={sorted(inter)[:5]}"
    union = set().union(*intervals) if intervals else set()
    if union != set(range(1, 118)):
        ok = False
        detail += (
            f" union missing={sorted(set(range(1, 118)) - union)[:5]} extra={sorted(union - set(range(1, 118)))[:5]}"
        )
    check(
        "5 part intervals (offset, offset+cap] pairwise disjoint, union == 1..117",
        ok,
        detail,
    )

    # ------------------------------------------------------- invariant 6
    n_unique = len({sd._segment_label for sd in ai_segment_descriptions})
    check(
        "6 uniqueness: 117 distinct _segment_label values",
        n_unique == 117,
        f"got {n_unique}",
    )

    n = 6  # total invariants
    print(f"{n - FAILURES} checks passed, {FAILURES} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
