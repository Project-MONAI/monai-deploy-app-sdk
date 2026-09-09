# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""ai_body_descriptions.py — 2-entry SNOMED table for the `body` single-model
task (Phase 19, plan 19-01, BODY-01).

Ported VERBATIM (semantic values) from the TS 2.18 mapping
`resources/totalsegmentator_snomed_mapping.csv` rows 162-163:
  body_trunc       -> SCT 22943007 "Trunk"
  body_extremities -> SCT 66019005 "Limb"
category SCT 123037004 (Anatomical Structure) for both.

NOTE: pydicom's codes.SCT has no `Limb` entry, so BOTH property types are
constructed via raw Code(value, scheme_designator, meaning) to keep the
table provably TS-CSV-faithful.

Consumed by app.py's single_model branch via the registry `snomed`
attribute (importlib + getattr, no hard-coded module name in app.py):
  * `body_descriptions`  -> SEG writer segment_descriptions (2)
  * `body_volume_labels` -> metrics labels_dict / SR rows (2 keys, NO
    "background" — same convention as the liver table)
  * `body_algorithm_name` / `body_algorithm_family` / `body_algorithm_version`
    -> ModelInfo for the P10 output chain
"""

# required for setting SegmentDescription attributes
# direct import as this is not part of App SDK package
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

try:  # package-style import (my_app.*)
    from my_app.dicom_seg_writer_operator import SegmentDescription
except ImportError:  # flat import (my_app dir on sys.path, as the app runner provides)
    from dicom_seg_writer_operator import SegmentDescription

# general algorithm information
_body_algorithm_name = "CT TotalSegmentator Body Segmentation"
_body_algorithm_family = codes.DCM.ArtificialIntelligence
_body_algorithm_version = "2.18.0"  # TS 2.18.0 task 299 (v2.0.0-weights)

__all__ = [
    "body_descriptions",
    "body_volume_labels",
    "body_algorithm_name",
    "body_algorithm_family",
    "body_algorithm_version",
]

# SNOMED codes from totalsegmentator_snomed_mapping.csv rows 162-163.
# pydicom lacks codes.SCT.Limb, so both rows use raw Code construction.
body_descriptions = [
    SegmentDescription(
        segment_label="body_trunc",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="22943007",
            scheme_designator="SCT",
            meaning="Trunk",
        ),
        algorithm_name=_body_algorithm_name,
        algorithm_family=_body_algorithm_family,
        algorithm_version=_body_algorithm_version,
    ),
    SegmentDescription(
        segment_label="body_extremities",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="66019005",
            scheme_designator="SCT",
            meaning="Limb",
        ),
        algorithm_name=_body_algorithm_name,
        algorithm_family=_body_algorithm_family,
        algorithm_version=_body_algorithm_version,
    ),
]

# define labels for custom volume transforms
# NOTE: excludes "background" (same convention as liver_volume_labels — the
# app's metrics dicts exclude background).
body_volume_labels = {
    "body_trunc": 1,
    "body_extremities": 2,
}

# ModelInfo fields for the P10 output chain (app.py resolves them from this
# module via the registry `snomed` attribute — getattr, no hard-coded names)
body_algorithm_name = _body_algorithm_name
body_algorithm_family = _body_algorithm_family
body_algorithm_version = _body_algorithm_version
