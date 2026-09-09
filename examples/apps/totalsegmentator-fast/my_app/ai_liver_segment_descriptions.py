# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""ai_liver_segment_descriptions.py — 8-entry 99COUINAUD table for the
`liver_segments` crop-cascade task (Phase 18, plan 18-02, LIV-01).

Ported VERBATIM (semantic values) from the MAP oracle
`ct-totalsegmentator-map/app_liver_segments/ai_segment_descriptions.py`
(read-only provenance). The SNOMED CT Browser contains no Couinaud liver
segment codes, so the oracle uses a PRIVATE scheme (`99COUINAUD`,
LIVSEG1..8) inside the standard SegmentDescription structure — that is
how this app satisfies the "SNOMED descriptions" requirement, exactly as
the oracle does.

Consumed by app.py's crop_cascade branch via the registry `snomed`
attribute (`liver_segment_descriptions` — importlib + getattr, no
hard-coded module name in app.py):
  * `liver_segment_descriptions` -> SEG writer segment_descriptions (8)
  * `liver_volume_labels`        -> metrics labels_dict / SR rows (8 keys,
    NO "background" — the app's metrics dicts exclude background, matching
    how the total path builds _metrics_labels)
  * `liver_algorithm_name` / `liver_algorithm_version` / `liver_algorithm_family`
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

# general algorithm information (MAP oracle values, verbatim)
_liver_algorithm_name = "CT TotalSegmentator Liver Segments Segmentation"
_liver_algorithm_family = codes.DCM.ArtificialIntelligence
_liver_algorithm_version = "2.17.0"  # weights pulled from PyPi TotalSegmentator==2.17.0

_LIVER_SEG_SCHEME = "99COUINAUD"  # private scheme

__all__ = [
    "liver_segment_descriptions",
    "liver_volume_labels",
    "liver_algorithm_name",
    "liver_algorithm_family",
    "liver_algorithm_version",
]

# The SNOMED CT Browser does not contain the Couinaud Liver Segment codes, so a
# private scheme is used
# https://browser.ihtsdotools.org/?perspective=full&conceptId1=404684003&edition=MAIN/2026-05-01&release=&languages=en

# create the required segment description for each segment with
# the actual algorithm and the pertinent organ/tissue; the segment_label,
# algorithm_name, and algorithm_version are of DICOM VR LO type, limited to
# 64 chars
# https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
liver_segment_descriptions = [
    SegmentDescription(
        segment_label="Liver Segment I",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG1",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment I",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment II",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG2",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment II",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment III",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG3",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment III",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment IV",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG4",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment IV",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment V",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG5",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment V",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment VI",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG6",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment VI",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment VII",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG7",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment VII",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
    SegmentDescription(
        segment_label="Liver Segment VIII",
        segmented_property_category=codes.SCT.BodyStructure,
        segmented_property_type=Code(
            value="LIVSEG8",
            scheme_designator=_LIVER_SEG_SCHEME,
            meaning="Couinaud Liver Segment VIII",
        ),
        algorithm_name=_liver_algorithm_name,
        algorithm_family=_liver_algorithm_family,
        algorithm_version=_liver_algorithm_version,
    ),
]

# define labels for custom volume transforms
# abbreviated in some cases to not exceed VR SH max length of 16
# NOTE: the MAP oracle's dict includes "background": 0; this app's metrics
# dicts EXCLUDE background (the total path builds
# _metrics_labels = {k: v for k, v in volume_labels.items() if k != "background"}),
# so this dict ships the 8 foreground keys only.
liver_volume_labels = {
    "cnd.lv.seg.1": 1,
    "cnd.lv.seg.2": 2,
    "cnd.lv.seg.3": 3,
    "cnd.lv.seg.4": 4,
    "cnd.lv.seg.5": 5,
    "cnd.lv.seg.6": 6,
    "cnd.lv.seg.7": 7,
    "cnd.lv.seg.8": 8,
}

# ModelInfo fields for the P10 output chain (app.py resolves them from this
# module via the registry `snomed` attribute — getattr, no hard-coded names)
liver_algorithm_name = _liver_algorithm_name
liver_algorithm_family = _liver_algorithm_family
liver_algorithm_version = _liver_algorithm_version
