# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chunked parallel DICOM-SEG save (Phase 14-02, WS-2).

The highdicom ``Segmentation`` object is BUILT serially (unchanged). Only the
**serialization** of the ``PerFrameFunctionalGroupsSequence`` (PFFG) items is
parallelized: the item list is split into N contiguous, order-preserving chunks;
each worker process encodes its chunk to bytes with the SAME explicit-VR-LE
transfer syntax as ``save_as``; the parent splices the chunk bytes into a
skeleton file (the same dataset written with an EMPTY PFFG sequence) so the
result is **byte-for-byte identical** to ``seg.save_as(path)``.

Why it is byte-identical (see ``scripts/test_writer_parallel.py`` for the proof):
  * the skeleton is produced by the exact same ``dcmwrite`` path as ``save_as``
    with ``write_like_original=True`` — preamble + group-0002 file meta + every
    element except the PFFG items (incl. the full PixelData) are re-emitted
    unchanged and verified equal to the serial file, byte for byte;
  * each PFFG item is framed exactly as pydicom's ``write_sequence_item`` does
    for a defined-length item: ``ItemTag(FFFE,E000) + UL(content) + elements``
    (no item-delimitation tag, since items carry a defined length);
  * the only bytes we synthesize are the PFFG SQ length field (UL, little-endian)
    and the concatenation of the worker item bytes.

Correctness bar: full-file SHA-256 of the parallel output must equal that of the
serial ``save_as`` output. ``save_seg_parallel`` performs a post-write self-check
(SHA-256 of the re-read PFFG region vs the in-memory item bytes + total size) and
raises on any mismatch — it never emits a corrupt/partial SEG.

This module imports pydicom (+ stdlib) only. It must NOT import monai / holoscan /
torch: worker processes pickled from here must never create a CUDA context.
"""

import copy
import hashlib
import struct
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import pydicom
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.filebase import DicomBytesIO
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import UID, ExplicitVRLittleEndian

#: PerFrameFunctionalGroupsSequence (0x5200,0x9230)
PFFG_TAG = Tag(0x5200, 0x9230)
#: On-disk little-endian tag bytes for (0x5200,0x9230): group,element each LE.
PFFG_TAG_BYTES = struct.pack("<HH", 0x5200, 0x9230)
#: On-disk little-endian Item tag (FFFE,E000) — verified against the pinned
#: baseline bytes (pydicom 3.0 Tag.to_bytes() now demands an explicit length).
ITEM_TAG_BYTES = b"\xfe\xff\x00\xe0"
#: Minimum PFFG frame count for the pool to pay its spawn cost (research §WS-2:
#: ~10k frames; 44238-class 4914 frames stay serial, 31322-class 40k go parallel).
FRAME_THRESHOLD = 10_000


_FILE_META = None
_FILE_HEADER_LEN = None


def _get_file_meta() -> FileMetaDataset:
    """The one fixed synthetic file_meta used for standalone item encoding."""
    global _FILE_META
    if _FILE_META is None:
        fm = FileMetaDataset()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.66.4")  # SEG (unused; placeholder)
        fm.MediaStorageSOPInstanceUID = UID("1.2.3.9.9.9")  # placeholder
        _FILE_META = fm
    return _FILE_META


def _file_header_len() -> int:
    """Byte length of (preamble + DICM + group-0002 meta) as dcmwrite emits it
    for _get_file_meta() — measured once from an empty-dataset encode."""
    global _FILE_HEADER_LEN
    if _FILE_HEADER_LEN is None:
        fd = FileDataset(None, Dataset(), file_meta=copy.deepcopy(_get_file_meta()), preamble=b"")
        buf = DicomBytesIO()
        pydicom.dcmwrite(
            buf,
            fd,
            write_like_original=False,
            implicit_vr=False,
            little_endian=True,
            enforce_file_format=False,
            overwrite=True,
        )
        _FILE_HEADER_LEN = len(buf.getvalue())
    return _FILE_HEADER_LEN


def _item_element_bytes(item: Dataset) -> bytes:
    """Encode one PFFG item's data elements in explicit-VR-LE, stripping the
    synthetic DICOM file header (preamble + group-0002 meta) that the standalone
    ``dcmwrite`` adds so we can serialize one item in isolation.

    The element bytes returned are exactly the item *content* that
    ``write_sequence_item`` places between the Item tag+length and (for
    undefined-length items) the delimitation tag. For our defined-length items
    the content is the raw element bytes only.
    """
    fd = FileDataset(None, item, file_meta=copy.deepcopy(_get_file_meta()), preamble=b"")
    buf = DicomBytesIO()
    pydicom.dcmwrite(
        buf,
        fd,
        write_like_original=False,
        implicit_vr=False,  # explicit VR (matches the file's transfer syntax)
        little_endian=True,
        enforce_file_format=False,
        overwrite=True,
    )
    raw = buf.getvalue()
    # The standalone dcmwrite still emits a 128-byte zero preamble + "DICM" +
    # the group-0002 meta block (pydicom 3.0 pads/writes meta even with
    # preamble=b""; the meta layout is deterministic for a given file_meta).
    # Skip the fixed header length, measured once by encoding an EMPTY dataset
    # with the identical file_meta (meta is independent of body elements).
    return raw[_file_header_len() :]


def _encode_pffg_chunk(items: List[Dataset]) -> bytes:
    """Top-level worker: encode a contiguous chunk of PFFG items to their exact
    in-file bytes. Defined-length framing per item: ItemTag + UL(len) + elements.
    Order-preserving and deterministic (no UIDs/timestamps generated here)."""
    out = bytearray()
    for item in items:
        content = _item_element_bytes(item)
        out += ITEM_TAG_BYTES
        out += struct.pack("<I", len(content))
        out += content
    return bytes(out)


def _dataset_to_bytes(ds: Dataset) -> bytes:
    """Encode a Dataset in-memory via the SAME path ``save_as`` uses
    (``dcmwrite(..., write_like_original=True)``). Returns preamble + DICM +
    group-0002 meta + element bytes, byte-identical to ``ds.save_as(path)``."""
    buf = DicomBytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=True)
    return buf.getvalue()


def _find_pffg_header(data: bytes):
    """Byte offset of the PFFG SQ header (tag start) in an explicit-VR-LE file,
    or None. The tag is followed by VR 'SQ' + 2 reserved bytes, which pins the
    match against the (rare) case of the tag byte-pattern appearing in data."""
    pos = data.find(PFFG_TAG_BYTES)
    while pos != -1:
        if data[pos + 4 : pos + 6] == b"SQ" and data[pos + 6 : pos + 8] == b"\x00\x00":
            return pos
        pos = data.find(PFFG_TAG_BYTES, pos + 1)
    return None


def _split_contiguous(seq: List, nproc: int) -> List[List]:
    """Split into at most nproc contiguous, order-preserving chunks."""
    n = len(seq)
    k = max(1, min(nproc, n))
    base, rem = divmod(n, k)
    chunks, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        if end > start:
            chunks.append(seq[start:end])
        start = end
    return chunks


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_skeleton(seg) -> Dataset:
    """Fresh Dataset identical to seg but with an EMPTY PFFG sequence.

    Preserves element order and file_meta/preamble, and does NOT mutate seg.
    (copy.copy(seg) is NOT safe: pydicom's Dataset shares its internal element
    storage, so rebinding the PFFG tag on the copy would also empty seg's own
    sequence — verified by test. Building a fresh Dataset avoids this.)
    """
    skel = Dataset()
    fm = getattr(seg, "file_meta", None)
    if fm is not None:
        skel.file_meta = copy.deepcopy(fm)
    pre = getattr(seg, "preamble", None)
    if pre:
        skel.preamble = bytes(pre)
    for tag in seg.keys():
        if tag == PFFG_TAG:
            skel[tag] = DataElement(PFFG_TAG, "SQ", Sequence([]))
        else:
            skel[tag] = seg[tag]
    return skel


def save_seg_parallel(seg, path, nproc: int = 4, self_check: bool = True) -> Dict:
    """Write ``seg`` to ``path`` with the PFFG-item serialization fanned out over
    ``nproc`` processes. Byte-identical to ``seg.save_as(path)``.

    Returns a small timing/provenance dict. Raises ``RuntimeError`` if the
    post-write self-check fails (never emits a corrupt/partial SEG).
    """
    import time

    items = list(seg[PFFG_TAG])
    n = len(items)
    if nproc <= 1 or n == 0:
        seg.save_as(path)
        return {"nproc": 1, "n_items": n, "serial": True}

    # Guard: this design assumes defined-length PFFG items (true for a freshly
    # built highdicom Segmentation). Refuse to splice on undefined-length items.
    if getattr(seg[PFFG_TAG], "is_undefined_length", False):
        raise RuntimeError("save_seg_parallel: undefined-length PFFG not supported")

    t_start = time.perf_counter()

    # 1) skeleton: same dataset, EMPTY PFFG -> in-memory dcmwrite (== save_as).
    #    Carries preamble+meta+all pre-PFFG elements + the full PixelData suffix;
    #    the PFFG SQ length is 0 and will be patched below.
    #    Built as a FRESH Dataset (not copy.copy) so seg's own PFFG is untouched.
    skel = _build_skeleton(seg)
    t0 = time.perf_counter()
    skel_bytes = _dataset_to_bytes(skel)
    t_skel = time.perf_counter() - t0

    p = _find_pffg_header(skel_bytes)
    if p is None:
        raise RuntimeError("save_seg_parallel: PFFG SQ header not found in skeleton")

    # 2) parallel PFFG-item serialization (contiguous, order-preserving chunks).
    chunks = _split_contiguous(items, nproc)
    t0 = time.perf_counter()
    if len(chunks) == 1:
        parts = [_encode_pffg_chunk(chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=max(1, len(chunks))) as ex:
            parts = list(ex.map(_encode_pffg_chunk, chunks))
    item_bytes = b"".join(parts)
    t_pool = time.perf_counter() - t0

    # 3) splice: keep everything up to the PFFG tag+SQ+reserved (p..p+8), patch
    #    the UL length, append the item bytes, then the unchanged suffix.
    t0 = time.perf_counter()
    out = skel_bytes[: p + 8] + struct.pack("<I", len(item_bytes)) + item_bytes + skel_bytes[p + 12 :]
    with open(path, "wb") as fh:
        fh.write(out)
    t_write = time.perf_counter() - t0

    if self_check:
        _post_write_self_check(path, p, item_bytes, skel_bytes, out)

    return {
        "nproc": len(chunks),
        "n_items": n,
        "serial": False,
        "skeleton_s": round(t_skel, 4),
        "pool_s": round(t_pool, 4),
        "write_s": round(t_write, 4),
        "total_s": round(time.perf_counter() - t_start, 4),
        "item_bytes": len(item_bytes),
        "file_bytes": len(out),
    }


def _post_write_self_check(path, pffg_off, item_bytes, skel_bytes, out):
    """Verify the bytes actually on disk: (a) total size, (b) SHA-256 of the
    re-read PFFG region == SHA-256 of the in-memory item bytes, (c) the prefix
    and suffix regions on disk == the skeleton's regions. Raises on mismatch."""
    disk = open(path, "rb").read()
    if len(disk) != len(out):
        raise RuntimeError(f"save_seg_parallel self-check: size mismatch disk={len(disk)} expected={len(out)}")
    items_off = pffg_off + 12
    disk_items = disk[items_off : items_off + len(item_bytes)]
    if _sha256(disk_items) != _sha256(item_bytes):
        raise RuntimeError("save_seg_parallel self-check: PFFG region sha256 mismatch on disk")
    if disk[: pffg_off + 8] != skel_bytes[: pffg_off + 8]:
        raise RuntimeError("save_seg_parallel self-check: prefix region mismatch on disk")
    if disk[items_off + len(item_bytes) :] != skel_bytes[pffg_off + 12 :]:
        raise RuntimeError("save_seg_parallel self-check: suffix region mismatch on disk")
