# Copyright 2022-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from monai.deploy.utils.importutil import optional_import

nib, _ = optional_import("nibabel")
sitk, _ = optional_import("SimpleITK")
label, _ = optional_import("skimage.measure", name="label")
measure, _ = optional_import("skimage", name="measure")
mesh, _ = optional_import("stl", name="mesh")
resize, _ = optional_import("skimage.transform", name="resize")
gaussian, _ = optional_import("skimage.filters", name="gaussian")
trimesh, _ = optional_import("trimesh")

from monai.deploy.core import ConditionType, Fragment, Image, Operator, OperatorSpec

__all__ = ["STLConversionOperator", "STLConverter"]

# MESHING CONSTANTS
#
# _CROP_MARGIN_VOXELS: source voxels kept around the mask bounding box; at least 1 is required so
# the real zero-valued neighbours are inside the crop and interpolation at the crop faces behaves
# exactly as it did on the full volume; 2 leaves room for a wider (cubic) interpolation kernel
_CROP_MARGIN_VOXELS = 2


# nibabel is required by the dependent class STLConverter.
# @md.env(
#     pip_packages=["numpy>=1.21", "nibabel >= 3.2.1", "numpy-stl>=2.12.0", "scikit-image>=0.17.2", "trimesh>=3.8.11"]
# )
class STLConversionOperator(Operator):
    """Converts volumetric image to surface mesh in STL format.

    If a file path is provided, the STL binary will be saved in the said output folder.
    This operator also saves the STL file as bytes in memory, identified by the named output.
    Being optional, this output does not require any downstream receiver.

    Named inputs:
        image: Image object for which to generate surface mesh.
        output_file: Optional, the path of the file to save the mesh in STL format.
                     If provided, this will override the output file path set on the object.

    Named output:
        stl_bytes: Bytes of the surface mesh STL file. Optional, not requiring a downstream receiver.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_file: Union[Path, str],
        class_id=None,
        is_smooth=True,
        keep_largest_connected_component=True,
        step_size: int = 1,
        smoothing_iterations: int = 20,
        presmooth_sigma: Optional[float] = 0.75,
        resize_order: int = 1,
        fail_on_empty: bool = False,
        **kwargs,
    ) -> None:
        """Creates an object to generate a surface mesh and saves it as an STL file if the path is provided.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            output_file ([Path,str], optional): output STL file path. None for no file output.
            class_id (array, optional): Class label ids. Defaults to None (all non-zero labels).
            is_smooth (bool, optional): Taubin smoothing of the mesh. Defaults to True.
            keep_largest_connected_component (bool, optional): Defaults to True. Applied per class,
                after the class filter.
            step_size (int, optional): marching_cubes step size. Defaults to 1. Values > 1 stride the
                grid, shrink the STL, and will delete small structures.
            smoothing_iterations (int, optional): Taubin smoothing iterations. Defaults to 20.
            presmooth_sigma (float, optional): Gaussian sigma, in isotropic voxels, applied to the
                resampled mask before meshing to suppress slice-direction terracing. Defaults to 0.75.
                None or 0 disables it. Affects mesh geometry only.
            resize_order (int, optional): interpolation order for the isotropic resample. Defaults to 1.
                Use 3 (cubic) for a smoother surface at some extra cost.
            fail_on_empty (bool, optional): raise instead of returning empty bytes when the requested
                class is absent. Defaults to False.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._class_id = class_id
        self._is_smooth = is_smooth
        self._keep_largest_connected_component = keep_largest_connected_component
        self._step_size = int(step_size)
        self._smoothing_iterations = int(smoothing_iterations)
        self._presmooth_sigma = presmooth_sigma
        self._resize_order = int(resize_order)
        self._fail_on_empty = bool(fail_on_empty)
        self._output_file = Path(output_file) if output_file and len(str(output_file)) > 0 else None
        self._converter = STLConverter(*args, **kwargs)

        self.input_name_image = "image"
        self.input_name_output_file = "output_file"
        self.output_name_stl_bytes = "stl_bytes"

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)
        spec.input(self.input_name_output_file).condition(ConditionType.NONE)  # Optional, set as needed.
        spec.output(self.output_name_stl_bytes).condition(ConditionType.NONE)  # No receivers required.

    def compute(self, op_input, op_output, context):
        """Gets the input (image), processes it and sets results in the output.

        This function sets the mesh in STL bytes in its named output, which requires no receivers.
        If provided, the mesh will be saved to the file in STL format.

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        input_image = op_input.receive(self.input_name_image)
        if input_image is None:
            raise ValueError("Input image is not received.")

        _output_file = op_input.receive(self.input_name_output_file)
        if not _output_file:
            # Use the object's attribute to get the STL output path, if any.
            if self._output_file and len(str(self._output_file)) > 0:
                _output_file = Path(self._output_file)
                _output_file.parent.mkdir(parents=True, exist_ok=True)
                self._logger.info(f"Output will be saved in file {_output_file}.")

        stl_bytes = self._convert(input_image, _output_file)
        op_output.emit(stl_bytes, self.output_name_stl_bytes)

    @property
    def last_volume_mm3(self) -> Optional[float]:
        """Volume in mm^3 of the segmentation meshed by the most recent call, or None."""
        return self._converter.last_volume_mm3

    @property
    def last_fov_truncated(self) -> bool:
        """True when the most recent segmentation reached a face of the acquired volume."""
        return self._converter.last_fov_truncated

    def _convert(self, image: Image, output_file: Optional[Path] = None):
        """
        Args:
            image (Image): object with the image (ndarray in DHW) and its metadata dictionary.
            output_file (Path, optional): output file path. Default None for no file output.

        Returns:
            Bytes: Bytes of the binary of STL file
        """

        # Use path in the output_file arg if provided
        # parents=True, so a nested output tree is created
        if isinstance(output_file, Path):
            output_file.parent.mkdir(parents=True, exist_ok=True)

        return self._converter.convert(
            image=image,
            output_file=output_file,
            class_ids=self._class_id,
            is_smooth=self._is_smooth,
            keep_largest_connected_component=self._keep_largest_connected_component,
            step_size=self._step_size,
            smoothing_iterations=self._smoothing_iterations,
            presmooth_sigma=self._presmooth_sigma,
            resize_order=self._resize_order,
            fail_on_empty=self._fail_on_empty,
        )


class STLConverter:
    """Converts volumetric image to surface mesh in STL"""

    def __init__(self, *args, **kwargs):
        """Creates an instance to generate a surface mesh in STL with an Image object."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        # Measurement state from the most recent convert() call, for the caller to read back
        self.last_volume_mm3: Optional[float] = None
        self.last_voxel_count: int = 0
        self.last_mesh_volume_mm3: Optional[float] = None
        self.last_fov_truncated: bool = False
        self.last_fov_contact_faces: List[str] = []

    def convert(
        self,
        image: Image,
        output_file: Optional[Path] = None,
        class_ids=None,
        is_smooth=True,
        keep_largest_connected_component=True,
        step_size: int = 1,
        smoothing_iterations: int = 20,
        presmooth_sigma: Optional[float] = 0.75,
        resize_order: int = 1,
        fail_on_empty: bool = False,
    ):
        """
        Args:
            image (Image): object with the image (ndarray of DHW index order) and its metadata dictionary.
            output_file (str): output STL file path. Default to None for not saving output file.
            class_ids (array, optional): Class label id(s). Defaults to None (all non-zero labels).
            is_smooth (bool, optional): Taubin smoothing of the mesh. Defaults to True.
            keep_largest_connected_component (bool, optional): Defaults to True. Applied per class,
                after the class filter.
            step_size (int, optional): marching_cubes step size. Defaults to 1. Values > 1 stride the
                grid, shrink the STL, and will delete small structures.
            smoothing_iterations (int, optional): Taubin smoothing iterations. Defaults to 20.
            presmooth_sigma (float, optional): Gaussian sigma, in isotropic voxels, applied to the
                resampled mask before meshing to suppress slice-direction terracing. Defaults to 0.75.
                None or 0 disables it. Affects mesh geometry only.
            resize_order (int, optional): interpolation order for the isotropic resample. Defaults to 1.
                Use 3 (cubic) for a smoother surface at some extra cost.
            fail_on_empty (bool, optional): raise instead of returning empty bytes when the requested
                class is absent. Defaults to False.

        Returns:
            Bytes of the binary STL file, or b"" when the requested class is absent.
        """

        if image is None or not isinstance(image, Image):
            raise ValueError("image is not an Image object.")

        if isinstance(output_file, Path):
            output_file.parent.mkdir(parents=True, exist_ok=True)

        self.last_volume_mm3 = None
        self.last_voxel_count = 0
        self.last_mesh_volume_mm3 = None
        self.last_fov_truncated = False
        self.last_fov_contact_faces = []

        s_image = self.SpatialImage(image)
        nda = s_image.image_array

        # Spacing ordered to match the array axes - the legacy .spacing tuple is (row, col, depth)
        # and must not be paired positionally with a (depth, row, col) array
        spacing_dhw = s_image.spacing_dhw
        if spacing_dhw is None or not np.all(np.isfinite(spacing_dhw)) or np.any(np.asarray(spacing_dhw) <= 0):
            raise ValueError(f"Image spacing/resolution is missing or invalid: {spacing_dhw}")

        self._logger.info(f"Image ndarray shape (depth, row, col): {nda.shape}")
        self._logger.info(f"Spacing matched to array axes (depth, row, col) mm: {spacing_dhw}")
        self._logger.info(f"class_ids: {class_ids}")
        self._logger.info(f"Unique label values in nda: {np.unique(nda)}")

        # This operator reads the volume in DICOM index order and does not re-orient it, so a
        # re-oriented affine means the geometry below cannot be trusted; _load_data assigns both
        # affines from the same source, so this is a guard rather than an expected path
        if (
            s_image.original_affine is not None
            and s_image.affine is not None
            and np.sum(np.abs(np.asarray(s_image.original_affine) - np.asarray(s_image.affine))) > 1e-7
        ):
            self._logger.warning(
                "original_affine differs from affine: the volume appears re-oriented. This operator "
                "assumes DICOM index order and does not re-orient; mesh geometry may be wrong"
            )

        # ------------------------------------------------------------------
        # Select the requested classes, then clean each one up independently
        # Order matters: binarising first would collapse the label identities, leaving every class
        # id except the largest component's resolving to an empty mask
        # ------------------------------------------------------------------
        id_list = self._normalize_class_ids(class_ids)
        binary_mask = np.zeros(nda.shape, dtype=np.uint8)

        if id_list is None:
            component = (np.asarray(nda) > 0).astype(np.uint8)
            self._logger.info(f"all non-zero labels: {int(component.sum())} voxels before CC filter")
            if keep_largest_connected_component:
                component = STLConverter.get_largest_cc(component)
                self._logger.info(f"all non-zero labels: {int(component.sum())} voxels after CC filter")
            np.maximum(binary_mask, component, out=binary_mask)
        else:
            # nda is cast to float32 by SpatialImage, so compare on rounded values rather
            # than relying on exact float equality with an integer label
            rounded = np.rint(np.asarray(nda))
            for class_id in id_list:
                component = (rounded == class_id).astype(np.uint8)
                voxels = int(component.sum())
                self._logger.info(f"class_id {class_id}: {voxels} voxels before CC filter")
                if voxels == 0:
                    self._logger.warning(f"class_id {class_id} is not present in the label volume")
                    continue
                if keep_largest_connected_component:
                    component = STLConverter.get_largest_cc(component)
                    self._logger.info(f"class_id {class_id}: {int(component.sum())} voxels after CC filter")
                np.maximum(binary_mask, component, out=binary_mask)

        # ------------------------------------------------------------------
        # Volume from the voxel count and the source spacing, before any resampling or smoothing
        # ------------------------------------------------------------------
        voxel_volume_mm3 = float(np.prod(np.asarray(spacing_dhw, dtype=np.float64)))
        self.last_voxel_count = int(binary_mask.sum())
        self.last_volume_mm3 = self.last_voxel_count * voxel_volume_mm3
        self._logger.info(
            f"Voxel volume {voxel_volume_mm3:.6f} mm^3; {self.last_voxel_count} voxels; "
            f"segmentation volume {self.last_volume_mm3:.1f} mm^3 ({self.last_volume_mm3 / 1000.0:.2f} mL)"
        )

        # An absent class is a valid result for a patient, not a reason to fail the fragment
        if self.last_voxel_count == 0:
            msg = f"No voxels selected for class_ids={class_ids}; no mesh generated"
            if fail_on_empty:
                raise ValueError(msg)
            self._logger.warning(msg)
            return b""

        # ------------------------------------------------------------------
        # Field-of-view contact - a mask that reaches a face of the acquired volume continues
        # outside it, so the volume derived from it is a lower bound and the mesh gets a flat cap
        # rather than the real anatomical surface
        # ------------------------------------------------------------------
        self.last_fov_contact_faces = STLConverter.find_boundary_contact(binary_mask)
        self.last_fov_truncated = bool(self.last_fov_contact_faces)
        if self.last_fov_truncated:
            self._logger.warning(
                f"Segmentation reaches the edge of the acquired volume on {self.last_fov_contact_faces}. "
                f"The organ is truncated by the field of view, so {self.last_volume_mm3 / 1000.0:.2f} mL "
                f"is a LOWER BOUND on the true volume"
            )
        else:
            self._logger.info("Segmentation is fully inside the acquired volume (no field-of-view contact)")

        # ------------------------------------------------------------------
        # Crop to the mask bounding box before resampling, so the cost of the resample and the
        # blur scales with the organ rather than the acquisition. The margin keeps real
        # zero-valued neighbours inside the crop, so the interpolated surface at the crop faces
        # is identical to what the full volume would have produced
        # ------------------------------------------------------------------
        crop_start, crop_stop = STLConverter.bounding_box(binary_mask, margin=_CROP_MARGIN_VOXELS)
        cropped = binary_mask[crop_start[0] : crop_stop[0], crop_start[1] : crop_stop[1], crop_start[2] : crop_stop[2]]
        crop_shape = cropped.shape
        self._logger.info(
            f"Cropped to mask bounding box {tuple(crop_start)}..{tuple(crop_stop)} -> {tuple(crop_shape)}, "
            f"{100.0 * float(np.prod(crop_shape)) / float(np.prod(nda.shape)):.1f}% of the volume"
        )

        # ------------------------------------------------------------------
        # Isotropic resample for meshing
        # Resample the cropped mask to an isotropic grid, so marching cubes sees equal sampling
        # in every direction; float32 keeps the allocation to half of what the default would be
        # ------------------------------------------------------------------
        target_spacing = float(np.amin(spacing_dhw))

        target_shape = [
            max(int(np.round(float(crop_shape[_j]) * float(spacing_dhw[_j]) / target_spacing)), 1) for _j in range(3)
        ]
        self._logger.info(
            f"Resampling {tuple(crop_shape)} @ "
            f"{tuple(round(float(v), 4) for v in spacing_dhw)} mm -> "
            f"{tuple(target_shape)} @ {target_spacing:.4f} mm isotropic (order={resize_order})"
        )
        resampled = STLConverter.resize_volume(
            cropped.astype(np.float32), output_shape=target_shape, order=int(resize_order)
        )
        resampled = np.clip(np.asarray(resampled, dtype=np.float32), 0.0, 1.0)

        # ------------------------------------------------------------------
        # Zero-pad every face so that a mask touching the edge of the acquisition is capped and
        # the surface closes; marching cubes leaves an open surface wherever the field runs off
        # the array
        # The pad is wide enough that the optional Gaussian kernel decays to zero inside it
        # ------------------------------------------------------------------
        sigma = float(presmooth_sigma) if presmooth_sigma else 0.0
        pad_width = (int(np.ceil(3.0 * sigma)) + 1) if sigma > 0 else 1
        padded = np.pad(resampled, pad_width, mode="constant", constant_values=0.0)

        # Optional Gaussian pre-smoothing; interpolating a binary mask across thin slices leaves
        # flat-topped contours, and Taubin smoothing is too local to remove ridges only a couple
        # of vertices wide; blurring the field before thresholding at 0.5 attenuates them
        # This affects mesh geometry only - the reported volume is computed above, pre-resample
        if sigma > 0:
            self._logger.info(f"Gaussian pre-smoothing of the meshing grid, sigma={sigma:.3f} voxels")
            padded = np.asarray(
                gaussian(padded, sigma=sigma, preserve_range=True, mode="constant", cval=0.0), dtype=np.float32
            )

        # ------------------------------------------------------------------
        # Marching cubes
        # A step_size above 1 strides the grid and will delete small or thin structures
        # allow_degenerate=False keeps degenerate triangles out, so the mesh can be watertight
        # and its enclosed volume meaningful
        # ------------------------------------------------------------------
        try:
            verts, faces, _, _ = measure.marching_cubes(
                padded, level=0.5, step_size=max(int(step_size), 1), allow_degenerate=False
            )
        except (ValueError, RuntimeError) as err:
            msg = f"marching_cubes failed for class_ids={class_ids} ({self.last_voxel_count} voxels): {err}"
            if fail_on_empty:
                raise ValueError(msg) from err
            self._logger.warning(msg)
            return b""

        self._logger.info(f"Mesh from marching_cubes: {verts.shape[0]} vertices, {faces.shape[0]} faces")

        # Remove the pad offset, map from the isotropic grid back to the cropped source grid,
        # then shift by the crop origin so the indices are in full-volume index space again
        verts = np.asarray(verts, dtype=np.float64) - float(pad_width)
        for _j in range(3):
            verts[:, _j] = (verts[:, _j] + 0.5) * float(crop_shape[_j]) / float(resampled.shape[_j]) - 0.5
            verts[:, _j] += float(crop_start[_j])

        # ------------------------------------------------------------------
        # Index -> physical (patient LPS, mm)
        # marching_cubes returns vertices in numpy axis order (depth, row, col), while a
        # SimpleITK continuous index is (i, j, k) == (col, row, depth). Passing them through
        # unreversed transposes the mesh and applies the in-plane spacing to the slice axis
        # ------------------------------------------------------------------
        idx_xyz = verts[:, ::-1]
        verts_phys = s_image.index_to_physical(idx_xyz)

        # Cross-check the vectorized mapping against SimpleITK on one vertex
        try:
            itk_point = np.asarray(
                s_image.itk_image.TransformContinuousIndexToPhysicalPoint([float(v) for v in idx_xyz[0]])
            )
            delta = float(np.max(np.abs(itk_point - verts_phys[0])))
            self._logger.info(f"index->physical agreement with SimpleITK on first vertex: {delta:.3e} mm")
        except Exception as err:  # pragma: no cover - diagnostic only
            self._logger.warning(f"Could not cross-check index->physical against SimpleITK: {err}")

        # ------------------------------------------------------------------
        # Build, orient, smooth and export the mesh
        # Build the mesh directly from the marching-cubes output
        # ------------------------------------------------------------------
        mesh_data = trimesh.Trimesh(vertices=verts_phys, faces=faces, process=True)

        # A direction matrix with negative determinant flips triangle winding
        try:
            if mesh_data.is_watertight and mesh_data.volume < 0:
                self._logger.info("Inverting mesh winding (negative enclosed volume)")
                mesh_data.invert()
        except Exception as err:  # pragma: no cover - diagnostic only
            self._logger.warning(f"Could not evaluate mesh orientation: {err}")

        if is_smooth and int(smoothing_iterations) > 0:
            trimesh.smoothing.filter_taubin(mesh_data, iterations=int(smoothing_iterations))

        # QC: the mesh volume should track the voxel volume closely; A large gap means the mesh
        # degraded (step_size too coarse, over-smoothing, non-manifold surface) or, for a
        # truncated organ, that the flat cap does not follow the anatomy
        if mesh_data.is_watertight:
            self.last_mesh_volume_mm3 = float(abs(mesh_data.volume))
            rel = 100.0 * (self.last_mesh_volume_mm3 - self.last_volume_mm3) / self.last_volume_mm3
            self._logger.info(
                f"Mesh volume {self.last_mesh_volume_mm3 / 1000.0:.2f} mL vs voxel volume "
                f"{self.last_volume_mm3 / 1000.0:.2f} mL ({rel:+.2f}%)"
            )
        else:
            self._logger.warning(
                "Mesh is not watertight; mesh volume not computed. Reported volume remains voxel-based"
            )

        self._logger.info(f"Mesh extents (mm): {np.round(mesh_data.extents, 2)}")
        self._logger.info(f"Mesh bounds (mm): {np.round(mesh_data.bounds, 2).tolist()}")

        stl_bytes = mesh_data.export(file_type="stl")
        if isinstance(stl_bytes, str):
            stl_bytes = stl_bytes.encode("utf-8")

        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(out_path), "wb") as w_file:
                w_file.write(stl_bytes)
            self._logger.info(f"Wrote {len(stl_bytes)} bytes to {out_path}")

        return stl_bytes

    # Helper functions
    @staticmethod
    def _normalize_class_ids(class_ids) -> Optional[List[int]]:
        """Returns a list of int class ids, or None meaning 'all non-zero labels'."""
        if class_ids is None:
            return None
        if isinstance(class_ids, (list, tuple, set, np.ndarray)):
            ids = [int(np.rint(float(c))) for c in np.asarray(list(class_ids)).ravel()]
            return ids if ids else None
        try:
            return [int(np.rint(float(class_ids)))]
        except (TypeError, ValueError) as err:
            raise ValueError(f"That was no valid value for class_id: {class_ids!r}") from err

    @staticmethod
    def bounding_box(binary_mask, margin: int = 0):
        """Start and stop indices of the mask's bounding box, expanded by margin and clipped.

        Returns two 3-tuples suitable for slicing, in (depth, row, col) order. The caller must
        ensure the mask is non-empty.
        """
        starts: List[int] = []
        stops: List[int] = []
        for axis in range(3):
            other_axes = tuple(a for a in range(3) if a != axis)
            present = np.flatnonzero(np.any(binary_mask, axis=other_axes))
            starts.append(max(int(present[0]) - int(margin), 0))
            stops.append(min(int(present[-1]) + 1 + int(margin), int(binary_mask.shape[axis])))
        return tuple(starts), tuple(stops)

    @staticmethod
    def find_boundary_contact(binary_mask) -> List[str]:
        """Names the faces of the volume that the mask touches, in (depth, row, col) index space.

        A non-empty result means the segmentation continues outside the acquired volume, so any
        volume derived from it is a lower bound. Axis 0 is the slice axis; whether its first face
        is superior or inferior depends on the acquisition direction, so faces are named by index
        rather than by anatomy.
        """
        axis_names = ("depth", "row", "col")
        contacts: List[str] = []
        for axis, name in enumerate(axis_names):
            if np.take(binary_mask, 0, axis=axis).any():
                contacts.append(f"{name}_first")
            if np.take(binary_mask, binary_mask.shape[axis] - 1, axis=axis).any():
                contacts.append(f"{name}_last")
        return contacts

    @staticmethod
    def get_largest_cc(nda):
        """Largest connected component of a binary mask, as uint8. Empty in, empty out.

        The input is binarised first, so this is never asked to label a multi-label array, where
        skimage would group by equal value and return whichever organ happened to be largest.
        An empty mask returns an empty mask rather than raising: a patient with no target organ
        is a legitimate result.
        """
        logger = logging.getLogger("{}.{}".format(__name__, "STLConverter"))
        binary = (np.rint(np.asarray(nda)) > 0).astype(np.uint8)
        logger.debug("get_largest_cc ndarray shape: {}".format(binary.shape))

        labels = label(binary)
        if labels.max() == 0:
            logger.warning("get_largest_cc received an empty mask.")
            return np.zeros(binary.shape, dtype=np.uint8)

        counts = np.bincount(labels.ravel())
        counts[0] = 0  # background
        largest_label = int(np.argmax(counts))
        logger.debug("get_largest_cc: {} components, largest has {} voxels".format(labels.max(), counts[largest_label]))
        return (labels == largest_label).astype(np.uint8)

    @staticmethod
    def resize_volume(nda, output_shape, order=1, preserve_range=True, anti_aliasing=False):
        return resize(
            nda, output_shape, order=order, mode="constant", preserve_range=preserve_range, anti_aliasing=anti_aliasing
        )

    @staticmethod
    def write_stl(verts, faces, filename):
        """Writes verts/faces to an STL file via numpy-stl.

        Not used by convert(), which builds the mesh in trimesh and exports it in memory.
        Kept as a standalone helper for callers that have raw marching-cubes output.
        """
        # Create the mesh
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = verts[f[j], :]

        cube.save(os.path.splitext(filename)[0] + ".stl")

    # Helper class for wrapping the App SDK Image object
    #
    class SpatialImage:
        """Object encapsulating a spatial volume image instance of Image.

        Channel is not supported in this version.

        Index conventions used throughout:
            - the App SDK array is (depth, row, col)
            - a SimpleITK / DICOM continuous index is (i, j, k) == (col, row, depth)
            - physical coordinates are DICOM patient coordinates (LPS) in mm
        """

        def __init__(self, image: Image, dtype=np.float32):
            """Creates an instance.

            Args:
                image(Image): An instance of Image.
                dtype (Numpy type, optional): Defaults to np.float32.
            """

            self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
            if image is None or not isinstance(image, Image):
                raise ValueError("Argument is not an Image object.")
            self._image = image
            self._dtype = dtype

            self._props: Dict = {}
            """ Properties may include some or all of the following
                img_array
                shape
                spacing            (row, col, depth) - do not index against the array with this
                spacing_dhw        (depth, row, col) - matched to the array axes
                origin             physical position of voxel [0, 0, 0], LPS mm
                index_to_physical  3x3 matrix mapping an (i, j, k) index to LPS mm
                original_affine
                affine
                itk_image
            """
            self._read_from_in_mem_image(self._image)

        @property
        def image_array(self):
            """Image data in Numpy array, or None"""
            return self._props.get("img_array", None)

        @property
        def itk_image(self):
            """ITK image object created from the encapsulated image object, or None"""
            return self._props.get("itk_image", None)

        @property
        def shape(self):
            """Shape of image array, or None"""
            return self._props.get("shape", None)

        @property
        def spacing(self):
            """Pixel spacing in the legacy (row, col, depth) order, or None.

            Kept for backwards compatibility only. Use spacing_dhw for anything indexed
            against the image array.
            """
            return self._props.get("spacing", None)

        @property
        def spacing_dhw(self):
            """Pixel spacing in (depth, row, col) order, matched to the array axes, or None."""
            return self._props.get("spacing_dhw", None)

        @property
        def origin(self):
            """Physical position (LPS, mm) of voxel [0, 0, 0], or None"""
            return self._props.get("origin", None)

        @property
        def original_affine(self):
            """Original affine of the image, or None"""
            return self._props.get("original_affine", None)

        @property
        def affine(self):
            """Affine of the re-oriented image data, or None"""
            return self._props.get("affine", None)

        def set_property(self, key: str, value):
            """Sets an image property

            Args:
                key (str): key of the property
                value: value of the property
            """
            self._props[key] = value

        def get_property(self, key: str, default=None):
            """Gets value of the specified property

            Args:
                key (str): key of the property
                default: default value if the property does not exist.

            Returns:
                the value of the property, or the default value if property does not exist.

            """
            return self._props.get(key, default)

        def get_data(self):
            """Returns the image array in ndarray"""
            return self._props.get("img_array", None)

        def index_to_physical(self, idx_xyz):
            """Maps continuous indices to physical points, vectorized.

            Args:
                idx_xyz: (N, 3) array of continuous indices in (i, j, k) == (col, row, depth)
                    order, i.e. the same order SimpleITK expects.

            Returns:
                (N, 3) array of physical points in patient coordinates (LPS, mm).
            """
            matrix = self._props["index_to_physical"]
            origin = self._props["origin"]
            return np.asarray(idx_xyz, dtype=np.float64) @ matrix.T + origin

        @staticmethod
        def _derive_origin(img_meta_dict):
            """Physical position (LPS, mm) of voxel [0, 0, 0], with the source it came from.

            Without this the mesh is translated by the first slice's ImagePositionPatient and is
            not in patient coordinates at all, which also makes any frame of reference it claims
            meaningless.
            """
            dicom_affine = img_meta_dict.get("dicom_affine_transform", None)
            if dicom_affine is not None:
                return np.asarray(dicom_affine, dtype=np.float64)[:3, 3], "dicom_affine_transform"

            nifti_affine = img_meta_dict.get("nifti_affine_transform", None)
            if nifti_affine is not None:
                translation = np.asarray(nifti_affine, dtype=np.float64)[:3, 3]
                # NIfTI affines are RAS; DICOM patient coordinates are LPS
                return np.array([-translation[0], -translation[1], translation[2]]), "nifti_affine_transform (RAS->LPS)"

            return np.zeros(3, dtype=np.float64), "fallback (0, 0, 0)"

        def _load_data(self, image):
            img_array = image.asnumpy()
            img_meta_dict = image.metadata()
            shape = np.asarray(img_array.shape)

            row_pixel_spacing = float(img_meta_dict["row_pixel_spacing"])
            col_pixel_spacing = float(img_meta_dict["col_pixel_spacing"])
            depth_pixel_spacing = float(img_meta_dict["depth_pixel_spacing"])

            # Legacy tuple, unchanged, so existing callers of .spacing keep working
            spacing = np.asarray((row_pixel_spacing, col_pixel_spacing, depth_pixel_spacing))

            # Spacing matched to the (depth, row, col) array axes. DICOM PixelSpacing[0]
            # ("adjacent row spacing", here row_pixel_spacing) is the step along the ROW index;
            # PixelSpacing[1] is the step along the COLUMN index
            spacing_dhw = np.asarray((depth_pixel_spacing, row_pixel_spacing, col_pixel_spacing))

            original_affine = img_meta_dict["nifti_affine_transform"]
            affine = original_affine

            itk_image = sitk.GetImageFromArray(img_array)

            # SimpleITK spacing is given in (x, y, z) == (col, row, depth) order
            itk_image.SetSpacing([col_pixel_spacing, row_pixel_spacing, depth_pixel_spacing])

            # The ITK direction matrix holds the physical directions of the index axes as its
            # COLUMNS. Stacking them as rows gives the transpose, which for any oblique
            # ImageOrientationPatient is the inverse rotation
            # Note the DICOM naming trap: ImageOrientationPatient[0:3] is the "row direction",
            # meaning the direction of travel ALONG a row, i.e. of increasing COLUMN index (x)
            x_direction = np.asarray(img_meta_dict["row_direction_cosine"], dtype=np.float64).ravel()[:3]
            y_direction = np.asarray(img_meta_dict["col_direction_cosine"], dtype=np.float64).ravel()[:3]
            z_direction = np.asarray(img_meta_dict["depth_direction_cosine"], dtype=np.float64).ravel()[:3]
            direction_matrix = np.column_stack((x_direction, y_direction, z_direction))
            itk_image.SetDirection([float(v) for v in direction_matrix.ravel()])

            origin, origin_source = self._derive_origin(img_meta_dict)
            if origin_source.startswith("fallback"):
                self._logger.warning(
                    "No affine in the image metadata; mesh origin defaults to (0, 0, 0) and the mesh "
                    "will NOT be in patient coordinates"
                )
            itk_image.SetOrigin([float(v) for v in origin])
            self._logger.info(f"Origin {np.round(origin, 3)} mm (LPS) from {origin_source}")
            self._logger.info(f"Direction matrix (columns = i, j, k directions):\n{np.round(direction_matrix, 6)}")

            # index (i, j, k) -> physical: origin + Direction @ diag(spacing_xyz) @ index
            index_to_physical = direction_matrix @ np.diag(
                np.array([col_pixel_spacing, row_pixel_spacing, depth_pixel_spacing], dtype=np.float64)
            )

            return (
                img_array,
                affine,
                original_affine,
                shape,
                spacing,
                spacing_dhw,
                origin,
                index_to_physical,
                itk_image,
            )

        def _read_from_in_mem_image(self, image):
            """Parse the in-memory image for the attributes.

            Args:
                image (Image): App SDK Image instance.

            Returns:
                An instance of SpatialImage.
            """
            (
                img_array,
                affine,
                original_affine,
                shape,
                spacing,
                spacing_dhw,
                origin,
                index_to_physical,
                itk_image,
            ) = self._load_data(image)

            num_dims = len(img_array.shape)
            self._logger.info(f"num_dims: {num_dims}")
            img_array = img_array.astype(self._dtype)

            if num_dims == 2:
                self._logger.info("2D image")
            elif num_dims == 3:
                self._logger.info("3D image")
            elif num_dims <= 5:
                # if 4d data, we assume 4th dimension is channels
                # if 5d data, try to squeeze 5th dimension
                if num_dims == 5:
                    img_array = np.squeeze(img_array)
                    if len(img_array.shape) != 4:
                        raise ValueError("Cannot squeeze 5D image to 4D; object doesn't support time based data")
                # Multi-channel data is out of scope for a segmentation-to-mesh operator
                self._logger.info(f"4D image with shape {img_array.shape}; channel handling is not implemented")
                raise NotImplementedError("Object does not support multi-channel image data")
            else:
                raise NotImplementedError("Object does not support image of dims {}".format(num_dims))

            self._props["original_affine"] = original_affine
            self._props["affine"] = affine
            self._props["spacing"] = spacing
            self._props["spacing_dhw"] = spacing_dhw
            self._props["origin"] = origin
            self._props["index_to_physical"] = index_to_physical
            self._props["shape"] = shape
            self._props["img_array"] = img_array
            self._props["itk_image"] = itk_image


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

    logging.basicConfig(level=logging.INFO)

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/spleen_ct/dcm")
    output_path = Path.cwd() / "output_stl/test.stl"

    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="dcm_loader")
    series_selector = DICOMSeriesSelectorOperator(fragment, name="series_selector")
    dcm_to_volume_op = DICOMSeriesToVolumeOperator(fragment, name="dcm_to_vol")
    stl_writer = STLConversionOperator(fragment, output_file=output_path, name="stl_writer")

    # Testing with the main entry functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    study_selected_series_list = series_selector.filter(None, study_list)
    image = dcm_to_volume_op.convert_to_image(study_selected_series_list)
    stl_writer._convert(image, output_path)
    print(f"Segmentation volume: {stl_writer.last_volume_mm3 / 1000.0:.2f} mL")
    print(f"Field-of-view truncated: {stl_writer.last_fov_truncated}")


if __name__ == "__main__":
    test()
