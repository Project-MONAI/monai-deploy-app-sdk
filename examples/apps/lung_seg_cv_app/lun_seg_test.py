# # Copyright 2021 MONAI Consortium
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import os
# from glob import glob

# import matplotlib.pyplot as plt
# import numpy as np
# import pydicom as dicom
# import scipy.ndimage
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from plotly.graph_objs import *
# from plotly.offline import iplot
# from plotly.tools import FigureFactory as FF
# from skimage import measure, morphology
# from sklearn.cluster import KMeans


# def load_scan(path):
#     slices = [dicom.read_file(path + "/" + s) for s in os.listdir(path)]
#     slices.sort(key=lambda x: int(x.InstanceNumber))
#     try:
#         slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
#     except RuntimeError:
#         slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

#     for s in slices:
#         s.SliceThickness = slice_thickness

#     return slices


# def get_pixels_hu(scans):
#     image = np.stack([s.pixel_array for s in scans])
#     image = image.astype(np.int16)
#     image[image == -2000] = 0

#     intercept = scans[0].RescaleIntercept
#     slope = scans[0].RescaleSlope

#     if slope != 1:
#         image = slope * image.astype(np.float64)
#         image = image.astype(np.int16)

#     image += np.int16(intercept)

#     return np.array(image, dtype=np.int16)


# def save_images_to_npy(full_filename, imgs):
#     np.save(full_filename, imgs)


# def load_images_from_npy(full_filename):
#     imgs = np.load(full_filename)
#     return imgs


# def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
#     fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
#     for i in range(rows * cols):
#         ind = start_with + i * show_every
#         ax[int(i / rows), int(i % rows)].set_title("slice %d" % ind)
#         ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap="gray")
#         ax[int(i / rows), int(i % rows)].axis("off")
#     plt.show()


# def resample(image, scan, new_spacing=[1, 1, 1]):
#     px = scan[0].PixelSpacing
#     spacing_temp = []
#     spacing_temp.append(scan[0].SliceThickness)
#     spacing_temp.append(px[0])
#     spacing_temp.append(px[1])
#     spacing = np.array(spacing_temp)
#     resize_factor = spacing / new_spacing
#     new_real_shape = image.shape * resize_factor
#     new_shape = np.round(new_real_shape)
#     real_resize_factor = new_shape / image.shape
#     new_spacing = spacing / real_resize_factor

#     image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

#     return image, new_spacing


# def make_mesh(image, threshold=-300, step_size=1):

#     print("Transposing surface")
#     p = image.transpose(2, 1, 0)

#     print("Calculating surface")
#     verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
#     return verts, faces


# def plotly_3d(verts, faces):
#     x, y, z = zip(*verts)

#     print("Drawing")

#     # Make the colormap single color since the axes are positional not intensity.
#     # colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
#     colormap = ["rgb(236, 236, 212)", "rgb(236, 236, 212)"]

#     fig = FF.create_trisurf(
#         x=x,
#         y=y,
#         z=z,
#         plot_edges=False,
#         colormap=colormap,
#         simplices=faces,
#         backgroundcolor="rgb(64, 64, 64)",
#         title="Interactive Visualization",
#     )
#     iplot(fig)


# def plt_3d(verts, faces):
#     print("Drawing")
#     x, y, z = zip(*verts)
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection="3d")
#     mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
#     face_color = [1, 1, 0.9]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)

#     ax.set_xlim(0, max(x))
#     ax.set_ylim(0, max(y))
#     ax.set_zlim(0, max(z))  # print("appending")

#     img = img / std
#     # Find the average pixel value near the lungs
#     # to renormalize washed out images
#     middle = img[int(col_size / 5) : int(col_size / 5 * 4), int(row_size / 5) : int(row_size / 5 * 4)]
#     mean = np.mean(middle)
#     max = np.max(img)
#     min = np.min(img)
#     # To improve threshold finding, I'm moving the
#     # underflow and overflow on the pixel spectrum
#     img[img == max] = mean
#     img[img == min] = mean
#     #
#     # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
#     #
#     kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
#     centers = sorted(kmeans.cluster_centers_.flatten())
#     threshold = np.mean(centers)
#     thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

#     # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
#     # We don't want to accidentally clip the lung.

#     eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
#     dilation = morphology.dilation(eroded, np.ones([8, 8]))

#     labels = measure.label(dilation)  # Different labels are displayed in different colors
#     label_vals = np.unique(labels)
#     regions = measure.regionprops(labels)
#     good_labels = []
#     for prop in regions:
#         B = prop.bbox
#         if (
#             B[2] - B[0] < row_size / 10 * 9
#             and B[3] - B[1] < col_size / 10 * 9
#             and B[0] > row_size / 5
#             and B[2] < col_size / 5 * 4
#         ):
#             good_labels.append(prop.label)
#     mask = np.ndarray([row_size, col_size], dtype=np.int8)
#     mask[:] = 0

#     #
#     #  After just the lungs are left, we do another large dilation
#     #  in order to fill in and out the lung mask
#     #
#     for N in good_labels:
#         mask = mask + np.where(labels == N, 1, 0)
#     mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

#     if display:
#         fig, ax = plt.subplots(3, 2, figsize=[12, 12])
#         ax[0, 0].set_title("Original")
#         ax[0, 0].imshow(img, cmap="gray")
#         ax[0, 0].axis("off")
#         ax[0, 1].set_title("Threshold")
#         ax[0, 1].imshow(thresh_img, cmap="gray")
#         ax[0, 1].axis("off")
#         ax[1, 0].set_title("After Erosion and Dilation")
#         ax[1, 0].imshow(dilation, cmap="gray")
#         ax[1, 0].axis("off")
#         ax[1, 1].set_title("Color Labels")
#         ax[1, 1].imshow(labels)
#         ax[1, 1].axis("off")
#         ax[2, 0].set_title("Final Mask")
#         ax[2, 0].imshow(mask, cmap="gray")
#         ax[2, 0].axis("off")
#         ax[2, 1].set_title("Apply Mask on Original")
#         ax[2, 1].imshow(mask * img, cmap="gray")
#         ax[2, 1].axis("off")

#         plt.show()
#     return mask * img


# data_path = "/home/rahul/medical-images/lung-ct-2/"
# output_path = working_path = "/home/rahul/monai-output/"
# g = glob(data_path + "/*.dcm")

# # Print out the first 5 file names to verify we're in the right folder.
# print("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
# print("\n".join(g[:5]))

# patient = load_scan(data_path)
# imgs = get_pixels_hu(patient)
# np.save(output_path + "fullimages.npy", imgs)

# imgs_to_process = np.load(output_path + "fullimages.npy")
# sample_stack(imgs_to_process)

# print("Shape before resampling\t", imgs_to_process.shape)
# imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])
# print("Shape after resampling\t", imgs_after_resamp.shape)


# masked_lung = []
# for img in imgs_after_resamp:
#     masked_lung.append(make_lungmask(img))

# sample_stack(masked_lung, show_every=2)
# np.save(output_path + "maskedimages.npy", masked_lung)
