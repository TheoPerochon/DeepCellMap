# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
 
# Standard library imports
import math
import os
import pandas as pd 

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from simple_colors import *
from tqdm.notebook import tqdm
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from skimage.morphology import  disk
# Project-specific imports
from preprocessing import slide
from utils import util
from utils.util import Time
from utils.util import * 

# #from python_files.const import *
from config.html_generation_config import HtmlGenerationConfig

# from config.base_config import BaseConfig
# from config.datasets_config import *
# base_config = BaseConfig()
# # Get the configuration
# if base_config.dataset_name == "ihc_microglia_fetal_human_brain_database":
#     dataset_config = IhcMicrogliaFetalHumanBrain()
# elif base_config.dataset_name == "cancer_data_immunofluorescence":
#     dataset_config = FluorescenceCancerConfig()

def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.

  Shape (h, w, c) to (h, w).

  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  t = Time()
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  #util.np_info(grayscale, "Gray", t.elapsed())
  return grayscale


def filter_complement(np_img, output_type="uint8"):
  """
  Obtain the complement of an image as a NumPy array.

  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).

  Returns:
    Complement image as Numpy array.
  """
  t = Time()
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  #util.np_info(complement, "Complement", t.elapsed())
  return complement


def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
  """
  Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

  Args:
    np_img: Image as a NumPy array.
    low: Low threshold.
    high: High threshold.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
  """
  t = Time()
  hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
  if output_type == "bool":
    pass
  elif output_type == "float":
    hyst = hyst.astype(float)
  else:
    hyst = (255 * hyst).astype("uint8")
  util.np_info(hyst, "Hysteresis Threshold", t.elapsed())
  return hyst


def filter_otsu_threshold(np_img,min_thresh=-1, output_type="uint8", return_thresh = False):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

    Args:
      np_img: Image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """
    t = Time()
    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    final_thresh = max(min_thresh,otsu_thresh_value)
    otsu = (np_img > final_thresh)
    if output_type == "bool":
      pass
    elif output_type == "float":
      otsu = otsu.astype(float)
    else:
      otsu = otsu.astype("uint8") * 255
    #util.np_info(otsu, "Otsu Threshold", t.elapsed())
    return otsu, final_thresh if return_thresh else otsu




def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8"):
  """
  Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
  local Otsu threshold.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
  """
  t = Time()
  local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
  if output_type == "bool":
    pass
  elif output_type == "float":
    local_otsu = local_otsu.astype(float)
  else:
    local_otsu = local_otsu.astype("uint8") * 255
  #util.np_info(local_otsu, "Otsu Local Threshold", t.elapsed())
  return local_otsu


def filter_entropy(np_img, neighborhood=9, threshold=5, output_type="uint8"):
  """
  Filter image based on entropy (complexity).

  Args:
    np_img: Image as a NumPy array.
    neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
    threshold: Threshold value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
  """
  t = Time()
  entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
  if output_type == "bool":
    pass
  elif output_type == "float":
    entr = entr.astype(float)
  else:
    entr = entr.astype("uint8") * 255
  #util.np_info(entr, "Entropy", t.elapsed())
  return entr


def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
  """
  Filter image based on Canny algorithm edges.

  Args:
    np_img: Image as a NumPy array.
    sigma: Width (std dev) of Gaussian.
    low_threshold: Low hysteresis threshold value.
    high_threshold: High hysteresis threshold value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
  """
  t = Time()
  can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    can = can.astype(float)
  else:
    can = can.astype("uint8") * 255
  #util.np_info(can, "Canny Edges", t.elapsed())
  return can


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = util.mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print(new_min_size)
    #print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  #util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img

def filter_remove_big_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = util.mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print(new_min_size)
    #print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  #util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img


def remove_large_objects(np_img, max_size, output_type="uint8"):
    # Label connected components in the binary imag
    if len(np.unique(np_img)) <= 2:
        labeled_image = sk_morphology.label(np_img)
    else : 
        labeled_image = np_img
    out = np.copy(labeled_image)
    object_sizes = np.bincount(labeled_image.ravel())
    too_big = object_sizes > max_size
    too_big_mask = too_big[labeled_image]
    out[too_big_mask] = 0

    if output_type == "bool":
      pass
    elif output_type == "float":
      out = out.astype(float)
    else:
      out = out.astype("uint8") * 255

    #util.np_info(np_img, "Remove Small Objs", t.elapsed())
    return out




def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8"):
  """
  Filter image to remove small holes less than a particular size.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  #rem_sm = sk_morphology.remove_small_holes(np_img, min_size=min_size)
  rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=min_size)

  if output_type == "bool":
    pass
  elif output_type == "float":
    rem_sm = rem_sk_morphology.astype(float)
  else:
    rem_sm = rem_sk_morphology.astype("uint8") * 255

  #util.np_info(rem_sm, "Remove Small Holes", t.elapsed())
  return rem_sm


def filter_contrast_stretch(np_img, low=40, high=60):
  """
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    low: Range low value (0 to 255).
    high: Range high value (0 to 255).

  Returns:
    Image as NumPy array with contrast enhanced.
  """
  t = Time()
  low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
  contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
  #util.np_info(contrast_stretch, "Contrast Stretch", t.elapsed())
  return contrast_stretch


def filter_histogram_equalization(np_img, nbins=256, output_type="uint8"):
  """
  Filter image (gray or RGB) using histogram equalization to increase contrast in image.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    output_type: Type of array to return (float or uint8).

  Returns:
     NumPy array (float or uint8) with contrast enhanced by histogram equalization.
  """
  t = Time()
  # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
  if np_img.dtype == "uint8" and nbins != 256:
    np_img = np_img / 255
  hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
  if output_type == "float":
    pass
  else:
    hist_equ = (hist_equ * 255).astype("uint8")
  #util.np_info(hist_equ, "Hist Equalization", t.elapsed())
  return hist_equ


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
  """
  Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
  is enhanced.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    clip_limit: Clipping limit where higher value increases contrast.
    output_type: Type of array to return (float or uint8).

  Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
  """
  t = Time()
  adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
  if output_type == "float":
    pass
  else:
    adapt_equ = (adapt_equ * 255).astype("uint8")
  #util.np_info(adapt_equ, "Adapt Equalization", t.elapsed())
  return adapt_equ


def filter_local_equalization(np_img, disk_size=50):
  """
  Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used for the local histograms

  Returns:
    NumPy array with contrast enhanced using local equalization.
  """
  t = Time()
  local_equ = sk_filters.rank.equalize(np_img, selem=sk_morphology.disk(disk_size))
  #util.np_info(local_equ, "Local Equalization", t.elapsed())
  return local_equ


def filter_rgb_to_hed(np_img, output_type="uint8"):
  """
  Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.

  Args:
    np_img: RGB image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array (float or uint8) with HED channels.
  """
  t = Time()
  hed = sk_color.rgb2hed(np_img)
  if output_type == "float":
    hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
  else:
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

  #util.np_info(hed, "RGB to HED", t.elapsed())
  return hed


def filter_rgb_to_hsv(np_img, display_np_info=True):
  """
  Filter RGB channels to HSV (Hue, Saturation, Value).

  Args:
    np_img: RGB image as a NumPy array.
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    Image as NumPy array in HSV representation.
  """

  if display_np_info:
    t = Time()
  hsv = sk_color.rgb2hsv(np_img)
  if display_np_info:
    util.np_info(hsv, "RGB to HSV", t.elapsed())
  return hsv


def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
  """
  Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
  values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
  https://en.wikipedia.org/wiki/HSL_and_HSV

  Args:
    hsv: HSV image as a NumPy array.
    output_type: Type of array to return (float or int).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
  """
  if display_np_info:
    t = Time()
  h = hsv[:, :, 0]
  h = h.flatten()
  if output_type == "int":
    h *= 360
    h = h.astype("int")
  if display_np_info:
    util.np_info(hsv, "HSV to H", t.elapsed())
  return h


def filter_hsv_to_s(hsv):
  """
  Experimental HSV to S (saturation).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Saturation values as a 1-dimensional NumPy array.
  """
  s = hsv[:, :, 1]
  s = s.flatten()
  return s


def filter_hsv_to_v(hsv):
  """
  Experimental HSV to V (value).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Value values as a 1-dimensional NumPy array.
  """
  v = hsv[:, :, 2]
  v = v.flatten()
  return v


def filter_hed_to_hematoxylin(np_img, output_type="uint8"):
  """
  Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array for Hematoxylin channel.
  """
  t = Time()
  hema = np_img[:, :, 0]
  if output_type == "float":
    hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
  else:
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
  #util.np_info(hema, "HED to Hematoxylin", t.elapsed())
  return hema


def filter_hed_to_eosin(np_img, output_type="uint8"):
  """
  Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array for Eosin channel.
  """
  t = Time()
  eosin = np_img[:, :, 1]
  if output_type == "float":
    eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
  else:
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
  #util.np_info(eosin, "HED to Eosin", t.elapsed())
  return eosin


def filter_binary_fill_holes(np_img, output_type="bool"):
  """
  Fill holes in a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where holes have been filled.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Fill Holes", t.elapsed())
  return result


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Erode a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for erosion.
    iterations: How many times to repeat the erosion.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where edges have been eroded.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Erosion", t.elapsed())
  return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Dilate a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for dilation.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Dilation", t.elapsed())
  return result


def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
  Opening can be used to remove small objects.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for opening.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) following binary opening.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Opening", t.elapsed())
  return result


def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
  Closing can be used to remove small holes.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for closing.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) following binary closing.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Closing", t.elapsed())
  return result


def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800):
  """
  Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
  colored based on the average color for that segment.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  result = sk_color.label2rgb(labels, np_img, kind='avg')
  #util.np_info(result, "K-Means Segmentation", t.elapsed())
  return result


def filter_rag_threshold(np_img, compactness=10, n_segments=800, threshold=9):
  """
  Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
  similar regions based on threshold value, and then output these resulting region segments.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    threshold: Threshold value for combining regions.

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  g = sk_future.graph.rag_mean_color(np_img, labels)
  labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
  result = sk_color.label2rgb(labels2, np_img, kind='avg')
  #util.np_info(result, "RAG Threshold", t.elapsed())
  return result


def filter_threshold(np_img, threshold, output_type="bool"):
  """
  Return mask where a pixel has a value if it exceeds the threshold value.

  Args:
    np_img: Binary image as a NumPy array.
    threshold: The threshold value to exceed.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
  t = Time()
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Threshold", t.elapsed())
  return result


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
  t = Time()

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = util.mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    #print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  #util.np_info(np_img, "Filter Green Channel", t.elapsed())
  return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Red", t.elapsed())
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Filter Red Pen", t.elapsed())
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
  """
  Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
  Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
  lower threshold value rather than a blue channel upper threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Green", t.elapsed())
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Filter Green Pen", t.elapsed())
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Blue", t.elapsed())
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Blue Pen", t.elapsed())
  return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  t = Time()
  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Grays", t.elapsed())
  return result


def uint8_to_bool(np_img):
  """
  Convert NumPy array of uint8 (255,0) values to bool (True,False) values

  Args:
    np_img: Binary image as NumPy array of uint8 (255,0) values.

  Returns:
    NumPy array of bool (True,False) values.
  """
  result = (np_img / 255).astype(bool)
  return result


def filter_cellpose(img, param_cellpose):
    """ Apply cellpose to the image """
    from cellpose import models
    from cellpose.io import imread
    model_type = param_cellpose["model_type"]
    diameter = param_cellpose["diameter"]
    channels = param_cellpose["channels"]
    normalisation = param_cellpose["normalisation"]
    net_avg = param_cellpose["net_avg"]
    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diam = model.eval(img, diameter=diameter, channels=channels,normalize=normalisation, net_avg = net_avg)
    # print(type(masks))
    # util.info_np(masks)
    return masks, diam



from skimage.filters import threshold_multiotsu 


def filter_multi_otsu_threshold(np_img,param, output_type="uint8", return_thresh = False):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  t = Time()
  otsu_thresh_value = threshold_multiotsu(np_img)
  # regions = np.digitize(np_img, bins=otsu_thresh_value)
  # util.info_np(regions, "regions")
  thresh = max(param[1], otsu_thresh_value[-1])
  otsu = (np_img >= thresh)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  #util.np_info(otsu, "Otsu Threshold", t.elapsed())
  return otsu, thresh if return_thresh else otsu

def filter_center_cells(img, size_middle_square,preserve_identity=False): 
    """ Filter cells that touch the center of the image, the center of mass can be outside
    
    if preserve_identity : but conserve different labels that can be in the cells. Used during reconstruction of cells rom different channels 
    """
    # filter_img = np.zeros((img.shape[0],img.shape[1]))
    labeled_image = sk_morphology.label(img)
    out = np.copy(labeled_image)

    img_filter_center = np.zeros(img.shape).astype(bool)
    img_filter_center[int(img_filter_center.shape[0]/2-size_middle_square/2):int(img_filter_center.shape[0]/2+size_middle_square/2), int(img_filter_center.shape[1]/2-size_middle_square/2):int(img_filter_center.shape[1]/2+size_middle_square/2)] = True

    out = out*img_filter_center
    cells_in_center = np.zeros(img.shape)
    for cell_id in np.unique(out): 
        if cell_id == 0 : 
            continue 
        cell_mask = labeled_image == cell_id
        # util.display_mask(cell_mask)
        cells_in_center+= cell_mask
    # cells_centre_label = np.where(np.logical_and(out,img_filter_center),labeled_image,0)
    out = cells_in_center
    if preserve_identity : 
        out = np.where(out,img,0)
        return out 
    else : 
      return out.astype(bool)

def apply_process_from_name(img, process_name, param, output_type = "uint8"):
    """ Apply process_name to the image """
    # info_np(img,"img debut apply_process_from_name ")
    # if len(img.shape)!=2:
    #     img = np.mean(img,axis = 2).astype(np.uint8)
    #     info_np(img,"img apres mean ")  
    # print("process_name",process_name)
    # print("param",param)
    if process_name == "manual_threshold":
        img_binary = np.mean(img,axis = 2).astype(np.uint8) if len(img.shape)!=2 else img
        # img_binary = img_binary[:,:,np.newaxis]
        img_binary = (img_binary>=param)
        name = "Img>"+str(param) +"=1 "
        if output_type == "uint8":
            img_binary = img_binary.astype(np.uint8)
        if output_type == "bool":
            img_binary = img_binary.astype(bool)
        return img_binary,name
    if process_name == "otsu_thresholding":
        if param[0] == "dont_consider_background" : 
           img[img==0]=np.mean(img[img!=0])
        img_binary, otsu_thresh_values = filter_otsu_threshold(img,min_thresh=param[1],output_type=output_type,return_thresh =True)
        name = "Otsu based binarization -> (Img>"+str(otsu_thresh_values) +")=1 "
        return img_binary,name
    if process_name == "multi_otsu_thresholding":
        if param[0] == "dont_consider_background" : 
           img[img==0]=np.mean(img[img!=0])
        img_binary, otsu_thresh_value = filter_multi_otsu_threshold(img,param,output_type=output_type,return_thresh =True)
        name = "Multi otsu based binarization -> (Img>"+str(otsu_thresh_value) +")=1 "
        return img_binary,name
    elif process_name == "dilation" :
        img_filtered =  filter_binary_dilation(img, disk_size=param, iterations=1, output_type=output_type)
        name = "Dilation(disk("+str(param)+'))'
        if output_type == "uint8":
            img_filtered = img_filtered.astype(np.uint8)
        if output_type == "bool":
            img_filtered = img_filtered.astype(bool)
        return img_filtered, name
    elif process_name == "dilation2" :
        img_filtered =  filter_binary_dilation(img, disk_size=param, iterations=1, output_type=output_type)
        name = "Dilation(disk("+str(param)+'))'
        if output_type == "uint8":
            img_filtered = img_filtered.astype(np.uint8)
        if output_type == "bool":
            img_filtered = img_filtered.astype(bool)
        return img_filtered, name
    elif process_name == "erosion" :
        img_filtered =  filter_binary_erosion(img, disk_size=param, iterations=1, output_type=output_type)
        name = "Erosion(disk("+str(param)+'))'
        if output_type == "uint8":
            img_filtered = img_filtered.astype(np.uint8)
        if output_type == "bool":
            img_filtered = img_filtered.astype(bool)
        return img_filtered, name
    elif process_name == "opening" :
        img_filtered =  filter_binary_opening(img, disk_size=param, iterations=1, output_type=output_type)
        name = "Opening(disk("+str(param)+'))'
        if output_type == "uint8":
            img_filtered = img_filtered.astype(np.uint8)
        if output_type == "bool":
            img_filtered = img_filtered.astype(bool)
        return img_filtered, name
    elif process_name == "fill_holes": 
        img_filtered =  filter_binary_fill_holes(img, output_type=output_type)
        name = "Holes filling"
        return img_filtered, name
    elif process_name == "remove_small_objects": 
        img_filtered =  filter_remove_small_objects(img, min_size=param,avoid_overmask=False)
        name = "Removing small objects (size<"+str(param)+")"
        return img_filtered, name
    elif process_name == "remove_large_objects": 
        img_filtered =  remove_large_objects(img, max_size=param)
        name = "Removing large objects (size>"+str(param)+")"
        return img_filtered, name
    elif process_name == "cellpose":
        img_filtered,diam =  filter_cellpose(img, param)
        name = "Cellpose (" + str(np.max(img_filtered)) + " cells) - diam = "+str(diam)
        return img_filtered, name
    elif process_name == "filter_center_cells":
        img_filtered =  filter_center_cells(img, param)
        name = "Filtering cells in center"
        return img_filtered, name
    elif process_name == "rgb_to_eosin":
        rgb_eosin =  util.rgb_to_eosin(img)
        img_filtered = 255*np.mean(rgb_eosin, axis=2)
        name = "rgb -> eosin"
        return img_filtered, name  
    elif process_name == "rgb_to_d":
        rgb_d =  util.rgb_to_d(img)
        img_filtered = 255*np.mean(rgb_d, axis=2)
        name = "rgb -> d"
        return img_filtered, name  
    elif process_name == "rgb_threshold_based_binarization_microglia_IHC":
        #TODO
        img_filtered =  segmentation_microglia_adaptative_threshold(img, param)
        name = "otsu_based_binarization_microglia_IHC"
        return img_filtered, name    
    elif process_name == "take_largest_component":
        img_filtered =  take_largest_component(img, param)
        name = "Selection_"+str(param)+"_largest_component"
        return img_filtered, name   
    elif process_name == "test_cas1__cas_2" :
        list_img_filtered, list_names = test_cas1_cas2(img, param)
        return list_img_filtered, list_names
    else : 
      print("process_name",process_name)
      print(" hasn't been implemented yet")

def test_cas1_cas2(img, param): 
    check_stats_process = True
    Save_img_intermediaire=True 
    list_imgs = []
    list_names = []

    dilation_test = param["dilation_test"]
    thresh_fraction_cells_discovered = param["thresh_fraction_cells_discovered"]
    param_erosion_cas_1 = param["param_erosion_cas_1"]
    param_dil_cas_2 = param["param_dil_cas_2"]
    dont_consider_background = param["dont_consider_background"]
    multi_otsu = param["multi_otsu"]
    min_thresh_second_binarization = param["min_thresh_second_binarization"]
    if dont_consider_background:
        img[img==0]=np.mean(img[img!=0])
    img_binary, otsu_thresh_values = filter_otsu_threshold(img,min_thresh=10,output_type = "uint8",return_thresh =True)
    name = "Otsu binarization -> (Img>"+str(otsu_thresh_values) +")=1 "
    if check_stats_process : 
        n_components, fract_masked = get_n_components_mask_coverage(img_binary)
        name+="\n"+str(n_components)+" components\n"+str(np.round(fract_masked*100,2))+"% masked"
    list_imgs.append(img_binary)
    list_names.append(name)
    
    img_dilated =  filter_binary_dilation(img_binary, disk_size=dilation_test, iterations=1, output_type= "uint8")
    name = "Dilation(disk("+str(dilation_test)+'))'
    if check_stats_process : 
        n_components, fract_masked = get_n_components_mask_coverage(img_dilated)
        name+="\n"+str(n_components)+" components\n"+str(np.round(fract_masked*100,2))+"% masked"
    list_imgs.append(img_dilated)
    list_names.append(name)
    if multi_otsu : 
        img_binary, otsu_thresh_value = filter_multi_otsu_threshold(img,["dont_consider_background", 20],output_type="uint8",return_thresh =True)
        name = "Multi Otsu binarization -> (Img>"+str(otsu_thresh_value) +")=1 "
    else : 
        img_binary, otsu_thresh_values = filter_otsu_threshold(img,min_thresh=min_thresh_second_binarization,output_type = "uint8",return_thresh =True)
        name = "Otsu binarization -> (Img>"+str(otsu_thresh_values) +")=1 "
    list_imgs.append(img_binary)
    list_names.append(name)
    n_components, fract_masked = get_n_components_mask_coverage(img_dilated)
    # print("fract_masked",fract_masked)
    # print("thresh_fraction_cells_discovered",thresh_fraction_cells_discovered)
    if fract_masked > thresh_fraction_cells_discovered : 
  
        img_filtered =  filter_binary_erosion(img_binary, disk_size=param_erosion_cas_1, iterations=1, output_type="uint8")
        name = "Erosion(disk("+str(param_erosion_cas_1)+'))'

    else : 
        img_filtered =  filter_binary_dilation(img_binary, disk_size=param_dil_cas_2, iterations=1, output_type="uint8")
        name = "Dilation(disk("+str(param_dil_cas_2)+'))'
    if check_stats_process : 
        n_components, fract_masked = get_n_components_mask_coverage(img_filtered)
        name+="\n"+str(n_components)+" components\n"+str(np.round(fract_masked*100,2))+"% masked"
    list_imgs.append(img_filtered)
    list_names.append(name)

    return list_imgs, list_names 

def get_n_components_mask_coverage(img):
    labelised = sk_morphology.label(img)
    n_components = len(np.unique(labelised))-1
    ratio_masked = np.sum(img.astype(bool))/img.size
    return n_components, ratio_masked



def take_largest_component(img, param):
    n_largest_components = param
    labeled_img = sk_morphology.label(img)

    component_ims = []
    component_size = []
    for i in range(1,np.max(labeled_img)+1):
        component_ims.append(np.where(labeled_img==i,1,0))
        component_size.append(np.sum(component_ims[-1]))
        
    component_size_df = pd.DataFrame(component_size, columns=["size"])
    n_largest = component_size_df.nlargest(n_largest_components,"size")

    for i in range(n_largest_components):
        if i == 0 : 
            img_filtered = component_ims[n_largest.index[i]]
        else : 
            img_filtered += component_ims[n_largest.index[i]]
    return img_filtered
def  extract_tissue_ihc_microglia_fetal_human_brain_database(np_img, slide_num=None, info=None, dataset_config=None, save=False, display=False):
    """ 
    Apply filters to extract tissue on IHC human brain slides dataset 

    ihc_microglia_fetal_human_brain_database
    """
    disk_dilation = 10
    disk_erosion = 9
    if dataset_config.preprocessing_config.tissue_extraction_accept_holes:
        mask_not_green = filter_green_channel(np_img)
        mask_not_green_dilate = filter_binary_dilation(mask_not_green,disk_size=disk_dilation)
        mask_not_green_dilate_removeobject = filter_remove_small_objects(mask_not_green_dilate, min_size=5000)
        mask_not_green_dilate_removeobject = filter_binary_erosion(mask_not_green_dilate_removeobject, disk_size=disk_erosion)
        binary_mask_tissue = np.where(mask_not_green_dilate_removeobject == 255, True,False)
        filtered_rgb = util.mask_rgb(np_img, binary_mask_tissue)
        return filtered_rgb, binary_mask_tissue
      
    else : 
        mask_not_green = filter_green_channel(np_img)
        mask_not_green_dilate = filter_binary_dilation(mask_not_green,disk_size=10)
        mask_not_green_dilate_fillholes = filter_binary_fill_holes(mask_not_green_dilate)
        mask_not_green_dilate_fillholes_removeobject = filter_remove_small_objects(mask_not_green_dilate_fillholes, min_size=5000)
        binary_mask_tissue = np.where(mask_not_green_dilate_fillholes_removeobject == 255, True,False)
        filtered_np_rgb = util.mask_rgb(np_img, binary_mask_tissue)
        return filtered_np_rgb, binary_mask_tissue


def  extract_tissue_cancer_data_immunofluorescence_database(np_img, slide_num=None, info=None, dataset_config=None, save=False, display=False):
    """ 
    Apply filters to extract tissue on IHC human brain slides dataset 

    ihc_microglia_fetal_human_brain_database
    """
    dataset_config.sequence_processing

def singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to training images and optionally save and/or display the filtered images.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  #print("Applying filters to images\n")

  if image_num_list is not None:
    _, info = apply_filters_to_image_list(image_num_list, save, display)
  else:
    num_training_slides = slide.get_num_training_slides()
    (s, e, info) = apply_filters_to_image_range(1, num_training_slides, save, display)

  #print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

  if html:
    generate_filter_html_result(info)


# if __name__ == "__main__":
# slide.training_slide_to_image(2)
# singleprocess_apply_filters_to_images(image_num_list=[2], display=True)

# singleprocess_apply_filters_to_images()




#### Filtres tiles theo 

def filter_remplissage_trous(im_b):
    # Remplissage des trous 
    bords = np.copy(im_b)
    bords = 1-bords
    bords[1:bords.shape[0]-1, 1:bords.shape[1]-1] = 0

    #Reconstruction de la plus grande composante connexe de l'image : le fond 
    im_fond_image = sk_morphology.reconstruction(bords,1-im_b)
    im_fond_image = 1 - im_fond_image

    return im_fond_image



def filter_suppression_fausses_composantes_connexes(tile_mask, min_cell_size= 300, verbose = 0): 
    """ Supprime les cellules de taille inférieur ou egale a min_cell_size 
    Renvoie une matrice avec : cells = taille_cell, background = 0 
    """
    im_labelised = sk_morphology.label(tile_mask)
    number_composantes_connexes = 1 

    if np.max(im_labelised)==0:
      return ("Il n'y a pas de microglie") 
    im_labelised_new = np.zeros(im_labelised.shape)
    nb_composante = np.max(im_labelised)
    print("Il y a "+str(nb_composante) + " cellules microgliales sur l'image avant filtrage") if verbose >= 3 else None

    for i in range(1,nb_composante+1):
      composante_i = np.where(im_labelised == i, 1, 0)
      size_comp_i = np.sum(composante_i)
      if size_comp_i >= min_cell_size:
        im_labelised_new += composante_i*size_comp_i
        number_composantes_connexes+=1

    print("Il y a "+str(number_composantes_connexes-1) + " cellules microgliales sur l'image après filtrage") if verbose >= 3 else None

    im_labelised_new = im_labelised_new.astype(int)
    #im_labelised_new = np.where(im_labelised_new>0, 1,0)
    return im_labelised_new


def filter_seuillage_canaux_rgb_microglia(im_rgb, display = False):

    """
    Prends une image rgb et renvoie un mask de la microglie (1 pour la microglie, 0 pour le background) 
    """
    
    im_canal_r = im_rgb[:,:,0]
    im_canal_gb = im_rgb[:,:,1:2]
    im_canal_gb = np.mean(im_canal_gb, axis =2)

    im_canal_r[im_canal_r<= COEF_SEUILLAGE_CANAL_ROUGE*im_canal_gb]= 0
    im_canal_r[im_canal_r> COEF_SEUILLAGE_CANAL_ROUGE*im_canal_gb]= 1

    if display:
        fig1 = plt.figure(figsize=(20,9), constrained_layout=True)
        spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
        f1_ax1 = fig1.add_subplot(spec1[0, 0])
        f1_ax2 = fig1.add_subplot(spec1[0, 1])
    
        f1=f1_ax1.imshow(im_rgb)
        f1_ax1.set_title("Image RGB avant binarisation", fontsize=20)
        f1_ax1.axis('off')
    
        divider = make_axes_locatable(f1_ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f2=f1_ax2.imshow(im_canal_r,cmap='viridis')
        fig1.colorbar(f1, cax=cax, orientation='vertical')
        f1_ax2.set_title("Image binarisée avec coef de seuillage =" + str(COEF_SEUILLAGE_CANAL_ROUGE)+" \ncanal R > coef * mean(canal G,canal B)", fontsize=20)
        f1_ax2.axis('off')

        plt.show()

    return im_canal_r

#### Segmentation of the pink and brown cells 

#Les exemples utilisés pour construire le model sont ceux de DICT_ROI_MODEL_SEGMENTATION_PINK_BROWN 
 

def imageHist(image):
    """
    La fonction plot l'histogramme rgb de l'image 2D ou 3D
    """
    _, axis = plt.subplots(ncols=2, figsize=(20, 10))
    if (image.ndim == 2):
        # Grascale Image
        axis[0].imshow(image, cmap=plt.get_cmap('gray'))
        axis[1].set_title('Histogram')
        axis[0].set_title('Grayscale Image')
        hist = exposure.histogram(image)
        axis[1].plot(hist[0])
    else:
        # Color image
        axis[0].imshow(image, cmap='gray')
        axis[1].set_title('Histogram')
        axis[0].set_title('Colored Image')
        rgbcolors = ['red', 'green', 'blue']
        for i, mycolor in enumerate(rgbcolors): 
            axis[1].plot(exposure.histogram(image[...,i])[0], color=mycolor)

def build_param_filtering_from_examples():
    """
    IMPORTANT : Les exemples utilisés pour construire les valeurs sont celles du MACBOOK  
    Construit pour chaque canal RGB la valeur moyenne et l'écart type des valeurs des pixels à partir d'exemples dans un dossier (ici segmentation_pink_brown) 
    Ces valeurs sont ensuite utilisées (par example dans model_segmentation_pink_brown_cells) pour seuiller les canaux afin de binariser l'image
    """

    path_exemple_pink_brown_cells = os.path.join(dataset_config.dir_output, "segmentation_pink_brown")
    i=0
    valeurs_canal_r = []
    valeurs_canal_g = []
    valeurs_canal_b = []
    for path_ex in supprimer_DS_Store(os.listdir(path_exemple_pink_brown_cells)):

        path_ex = os.path.join(path_exemple_pink_brown_cells,path_ex)
        im = plt.imread(path_ex)*255
        im = im.astype(int)

        valeurs_canal_r = valeurs_canal_r + list(im[7:-7,7:-7,0].flatten())
        valeurs_canal_g =valeurs_canal_g + list(im[7:-7,7:-7,1].flatten())
        valeurs_canal_b = valeurs_canal_b + list(im[7:-7,7:-7,2].flatten())

        #imageHist(im[:,:,:])
        #print(im[7:-7,7:-7,:])

    #valeurs_canal_r = np.reshape(valeurs_canal_r,-1).flatten()

    list_mean=[np.mean(valeurs_canal_r), np.mean(valeurs_canal_g), np.mean(valeurs_canal_b)]
    list_std = [np.std(valeurs_canal_r),np.std(valeurs_canal_g),np.std(valeurs_canal_b)]
    return list_mean,list_std

def keep_biggest_tissue(binary_img, verbose = 0):
    binary_img_label = sk_morphology.label(binary_img)
    nb_components = np.max(binary_img_label)
    size_component=[]
    print("nb_components =", nb_components) 
    for i in tqdm(range(1,nb_components+1)):
        component_i = np.where(binary_img_label==i,1,0)
        size_component.append(np.sum(component_i))
    print("tailles composants = ", size_component) if verbose == 2 else None
    bigger_component = np.argmax(size_component)+1
    binary_tissue_of_interest = np.where(binary_img_label ==bigger_component,1,0 )
    return binary_tissue_of_interest

def filter_dense_cells(rgb_image):
    def keep_biggest_tissue(binary_img, verbose = 0):
        binary_img_label = sk_morphology.label(binary_img)
        nb_components = np.max(binary_img_label)
        size_component=[]
        print("nb_components =", nb_components) 
        for i in tqdm(range(1,nb_components+1)):
            component_i = np.where(binary_img_label==i,1,0)
            size_component.append(np.sum(component_i))
        print("tailles composants = ", size_component) if verbose == 2 else None
        bigger_component = np.argmax(size_component)+1
        binary_tissue_of_interest = np.where(binary_img_label ==bigger_component,1,0 )
        return binary_tissue_of_interest
  
    f1 = np.logical_and(rgb_image[:,:,0]<140,rgb_image[:,:,1]<140,rgb_image[:,:,2]<140)


    # f1_1 = sk_morphology.opening(a,disk(2))
    
    # f1_2= sk_morphology.dilation(f1_1,disk(3))
    # f1_3 = ndimage.binary_fill_holes(f1_2)
    # f1_4 = sk_morphology.dilation(f1_3,disk(3))
    # f1_5 = sk_morphology.dilation(f1_4,disk(5))
    # f1_6 = keep_biggest_tissue(f1_5)

    print( rgb_image.dtype)
    print(np.max(rgb_image))
    f2_1 = sk_morphology.dilation(f1,disk(5))
    f2_2= sk_morphology.opening(f2_1,disk(7))
    f2_3 = sk_morphology.dilation(f2_2,disk(4))
    f2_4 = keep_biggest_tissue(f2_3)

    #display_rgb_mask(rgb_image,a, "RGB img","Segmentation of dense cells", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f1_1, "RGB img","f1_1", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f1_2, "RGB img","f1_2", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f1_3, "RGB img","f1_3", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f1_4, "RGB img","f1_4", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f1_5, "RGB img","f1_5", figsize = (18,15), cmap = "Purples")
    #isplay_rgb_mask(rgb_image,f1_6, "RGB img","f1_6", figsize = (18,15), cmap = "Purples")

    #display_rgb_mask(rgb_image,f2_1, "RGB img","f2_1", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f2_2, "RGB img","f2_1", figsize = (18,15), cmap = "Purples")
    #display_rgb_mask(rgb_image,f2_3, "RGB img","f2_3", figsize = (18,15), cmap = "Purples")
    util.display_rgb_mask(rgb_image,f2_4, "RGB img","f2_4", figsize = (18,15), cmap = "Purples")
    #util.display_rgb_mask(rgb_image,f3_4, "RGB img","Segmentation of dense cells", figsize = (15,15), cmap = "Purples")
    return f2_4



######## Forebrain specific functions 

def filter_components_tissue(binary_img_label, verbose = 0):
  """
  Usage spécific à la slide 1 
  Supprime les 2, 3, 4, 5e plus grands tissus 
  
  """
  nb_components = np.max(binary_img_label)
  size_component=[]
  print("nb_components =", nb_components) if verbose == 2 else None
  binary_tissue_of_interest = np.copy(binary_img_label)
  for i in tqdm(range(1,nb_components+1)):
      component_i = np.where(binary_img_label==i,1,0)
      size_component.append(np.sum(component_i))
  print("tailles composants = ", size_component) if verbose == 2 else None
  #size_component a les tailles de composantes 
  """'est ici que je fais le choix des quelles je garde et de squelles je supprime 
  #Car je veux supprimer les 2, 3 et 4e plus grosses composantes 
  """
  for i in range(5): 
      arg_max = np.argmax(size_component)
      print("taille composante = ", size_component[arg_max]) if verbose == 2 else None
      size_component[arg_max]=0
      if i == 0: #Je laisse la première
          continue
      else : 
          binary_tissue_of_interest = np.where(binary_img_label ==arg_max+1,0,binary_tissue_of_interest)
  binary_tissue_of_interest = np.where(binary_tissue_of_interest>0,1,0)
  return binary_tissue_of_interest

def keep_biggest_tissue(binary_img, verbose = 0):
  nb_components = np.max(binary_img)
  size_component=[]
  print("nb_components =", nb_components) if verbose == 2 else None
  for i in tqdm(range(1,nb_components+1)):
      component_i = np.where(binary_img==i,1,0)
      size_component.append(np.sum(component_i))
  print("tailles composants = ", size_component) if verbose == 2 else None
  bigger_component = np.argmax(size_component)+1
  binary_tissue_of_interest = np.where(binary_img ==bigger_component,1,0 )
  return binary_tissue_of_interest

def extract_interest_area_forebrain(img_rgb, verbose = 0, display = True):
  """
  Extract the tissue of interest of slide 1
   """
  binary = img_rgb[:,:,0] < 0.85
  print("binarisation done") if verbose > 0 else None
  print("type = ", binary.dtype) if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_2 = ndimage.binary_fill_holes(binary)
  print("binary_fill_holes done") if verbose > 0 else None
  print("type = ", binary_2.dtype) if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_2, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_3 = sk_morphology.erosion(binary_2, disk(5))
  print("erosion done") if verbose > 0 else None
  print("type = ", binary_3.dtype) if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_3, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_4 = sk_morphology.label(binary_3)
  print("type = ", binary_4.dtype) if verbose > 0 else None
  print("label done") if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_4, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_5 = filter_components_tissue(binary_4)
  print("type = ", binary_5.dtype) if verbose > 0 else None
  print("filter_components_tissue done") if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_5, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_6 = sk_morphology.dilation(binary_5, disk(25))
  print("filter_components_tissue done") if verbose > 0 else None
  print("type = ", binary_6.dtype) if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_6, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_7 = sk_morphology.label(binary_6)
  print("type = ", binary_7.dtype) if verbose > 0 else None
  print("filter_components_tissue done") if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_7, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_8 = keep_biggest_tissue(binary_7)
  print("filter_components_tissue done") if verbose > 0 else None
  print("type = ", binary_8.dtype) if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_8, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_9 = sk_morphology.dilation(binary_8, disk(51))
  print("type = ", binary_9.dtype) if verbose > 0 else None
  print(" done") if verbose > 0 else None
  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_9, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
  binary_10 = sk_morphology.erosion(binary_9,disk(31))
  print("type binary_10= ", binary_10.dtype) if verbose > 0 else None



  if display :
      f = plt.figure(figsize = (20,20))
      plt.imshow(binary_10, cmap = 'gray')
      plt.colorbar(shrink=0.5)
      plt.show() 
      binary_tissue_of_interest_3D = np.dstack([binary_10, binary_10, binary_10])
      print("type binary_tissue_of_interest_3D= ", binary_tissue_of_interest_3D.dtype) if verbose > 0 else None

      img_rgb_tissue = np.where(binary_tissue_of_interest_3D==1,img_rgb,0)
      print("type img_rgb_tissue= ", img_rgb_tissue.dtype) if verbose > 0 else None

      f = plt.figure(figsize=(20,10))

      plt.subplot(1,2,1)
      plt.imshow(img_rgb)
      plt.colorbar(shrink=0.5)

      plt.subplot(1,2,2)
      plt.imshow(img_rgb_tissue)
      plt.colorbar(shrink=0.5)    

  return binary_10


def specific_tissue_mask(slide_num, image_rgb, verbose = 0, display = False):
  """ 
  Renvoie le mask binaire du tissue d'intéret dans la slide (dépend de chaque slide) 
  """

  if slide_num == 1:
    print("Computing mask tissue...") if verbose >0 else None 
    mask_binaire = extract_interest_area_forebrain(image_rgb, verbose = verbose, display = display)
    return mask_binaire
  else :
    print("Pas de filtre spécial pour cette slide")


