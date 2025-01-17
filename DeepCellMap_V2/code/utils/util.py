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

# Importation des librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.random as npr
import datetime
#import seaborn as sns



from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from simple_colors import *
# from config.const import *

import skimage.morphology as sk_morphology
from skimage.morphology import square, disk, diamond
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from rasterio import features, Affine

from shapely.geometry import (
    MultiPoint,
    MultiPolygon,
    polygon,
    Point,
    LineString,
    Polygon,
)
from shapely import geometry

from matplotlib import cm
from matplotlib.colors import ListedColormap


# from palettable.colorbrewer.sequential import YlGnBu_9
# import palettable


# Viennent de https://jiffyclub.github.io/palettable/
# CMAP_COLOCALISATION_CONTINUE = YlGnBu_9.mpl_colormap
# CMAP_COLOCALISATION_DISCRETE = ListedColormap(palettable.colorbrewer.sequential.YlGnBu_9.mpl_colors)


######################### Color related util --------------------------------------------------------------
"""
Liens utiles 

Couleurs matplotlib (étendues) 
https://xkcd.com/color/rgb/tile_coord

TIME : 
# #Au debut de la fonction
# t = Time()
# t___fonction___tot = t.elapsed()

# #Pour enregistrer le temps d'une fonction particulière dans une boucle
# t___fonction = Time()
# t___fonction___tot+=t___fonction.elapsed()
# print("%-20s | Time: %-14s " % ("Total fonction", str(t___fonction___tot)))

"""

class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed
    

def find_path_last(dir,figname):
    path_figure = os.path.join(dir, figname)
    list_imgs = [f for f in os.listdir(dir) if f.endswith(figname+".png")]
    liste = [int(i.split("_")[0]) for i in list_imgs]
    if len(liste) == 0:
        last_fig_number = 0
    else:
        last_fig_number = max(liste)

    figname = (
        str(last_fig_number + 1).zfill(3) + "_"+figname+".png"
    )

    path = os.path.join(dir, figname)
    return path


def get_path_table_cells(slide_num,dataset_config,particular_roi = None):
    """
    Get path of the table containing cells info
    """
    if particular_roi is not None :
        path_table_cells = os.path.join(dataset_config.dir_classified_img,"slide_{}".format(str(slide_num).zfill(3)), "slide_{}_cells_incomplete.csv".format(str(slide_num).zfill(3)))
    else : 
        path_table_cells = os.path.join(dataset_config.dir_classified_img,"slide_{}".format(str(slide_num).zfill(3)), "slide_{}_cells.csv".format(str(slide_num).zfill(3)))
    return path_table_cells


def create_path_roi(dataset_config,slide_num,origin_row,origin_col,end_row,end_col,entire_image=False):


    path_roi = os.path.join(dataset_config.dir_base_roi,"slide_{}".format(str(slide_num).zfill(3)),
                                "ro" +
                                str(origin_row).zfill(3) + "-" + "co" +
                                str(origin_col).zfill(3) + "-" + "re" +
                                str(end_row).zfill(3) + "-" + "ce" +
                                str(end_col).zfill(3))
    if entire_image:
        path_roi = os.path.join(dataset_config.dir_base_roi,"slide_{}".format(str(slide_num).zfill(3)),"entire_image")
    path_classified_slide = os.path.join(dataset_config.dir_classified_img,"slide_{}".format(str(slide_num).zfill(3)))
    #create path 
    if not os.path.exists(path_roi):
        os.makedirs(path_roi)
    if not os.path.exists(path_classified_slide):
        os.makedirs(path_classified_slide)
    return path_roi

#Related to paths 
def mkdir_if_nexist(pth):
    """Create a directory if the path does not already exist
    Parameters
    ----------
    pth : str
        Directory path to be created if it does not already exist
    """
    if not os.path.exists(pth):
        os.makedirs(pth)


def mkdirs_if_nexists(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.makedirs(path)


def supprimer_DS_Store(liste_filename):
    for noms in liste_filename:
        if "DS_Store" in noms:
            liste_filename.remove(noms)
    return liste_filename


def get_last_fig_number(path_directory):
    liste_imgs = [f for f in os.listdir(path_directory) if f.endswith(".png")]
    liste = [int(i.split("_")[0]) for i in liste_imgs]
    if len(liste) == 0:
        return 0
    else:
        return max(liste)


def decompose_filename(filename, type=int):
    """
    From filename like "006-R019-C035-x0768-y0561_mask.png" or "ready_for_training/mask/001-R024-C017-x1069-y0966_mask.png" give
    slide_num, tile_row, tile_col, x_coord, y_coord in string of int (by default)
    """
    dir, filename = os.path.split(filename)

    slide_num, tile_row, tile_col, x_coord, y_coord = (
        filename[0:3],
        filename[5:8],
        filename[10:13],
        filename[15:19],
        filename[21:25],
    )
    if type is int:
        return int(slide_num), int(tile_row), int(tile_col), int(x_coord), int(y_coord)
    else:
        return slide_num, tile_row, tile_col, x_coord, y_coord


# Related to image basic manipulations 

def crop_center_img(img, output_img_size):
    if len(img.shape) == 2:
        y, x = img.shape
        startx = x // 2 - (output_img_size // 2)
        starty = y // 2 - (output_img_size // 2)
        return img[
            starty : starty + output_img_size, startx : startx + output_img_size, :
        ]
    else:
        y, x, _ = img.shape
        startx = x // 2 - (output_img_size // 2)
        starty = y // 2 - (output_img_size // 2)
        return img[
            starty : starty + output_img_size, startx : startx + output_img_size, :
        ]


# Renvoie une image np_array rgb à partir du path de l'image
def tif_to_rgb(path):
    im = Image.open(path)
    im = np.asarray(im)
    return im


def rgb_to_hed(im):
    im_hed = rgb2hed(im)

    # Create an RGB image for each of the stains
    null = np.zeros_like(im_hed[:, :, 0])
    im_h = hed2rgb(np.stack((im_hed[:, :, 0], null, null), axis=-1))
    im_h = rescale_intensity(im_h, out_range=(0.0, 1.0))
    im_e = hed2rgb(np.stack((null, im_hed[:, :, 1], null), axis=-1))
    im_e = rescale_intensity(im_e, out_range=(0.0, 1.0))
    im_d = hed2rgb(np.stack((null, null, im_hed[:, :, 2]), axis=-1))
    im_d = rescale_intensity(im_d, out_range=(0.0, 1.0))

    return (im_h, im_e, im_d)


def rgb_to_eosin(im, display=False):
    im_hed = rgb2hed(im)
    null = np.zeros_like(im_hed[:, :, 0])
    im_e = hed2rgb(np.stack((null, im_hed[:, :, 1], null), axis=-1))
    im_e = rescale_intensity(im_e, out_range=(0.0, 1.0))
    im_e = 1-im_e
    return im_e

def rgb_to_d(im, display=False):
    im_hed = rgb2hed(im)
    null = np.zeros_like(im_hed[:, :, 0])
    im_d = hed2rgb(np.stack((null, im_hed[:, :, 2], null), axis=-1))
    im_d = rescale_intensity(im_d, out_range=(0.0, 1.0))
    im_d = 1-im_d
    return im_d

def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.

    Note that RGB PIL (w, h) -> NumPy (h, w, 3).

    Args:
      pil_img: The PIL Image.

    Returns:
      The PIL image converted to a NumPy array.
    """
    t = Time()
    rgb = np.asarray(pil_img)
    # np_info(rgb, "RGB", t.elapsed())
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.

    Args:
      np_img: The image represented as a NumPy array.

    Returns:
       The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64" or np_img.dtype == "float32":
        np_img = (np_img * 255).astype("uint8")
    else : 

        np_img = np_img.astype("uint8")
    return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.

    Args:
      np_arr: The NumPy array.
      name: The (optional) name of the array.
      elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print(
            "%-20s | Time: %-14s  Type: %-7s Shape: %s"
            % (name, str(elapsed), np_arr.dtype, np_arr.shape)
        )
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print(
            "%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s"
            % (
                name,
                str(elapsed),
                min,
                max,
                mean,
                is_binary,
                np_arr.dtype,
                np_arr.shape,
            )
        )



def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    t = Time()
    # mask = 255*mask.astype(int)
    result = rgb * np.dstack([mask, mask, mask])
    # np_info(result, "Mask RGB", t.elapsed())
    return result


def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)

def info_np(img, txt = ""):
    print(blue(txt), "-> shape ", img.shape, " dtype :",img.dtype, " min :",np.min(img), " max :",np.max(img))


def clip_and_rescale_image(img):
    ratio = 3/4
    # Calculate the maximum pixel value
    max_value = img.max()
    if max_value== 0 :
        return img
    # Clip values greater than 3/4 of the maximum to 3/4 of the maximum
    img[img > max_value * ratio] = max_value * ratio

    # Define the rescaling range (0 to 3/4 of the maximum)
    original_min = 0
    original_max = max_value * ratio
    new_min = 0
    new_max = 255  # You can adjust this value if you want a different max value

    # Rescale the image
    rescaled_image = np.interp(img, (original_min, original_max), (new_min, new_max)).astype(np.uint8)

    return rescaled_image

