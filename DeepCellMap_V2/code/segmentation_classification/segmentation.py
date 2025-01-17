# Importation des librairies
import numpy as np
import matplotlib.pyplot as plt
from simple_colors import *


from preprocessing import filter

# from python_files.const import *
from skimage.filters import threshold_multiotsu
import multiprocessing
from preprocessing import filter
from utils import util
from preprocessing import slide
from config.base_config import BaseConfig
from config.datasets_config import *
from utils.util import * 

from config.html_generation_config import HtmlGenerationConfig

#Pour le saving html
import os
import plotly.subplots as sp
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
import plotly.express as px


def segmentation_microglia_otsu(rgb, param=2, verbose=0):
    """Segmenting microglial cells using Otsu multi-thresholding-based binarisation

    Remark: to save calculation util.Time, filtering by size is carried out in the function calling this function
    """
    dilation_radius = param
    eosin = util.rgb_to_eosin(np.copy(rgb))
    eosin_grayscale = np.mean(eosin, axis=2)
    eosin_grayscale *= 255
    thresholds = threshold_multiotsu(eosin_grayscale)
    regions = np.digitize(eosin_grayscale, bins=thresholds)
    mask_1 = regions < 2
    mask_2 = sk_morphology.opening(mask_1, disk(1))  # Avant 1 mais j'essaye 2
    mask_3 = sk_morphology.dilation(mask_2, disk(dilation_radius))
    final_mask = filter.filter_remplissage_trous(mask_3)
    return final_mask

def segmentation_microglia_adaptative_threshold(
    tile_np,
    dilation_radius=2,
    min_cell_size=400,
    display=False,
    save=False,
    verbose=0,
):
    """
    Segmenting microglial cells using adaptative thresholding on RGB canals
    """
    # Seuillage de la microglie grace au canal rouge (cf explication de la fonction seuillage_microglie)
    tile_debut = np.copy(tile_np)
    tile_mask = filter.filter_seuillage_canaux_rgb_microglia(tile_debut, display=False)
    tile_mask = sk_morphology.opening(tile_mask, disk(1))
    if dilation_radius != False:
        tile_mask = sk_morphology.dilation(tile_mask, disk(dilation_radius))
    final_mask = filter.filter_remplissage_trous(tile_mask)
    final_mask = final_mask.astype(np.uint8)
    return final_mask

def segment_microglia_IHC(
    img,
    segment_param,
    verbose=0,
):
    """
    Input : RGB image 3x3tiles
    Output : Mask microglia 3x3tiles
    Note : 2 segmentation methods possible
    """
    img = np.copy(img)
    if segment_param["model_segmentation_type"] == "otsu_based_binarization_microglia_IHC":
        mask_cells = segmentation_microglia_otsu(img, verbose=verbose)
    elif segment_param["model_segmentation_type"] == "rgb_threshold_based_binarization_microglia_IHC":
        mask_cells = segmentation_microglia_adaptative_threshold(
            img, segment_param["dilation_radius"], segment_param["min_cell_size"], verbose=verbose
        )
    mask_cells = np.where(mask_cells > 0, 1, 0)
    return mask_cells

#### Tissue segmentation 

def multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None, dataset_config=None):
    """
    Apply a set of filters to all training images using multiple processes (one process per core).
    Args:
        save: If True, save filtered images.
        display: If True, display filtered images to screen (multiprocessed display not recommended).
        html: If True, generate HTML page to display filtered images.
        image_num_list: Optionally specify a list of image slide numbers.
    """
    util.Timer = util.Time()
    #print("Applying filters to images (multiprocess)\n")
    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    if image_num_list is not None:
        num_train_images = len(image_num_list)
    else:
        num_train_images = slide.get_num_training_slides()
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    # print("Number of processes: " + str(num_processes))
    # print("Number of training images: " + str(num_train_images))

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        if image_num_list is not None:
            sublist = image_num_list[start_index - 1:end_index]
            tasks.append((sublist, save, display,dataset_config))
        #print("Task #" + str(num_process) + ": Process slides " + str(sublist))
        else:
            tasks.append((start_index, end_index, save, display, dataset_config))
        #if start_index == end_index:
            #print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        #else:
            #print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        if image_num_list is not None:
            results.append(pool.apply_async(apply_filters_to_image_list, t))
        else:
            results.append(pool.apply_async(apply_filters_to_image_range, t))

    html_page_info = dict()
    for result in results:
        if image_num_list is not None:
            (image_nums, html_page_info_res) = result.get()
            html_page_info.update(html_page_info_res)
        #print("Done filtering slides: %s" % image_nums)
        else:
            (start_ind, end_ind, html_page_info_res) = result.get()
            html_page_info.update(html_page_info_res)
        #if (start_ind == end_ind):
            #print("Done filtering slide %d" % start_ind)
        #else:
            #print("Done filtering slides %d through %d" % (start_ind, end_ind))
    if html:
        generate_filter_html_result(html_page_info)

    # print("util.Time to apply filters to all images (multiprocess): %s\n" % str(util.Timer.elapsed()))

def apply_filters_to_image_list(image_num_list , save, display, dataset_config):
  """
  Apply filters to a list of images.

  Args:
    image_num_list: List of image numbers.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in image_num_list:
      if dataset_config.consider_image_with_channels:
        n_channels, channels = slide.get_info_img_from_column_name(slide_num,dataset_config=dataset_config,column_name="n_channels")
        channel_number = dataset_config.channel_used_to_segment_tissue
        _, info = apply_filters_to_image(slide_num, save=save, display=display,dataset_config=dataset_config,channel_number=channel_number)
        html_page_info.update(info)
        for channel_number in range(-1,n_channels):
              if channel_number == dataset_config.channel_used_to_segment_tissue : 
                  continue 
            #   print("Channel _ "+str(channel_number))
              _, info = apply_filters_to_image(slide_num, save=save, display=display,dataset_config=dataset_config,channel_number=channel_number)
              html_page_info.update(info)

      else : 
        print("slide_num", slide_num)
        _, info = apply_filters_to_image(slide_num, save=save, display=display,dataset_config=dataset_config)
        html_page_info.update(info)
  return image_num_list, html_page_info


def apply_filters_to_image_range(start_ind, end_ind, save, display,dataset_config):
    """
    Apply filters to a range of images.

    Args:
        start_ind: Starting index (inclusive).
        end_ind: Ending index (inclusive).
        save: If True, save filtered images.
        display: If True, display filtered images to screen.

    Returns:
        Tuple consisting of 1) staring index of slides converted to images, 2) ending index of slides converted to images,
        and 3) a dictionary of image filter information.
    """
    html_page_info = dict()
    for slide_num in range(start_ind, end_ind + 1):
        if dataset_config.consider_image_with_channels:
            n_channels, channels = slide.get_info_img_from_column_name(slide_num,dataset_config=dataset_config,column_name="n_channels")
            for channel_number in range(-1,n_channels):
                print("Channel _ "+str(channel_number))
                _, info = apply_filters_to_image(slide_num, save=save, display=display,dataset_config=dataset_config,channel_number=channel_number)
                html_page_info.update(info)
        else : 
            _, info = apply_filters_to_image(slide_num, save=save, display=display,dataset_config=dataset_config)
            html_page_info.update(info)
    return start_ind, end_ind, html_page_info


def apply_filters_to_image(slide_num,dataset_config=None, save=True, display=False,channel_number=None):
    """
    Apply a set of filters to an image and optionally save and/or display filtered images.

    Args:
      slide_num: The slide number.
      save: If True, save filtered images.
      display: If True, display filtered images to screen.

    Returns:
      Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
      (used for HTML page generation).
    """
    t = util.Time()
    info = dict()
    # img_path = slide.get_training_image_path(slide_num,channel_number=channel_number)
    img_path = slide.get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number,dataset_config=dataset_config)
    
    np_orig = slide.open_image_np(img_path)
    if channel_number is None :
        filtered_np_rgb, mask_tissue_np = apply_image_filters(np_orig, slide_num, info,dataset_config=dataset_config, save=save, display=display,channel_number=channel_number)
    else : 
        if channel_number == dataset_config.channel_used_to_segment_tissue : 
            filtered_np_rgb, mask_tissue_np = apply_image_filters(np_orig, slide_num, info,dataset_config=dataset_config, save=save, display=display,channel_number=channel_number)
        else : 
            path_rgb_filtered, path_mask = slide.get_filter_image_result(slide_num,channel_number = dataset_config.channel_used_to_segment_tissue,dataset_config=dataset_config)
            mask_tissue_np = slide.open_image_np(path_mask)
            mask_tissue_np = mask_tissue_np.astype(bool)
            filtered_np_rgb = np_orig*mask_tissue_np if len(np_orig.shape) == 2 else np_orig * np.dstack([mask_tissue_np, mask_tissue_np, mask_tissue_np])
            # info_np(filtered_np_rgb,"filtered_np_rgb apres filtrage")
    if save:
      # t1 = util.Time()
      # path_rgb_filtered, path_mask = slide.get_filter_image_result(slide_num,channel_number = channel_number)
      path_rgb_filtered, path_mask = slide.get_filter_image_result(slide_num,channel_number = channel_number,dataset_config=dataset_config)
      pil_img = util.np_to_pil(filtered_np_rgb)
      pil_img.save(path_rgb_filtered)
      pil_img_binary = util.np_to_pil(mask_tissue_np)
      pil_img_binary.save(path_mask)

      #print("%-20s | util.Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

      # t1 = util.Time()
      # thumbnail_path = slide.get_filter_thumbnail_result(slide_num)
      path_rgb_filtered_thumb, _  = slide.get_filter_image_result(slide_num,thumbnail = True,channel_number = channel_number,dataset_config=dataset_config)

      slide.save_thumbnail(pil_img, HtmlGenerationConfig.thumbnail_size, path_rgb_filtered_thumb)
      #print("%-20s | util.Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

    #print("Slide #%03d processing util.Time: %s\n" % (slide_num, str(t.elapsed())))
    return filtered_np_rgb, info

def apply_image_filters(np_img, slide_num=None, info=None, dataset_config=None, save=False, display=False,channel_number=None):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images.

    Args:
      np_img: Image as NumPy array.
      slide_num: The slide number (used for saving/displaying).
      info: Dictionary of slide information (used for HTML display).
      save: If True, save image.
      display: If True, display 
    Returns:
      Resulting filtered image as a NumPy array.
    """
    #Tissue extraction modularity 
    if dataset_config.dataset_name == "ihc_microglia_fetal_human_brain_database":
        filtered_np_rgb, mask_tissue_np = filter.extract_tissue_ihc_microglia_fetal_human_brain_database(np_img, slide_num=slide_num, info=info, dataset_config=dataset_config, save=save, display=display)
        info_np(filtered_np_rgb,"filtered_np_rgb")
        return filtered_np_rgb, mask_tissue_np
    else :
        model_segmentation = ModelSegmentation(dataset_config=dataset_config)
        list_images_processed, list_process_names = model_segmentation.test_segmentation_grayscale([np_img],slide_num=slide_num)
        if dataset_config.save_tissue_segmentation_steps:
            model_segmentation.laboratory_segmentation_save_images(list_images_processed, list_process_names)
        mask_tissue = list_images_processed[0][-2].astype(bool)
        filtered_np_rgb = np_img*mask_tissue if len(np_img.shape) == 2 else np_img * np.dstack([mask_tissue, mask_tissue, mask_tissue])
    return filtered_np_rgb, mask_tissue

### About html 
def generate_filter_html_result(html_page_info):
  """
  Generate HTML to view the filtered images. If HtmlGenerationConfig.filter_paginate is True, the results will be paginated.

  Args:
    html_page_info: Dictionary of image information.
  """
#   print(html_page_info)

  if not HtmlGenerationConfig.filter_paginate:
    html = ""
    html +=util_html.html_header("Filtered Images")
    html += "  <table>\n"

    row = 0
    for key in sorted(html_page_info):
      value = html_page_info[key]
      current_row = value[0]
      if current_row > row:
        html += "    <tr>\n"
        row = current_row
      html += image_cell(value[0], value[1], value[2], value[3])
      next_key = key + 1
      if next_key not in html_page_info:
        html += "    </tr>\n"

    html += "  </table>\n"
    html +=util_html.html_footer()
    text_file = open(os.path.join(dataset_config.dir_output_dataset, "filters.html"), "w")

    text_file.write(html)
    text_file.close()
  else:
    slide_nums = set()
    for key in html_page_info:
      slide_num = floor(key / 1000)
      slide_nums.add(slide_num)
    slide_nums = sorted(list(slide_nums))
    total_len = len(slide_nums)
    page_size = HtmlGenerationConfig.filter_pagination_size
    num_pages = ceil(total_len / page_size)

    for page_num in range(1, num_pages + 1):
      start_index = (page_num - 1) * page_size
      end_index = (page_num * page_size) if (page_num < num_pages) else total_len
      page_slide_nums = slide_nums[start_index:end_index]

      html = ""
      html +=util_html.html_header("Filtered Images, Page %d" % page_num)

      html += "  <div style=\"font-size: 20px\">"
      if page_num > 1:
        if page_num == 2:
          html += "<a href=\"filters.html\">&lt;</a> "
        else:
          html += "<a href=\"filters-%d.html\">&lt;</a> " % (page_num - 1)
      html += "Page %d" % page_num
      if page_num < num_pages:
        html += " <a href=\"filters-%d.html\">&gt;</a> " % (page_num + 1)
      html += "</div>\n"

      html += "  <table>\n"
      for slide_num in page_slide_nums:
        html += "  <tr>\n"
        filter_num = 1

        displayup_key = slide_num * 1000 + filter_num
        while displayup_key in html_page_info:
          value = html_page_info[displayup_key]
          html += image_cell(value[0], value[1], value[2], value[3])
          displayup_key += 1
        html += "  </tr>\n"

      html += "  </table>\n"

      html +=util_html.html_footer()
      if page_num == 1:
        text_file = open(os.path.join(dataset_config.dir_output_dataset, "filters.html"), "w")
      else:
        text_file = open(os.path.join(dataset_config.dir_output_dataset, "filters-%d.html" % page_num), "w")
      text_file.write(html)
      print(blue("Generated HTML at " + os.path.join(dataset_config.dir_output_dataset, "filters.html")))
      text_file.close()

def image_cell(slide_num, filter_num, display_text, file_text):
    """
    Generate HTML for viewing a processed image.

    Args:
      slide_num: The slide number.
      filter_num: The filter number.
      display_text: Filter display name.
      file_text: Filter name for file.

    Returns:
      HTML for a table cell for viewing a filtered image.
    """
    print(red("Rq : use get_filter_image_result to get the path"))
    # filt_img = slide.get_filter_image_path(slide_num, filter_num, file_text)
    filt_img = slide.get_filter_image_result(slide_num,dataset_config=dataset_config)
    # filt_thumb = slide.get_filter_thumbnail_result(slide_num)
    filt_thumb = get_filter_image_result(slide_num,thumbnail = True,channel_number = channel_number,dataset_config=dataset_config)
    # filt_thumb = slide.get_filter_thumbnail_path(slide_num, filter_num, file_text)
    img_name = slide.get_filter_image_filename(slide_num, filter_num, file_text)
    return "      <td>\n" + \
          "        <a target=\"_blank\" href=\"%s\">%s<br/>\n" % (filt_img, display_text) + \
          "          <img src=\"%s\" />\n" % (filt_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

def mask_percentage_text(mask_percentage):
  """
  Generate a formatted string representing the percentage that an image is masked.

  Args:
    mask_percentage: The mask percentage.

  Returns:
    The mask percentage formatted as a string.
  """
  return "%3.2f%%" % mask_percentage





# def save_display(save, display, info, np_img, slide_num, filter_num, display_text, filter_file_text,
#                  display_mask_percentage=True, channel_number = None):
#   """
#   Optionally save an image and/or display the image.

#   Args:
#     save: If True, save filtered images.
#     display: If True, display filtered images to screen.
#     info: Dictionary to store filter information.
#     np_img: Image as a NumPy array.
#     slide_num: The slide number.
#     filter_num: The filter number.
#     display_text: Filter display name.
#     filter_file_text: Filter name for file.
#     display_mask_percentage: If True, display mask percentage on displayed slide.
#   """
#   mask_percentage = None
#   if channel_number is not None : 
#      display_text+="_channel_"+str(channel_number)+"_"
#   if display_mask_percentage:
#     mask_percentage = util.mask_percent(np_img)
#     display_text = display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
#   if slide_num is None and filter_num is None:
#     pass
#   elif filter_num is None:
#     display_text = "S%03d " % slide_num + display_text
#   elif slide_num is None:
#     display_text = "F%03d " % filter_num + display_text
#   else:
#     display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
#   if display:
#     util.display_img(np_img, display_text)
#   if save:

#     save_filtered_image(np_img, slide_num, filter_num, filter_file_text,channel_number)
#   if info is not None:
#     if channel_number is not None : 
#         info[slide_num * 1000 + filter_num+10*channel_number] = (slide_num, filter_num, display_text, filter_file_text, mask_percentage, channel_number)
#     else :
#         info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, filter_file_text, mask_percentage, channel_number)


# def save_filtered_image(np_img, slide_num, filter_num, filter_file_text, channel_number = None):
#   """
#   Save a filtered image to the file system.

#   Args:
#     np_img: Image as a NumPy array.
#     slide_num:  The slide number.
#     filter_num: The filter number.
#     filter_file_text: Descriptive text to add to the image filename.
#   """
#   # t = Time()
  
#   filepath = slide.get_filter_image_path(slide_num, filter_num, filter_file_text,channel_number)
#   print(blue("Saved with save_display "),filepath)
#   pil_img = util.np_to_pil(np_img)
#   pil_img.save(filepath)
#   #print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))

#   # t1 = Time()
#   thumbnail_filepath = slide.get_filter_thumbnail_path(slide_num, filter_num, filter_file_text ,channel_number)
#   print(blue("Saved with save_display "),thumbnail_filepath)
#   slide.save_thumbnail(pil_img, HtmlGenerationConfig.filter_pagination_size, thumbnail_filepath)
#   #print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_filepath))

def segment_from_cell_segmentation_param(dataset_config, img, cell_segmentation_param,channel_number = None, verbose = 0):
    list_results = [img]
    if dataset_config.data_type == "fluorescence":
        for process_name,param_process in cell_segmentation_param[channel_number].items():
            # print("process_name",process_name)
            img_filtered, name = filter.apply_process_from_name(list_results[-1], process_name, param_process, output_type = "bool")

            if type(img_filtered) is list:
                list_results.append(img_filtered[-1])
            else : 
                list_results.append(img_filtered)

    elif dataset_config.data_type == "wsi":
        #Just different pour la selection de l'image
        for process_name,param_process in cell_segmentation_param.items():
            img_filtered, name = filter.apply_process_from_name(list_results[-1], process_name, param_process, output_type = "bool")
            # print(name, " is done")
            list_results.append(img_filtered)
    else : 
        raise Exception("data_type not implemented")
    return list_results[-1].astype(bool)

def find_good_tissue_segmentation_param(slide_num,dataset_config):
    if "image_"+str(slide_num) in list(dataset_config.tissue_segmentation_param.keys()):
        tissue_segmentation_param = dataset_config.tissue_segmentation_param["image_"+str(slide_num)]
    else : 
        tissue_segmentation_param = dataset_config.tissue_segmentation_param["default"]
    return tissue_segmentation_param

def test_existing_segmentation_config(dataset_config,tiles_list, tiles_names,channels_to_segment,image_list ):
    """Test all segmentation parameters that exists on already processed images
    Save the details of each steps 
    """

    list_imgs_process = []   
    list_img_process_names = []
    for num_image, image in enumerate(tiles_list):
        if dataset_config.data_type == "fluorescence":
            for channel_number, image_channel in image.items():
                if channels_to_segment == "all" or channel_number in channels_to_segment:
                    
                    img_name = tiles_names[num_image]+"-Ch"+str(channel_number)+"-"+dataset_config.channel_names[channel_number]
                    list_config_process_names = [img_name]
                    # info_np(image_channel, img_name)
                    # im_mask_complete = (np.ones(image_channel.shape)*50).astype(np.uint8)
                    # image_channel = image_channel*im_mask_complete
                    # info_np(image_channel, img_name+ " after multiplication")
                    list_img_processed = [image_channel]

                    for existing_config_name, dict_processing_steps in dataset_config.cell_segmentation_param_by_cannnel.items():
                        list_img_process_temp = [image_channel]
                        for process_name,param_process in dict_processing_steps[channel_number].items():
                            if process_name == "filter_center_cells":
                                continue 
                            img_filtered, name = filter.apply_process_from_name(list_img_process_temp[-1], process_name, param_process, output_type = "bool")
                            if type(img_filtered) is list:
                                list_img_process_temp+=img_filtered  
                            else : 
                                list_img_process_temp.append(img_filtered)         
                        list_config_process_names.append(existing_config_name)
                        list_img_processed.append(list_img_process_temp[-1])
                    list_img_processed.append(image_channel)
                    list_config_process_names.append("Original")
                    list_img_process_names+=list_config_process_names
                    list_imgs_process.append(list_img_processed)
    ModelSegmentation.laboratory_segmentation_save_images(dataset_config,list_imgs_process, list_img_process_names, segmentation_type = "cells",channels_to_segment=channels_to_segment,image_list=image_list,existing_config=True)

def test_existing_segmentation_config_with_details(dataset_config,tiles_list, tiles_names,channels_to_segment):
    """Test all segmentation parameters that exists on already processed images
    Save the details of each steps 
    """

    for existing_config_name, dict_processing_steps in dataset_config.cell_segmentation_param_by_cannnel.items():
        print(existing_config_name)
        print("dict_processing_steps",dict_processing_steps)
        cell_segmentor = segmentation.ModelSegmentation(dataset_config=dataset_config, object_to_segment = "cells", cell_segmentation_param=dict_processing_steps)
        list_images_processed, list_process_names = cell_segmentor.test_cell_segmentation_from_tiles(tiles_list,tiles_names,channels_to_segment = channels_to_segment, save_html = True)
        ModelSegmentation.laboratory_segmentation_save_images(dataset_config,list_images_processed, list_process_names, segmentation_type = "cells",channels_to_segment=channels_to_segment,image_list=image_list,existing_config=existing_config_name)
        print("Done :)")

class ModelSegmentation:
    """
    Classe de base d'un model de segmentation 

    Required key of cell_segmentation_param : 
    - segmentation name : name of the segmentation method
    - min_cell_size : minimum size of a cell
    """
    def __init__(
        self,
        dataset_config = None ,#Pe créer une classe générique 
        object_to_segment = None,
        cell_segmentation_param= None, 
        tissue_segmentation_param = None,
        verbose=0,
    ):
        self.object_to_segment = object_to_segment
        self.dataset_config = dataset_config
        self.cell_segmentation_param = cell_segmentation_param
        self.tissue_segmentation_param = tissue_segmentation_param

    def segment_cells(self, img, channel_number=None, verbose=0):
        """
        Segment cells
        """
        if self.object_to_segment == "microglia_IHC" :
            #Ici je dois filtrer les cellules du centre 
            return segment_from_cell_segmentation_param(self.dataset_config, img, self.cell_segmentation_param,channel_number=channel_number, verbose = verbose)

            # return segment_microglia_IHC(img,self.cell_segmentation_param,verbose=verbose)
        else : 
            return segment_from_cell_segmentation_param(self.dataset_config, img, self.cell_segmentation_param,channel_number=channel_number, verbose = verbose)

            # # Segmentation modularity 
            # raise NotImplementedError("model_segmentation_type not implemented")


    def test_cell_segmentation_from_tiles(self, tiles_list,tiles_names,channels_to_segment = "all", save_html = True):
        """
        tiles_list is a list of dict where each object = dict[channel]
        """ 
        list_images_processed = []
        list_process_names = []
        check_stats_process = True 
        for num_image, image in enumerate(tiles_list):
            if self.dataset_config.data_type == "fluorescence":

                for channel_number, image_channel in image.items():

                    if channels_to_segment == "all" or channel_number in channels_to_segment:
                        list_img_process = [image_channel]
                        # img_name = "Image "+tiles_names[num_image]+" Channel "+str(channel_number)
                        img_name = tiles_names[num_image]+"-Ch"+str(channel_number)+"-"+self.dataset_config.channel_names[channel_number]
                        list_process_names.append(img_name)
                        for process_name,param_process in self.cell_segmentation_param[channel_number].items():
                            img_filtered, name = filter.apply_process_from_name(list_img_process[-1], process_name, param_process, output_type = "bool")
                            if type(img_filtered) is list:
                                list_img_process+=img_filtered  
                                list_process_names+=name
                            else : 
                                if check_stats_process : 
                                    n_components, ratio_masked = filter.get_n_components_mask_coverage(list_img_process[-1])
                                    name+="\n"+str(n_components)+" components\n"+str(np.round(ratio_masked*100,2))+"% masked"

                                list_img_process.append(img_filtered)
                                list_process_names.append(name) 

                            # name_img = "Image "+ str(num_image) +"->"+ name
                        list_img_process.append(image_channel)
                        list_process_names.append("original") 
                        list_images_processed.append(list_img_process)
            else : 
                list_img_process = [image]
                list_process_names.append("Image "+str(num_image))
                # print("cell_segmentation_param",self.cell_segmentation_param)
                for process_name,param_process in self.cell_segmentation_param.items():
                    img_filtered, name = filter.apply_process_from_name(list_img_process[-1], process_name, param_process, output_type = "bool")
                    list_img_process.append(img_filtered)
                    name_img = name
                    list_process_names.append(name_img)
                list_img_process.append(image)
                list_process_names.append("original") 
                list_images_processed.append(list_img_process)
        return list_images_processed, list_process_names

    def test_segmentation_grayscale(self,list_images,slide_num=None,sequence_processing=None):
        """
        Return 
        image_lists [[im_1_process_1,im_1_process_2,..],[im_2_process_1,im_2_process_2,..]]
        """
        if sequence_processing is None : 
            sequence_processing = find_good_tissue_segmentation_param(slide_num,self.dataset_config)
        list_images_processed = []
        list_process_names = []
        for num_image, image in enumerate(list_images):
            list_img_process = [image]
            list_process_names.append("Image "+str(num_image))

            for process_name,param_process in sequence_processing.items():
                img_filtered, name = filter.apply_process_from_name(list_img_process[-1], process_name, param_process, output_type = "bool")
                list_img_process.append(img_filtered)
                name_img = name
                list_process_names.append(name_img)
            list_img_process.append(image)
            list_process_names.append("original")
            list_images_processed.append(list_img_process)
        return list_images_processed, list_process_names

    @classmethod
    def get_n_images_col(cls,image_lists):
        max_process = 0
        # if self.dataset_config.data_type == "fluorescence": 
        #     if self.cell_segmentation_param is not None : 
        #         for row_i in range(len(image_lists)):

        #             nb_process = len(image_lists[row_i])
        #             if nb_process > max_process:
        #                     max_process = nb_process
        #     else : 
        #         max_process = len(self.tissue_segmentation_param.keys())

        # else : 
        #     nb_process = len(self.cell_segmentation_param.keys())
        #     if nb_process > max_process:
        #             max_process = nb_process
        for row_i in range(len(image_lists)):

            nb_process = len(image_lists[row_i])
            if nb_process > max_process:
                    max_process = nb_process
        return max_process

    @classmethod
    def laboratory_segmentation_save_images(cls,dataset_config,image_lists, processing_names ,segmentation_type = "Tissue",channels_to_segment=None,image_list=None,existing_config=None ):
        """
        Save images to a specified directory and generate an HTML file to visualize them in a grid.

        Args:
            image_lists (list of lists of np.array): List of lists of images. Each inner list represents a row.
            image_names (list): List of image names corresponding to each row.
            processing_names (list): List of processing names corresponding to each column.
            output_directory (str): The directory where images will be saved.
        """
        # Ensure the output directory exists

        output_directory = os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_base"],segmentation_type+"_filtering_laboratory")
        os.makedirs(output_directory, exist_ok=True)

        # Image size 
        image_height, image_width = image_lists[0][0].shape[0],image_lists[0][0].shape[1]
        ratio_rect = image_width/image_height

        # Create subplots
        n_rows = len(image_lists)


        # n_cols = len(image_lists[0])
        n_cols = cls.get_n_images_col(image_lists)#+1 because original image
        fig = sp.make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, shared_yaxes=False,vertical_spacing = 0.03,horizontal_spacing = 0.00005)
        plot_number = 0 
        for row in range(n_rows):
            for col in range(n_cols):
                if len(image_lists[row]) <= col : 
                    img_data = 255*np.ones((image_height,image_width))
                    proc_name = " "
                else : 
                    img_data = image_lists[row][col]
                    proc_name = processing_names[plot_number]
                    plot_number +=1 
                fig.add_trace(px.imshow(img_data,binary_string=True).data[0], row=row + 1, col=col + 1)
                # Set the title of the subplot
                fig.update_xaxes(title_text=proc_name, row=row + 1, col=col + 1)

        # Set layout for the subplots
        if channels_to_segment is not None :
            title_text = segmentation_type+" segmentation laboratory - "+dataset_config.dataset_name+" <b> Channel "+str(dataset_config.channel_names[channels_to_segment[0]])
        else :
            title_text = segmentation_type+" segmentation laboratory - "+dataset_config.dataset_name
        fig.update_layout(
            title_text=title_text,title_x=0.01, title_font=dict(size=40),
            showlegend=False,
            height=n_rows * 800,  # Adjust the height based on the number of rows
            width = int(n_cols * 800 * ratio_rect),
            margin=dict(
            l=1,
            r=1,
            b=10,
            t=100,
            pad=1))

        fig.update_layout(showlegend=False)
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_xaxes(side="top")
        fig.update_yaxes(visible=False)

        # Generate HTML file to view the grid of images
        if channels_to_segment is not None : 
            name_fig = os.path.join("channel_"+str(channels_to_segment[0]) +".html")
        else : 
            name_fig = "test_segmentation.html"
        id_img = 1
        if image_list is not None :
            output_directory = os.path.join(output_directory,"img_"+str(image_list[0]).zfill(2))
            mkdir_if_nexist(output_directory)
        if existing_config is not None : 
            if type(existing_config) == str : 
                name_fig = "config_"+existing_config +"_"+ name_fig
            else : 
                name_fig = "existing_config_"+ name_fig
        path = os.path.join(output_directory, name_fig)
        while os.path.exists(path):
            path = os.path.join(output_directory, str(id_img).zfill(2) + "_" +name_fig)
            id_img += 1
        pio.write_html(fig, path)
        print(blue("HTML saved at " + path))
