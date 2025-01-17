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

import glob
from math import floor
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np


try:
    import openslide
except ImportError:
    print("Grrr")

import os
import PIL
from PIL import Image
import re
import sys
from simple_colors import *
import tifffile as tiff #To suppress ? 
import czifile #Usefull twice but may be replaced by pyczi (https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/pylibCZIrw/pylibCZIrw_3_3_0.ipynb#scrollTo=114e59e5)
from utils import util

try:
    from pylibCZIrw import czi as pyczi
except ImportError:
    print("Pb importing pylibCZIrw")
from ast import literal_eval 
import pandas as pd 
import openslide

# #from python_files.const import *
from config.html_generation_config import HtmlGenerationConfig

def _get_cells_statistics(dict_statistics_images, dataset_config, image_number):
    """
    Create the dictionnary with all the stats from table_cells 
    """
    path_table_cells = util.get_path_table_cells(image_number,dataset_config)
    if os.path.exists(path_table_cells):
        table_cells_slide = pd.read_csv(path_table_cells, sep = ";")
    else :
        raise Exception("Table cells not found -> all tiles hasn't been segmented and classified")

    dict_statistics_images["n_cells_slide"] = len(table_cells_slide)
    dict_statistics_images["mean_n_cells_per_tile_slide"] = table_cells_slide.groupby(['tile_row', 'tile_col'])['id_cell'].count().mean()
    dict_statistics_images["std_n_cells_per_tile_slide"] = table_cells_slide.groupby(['tile_row', 'tile_col'])['id_cell'].count().std()
    dict_statistics_images["mean_cell_size_slide"] = table_cells_slide['size'].mean()
    dict_statistics_images["std_cell_size_slide"] = table_cells_slide['size'].std()

    for idx_decision, cell_type_name in enumerate(dataset_config.cell_class_names,1):
        dict_statistics_images["n_cells_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide["cell_type"] == idx_decision].shape[0]
        dict_statistics_images["fraction_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide["cell_type"] == idx_decision].shape[0]/dict_statistics_images["n_cells_slide"]
        dict_statistics_images["mean_size_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide['cell_type'] == idx_decision]['size'].mean()
        dict_statistics_images["std_size_{}_slide".format(cell_type_name)] = table_cells_slide[table_cells_slide['cell_type'] == idx_decision]['size'].std()
        if dataset_config.statistics_with_proba:
            dict_statistics_images["n_cells_{}_proba_slide".format(cell_type_name)] = table_cells_slide["proba_{}".format(cell_type_name)].sum()
            dict_statistics_images["fraction_{}_proba_slide".format(cell_type_name)] = table_cells_slide["proba_{}".format(cell_type_name)].sum()/dict_statistics_images["n_cells_slide"]
    # if dataset_config.data_type == "fluorescence":
    #     pass
    # elif dataset_config.data_type == "wsi": 
    #     pass
    # else : 
    #     raise Exception("Data type not found")
    return dict_statistics_images

def _get_nuclei_statistics(dict_statistics_images, dataset_config, image_number):
    """Create the dictionnary with all the stats about nuclei
    """

    table_nuclei_slide = None #TODO
    dict_statistics_images["n_nuclei_in_slide"] = None
    dict_statistics_images["mean_nuclei_density_slide"] = None
    dict_statistics_images["std_nuclei_density_slide"] = None

    if dataset_config.data_type == "fluorescence":
        pass
    elif dataset_config.data_type == "wsi": 
        pass
    else : 
        pass
        # raise Exception("Data type not found")

    return dict_statistics_images

def _get_tissue_statistics(dict_statistics_images, dataset_config,image_number): 

    tiles_and_tissue_percentage = pd.read_csv(os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(image_number).zfill(3)),"tiles_and_tissue_percentage.csv"),sep = ";")
    tiles_and_tissue_percentage["area_tissue_in_tile"] = tiles_and_tissue_percentage["tissue_percentage"]*dataset_config.tile_width*dataset_config.tile_height
    dict_statistics_images["area_tissue_slide"] = tiles_and_tissue_percentage["area_tissue_in_tile"].sum()
    dict_statistics_images["fraction_tissue_slide"] = dict_statistics_images["area_tissue_slide"]/dict_statistics_images["area_slide"]
    if dataset_config.data_type == "fluorescence":
        pass
    elif dataset_config.data_type == "wsi": 
        pass
    else : 
        pass
        # raise Exception("Data type not found")
    
    return dict_statistics_images

def _get_dataset_specific_statistics(dict_statistics_images,dataset_config, image_number):
    """ Each dataset can have its proper features. Here, the function get the value to each features depending on the name of the feature"""
    #All 
    for feature_name in dataset_config.colnames_df_image:
        if feature_name in dict_statistics_images.keys():
            continue 
        else : 
            #WSI 
            if feature_name == "age": #MODIFIED_CODE_HERE_FOR_SIMULATION
                dict_statistics_images[feature_name] =  1#dataset_config.mapping_img_age[image_number-1]
            elif feature_name == "gender":
                dict_statistics_images[feature_name] =  1#dataset_config.mapping_img_gender[image_number-1]
            elif feature_name == "model_segmentation_slide":
                dict_statistics_images[feature_name] =  dataset_config.model_segmentation_name
            elif feature_name == "model_classification_slide":
                dict_statistics_images[feature_name] =  dataset_config.classification_param["model_name"]
            else : 
                print("feature_name",feature_name)
                raise Exception("Feature name not found")
    return dict_statistics_images

def get_statistics_images(dataset_config,from_image=None, to_image=None,image_list=None):
    list_dictionnaries_images = []
    if image_list is None : 
        image_list = range(from_image, to_image+1)
    path_df = os.path.join(dataset_config.dir_output_dataset, "statistics_images.csv")
    if os.path.exists(path_df):
        df_old = pd.read_csv(path_df, sep = ";")
    for image_number in image_list:
        if os.path.exists(path_df):
            if image_number in list(df_old["slide_num"].unique()): 
                continue
        basic_info_image = pd.read_csv(os.path.join(dataset_config.dir_dataset, "info_data.csv"), sep = ";")
        
        dict_statistics_images = basic_info_image[basic_info_image["slide_num"] == image_number].iloc[0].to_dict()

        dict_statistics_images = _get_cells_statistics(dict_statistics_images, dataset_config, image_number)
        dict_statistics_images = _get_nuclei_statistics(dict_statistics_images, dataset_config, image_number)
        dict_statistics_images = _get_tissue_statistics(dict_statistics_images, dataset_config,image_number)

        dict_statistics_images = _get_dataset_specific_statistics(dict_statistics_images, dataset_config,image_number)

        list_dictionnaries_images.append(dict_statistics_images)
    df = pd.DataFrame(list_dictionnaries_images)
    path_df = os.path.join(dataset_config.dir_output_dataset, "statistics_images.csv")
    if os.path.exists(path_df):
        df_old = pd.read_csv(path_df, sep = ";")
        df = pd.concat([df_old, df],ignore_index=True)
        # df = df.append(df_old, ignore_index=True)
    os.makedirs(dataset_config.dir_output_dataset, exist_ok=True)
    df.to_csv(path_df, sep = ";", index = False)
    blue(f"Df savd at {path_df}")
    return df


def create_load_df_slides_CHANGED():
    """Create and save temporal analysis dataframe or load it if exists

    Cols are ["Slide_num", "pcw", "nb_tiles_tot","model_model_segmentation_type", 'model_classification_name', "nb_cells_tot","nb_cells_per_tiles_mean","nb_cells_per_tiles_std","nb_cells_<Amoeboid>","mean_size_<Amoeboid>,"std_size_<Amoeboid>]

    Returns:
        df_slides (DataFrame): dataframe with all informations on slides
        path_temporal_analysis_csv (str): path to the csv file
    """
    state_names = dataset_config.cells_label_names[1:-1]
    path_temporal_analysis_csv = os.path.join(
        dataset_config.dir_base_stat_analysis, "df_slides.csv"
    )
    if os.path.exists(path_temporal_analysis_csv):
        # print("df_slides already exists and will be completed !")
        df_slides = pd.read_csv(path_temporal_analysis_csv, sep=";")
        df_slides = df_slides.sort_values("pcw")
        return df_slides, path_temporal_analysis_csv
    else:
        colnames_features_temporal_analysis = [
            "pcw",
            "slide_num",
            "slide_shape",
            "slide_shape_in_tile",
            "n_tiles_slide",
            "pixel_resolution",
            "gender",
            "area_tissue_slide",
            "fraction_tissue_slide",
        ]
        colnames_tissue_percent = [
            "n_tiles_tissue_100_slide",
            "n_tiles_tissue_sup_75_slide",
            "n_tiles_tissue_sup_5_slide",
        ]
        colnames_model_used = ["model_segmentation_slide", "model_classification_slide"]
        nuclei_density = [
            "n_nuclei_in_slide",
            "mean_nuclei_density_slide",
            "std_nuclei_density_slide",
        ]

        colnames_ncells = [
            "n_cells_slide",
            "mean_n_cells_per_tile_slide",
            "std_n_cells_per_tile_slide",
        ] + ["n_" + state + "_slide" for state in state_names]
        colnames_ncells_proba_conservation = [
            "n_" + state + "_proba_slide" for state in state_names
        ]
        colnames_size = (
            ["mean_cell_size_slide"]
            + ["mean_size_" + state + "_slide" for state in state_names]
            + ["std_cell_size_slide"]
            + ["std_size_" + state + "_slide" for state in state_names]
        )

        colnames_features_temporal_analysis += (
            colnames_tissue_percent
            + colnames_model_used
            + nuclei_density
            + colnames_ncells
            + colnames_ncells_proba_conservation
            + colnames_size
        )
        df_slides = pd.DataFrame(columns=colnames_features_temporal_analysis)
        df_slides.to_csv(path_temporal_analysis_csv, sep=";", index=False)
        return df_slides, path_temporal_analysis_csv


def add_slide_to_slides_df_CHANGED(slide_num, dataset_config):
    """
    Complète le dataframe sur les slides
    Option : df_slides = df_slides.sort_values('pcw')
    """
    model_classification = ModelClassification(dataset_config.classification_param)
    model_segmentation = ModelSegmentation(dataset_config.cell_segmentation_param)

    state_names = model_classification.param_model["param_training_set"]["state_names"][
        1:
    ]
    df_slides, path_temporal_analysis_csv = create_load_df_slides()
    table_cells_slide, path_table_cells_slide = load_table_cells_slide(
        slide_num, model_segmentation, model_classification
    )
    time_per_tile_slide, path_time_per_tile_slide = load_time_per_tile_slide(
        slide_num, model_segmentation, model_classification
    )
    if slide_num in list(df_slides["slide_num"]):
        None
    else:
        dict_info_slide = dict()
        tissue_percent_stat = get_tissue_percent_stats(time_per_tile_slide)
        dict_qte_size = _get_qte_size_from_table_cells(
            table_cells_slide, time_per_tile_slide, model_classification
        )
        dict_info_slide["pcw"] = dataset_config.mapping_img_number[slide_num]
        dict_info_slide["slide_num"] = slide_num
        dict_info_slide["slide_shape"] = _get_slide_shape(slide_num)
        dict_info_slide["slide_shape_in_tile"] = (
            int(time_per_tile_slide["tile_row"].max() + 1),
            int(time_per_tile_slide["tile_col"].max() + 1),
        )  # a verif
        dict_info_slide["n_tiles"] = (
            dict_info_slide["slide_shape_in_tile"][0]
            * dict_info_slide["slide_shape_in_tile"][1]
        )
        dict_info_slide["pixel_resolution"] = dataset_config.conversion_px_micro_meter
        dict_info_slide["gender"] = "TODO"
        dict_info_slide["area_tissue_slide"] = time_per_tile_slide[
            "tissue_percent"
        ].sum() * (dataset_config.tile_height**2)
        dict_info_slide["fraction_tissue_slide"] = time_per_tile_slide[
            "tissue_percent"
        ].mean()  # mean de celles qui ont du tissue ? Oui je crois car sinon elles sont pas dans le df
        dict_info_slide["n_tiles_tissue_100_slide"] = tissue_percent_stat[0]
        dict_info_slide["n_tiles_tissue_sup_75_slide"] = tissue_percent_stat[1]
        dict_info_slide["n_tiles_tissue_sup_5_slide"] = tissue_percent_stat[2]
        dict_info_slide[
            "model_segmentation_slide"
        ] = model_segmentation.model_segmentation_type
        dict_info_slide[
            "model_classification_slide"
        ] = model_classification.param_model["model_name"]
        dict_info_slide["n_nuclei_in_slide"] = 1
        dict_info_slide["mean_nuclei_density_slide"] = 1
        dict_info_slide["std_nuclei_density_slide"] = 1
        dict_info_slide["n_cells_slide"] = dict_qte_size["n_cells"]
        dict_info_slide["mean_n_cells_per_tile_slide"] = dict_qte_size[
            "mean_n_cells_per_tile"
        ]
        dict_info_slide["std_n_cells_per_tile_slide"] = dict_qte_size[
            "std_n_cells_per_tile"
        ]
        for state in state_names:
            dict_info_slide["n_" + state + "_slide"] = dict_qte_size["n_" + state]
            dict_info_slide["n_" + state + "_proba_slide"] = dict_qte_size[
                "n_" + state + "_proba"
            ]
        dict_info_slide["mean_cell_size_slide"] = dict_qte_size["mean_cell_size"]
        dict_info_slide["std_cell_size_slide"] = dict_qte_size["std_cell_size"]
        for state in state_names:
            dict_info_slide["mean_size_" + state + "_slide"] = dict_qte_size[
                "mean_size_" + state
            ]
            dict_info_slide["std_size_" + state + "_slide"] = dict_qte_size[
                "std_size_" + state
            ]
        df_slides = df_slides.append(dict_info_slide, ignore_index=True)
        df_slides.to_csv(path_temporal_analysis_csv, sep=";", index=False)
    return df_slides


# CORRESPONDANCE_NOMS
def correspondance_nom_slide_num(slide_num):
    # indice = np.where(CORRESPONDANCE_NOMS[:,1].astype(int)==slide_num)[0][0]
    # nom_slide = CORRESPONDANCE_NOMS[indice, 0]
    # return nom_slide
    return str(slide_num)


def get_channels(slide_filepath):
    with pyczi.open_czi(slide_filepath) as czidoc:
        metadata = czidoc.raw_metadata
        a = metadata.find("Before Exp")
        b = metadata.find("Smart</BeforeHardwareSetting>")
        part_with_channel = metadata[a:b]
        channels_list = part_with_channel[part_with_channel.find("[")+1:part_with_channel.find("]")].split(",")
        return channels_list

def get_info_data(dataset_config,image_list = None, from_image = None, to_image = None):
    list_dict = []
    if image_list is None : 
        image_list = list(range(from_image, to_image+1))
    else : 
        image_list = image_list
    for slide_num in image_list:
        padded_sl_num = str(slide_num).zfill(3)
        slide_filepath = os.path.join(dataset_config.dir_dataset, padded_sl_num + "." + dataset_config.data_format)
        dict_slide = dict()
        if dataset_config.data_format == "czi":
            # with pyczi.open_czi(slide_filepath) as czi:
            czi = czifile.imread(slide_filepath)
            dict_slide["dataset_name"] = dataset_config.dataset_name
            dict_slide["slide_num"] = int(padded_sl_num)
            dict_slide["slide_name"] = dataset_config.mapping_img_name[slide_num-1]
            dict_slide["slide_path"] = slide_filepath
            dict_slide["slide_type"] = dataset_config.data_format
            dict_slide["slide_shape"] = czi.shape
            shape_squeezed = [ k for k in dict_slide["slide_shape"] if k!=1]
            dict_slide["slide_height"] = shape_squeezed[-2]
            dict_slide["slide_width"] = shape_squeezed[-1]
            dict_slide["area_slide"] =  dict_slide["slide_height"]*dict_slide["slide_width"]
            dict_slide["n_tiles_row_slide"] = int(shape_squeezed[-2]/dataset_config.tile_height)+1
            dict_slide["n_tiles_col_slide"] = int(shape_squeezed[-1]/dataset_config.tile_width)+1
            dict_slide["slide_shape_in_tile"] = (dict_slide["n_tiles_row_slide"],dict_slide["n_tiles_col_slide"])
            dict_slide["n_tiles_slide"] = dict_slide["n_tiles_row_slide"]*dict_slide["n_tiles_col_slide"]
            dict_slide["pixel_resolution"] = dataset_config.conversion_px_micro_meter
            dict_slide["channels"] = get_channels(slide_filepath)
            dict_slide["slide_size"] = czi.size #depth*width*height*n_channels
            dict_slide["slide_dtype"] = czi.dtype
            dict_slide["slide_min"] = czi.min()
            dict_slide["slide_max"] = czi.max().astype(np.uint32)

            list_dict.append(dict_slide)
        else :
            slide_im = openslide.open_slide(slide_filepath)
            dict_slide["dataset_name"] = dataset_config.dataset_name
            dict_slide["slide_num"] = int(padded_sl_num)
            dict_slide["slide_name"] = dataset_config.mapping_img_name[slide_num-1]
            dict_slide["slide_path"] = slide_filepath
            dict_slide["slide_type"] = dataset_config.data_format
            dict_slide["slide_shape"] = slide_im.dimensions
            dict_slide["slide_height"] = slide_im.dimensions[1]
            dict_slide["slide_width"] = slide_im.dimensions[0]
            dict_slide["area_slide"] =  dict_slide["slide_height"]*dict_slide["slide_width"]
            shape_squeezed = [ k for k in dict_slide["slide_shape"] if k!=1]
            dict_slide["row_tile_size"] = dataset_config.tile_height
            dict_slide["col_tile_size"] = dataset_config.tile_width
            dict_slide["n_tiles_row_slide"] = int(shape_squeezed[-1]/dataset_config.tile_height)+1
            dict_slide["n_tiles_col_slide"] = int(shape_squeezed[-2]/dataset_config.tile_width)+1
            dict_slide["slide_shape_in_tile"] = (dict_slide["n_tiles_row_slide"],dict_slide["n_tiles_col_slide"])
            dict_slide["n_tiles_slide"] = dict_slide["n_tiles_row_slide"]*dict_slide["n_tiles_col_slide"]
            dict_slide["pixel_resolution"] = dataset_config.conversion_px_micro_meter
            dict_slide["slide_size"] = os.path.getsize(slide_filepath)
            dict_slide["slide_dtype"] = "np.uint8" #slide.read_region((0,0),0,slide.level_dimensions[0]).convert("RGB").dtype
            dict_slide["slide_min"] = 0
            dict_slide["slide_max"] = 1
            list_dict.append(dict_slide)
    df = pd.DataFrame(list_dict)
    # df = df.append(list_dict, ignore_index=True)
    path_df = os.path.join(dataset_config.dir_dataset, "info_data.csv")
    if os.path.exists(path_df):
        df_old = pd.read_csv(path_df, sep = ";")
        df = pd.concat([df_old, df], ignore_index=True)
        # df = df.append(df_old, ignore_index=True)
    df.to_csv(os.path.join(dataset_config.dir_dataset, "info_data.csv"), sep = ";", index = False)
    # print("Info data saved in: ", os.path.join(dataset_config.dir_dataset, "info_data.csv"))
    return df

def get_info_img_from_column_name(slide_num, column_name, dataset_config):
    """get_info_img_from_column_name
    Search the value of the column_name for the slide_num
    """
    path_csv = os.path.join(dataset_config.dir_dataset, "info_data.csv")
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv, sep = ";")


        df = df[df["slide_num"] == slide_num]

        if column_name == "n_channels" : 
            shape = literal_eval(df["slide_shape"].values[0])
            shape_without_1 = [x for x in shape if x != 1]
            n_channels = shape_without_1[dataset_config.dim_position["C"]]
            channels = literal_eval(df["channels"].values[0])

            return n_channels, channels
        if column_name == "slide_shape_in_tile":
            n_row_n_col = literal_eval(df["slide_shape_in_tile"].values[0])
            return n_row_n_col
    else : 
        raise Exception("Info data not found")

def get_z_stack_img(slide_num,dataset_config):
    path_csv = os.path.join(dataset_config.dir_dataset, "info_data.csv")
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv, sep = ";")
        df = df[df["slide_num"] == slide_num]
        shape = literal_eval(df["slide_shape"].values[0])
        shape_without_1 = [x for x in shape if x != 1]
        n_z_stack = shape_without_1[dataset_config.dim_position["Z"]]
        return n_z_stack

def get_n_row_col_img(slide_num,dataset_config):
    path_csv = os.path.join(dataset_config.dir_dataset, "info_data.csv")
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv, sep = ";")
        df = df[df["slide_num"] == int(slide_num)]
        n_tiles_row_slide = df["n_tiles_row_slide"].values[0]
        n_tiles_col_slide = df["n_tiles_col_slide"].values[0]
        return n_tiles_row_slide,n_tiles_col_slide

def open_slide(filename):
    """
    Open a whole-slide image (*.svs, etc).

    Args:
      filename: Name of the slide file.

    Returns:
      An OpenSlide object representing a whole-slide image.
    """
    try:
        slide = openslide.open_slide(filename)
    except FileNotFoundError:
        slide = None
    return slide


def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).

    Args:
      filename: Name of the image file.

    returns:
      A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image


def open_image_np(filename):
    """
    Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

    Args:
      filename: Name of the image file.

    returns:
      A NumPy representing an RGB image.
    """
    pil_img = open_image(filename)
    np_img = util.pil_to_np_rgb(pil_img)
    return np_img


def get_training_slide_path(slide_num,dataset_config):
    """
    Convert slide number to a path to the corresponding WSI training slide file.

    Example:
      5 -> ../data/training_slides/TUPAC-TR-005.svs

    Args:
      slide_num: The slide number.

    Returns:
      Path to the WSI training slide file.
    """
    # if channel_number is None :

    padded_sl_num = str(slide_num).zfill(3)
    ## slide_filepath = os.path.join(dataset_config.dir_dataset,  padded_sl_num + "." + dataset_config.data_format)
    slide_filepath = os.path.join(
        dataset_config.dir_dataset, padded_sl_num + "." + dataset_config.data_format
    )
    return slide_filepath
    # else : 
    #     padded_sl_num = str(slide_num).zfill(3)
    #     ## slide_filepath = os.path.join(dataset_config.dir_dataset,  padded_sl_num + "." + dataset_config.data_format)
    #     slide_filepath = os.path.join(
    #         dataset_config.dir_dataset, padded_sl_num + "_fluo_channel_" + str(channel_number) + "."+dataset_config.data_format
    #     )
    #     return slide_filepath   


def get_crop_image_path(crop):
    """
    Obtain crop image path based on crop information such as row, column, row pixel position, column pixel position,
    pixel width, and pixel height.

    Args:
      crop: Crop object.

    Returns:
      Path to image crop.
    """
    padded_sl_num = str(crop.slide_num).zfill(3)
    padded_til_num = (
        "R"
        + str(crop.tile_row_number).zfill(3)
        + "-C"
        + str(crop.tile_col_number).zfill(3)
    )

    crop_path = os.path.join(
        CROP_DIR,
        padded_sl_num,
        padded_til_num,
        padded_sl_num
        + "-r"
        + str(crop.tile_row_number).zfill(3)
        + "-c"
        + str(crop.tile_col_number).zfill(3)
        + "-"
        + "crop"
        + str(crop.num_crop).zfill(2)
        + "_RGB."
        + dataset_config.preprocessing_config.dest_train_ext,
    )
    tile_directory_path = os.path.join(
        CROP_DIR, padded_sl_num, padded_til_num
    )  # ex ./crop_png/009/R44-C12

    return tile_directory_path, crop_path


def get_tile_image_path(tile):
    """
    Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
    pixel width, and pixel height.

    Args:
      tile: Tile object.

    Returns:
      Path to image tile.
    """
    t = tile
    dataset_config = t.dataset_config
    padded_sl_num = str(t.slide_num).zfill(3)
    tile_path = os.path.join(
        dataset_config.preprocessing_config.preprocessing_path["dir_tiles"] ,
        padded_sl_num,
        padded_sl_num
        + "-"
        + dataset_config.preprocessing_config.tile_suffix
        + "-r%d-c%d-x%d-y%d-w%d-h%d"
        % (t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s)
        + "."
        + dataset_config.preprocessing_config.dest_train_ext,
    )
    return tile_path


def get_tile_image_path_by_slide_row_col(slide_num, row, col):
    """
    Obtain tile image path using wildcard displayup with slide number, row, and column.

    Args:
      slide_num: The slide number.
      row: The row.
      col: The column.

    Returns:
      Path to image tile.
    """
    padded_sl_num = str(slide_num).zfill(3)
    wilcard_path = os.path.join(
        dataset_config.preprocessing_config.preprocessing_path["dir_tiles"] ,
        padded_sl_num,
        + padded_sl_num
        + "-"
        + dataset_config.preprocessing_config.tile_suffix
        + "-r%d-c%d-*." % (row, col)
        + dataset_config.preprocessing_config.dest_train_ext,
    )
    if len(glob.glob(wilcard_path)) == 0:
        """
        Il n'y a pas la tile sur l'ordi
        """
        # print("The path to the tile image of slide " +str(slide_num) + " row " + str(row) + " col " + str(col) + " does not exist because the tile has not been downloaded to the computer.")
        return ""
    img_path = glob.glob(wilcard_path)[0]
    return img_path

def get_downscaled_paths(directory,slide_num,channel_number = None, large_w=None, large_h=None, small_w=None, small_h=None,dataset_config=None):
    """
    Args:
      slide_num: The slide number.
      large_w: Large image width.
      large_h: Large image height.
      small_w: Small image width.
      small_h: Small image height.

    Returns:
       Path to the image file.
    """
    "slide_numslide_num,",slide_num
    if dataset_config.preprocessing_config.tissue_extraction_accept_holes:
        txt_holes = "with_holes_"
    else :
        txt_holes = "without_holes_"
    padded_sl_num = str(slide_num).zfill(3)
    if channel_number is None : 
        txt_channel = ""
    elif channel_number == -1:
        txt_channel = "-C-all"
    else : 
        txt_channel = "-C-"+str(channel_number)
    if "thumbnail" in directory:
        ext = HtmlGenerationConfig.thumbnail_ext
    else:
        ext = dataset_config.preprocessing_config.dest_train_ext
    dir = dataset_config.preprocessing_config.preprocessing_path[directory]

    if directory != "dir_downscaled_img":
        training_img_path = get_training_image_path(slide_num, dataset_config=dataset_config)
        training_img_path = get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number, large_w=None, large_h=None, small_w=None, small_h=None,dataset_config=dataset_config)
        large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
        img_path = os.path.join(dir,
        padded_sl_num
        +txt_channel
        + "-"
        + str(dataset_config.preprocessing_config.scale_factor)
        + "x-"
        + str(large_w)
        + "x"
        + str(large_h)
        + "-"
        + str(small_w)
        + "x"
        + str(small_h)
        + "."
        + ext,
        )
        return img_path
    
    else : 
        if large_w is None and large_h is None and small_w is None and small_h is None: #Retrieve it when doesn't exist

            if channel_number is None : 
                # print(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"])
                # wildcard_path = os.path.join(
                #     dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,  padded_sl_num + "*." + dataset_config.preprocessing_config.dest_train_ext
                # )
                # print(os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"]))
                path_list = [
                    os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] , k)
                    for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] )
                    if k.startswith(padded_sl_num)]
                img_path = path_list[0]
                return img_path
            elif channel_number == -1 :
                # print("dataset_config.preprocessing_config.preprocessing_path[]",dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"])
                # print(os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"]))
                path_list = [
                    os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] , k)
                    for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] )
                    if k.startswith(padded_sl_num) and "channel_all" in k]
                img_path = path_list[0]
                return img_path
            else : 

                path_list = [
                    os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] , k)
                    for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] )
                    if k.startswith(padded_sl_num) and "channel_"+str(channel_number) in k]
                img_path = path_list[0]
                return img_path
        else : 
            if channel_number is None : 
                img_path = os.path.join(
                    dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,
                    padded_sl_num
                    +txt_channel
                    + "-"
                    + str(dataset_config.preprocessing_config.scale_factor)
                    + "x-"
                    + str(large_w)
                    + "x"
                    + str(large_h)
                    + "-"
                    + str(small_w)
                    + "x"
                    + str(small_h)
                    + "."
                    + ext,
                )
                return img_path
            elif channel_number == -1 :
                img_path = os.path.join(
                    dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,
                    padded_sl_num
                    +txt_channel
                    + "-"
                    + str(dataset_config.preprocessing_config.scale_factor)
                    + "x-"
                    + str(large_w)
                    + "x"
                    + str(large_h)
                    + "-"
                    + str(small_w)
                    + "x"
                    + str(small_h)
                    + "_channel_all" 
                    + "."
                    + ext,
                )
                return img_path
            else :  
                img_path = os.path.join(
                    dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,
                    padded_sl_num
                    +txt_channel
                    + "-"
                    + str(dataset_config.preprocessing_config.scale_factor)
                    + "x-"
                    + str(large_w)
                    + "x"
                    + str(large_h)
                    + "-"
                    + str(small_w)
                    + "x"
                    + str(small_h)
                    + "_channel_" 
                    + str(channel_number)
                    + "."
                    + ext,
                )
                return img_path    


#     return img_filename
    # if large_w is None and large_h is None and small_w is None and small_h is None: #Retrieve it when doesn't exist
        # if channel_number is None : 
        #     path_list = [
        #         os.path.join(dir, k)
        #         for k in os.listdir(dir)
        #         if k.startswith(padded_sl_num)]
        #     img_path = path_list[0]
        #     return img_path
        # elif channel_number == -1 :
        #     path_list = [
        #         os.path.join(dir, k)
        #         for k in os.listdir(dir)
        #         if k.startswith(padded_sl_num) and "channel_all" in k]
        #     img_path = path_list[0]
        #     return img_path
        # else : 
        #     path_list = [
        #         os.path.join(dir, k)
        #         for k in os.listdir(dir)
        #         if k.startswith(padded_sl_num) and "channel_"+str(channel_number) in k]
        #     img_path = path_list[0]
        #     return img_path
    # else: #Create it when doesn't exist 



def get_training_image_path(
    slide_num, large_w=None, large_h=None, small_w=None, small_h=None, channel_number = None,dataset_config=None
):
    """
    Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
    the corresponding file based on the slide number will be displayed up in the file system using a wildcard.

    Example:
      5 -> ../data/training_png/TUPAC-TR-005-32x-49920x108288-1560x3384.png

    Args:
      slide_num: The slide number.
      large_w: Large image width.
      large_h: Large image height.
      small_w: Small image width.
      small_h: Small image height.

    Returns:
       Path to the image file.
    """
    padded_sl_num = str(slide_num).zfill(3)
    if large_w is None and large_h is None and small_w is None and small_h is None: #Retrieve it when doesn't exist
        if channel_number is None : 
            # wildcard_path = os.path.join(
            #     dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,  padded_sl_num + "*." + dataset_config.preprocessing_config.dest_train_ext
            # )

            path_list = [
                os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] , k)
                for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] )
                if k.startswith(padded_sl_num)]
            img_path = path_list[0]
            return img_path
        elif channel_number == -1 :
            #"concatenation des canaux"
            # wildcard_path = os.path.join(
            #     dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,  padded_sl_num + "*." + dataset_config.preprocessing_config.dest_train_ext
            # )
            path_list = [
                os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] , k)
                for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] )
                if k.startswith(padded_sl_num) and "channel_all" in k]


            # print(glob.glob(wildcard_path))
            # img_path = glob.glob(wildcard_path)[0]

            img_path = path_list[0]
            return img_path
        else : 
            # wildcard_path = os.path.join(
            #     dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,  padded_sl_num + "*." + dataset_config.preprocessing_config.dest_train_ext
            # )
            path_list = [
                os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] , k)
                for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] )
                if k.startswith(padded_sl_num) and "channel_"+str(channel_number) in k]
            # print(glob.glob(wildcard_path))
            # img_path = glob.glob(wildcard_path)[0]

            img_path = path_list[0]
            return img_path
    else: #Create it when doesn't exist 
        if channel_number is None : 
            img_path = os.path.join(
                dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,
                padded_sl_num
                + "-"
                + str(dataset_config.preprocessing_config.scale_factor)
                + "x-"
                + str(large_w)
                + "x"
                + str(large_h)
                + "-"
                + str(small_w)
                + "x"
                + str(small_h)
                + "."
                + dataset_config.preprocessing_config.dest_train_ext,
            )
            return img_path
        elif channel_number == -1 :
            img_path = os.path.join(
                dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,
                padded_sl_num
                + "-"
                + str(dataset_config.preprocessing_config.scale_factor)
                + "x-"
                + str(large_w)
                + "x"
                + str(large_h)
                + "-"
                + str(small_w)
                + "x"
                + str(small_h)
                + "_channel_all" 
                + "."
                + dataset_config.preprocessing_config.dest_train_ext,
            )
            return img_path
        else :  
            img_path = os.path.join(
                dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_img"] ,
                padded_sl_num
                + "-"
                + str(dataset_config.preprocessing_config.scale_factor)
                + "x-"
                + str(large_w)
                + "x"
                + str(large_h)
                + "-"
                + str(small_w)
                + "x"
                + str(small_h)
                + "_channel_" 
                + str(channel_number)
                + "."
                + dataset_config.preprocessing_config.dest_train_ext,
            )
            return img_path
            


# def get_training_thumbnail_path(
#     slide_num, large_w=None, large_h=None, small_w=None, small_h=None, channel_number = None
# ):
#     """
#     Convert slide number and optional dimensions to a training thumbnail path. If no dimensions are
#     supplied, the corresponding file based on the slide number will be displayed up in the file system using a wildcard.

#     Example:
#       5 -> ../data/training_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384.jpg

#     Args:
#       slide_num: The slide number.
#       large_w: Large image width.
#       large_h: Large image height.
#       small_w: Small image width.
#       small_h: Small image height.

#     Returns:
#        Path to the thumbnail file.
#     """
#     padded_sl_num = str(slide_num).zfill(3)
#     if large_w is None and large_h is None and small_w is None and small_h is None:
#         # wilcard_path = os.path.join(
#         #     dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] ,
#         #      padded_sl_num + "*." + HtmlGenerationConfig.thumbnail_ext,
#         # )

#         # img_path = glob.glob(wilcard_path)[0]
#         if channel_number is None : 
#             print("Channel_number is none")
#             print(os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] ))
#             print(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"])
#             path_list = [
#                 os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] , k)
#                 for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] )
#                 if k.startswith(padded_sl_num)]
#             img_path = path_list[0]
#             return img_path
#         elif channel_number == -1 :
#             path_list = [
#                 os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] , k)
#                 for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] )
#                 if k.startswith(padded_sl_num) and "channel_all" in k]
#             img_path = path_list[0]
#             return img_path
#         else : 
#             path_list = [
#                 os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] , k)
#                 for k in os.listdir(dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] )
#                 if k.startswith(padded_sl_num) and "channel_"+str(channel_number) in k]
#             img_path = path_list[0]
#             return img_path
#     else:
#         if channel_number is None :
#             img_path = os.path.join(
#                 dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] ,
#                 padded_sl_num
#                 + "-"
#                 + str(dataset_config.preprocessing_config.scale_factor)
#                 + "x-"
#                 + str(large_w)
#                 + "x"
#                 + str(large_h)
#                 + "-"
#                 + str(small_w)
#                 + "x"
#                 + str(small_h)
#                 + "."
#                 + HtmlGenerationConfig.thumbnail_ext,
#             )
#         elif channel_number == -1 : 
#             img_path = os.path.join(
#                 dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] ,
#                 padded_sl_num
#                 + "-"
#                 + str(dataset_config.preprocessing_config.scale_factor)
#                 + "x-"
#                 + str(large_w)
#                 + "x"
#                 + str(large_h)
#                 + "-"
#                 + str(small_w)
#                 + "x"
#                 + str(small_h)
#                 + "_channel_all" 
#                 + "."
#                 + HtmlGenerationConfig.thumbnail_ext,
#             )
#         else : 
#             img_path = os.path.join(
#                 dataset_config.preprocessing_config.preprocessing_path["dir_thumbnail_original"] ,
#                 padded_sl_num
#                 + "-"
#                 + str(dataset_config.preprocessing_config.scale_factor)
#                 + "x-"
#                 + str(large_w)
#                 + "x"
#                 + str(large_h)
#                 + "-"
#                 + str(small_w)
#                 + "x"
#                 + str(small_h)
#                 + "_channel_" 
#                 + str(channel_number)
#                 + "."
#                 + HtmlGenerationConfig.thumbnail_ext,
#             )
#     return img_path


# def get_filter_image_path(slide_num, filter_number, filter_name_info, channel_number = None):
#     """
#     Convert slide number, filter number, and text to a path to a filter image file.

#     Example:
#       5, 1, "rgb" -> ../data/filter_png/TUPAC-TR-005-001-rgb.png

#     Args:
#       slide_num: The slide number.
#       filter_number: The filter number.
#       filter_name_info: Descriptive text describing filter.

#     Returns:
#       Path to the filter image file.
#     """
#     dir = dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_filtered_img"] 
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     if channel_number is None : 
#         img_path = os.path.join(
#             dir, get_filter_image_filename(slide_num, filter_number, filter_name_info)
#         )
#     else : 
#         img_path = os.path.join(
#             dir, get_filter_image_filename(slide_num, filter_number, filter_name_info, channel_number)
#         )
#     return img_path


# def get_filter_thumbnail_path(slide_num, filter_number, filter_name_info, channel_number = None):
#     """
#     Convert slide number, filter number, and text to a path to a filter thumbnail file.

#     Example:
#       5, 1, "rgb" -> ../data/filter_thumbnail_jpg/TUPAC-TR-005-001-rgb.jpg

#     Args:
#       slide_num: The slide number.
#       filter_number: The filter number.
#       filter_name_info: Descriptive text describing filter.

#     Returns:
#       Path to the filter thumbnail file.
#     """
#     dir = dataset_config.preprocessing_config.preprocessing_path["dir_filtered_thumbnail_img"] 
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     img_path = os.path.join(
#         dir,
#         get_filter_image_filename(
#             slide_num, filter_number, filter_name_info, thumbnail=True, channel_number = channel_number),
#     )
#     return img_path


def get_filter_image_filename(
    slide_num, filter_number, filter_file_text, thumbnail=False, channel_number = None
):
    """
    Convert slide number, filter number, and text to a filter file name.

    Example:
      5, 1, "rgb", False -> TUPAC-TR-005-001-rgb.png
      5, 1, "rgb", True -> TUPAC-TR-005-001-rgb.jpg

    Args:
      slide_num: The slide number.
      filter_number: The filter number.
      filter_name_info: Descriptive text describing filter.
      thumbnail: If True, produce thumbnail filename.

    Returns:
      The filter image or thumbnail file name.
    """
    if thumbnail:
        ext = HtmlGenerationConfig.thumbnail_ext
    else:
        ext = dataset_config.preprocessing_config.dest_train_ext
    padded_sl_num = str(slide_num).zfill(3)
    padded_fi_num = str(filter_number).zfill(3)
    if channel_number is None : 
        
        img_filename = (
            padded_sl_num
            + "-"
            + padded_fi_num
            + "-"
            + filter_file_text
            + "."
            + ext
        )
    else : 
        txt_channel = "_channel_"+str(channel_number) if channel_number != -1 else  "_channel_all"
        img_filename = (
            padded_sl_num
            + "-"
            + padded_fi_num
            + "-"
            + filter_file_text
            +txt_channel
            + "."
            + ext
        )
    return img_filename


# def get_tile_summary_image_path(slide_num,channel_number = None):
#     """
#     Convert slide number to a path to a tile summary image file.

#     Example:
#       5 -> ../data/tile_summary_png/TUPAC-TR-005-tile_summary.png

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the tile summary image file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_filtered"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_filtered"] )
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_tiles_filtered"] , get_tile_summary_image_filename(slide_num,channel_number=channel_number)
#     )
#     return img_path


# def get_tile_summary_thumbnail_path(slide_num):
#     """
#     Convert slide number to a path to a tile summary thumbnail file.

#     Example:
#       5 -> ../data/tile_summary_thumbnail_jpg/TUPAC-TR-005-tile_summary.jpg

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the tile summary thumbnail file.
#     """
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_tiles_thumbnail_filtered"] ,
#         get_tile_summary_image_filename(slide_num, thumbnail=True,channel_number=channel_number),
#     )
#     return img_path


# def get_tile_summary_on_original_image_path(slide_num, channel_number = None):
#     """
#     Convert slide number to a path to a tile summary on original image file.

#     Example:
#       5 -> ../data/tile_summary_on_original_png/TUPAC-TR-005-tile_summary.png

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the tile summary on original image file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_original"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_original"] )
#     img_path = os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_original"], get_tile_summary_image_filename(slide_num,channel_number=channel_number))
#     return img_path


# def get_tile_summary_on_original_thumbnail_path(slide_num):
#     """
#     Convert slide number to a path to a tile summary on original thumbnail file.

#     Example:
#       5 -> ../data/tile_summary_on_original_thumbnail_jpg/TUPAC-TR-005-tile_summary.jpg

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the tile summary on original thumbnail file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_thumbnail_original"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_thumbnail_original"] )
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_tiles_thumbnail_original"] ,
#         get_tile_summary_image_filename(slide_num, thumbnail=True,channel_number=channel_number),
#     )
#     return img_path


# def get_top_tiles_on_original_image_path(slide_num):
#     """
#     Convert slide number to a path to a top tiles on original image file.

#     Example:
#       5 -> ../data/top_tiles_on_original_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the top tiles on original image file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_original"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_original"] )
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_original"] , get_top_tiles_image_filename(slide_num)
#     )
#     return img_path


# def get_top_tiles_on_original_thumbnail_path(slide_num):
#     """
#     Convert slide number to a path to a top tiles on original thumbnail file.

#     Example:
#       5 -> ../data/top_tiles_on_original_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the top tiles on original thumbnail file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_thumbnail_original"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_thumbnail_original"] )
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_thumbnail_original"] ,
#         get_top_tiles_image_filename(slide_num, thumbnail=True,channel_number=channel_number),
#     )
#     return img_path


# def get_tile_summary_image_filename(slide_num, thumbnail=False,channel_number=None):
#     """
#     Convert slide number to a tile summary image file name.

#     Example:
#       5, False -> TUPAC-TR-005-tile_summary.png
#       5, True -> TUPAC-TR-005-tile_summary.jpg

#     Args:
#       slide_num: The slide number.
#       thumbnail: If True, produce thumbnail filename.

#     Returns:
#       The tile summary image file name.
#     """
#     if thumbnail:
#         ext = HtmlGenerationConfig.thumbnail_ext
#     else:
#         ext = dataset_config.preprocessing_config.dest_train_ext

#     padded_sl_num = str(slide_num).zfill(3)
#     # training_img_path = get_training_image_path(slide_num)
#     large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(
#         training_img_path
#     )
#     img_filename = (
#         padded_sl_num
#         + "-"
#         + str(dataset_config.preprocessing_config.scale_factor)
#         + "x-"
#         + str(large_w)
#         + "x"
#         + str(large_h)
#         + "-"
#         + str(small_w)
#         + "x"
#         + str(small_h)
#         + "-"
#         + "tile_summary"
#         + "."
#         + ext
#     )

#     return img_filename


# def get_top_tiles_image_filename(slide_num, thumbnail=False):
#     """
#     Convert slide number to a top tiles image file name.

#     Example:
#       5, False -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png
#       5, True -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg

#     Args:
#       slide_num: The slide number.
#       thumbnail: If True, produce thumbnail filename.

#     Returns:
#       The top tiles image file name.
#     """
#     if thumbnail:
#         ext = HtmlGenerationConfig.thumbnail_ext
#     else:
#         ext = dataset_config.preprocessing_config.dest_train_ext
#     padded_sl_num = str(slide_num).zfill(3)

#     training_img_path = get_training_image_path(slide_num)
#     large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(
#         training_img_path
#     )
#     img_filename = (
#         padded_sl_num
#         + "-"
#         + str(dataset_config.preprocessing_config.scale_factor)
#         + "x-"
#         + str(large_w)
#         + "x"
#         + str(large_h)
#         + "-"
#         + str(small_w)
#         + "x"
#         + str(small_h)
#         + "-"
#         + "random_tiles"
#         + "."
#         + ext
#     )

#     return img_filename


# def get_top_tiles_image_path(slide_num):
#     """
#     Convert slide number to a path to a top tiles image file.

#     Example:
#       5 -> ../data/top_tiles_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the top tiles image file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_filtered"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_filtered"] )
#     img_path = os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_filtered"] , get_top_tiles_image_filename(slide_num))
#     return img_path


# def get_top_tiles_thumbnail_path(slide_num):
#     """
#     Convert slide number to a path to a tile summary thumbnail file.

#     Example:
#       5 -> ../data/top_tiles_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg
#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the top tiles thumbnail file.
#     """
#     if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_thumbnail_filtered"] ):
#         os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_thumbnail_filtered"] )
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_random_tiles_thumbnail_filtered"] ,
#         get_top_tiles_image_filename(slide_num, thumbnail=True),
#     )
#     return img_path


def get_tile_data_filename(slide_num,channel_number=None,dataset_config=None):
    """
    Convert slide number to a tile data file name.

    Example:
      5 -> TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

    Args:
      slide_num: The slide number.

    Returns:
      The tile data file name.
    """
    padded_sl_num = str(slide_num).zfill(3)

    # training_img_path = get_training_image_path(slide_num)
    training_img_path = get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number,dataset_config=dataset_config)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(
        training_img_path
    )
    if channel_number is None : 
        txt_channel = ""
    elif channel_number == -1:
        txt_channel = "_channel_all"
    else : 
        txt_channel = "_channel_"+str(channel_number)
    data_filename = (
        padded_sl_num
        + txt_channel
        + "-"
        + str(dataset_config.preprocessing_config.scale_factor)
        + "x-"
        + str(large_w)
        + "x"
        + str(large_h)
        + "-"
        + str(small_w)
        + "x"
        + str(small_h)
        + "-"
        + "tile_data"
        + ".csv"
    )

    return data_filename


def get_tile_data_path(slide_num,dataset_config):
    """
    Convert slide number to a path to a tile data file.

    Example:
      5 -> ../data/tile_data/TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

    Args:
      slide_num: The slide number.

    Returns:
      Path to the tile data file.
    """
    if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_summary"] ):
        os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_summary"] )
    file_path = os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_tiles_summary"] , get_tile_data_filename(slide_num,dataset_config=dataset_config))
    return file_path


def get_filter_image_result(slide_num,thumbnail = False,channel_number = None,dataset_config=None):
    """
    Convert slide number to the path to the file that is the final result of filtering.

    Example:
      5 -> ../data/filter_png/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.png

    Args:
      slide_num: The slide number.

    Returns:
      Path to the filter image file.
    """
    if dataset_config.preprocessing_config.tissue_extraction_accept_holes:
        txt_holes = "with_holes_"
    else :
        txt_holes = "without_holes_"
    if thumbnail:
        ext = HtmlGenerationConfig.thumbnail_ext
        dir = dataset_config.preprocessing_config.preprocessing_path["dir_filtered_thumbnail_img"]
    else:
        ext = dataset_config.preprocessing_config.dest_train_ext
        dir = dataset_config.preprocessing_config.preprocessing_path["dir_downscaled_filtered_img"]
    
    if channel_number is None : 
        txt_channel = ""
    elif channel_number == -1:
        txt_channel = "-C-all"
    else : 
        txt_channel = "-C-"+str(channel_number)
    padded_sl_num = str(slide_num).zfill(3)
    # training_img_path = get_training_image_path(slide_num,channel_number=channel_number)
    training_img_path = get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number,dataset_config=dataset_config)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    
    # path_rgb_filtered = os.path.join(dir,
    #     padded_sl_num
    #     + "-"
    #     + str(dataset_config.preprocessing_config.scale_factor)
    #     + "x-"
    #     + str(large_w)
    #     + "x"
    #     + str(large_h)
    #     + "-"
    #     + str(small_w)
    #     + "x"
    #     + str(small_h)
    #     + "-"
    #     + txt_holes
    #     + "filtered"
    #     + txt_channel
    #     + "."
    #     + ext,
    # )
    path_rgb_filtered = os.path.join(dir,
        padded_sl_num
        +txt_channel
        + "-"
        + str(dataset_config.preprocessing_config.scale_factor)
        + "x-"
        + str(large_w)
        + "x"
        + str(large_h)
        + "-"
        + str(small_w)
        + "x"
        + str(small_h)
        + "."
        + ext,
        )
    # path_mask = os.path.join(dir,
    #     padded_sl_num
    #     + "-"
    #     + str(dataset_config.preprocessing_config.scale_factor)
    #     + "x-"
    #     + str(large_w)
    #     + "x"
    #     + str(large_h)
    #     + "-"
    #     + str(small_w)
    #     + "x"
    #     + str(small_h)
    #     + "-"
    #     + txt_holes
    #     + "filtered"
    #     + "_binary"
    #     +txt_channel
    #     +"."
    #     + ext,
    # )
    path_mask = os.path.join(dir,
        padded_sl_num
        +txt_channel
        + "-"
        + str(dataset_config.preprocessing_config.scale_factor)
        + "x-"
        + str(large_w)
        + "x"
        + str(large_h)
        + "-"
        + str(small_w)
        + "x"
        + str(small_h)
        +"_binary"
        + "."
        + ext,
        )
    return path_rgb_filtered, path_mask


# def get_filter_thumbnail_result(slide_num):
#     """
#     Convert slide number to the path to the file that is the final thumbnail result of filtering.

#     Example:
#       5 -> ../data/filter_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.jpg

#     Args:
#       slide_num: The slide number.

#     Returns:
#       Path to the filter thumbnail file.
#     """
#     padded_sl_num = str(slide_num).zfill(3)
#     training_img_path = get_training_image_path(slide_num)
#     large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(
#         training_img_path
#     )
#     img_path = os.path.join(
#         dataset_config.preprocessing_config.preprocessing_path["dir_filtered_thumbnail_img"] ,
#         padded_sl_num
#         + "-"
#         + str(dataset_config.preprocessing_config.scale_factor)
#         + "x-"
#         + str(large_w)
#         + "x"
#         + str(large_h)
#         + "-"
#         + str(small_w)
#         + "x"
#         + str(small_h)
#         + "-"
#         + "filtered"
#         + "."
#         + HtmlGenerationConfig.thumbnail_ext,
#     )
#     return img_path


def parse_dimensions_from_image_filename(filename):
    """
    Parse an image filename to extract the original width and height and the converted width and height.

    Example:
      "TUPAC-TR-011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)

    Args:
      filename: The image filename.

    Returns:
      Tuple consisting of the original width, original height, the converted width, and the converted height.
    """
    m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
    large_w = int(m.group(1))
    large_h = int(m.group(2))
    small_w = int(m.group(3))
    small_h = int(m.group(4))
    return large_w, large_h, small_w, small_h


def small_to_large_mapping(small_pixel, large_dimensions, dataset_config):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.

    Args:
      small_pixel: The scaled-down width and height.
      large_dimensions: The width and height of the original whole-slide image.

    Returns:
      Tuple consisting of the scaled-up width and height.
    """
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round(
        (large_w / dataset_config.preprocessing_config.scale_factor)
        / floor(large_w / dataset_config.preprocessing_config.scale_factor)
        * (dataset_config.preprocessing_config.scale_factor * small_x)
    )
    large_y = round(
        (large_h / dataset_config.preprocessing_config.scale_factor)
        / floor(large_h / dataset_config.preprocessing_config.scale_factor)
        * (dataset_config.preprocessing_config.scale_factor * small_y)
    )
    return large_x, large_y

def split_img_into_channels_img_range(start_ind, end_ind,dataset_config):
    for slide_num in range(start_ind, end_ind + 1):
        # print(slide_num)
        split_img_into_channels(slide_num,dataset_config)
    

def split_img_into_channels(slide_num,dataset_config):
    """
    Somme les Z s'il y a de la profondeur 
    """
    padded_sl_num = str(slide_num).zfill(3)
    slide_filepath = os.path.join(
        dataset_config.dir_dataset, padded_sl_num + "." + dataset_config.data_format
    )
    # print(slide_filepath)
    if dataset_config.data_format == "czi":
        # print("Image lecture")
        czi = czifile.imread(slide_filepath)
        # with pyczi.open_czi(slide_filepath) as czi:
        array = np.squeeze(czi)#, axis=(0, 1, -1))
        dim_position = dataset_config.dim_position
            
        if "Z" in dim_position.keys():
            czi_transposed = np.transpose(array, (dim_position["X"],dim_position["Y"],dim_position["C"],dim_position["Z"]))
            array_after_transform = np.sum(czi_transposed, axis=3)
        # czi_transposed = np.transpose(array, (1,2,0))
        else :
            array_after_transform = np.transpose(array, (dim_position["X"],2,0))

        for i in range(array_after_transform.shape[2]):
            # print("i = "+str(i))
            img = array_after_transform[:,:,i]
            img_rescales = util.clip_and_rescale_image(img)
            output_file = os.path.join(dataset_config.dir_dataset, padded_sl_num + "_fluo_channel_" + str(i) + ".png")
            tiff.imwrite(output_file, img_rescales, photometric='minisblack')  # 'minisblack' for grayscale
    else : 
        print("split_img_into_channels -> not czi file")
        return None

# def create_channel_order(channels,dataset_config):

def create_channel_order(channels,dataset_config):
    if hasattr(dataset_config,"channel_order"):
        return dataset_config.channel_order
    channel_order = []  
    for fluorophore in channels:
        # print("fluorophore",fluorophore)
        fluorophore = fluorophore.replace(" ","")
        # print(fluorophore)
        if fluorophore not in dataset_config.FLUOROPHORE_MAPPING_RGB.keys():
            raise Exception("Fluorophore not found in mapping")
        else : 
            channel_order.append(dataset_config.FLUOROPHORE_MAPPING_RGB[fluorophore])
    return channel_order

def training_slide_to_image(slide_num,dataset_config):
    """
    Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
    Doc for czi reading : https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/pylibCZIrw/pylibCZIrw_3_3_0.ipynb#scrollTo=114e59e5 
    Args:
      slide_num: The slide number.
    """
    # ICI : if split channel alors je creer des nouveaux fichiers avec les noms des channels 
    print(blue("Pre-processing : downscalling image {}...".format(slide_num)))
    if dataset_config.consider_image_with_channels :
        n_channels, channels = get_info_img_from_column_name(slide_num,dataset_config=dataset_config,column_name="n_channels")
        downscaled_img_canals = []
        for channel_number in range(n_channels):
            img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_num,dataset_config=dataset_config,channel_number=channel_number)
            downscaled_img_canals.append(img)
            # img_path = get_training_image_path(slide_num, large_w, large_h, new_w, new_h,channel_number)
            img_path = get_downscaled_paths("dir_downscaled_img", slide_num,channel_number, large_w, large_h, new_w, new_h,dataset_config=dataset_config)
            img.save(img_path)
            thumbnail_path = get_downscaled_paths("dir_thumbnail_original", slide_num,channel_number, large_w, large_h, new_w, new_h,dataset_config=dataset_config)
            # thumbnail_path = get_training_thumbnail_path(slide_num, large_w, large_h, new_w, new_h,channel_number)
            save_thumbnail(img, HtmlGenerationConfig.thumbnail_size, thumbnail_path)

        which_channel_to_find_RGB = create_channel_order(channels,dataset_config)

        list_ordered_imgs = [downscaled_img_canals[which_channel_to_find_RGB[k]] for k in range(n_channels)]
        # print(len(list_ordered_imgs))
        img_all_channels = np.dstack(list_ordered_imgs[:3])
        # img_all_channels = np.dstack([downscaled_img_canals[channel_order[0]],downscaled_img_canals[channel_order[1]],downscaled_img_canals[channel_order[2]]])
        # img_path = get_training_image_path(slide_num, large_w, large_h, new_w, new_h,channel_number=-1)#all canals
        img_path = get_downscaled_paths("dir_downscaled_img", slide_num,-1, large_w, large_h, new_w, new_h,dataset_config=dataset_config)
        img_all_channels_pil = Image.fromarray(img_all_channels)
        img_all_channels_pil.save(img_path)
        # thumbnail_path = get_training_thumbnail_path(slide_num, large_w, large_h, new_w, new_h,channel_number=-1)
        thumbnail_path = get_downscaled_paths("dir_thumbnail_original", slide_num,-1, large_w, large_h, new_w, new_h,dataset_config=dataset_config)

        save_thumbnail(img_all_channels_pil, HtmlGenerationConfig.thumbnail_size, thumbnail_path)
    else : 
        img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_num,dataset_config=dataset_config)
        # img_path = get_training_image_path(slide_num, large_w, large_h, new_w, new_h)
        img_path = get_downscaled_paths("dir_downscaled_img", slide_num, None,large_w, large_h, new_w, new_h,dataset_config=dataset_config)
        img.save(img_path)
        thumbnail_path = get_downscaled_paths("dir_thumbnail_original", slide_num,None, large_w, large_h, new_w, new_h,dataset_config=dataset_config)
        save_thumbnail(img, HtmlGenerationConfig.thumbnail_size, thumbnail_path)


def slide_to_scaled_pil_image(slide_num,dataset_config = None,channel_number = None):
    """
    Convert a WSI training slide to a scaled-down PIL image.

    Args:
      slide_num: The slide number.

    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    slide_filepath = get_training_slide_path(slide_num, dataset_config=dataset_config)
    if dataset_config.data_format =="czi":
        with pyczi.open_czi(slide_filepath) as czi_file:
            total_bounding_rectangle = czi_file.total_bounding_rectangle
            # print("total_bounding_rectangle",total_bounding_rectangle)
            # large_w, large_h = total_bounding_rectangle[2]-total_bounding_rectangle[0],total_bounding_rectangle[3]-total_bounding_rectangle[1]
            large_w, large_h = total_bounding_rectangle[2],total_bounding_rectangle[3]
            # print("large_w, large_h",large_w, large_h)
            new_w = floor(large_w / dataset_config.preprocessing_config.scale_factor)
            new_h = floor(large_h / dataset_config.preprocessing_config.scale_factor)
            # print(red("Tentative sommation selon z"))
            if dataset_config.has_Z:
                n_z = get_z_stack_img(slide_num,dataset_config)
                channel_i = np.zeros((large_h,large_w,1))
                for i in range(n_z):
                    channel_i += czi_file.read(plane={'C': channel_number,'Z': i})
            else :
                channel_i = czi_file.read(plane={'C': channel_number})
            # channel_i = czi_file.read(plane={'C': channel_number})
            channel_i = np.squeeze(channel_i)
            channel_i_rescaled = util.clip_and_rescale_image(channel_i)
            channel_i_pil = Image.fromarray(channel_i_rescaled)
            channel_i_pil = channel_i_pil.convert('L')
            # print("new_w, new_h",new_w, new_h)
            img = channel_i_pil.resize((new_w, new_h), PIL.Image.BILINEAR)
            return img, large_w, large_h, new_w, new_h
    else : 
        slide = open_slide(slide_filepath)
        large_w, large_h = slide.dimensions
        new_w = floor(large_w / dataset_config.preprocessing_config.scale_factor)
        new_h = floor(large_h / dataset_config.preprocessing_config.scale_factor)
        level = slide.get_best_level_for_downsample(dataset_config.preprocessing_config.scale_factor)
        whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        whole_slide_image = whole_slide_image.convert("RGB")
        img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        return img, large_w, large_h, new_w, new_h


def slide_to_scaled_np_image(slide_num,dataset_config):
    """
    Convert a WSI training slide to a scaled-down NumPy image.

    Args:
      slide_num: The slide number.

    Returns:
      Tuple consisting of scaled-down NumPy image, original width, original height, new width, and new height.
    """
    pil_img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_num, dataset_config=dataset_config)
    np_img = util.pil_to_np_rgb(pil_img)
    return np_img, large_w, large_h, new_w, new_h


def show_slide(slide_num,dataset_config):
    """
    Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.

    Args:
      slide_num: The slide number.
    """
    pil_img = slide_to_scaled_pil_image(slide_num,dataset_config=dataset_config)[0]
    pil_img.show()


def save_thumbnail(pil_img, size, path, display_path=False):
    """
    Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.

    Args:
      pil_img: The PIL image to save as a thumbnail.
      size:  The maximum width or height of the thumbnail.
      path: The path to the thumbnail.
      display_path: If True, display thumbnail path in console.
    """
    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    if display_path:
        print("Saving thumbnail to: " + path)
    dir = os.path.dirname(path)
    if dir != "" and not os.path.exists(dir):
        os.makedirs(dir)
    img.save(path)


def get_num_training_slides():
    """
    Obtain the total number of WSI training slide images.

    Returns:
      The total number of WSI training slide images.
    """
    num_training_slides = len(
        glob.glob1(dataset_config.dir_dataset, "*." + dataset_config.data_format)
    )
    return num_training_slides


def training_slide_range_to_images(dataset_config,image_list = None, from_image = None, to_image = None):
    """
    Convert a range of WSI training slides to smaller images (in a format such as jpg or png).

    Args:
      start_ind: Starting index (inclusive).
      end_ind: Ending index (inclusive).

    Returns:
      The starting index and the ending index of the slides that were converted.
    """
    if image_list is None : 
        image_list = list(range(from_image, to_image+1))
    else : 
        image_list = image_list
    for slide_num in image_list:
        training_slide_to_image(slide_num,dataset_config)


def singleprocess_training_slides_to_images():
    """
    Convert all WSI training slides to smaller images using a single process.
    """
    t = util.Time()

    num_train_images = get_num_training_slides()
    training_slide_range_to_images(1, num_train_images)

    t.elapsed_display()


def multiprocess_training_slides_to_images(image_num_list=None):
    """
    Convert all WSI training slides to smaller images using multiple processes (one process per core).
    Each process will process a range of slide numbers.
    """
    timer = util.Time()

    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    if image_num_list is not None:
        num_train_images = len(image_num_list)
    else:
        num_train_images = get_num_training_slides()
    
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    # each task specifies a range of slides
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((start_index, end_index))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print(
                "Task #"
                + str(num_process)
                + ": Process slides "
                + str(start_index)
                + " to "
                + str(end_index)
            )

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(training_slide_range_to_images, t))

    for result in results:
        (start_ind, end_ind) = result.get()
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    timer.elapsed_display()


def slide_stats(dataset_config):
    """
    Display statistics/graphs about training slides.
    """
    t = util.Time()

    if not os.path.exists(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] ):
        os.makedirs(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] )

    num_train_images = get_num_training_slides()
    slide_stats = []
    for slide_num in range(1, num_train_images + 1):
        slide_filepath = get_training_slide_path(slide_num,dataset_config=dataset_config)
        # print("Opening Slide #%d: %s" % (slide_num, slide_filepath))
        slide = open_slide(slide_filepath)
        (width, height) = slide.dimensions
        print("  Dimensions: {:,d} x {:,d}".format(width, height))
        slide_stats.append((width, height))

    max_width = 0
    max_height = 0
    min_width = sys.maxsize
    min_height = sys.maxsize
    total_width = 0
    total_height = 0
    total_size = 0
    which_max_width = 0
    which_max_height = 0
    which_min_width = 0
    which_min_height = 0
    max_size = 0
    min_size = sys.maxsize
    which_max_size = 0
    which_min_size = 0
    for z in range(0, num_train_images):
        (width, height) = slide_stats[z]
        if width > max_width:
            max_width = width
            which_max_width = z + 1
        if width < min_width:
            min_width = width
            which_min_width = z + 1
        if height > max_height:
            max_height = height
            which_max_height = z + 1
        if height < min_height:
            min_height = height
            which_min_height = z + 1
        size = width * height
        if size > max_size:
            max_size = size
            which_max_size = z + 1
        if size < min_size:
            min_size = size
            which_min_size = z + 1
        total_width = total_width + width
        total_height = total_height + height
        total_size = total_size + size

    avg_width = total_width / num_train_images
    avg_height = total_height / num_train_images
    avg_size = total_size / num_train_images

    stats_string = ""
    stats_string += "%-11s {:14,d} pixels (slide #%d)".format(max_width) % (
        "Max width:",
        which_max_width,
    )
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_height) % (
        "Max height:",
        which_max_height,
    )
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_size) % (
        "Max size:",
        which_max_size,
    )
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_width) % (
        "Min width:",
        which_min_width,
    )
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_height) % (
        "Min height:",
        which_min_height,
    )
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_size) % (
        "Min size:",
        which_min_size,
    )
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_width)) % "Avg width:"
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_height)) % "Avg height:"
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_size)) % "Avg size:"
    stats_string += "\n"

    stats_string += "\nslide number,width,height"
    for i in range(0, len(slide_stats)):
        (width, height) = slide_stats[i]
        stats_string += "\n%d,%d,%d" % (i + 1, width, height)
    stats_string += "\n"

    stats_file = open(os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] , "stats.txt"), "w")
    stats_file.write(stats_string)
    stats_file.close()

    t.elapsed_display()

    x, y = zip(*slide_stats)
    colors = np.random.rand(num_train_images)
    sizes = [10 for n in range(num_train_images)]
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.title("SVS Image Sizes")
    plt.set_cmap("prism")
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] , "svs-image-sizes.png"))
    plt.show()

    plt.clf()
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.title("SVS Image Sizes (Labeled with slide numbers)")
    plt.set_cmap("prism")
    for i in range(num_train_images):
        snum = i + 1
        plt.annotate(str(snum), (x[i], y[i]))
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] , "svs-image-sizes-slide-numbers.png"))
    plt.show()

    plt.clf()
    area = [w * h / 1000000 for (w, h) in slide_stats]
    plt.hist(area, bins=64)
    plt.xlabel("width x height (M of pixels)")
    plt.ylabel("# images")
    plt.title("Distribution of image sizes in millions of pixels")
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] , "distribution-of-svs-image-sizes.png"))
    plt.show()

    plt.clf()
    whratio = [w / h for (w, h) in slide_stats]
    plt.hist(whratio, bins=64)
    plt.xlabel("width to height ratio")
    plt.ylabel("# images")
    plt.title("Image shapes (width to height)")
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] , "w-to-h.png"))
    plt.show()

    plt.clf()
    hwratio = [h / w for (w, h) in slide_stats]
    plt.hist(hwratio, bins=64)
    plt.xlabel("height to width ratio")
    plt.ylabel("# images")
    plt.title("Image shapes (height to width)")
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_stats"] , "h-to-w.png"))
    plt.show()


def slide_info(display_all_properties=False):
    """
    Display information (such as properties) about training images.

    Args:
      display_all_properties: If True, display all available slide properties.
    """
    t = util.Time()

    num_train_images = get_num_training_slides()
    obj_pow_20_list = []
    obj_pow_40_list = []
    obj_pow_other_list = []
    for slide_num in range(1, num_train_images + 1):
        slide_filepath = get_training_slide_path(slide_num,dataset_config=dataset_config)
        # print("\nOpening Slide #%d: %s" % (slide_num, slide_filepath))
        slide = open_slide(slide_filepath)
        print("Level count: %d" % slide.level_count)
        print("Level dimensions: " + str(slide.level_dimensions))
        print("Level downsamples: " + str(slide.level_downsamples))
        print("Dimensions: " + str(slide.dimensions))
        ## theo objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        objective_power = 20  ##theo
        print("Objective power: " + str(objective_power))
        if objective_power == 20:
            obj_pow_20_list.append(slide_num)
        elif objective_power == 40:
            obj_pow_40_list.append(slide_num)
        else:
            obj_pow_other_list.append(slide_num)
        print("Associated images:")
        for ai_key in slide.associated_images.keys():
            print("  " + str(ai_key) + ": " + str(slide.associated_images.get(ai_key)))
        print("Format: " + str(slide.detect_format(slide_filepath)))
        if display_all_properties:
            print("Properties:")
            for prop_key in slide.properties.keys():
                print(
                    "  Property: "
                    + str(prop_key)
                    + ", value: "
                    + str(slide.properties.get(prop_key))
                )

    print("\n\nSlide Magnifications:")
    print("  20x Slides: " + str(obj_pow_20_list))
    print("  40x Slides: " + str(obj_pow_40_list))
    print("  ??x Slides: " + str(obj_pow_other_list) + "\n")

    t.elapsed_display()


# if __name__ == "__main__":
# show_slide(2)
# slide_info(display_all_properties=True)
# slide_stats()


## Théo


def get_num_training_crop():
    """
    Obtain the total number of crop to annotate

    Returns:
      The total number of crop to annotate
    """
    num_crops = len(glob.glob1(SRC_CROP_DIR, "*." + dataset_config.data_format))
    return num_crops

def get_images_to_segment(image_list,dataset_config):
    if dataset_config.data_type == "fluorescence" and dataset_config.consider_image_with_channels:
        if dataset_config.use_imgs_as_channels : 
            channel_number = None
        else : 
            channel_number = dataset_config.channel_used_to_segment_tissue
    else : 
        channel_number = None

    list_images = []
    list_images_names = []
    for slide_num in image_list: 
        img_path = get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number,dataset_config=dataset_config)
        np_orig = open_image_np(img_path)
        list_images.append(np_orig)
        img_name = dataset_config.mapping_img_name[slide_num-1]
        list_images_names.append(img_name)

    return list_images, list_images_names