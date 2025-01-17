# General import
from __future__ import print_function

#Importation des librairies 
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from PIL import Image
from os import listdir
import random 
import glob
from simple_colors import *

from segmentation_classification.segmentation import ModelSegmentation
from segmentation_classification import segmentation

from preprocessing import filter
from preprocessing import slide
from preprocessing import tiles
from utils.util import * 
from utils.util_fig_display import * 
from utils.util_colors_drawing import * 
#from python_files.const import *
from segmentation_classification.train_classification_model import unet 

# from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall 
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows
# Suppressing some warnings
import warnings
warnings.filterwarnings('ignore')

# Keras imports
from keras import models
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam 
from keras.metrics import SparseCategoricalAccuracy,MeanIoU

#### from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger # we currently don't use any other callbacks from ModelCheckpoints
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
# from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import load_img,plot_model
from keras import backend as keras
from keras.callbacks import Callback
from tensorflow.keras.utils import img_to_array
from tensorflow.keras import layers

from tensorflow.math import argmax
import tensorflow as tf
from skimage import img_as_ubyte, io, transform
import matplotlib as mpl
from matplotlib.pyplot import imread
from pathlib import Path
import shutil
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import ceil

from pip._internal.operations.freeze import freeze
import subprocess
# Imports for QC
from scipy import signal
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr


# For sliders and dropdown menu and progress bar
from ipywidgets import interact
import ipywidgets as widgets
# from tqdm import tqdm
from tqdm.notebook import tqdm

import skimage.morphology as sk_morphology
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows

from datetime import datetime
# from config.base_config import BaseConfig
# from config.datasets_config import *
# base_config = BaseConfig()
# # Get the configuration
# if base_config.dataset_name == "ihc_microglia_fetal_human_brain_database":
#     dataset_config = IhcMicrogliaFetalHumanBrain()
# elif base_config.dataset_name == "cancer_data_immunofluorescence":
#     dataset_config = FluorescenceCancerConfig()

def test_cell_truncated_by_3_tiles(x_min,x_max,y_min,y_max,n_tiles_border):
    border_max = dataset_config.tile_height*n_tiles_border-1
    if x_min == 0 or y_min == 0 or x_max == border_max or y_max == border_max:
        cell_is_truncated_by_3_tiles = True
    else :
        cell_is_truncated_by_3_tiles = False
    return cell_is_truncated_by_3_tiles


def segment_classify_cells_wsi_from_tile(slide_num,row, col, tile_cat,dataset_config, id_cell_max, channels_cells_to_segment=None):
    """
    Return 
    - Table cells with columns dataset_config.colnames_table_cells_base for row, col of slide_num
    - Save masks all cells 
    """
    save_mask = True
    # table_cells = pd.DataFrame(columns = dataset_config.colnames_table_cells_base)
    list_cells_tile = []
    cell_segmentation_param = find_good_cell_segmentation_param(slide_num,dataset_config)

    cell_segmentor = ModelSegmentation(dataset_config=dataset_config,object_to_segment = "microglia_IHC", cell_segmentation_param=cell_segmentation_param)
    model_classification = ModelClassification(dataset_config.classification_param,dataset_config=dataset_config)

    #Get tile 
    tile_rgb = tiles.get_tile(dataset_config, slide_num, row, col, (dataset_config.border_size_during_segmentation,dataset_config.border_size_during_segmentation),channels_of_interest = None)
    # display_rgb(tile_rgb, title = "row "+str(row)+" col "+str(col), figsize = (10,10))
    #Segment cells 
    segmented_cells_in_center = cell_segmentor.segment_cells(tile_rgb, verbose = 0)
    segmented_cells_in_center_label = sk_morphology.label(segmented_cells_in_center)

    id_cell = 0 
    # n_cells = np.max(segmented_cells_in_center_label)
    for cell in np.unique(segmented_cells_in_center_label)[1:]: #0 == background 
        tile_mask_cell = np.where(segmented_cells_in_center_label==cell,1,0).astype(bool)
        dict_cell = {
            "slide_num" : slide_num,
            "tile_row" : row,
            "tile_col" : col,
            "tile_cat" : tile_cat,
        }
        dict_cell = add_cell_coord_info(dict_cell,tile_mask_cell,dataset_config) #test
        if not dict_cell["check_in_centered_tile"] :#test
            # print("Cell outside center tile but should have been excluded earlier ")#test
            continue #test
        id_cell += 1 #test
        # dict_cell = add_cell_classification_info_wsi(dataset_config, dict_cell, slide_num, tile_rgb, mask_cell, id_cell, save_mask = save_mask)
        cell_mask = crop_around_cell(tile_mask_cell,dict_cell["x_tile_border"],dict_cell["y_tile_border"],max(dict_cell["length_max"],256)).astype(bool)
        crop_cell_rgb = crop_around_cell(tile_rgb,dict_cell["x_tile_border"],dict_cell["y_tile_border"],crop_larger = dataset_config.crop_size) 
        # if row == 15 and col == 40 : 
        #     display_mask(cell_mask,title="cell x_tile"+str(dict_cell["x_tile"])+"y_tile : "+str(dict_cell["y_tile"]))
        #     display_rgb(crop_cell_rgb)
        dict_cell = model_classification.classify_cells(crop_rgb=crop_cell_rgb,cell_mask=cell_mask, dict_cell=dict_cell)
        dict_cell["id_cell"] = id_cell_max+id_cell

        if save_mask :
            path_mask_cell = get_path_cell(slide_num,0, dict_cell["tile_row"], dict_cell["tile_col"], dict_cell["x_tile"], dict_cell["y_tile"], dataset_config)
            crop_cell_pil = np_to_pil(cell_mask)
            crop_cell_pil.save(path_mask_cell)

        # df_cell = pd.DataFrame(dict_cell, index = [id_cell])
        # table_cells = table_cells.append(df_cell)
        #-> here dict_cell is complete 
        list_cells_tile.append(dict_cell)
        # table_cells.loc[id_cell] = [id_cell_max+id_cell,slide_num, None, row,col,x_tile,y_tile,x,y,size,x_min,x_max,y_min,y_max,length_max]
    return list_cells_tile

# def add_cell_classification_info_wsi(dataset_config, dict_cell, slide_num, tile_rgb, tile_mask_cell, id_cell, save_mask = False):
#     """ 
#     Build dictionary cell info and save mask (classify rgb image)

#     Inputs : 
#     - Mask of the cell in img of size tile_height+2*border_size_during_segmentation
#     - RGB of the same img 

#     Outputs : 
#     dict_cell is enriched with : 
#         - proba class i 
#         - decision
#         - bonification cluster 
#         - penalite cluster
#         - ID_big_cluster
#     """

#     model_classification = ModelClassification(dataset_config.classification_param,dataset_config=dataset_config)

#     cell_mask = crop_around_cell(tile_mask_cell,dict_cell["x_tile_border"],dict_cell["y_tile_border"],dict_cell["length_max"]).astype(bool)
#     crop_cell_rgb = crop_around_cell(tile_rgb,dict_cell["x_tile_border"],dict_cell["y_tile_border"],length_max = dataset_config.crop_size) 

#     dict_cell = model_classification.classify_cells(crop_rgb=crop_cell_rgb,cell_mask=cell_mask, dict_cell=dict_cell)

#     if save_mask :
#         path_mask_cell = get_path_cell(slide_num,0, dict_cell["tile_row"], dict_cell["tile_col"], dict_cell["x_tile"], dict_cell["y_tile"], dataset_config)
        # crop_cell_pil = np_to_pil(crop_cell)
        # crop_cell_pil.save(path_mask_cell)


    

def segment_classify_cells_wsi(slide_num, dataset_config, verbose = 0):
    """
    Pour accelerer, je pourrai mettre le param de la 'opening a 3 pour avoir moins de composantes connexes dans le mask de la microglie 
    Pseudo code de la fonction unique qui fait le travaille de 
    - get_mask_microglia : une bouche sur chaque tile 
    - get_table_cells : une boucle pour chaque cellules 
    - post_process_table_cells_cluster : rien 
    - get_pred_model_classification(mode_prediction = "summed_proba") : une boucle sur chaque cellules 
    - get_pred_model_classification(mode_prediction = "max_proba") : une boucle sur chaque cellules 



    Ici, la fonction segment_classify_cells_on_slide(slide_num, model_segmentation = dataset_config.cell_segmentation_param["min_cell_size"], model_classification = model_classification, verbose = 0) a permis de calculer pour une slide entière : 

    - les masks binaires de chaque cellules. Ainsi que la classification des cellules (qui se trouve sur l'image et dans table_cells) 
    - Data cells qui donne pour chaque cellule (ID_cell	tile_row	tile_col	x_tile	y_tile	size	coord_x_min	coord_x_max	coord_y_min	coord_y_max	length_max	Decision_MODEL	proba_label_1_MODEL ...) 
    - Le dataframe computation_per_tile qui donne les temps de calculs dpour les différentes tiles et le pourcentage de tissu 


    """
    print(red("Segmentation and classification of microglial cells in the slide " +str(slide_num)))
   
    interval_tile_to_plot = 500
    create_gif = False
    pixel_per_tiles = 10
    threshold_tissu = 5
    fixed_thresh_size_bonification_cluster = 12000
    thresh_cluster_size_min = 4000
    model_classification = ModelClassification(dataset_config.classification_param)
    segmentation_model = ModelSegmentation(dataset_config.cell_segmentation_param)
    t = Time()
    t_compute_1_tile_tot = t.elapsed()
    t_read_region_tot = t.elapsed()
    t_asarray_region_tot = t.elapsed()
    t_compute_mask_tot = t.elapsed()
    t_compute_classification_tot = t.elapsed()
    t_label_mask_tot = t.elapsed()
    t_filter_tile_center_tot = t.elapsed()
    t_np_where_filter_cell_tot = t.elapsed()
    t_save_mask_tot = t.elapsed()

    ## Textee utile a dire 
    cell_label_names = ["Decision"] + dataset_config.cells_label_names

    """On s'interesse juste aux tiles avec tissu"""
    path_csv_tiles_with_tissu = ""

    """Time dataframe"""
    colnames_time_df = ["Compute_1_tile", "Crop_rgb_9_tiles","Convert_to_rgb","Segment_microglial_cells","Label_mask","Filter_tile_center","cells_filtered","nb_cells_tile","tile_row","tile_col","tissue_percent"]
    df_time = pd.DataFrame(columns = colnames_time_df)
    df_time.loc[0] = [t_compute_1_tile_tot.seconds,t_read_region_tot.seconds,t_asarray_region_tot.seconds,t_compute_mask_tot.seconds,t_label_mask_tot.seconds,t_filter_tile_center_tot.seconds,0,0,0, 0,0]

    """Pour traiter les cells individuellement"""
    colnames_idx = ['ID_cell','ID_tile',]
    colnames_table_cells = ['tile_row', 'tile_col', 'x_tile', "y_tile", "size","coord_x_min","coord_x_max","coord_y_min","coord_y_max"]
    colnames_condition_cluster = ["length_max","bonification_cluster_size_sup","penalite_cluster_size_inf"]
    colnames_pred = ["Decision"] + ["proba_"+f for f in dataset_config.cells_label_names]
    colname_id_big_clusters = ["ID_big_cluster"]
    colnames_table_cells_with_preds = colnames_idx+ colnames_table_cells +colnames_condition_cluster+ colnames_pred + colname_id_big_clusters
    table_cells = pd.DataFrame(columns = colnames_table_cells_with_preds)
    idx_cell = 0
    idx_tile = 1
    cells_filtered=0
    cells_classified = 0
    tile_with_tissu = 0

    """ gestion des tiles sans tissue"""

    path_filtered_image, path_mask_tissu = slide.get_filter_image_result(slide_num,dataset_config=dataset_config)
    mask_tissu_filtered = Image.open(path_mask_tissu)
    mask_tissu_filtered_np = np.asarray(mask_tissu_filtered)

    mask_tissu = np.copy(mask_tissu_filtered_np)
    mask_tissu = np.where(mask_tissu>0,1,0 )

    pcw = dataset_config.mapping_img_number[slide_num]
    name_path = "slide_"+str(slide_num)+"_"+str(pcw)+"_pcw_NEW"
    path_classified_slide = os.path.join(dataset_config.dir_classified_img,name_path)
    mkdir_if_nexist(dataset_config.dir_classified_img)
    mkdir_if_nexist(path_classified_slide)
    path_folder_slide_models = os.path.join(path_classified_slide,"segmentation_"+dataset_config.cell_segmentation_param["model_segmentation_type"]+"_min_cell_size_"+str(dataset_config.cell_segmentation_param["min_cell_size"]))
    mkdir_if_nexist(path_folder_slide_models)
    path_folder_slide_models_classif = os.path.join(path_folder_slide_models,"classification_"+dataset_config.classification_param["model_name"])
    mkdir_if_nexist(path_folder_slide_models_classif)

    path_folder_slide_mask_cells = os.path.join(path_folder_slide_models_classif,"Mask_cells")
    mkdir_if_nexist(path_folder_slide_mask_cells)

    path_slide = slide.get_training_slide_path(slide_num)
    s = slide.open_slide(path_slide) 
    tot_row_tiles = s.dimensions[1]//dataset_config.tile_height+1
    tot_col_tiles = s.dimensions[0]//dataset_config.tile_height+1 

    print("Total number of tiles of size (1024x1024) : ",tot_col_tiles*tot_row_tiles , " (= ",tot_col_tiles*tot_row_tiles*1024," pixels RGB)")

    """Pour réduire les calculs on filtre a un moment"""
    mask_tile_centre = np.zeros((dataset_config.tile_height*3,dataset_config.tile_width*3), dtype = np.uint8)
    mask_tile_centre[dataset_config.tile_height:2*dataset_config.tile_height,dataset_config.tile_width:2*dataset_config.tile_width] =1

    show = 0
    tile_center = 0
    idx_enorme_cluster = 0
    liste_tiles_slide = [(r,c) for r in range(1,tot_row_tiles) for c in range(1,tot_col_tiles)]
    # liste_tiles_slide = [(r,c) for r in range(16,19) for c in range(56,58)]

    for tile_row, tile_col in tqdm(liste_tiles_slide[2555:2565]):
        # print("idx_tile = ",idx_tile,"tile_row, tile_col = ",tile_row, tile_col)
        # if (tile_row, tile_col) != (17,57) and (tile_row, tile_col) != (18,58) :
        #     continue
        nb_cells_in_tile = 0
        #print("idx tile : ",idx_tile)
        t_compute_1_tile = Time()
        #print(tile_row, tile_col)
        if idx_tile%interval_tile_to_plot == 0:
            # print("idx_tile = ",idx_tile,"cells_filtered = ",cells_filtered,", idx_cell = ",idx_cell)
            path_mask_tissu_progression = os.path.join(path_folder_slide_models,"progression_computation")
            mkdir_if_nexist(path_mask_tissu_progression)
            path_mask_tissu_progression = os.path.join(path_mask_tissu_progression,str(idx_tile)+"_on_"+str(len(liste_tiles_slide))+".png" )
            f = display_mask(mask_tissu,title1="Progression",figsize = (30,20),vmax = 255,pathsave = path_mask_tissu_progression, cmap = CMAP_TISSU_SLIDE)
            path_csv = os.path.join(path_folder_slide_models_classif,"table_cells.csv")
            table_cells.to_csv(path_csv, sep = ";", index = False)
            del f
            df_time.to_csv(os.path.join(path_folder_slide_models_classif,"computation_time_per_tile.csv"), sep = ";", index = True)
            df_time_per_cells = pd.DataFrame({"Filter_1_cell": t_np_where_filter_cell_tot.seconds,"Compute_classification":t_compute_classification_tot.seconds,"Mask_saving":t_save_mask_tot.seconds,"cells_filtered" :cells_filtered, "cells_classified":cells_classified},index = ["Total"])
            df_time_per_cells.to_csv(os.path.join(path_folder_slide_models_classif,"computation_time_per_cells.csv"), sep = ";", index = True)
        if tile_row == 1 or tile_row==tot_row_tiles or tile_col==1 or tile_col==tot_col_tiles: #tile a un bord
            pourcentage_tissu = 0
            mask_tissu[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32] = 2 #tissu border
            mask_tissu[(tile_row-1)*32:(tile_row-1)*32+10,(tile_col-1)*32:(tile_col-1)*32+10] =3
        else : #tile du centre
            tile_center +=1
            mask_tile = mask_tissu_filtered_np[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32]
            #mask_tissu[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32] *=3
            mask_tissu[(tile_row-1)*32:(tile_row-1)*32+10,(tile_col-1)*32:(tile_col-1)*32+10] =3
            pourcentage_tissu = np.sum(mask_tile)*100/(32*32*255)
            if pourcentage_tissu > threshold_tissu : 
                #mask_tissu[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32] =4
                mask_tissu[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32] = np.where(mask_tissu[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32]==1, 4,mask_tissu[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32])
                tile_with_tissu +=1
                coord_origin_row = (tile_row-2)*dataset_config.tile_height
                coord_origin_col = (tile_col-2)*dataset_config.tile_width
                weight = 3*dataset_config.tile_width
                height = 3*dataset_config.tile_height

                t_read_region = Time()
                rgb = s.read_region((coord_origin_col,coord_origin_row),0,(weight,height))
                t_read_region_elapsed=t_read_region.elapsed()
                t_read_region_tot+=t_read_region_elapsed

                t_asarray_region = Time()
                rgb_9_tiles = np.asarray(rgb)
                rgb_9_tiles = rgb_9_tiles[:,:,:3]
                rgb_9_tiles = (rgb_9_tiles/255).astype(np.float32)

                #print("rgb_9_tiles dif values", np.unique(rgb_9_tiles) )
                """ #############################   ICI, j'a une image RGB de 9 tiles, je peux faire le traitement dessus ############################# """
                
                t_asarray_region_elapsed=t_asarray_region.elapsed()
                t_asarray_region_tot+=t_asarray_region_elapsed

                t_compute_mask = Time()
                #### Jusquea la : get_tile
                mask_cells_9_tiles = segmentation_model.segment_cells(rgb_9_tiles, verbose = verbose)
                t_compute_mask_elapsed = t_compute_mask.elapsed()
                t_compute_mask_tot+=t_compute_mask_elapsed
                #display_mask(mask_cells_9_tiles, figsize = (10,10))

                t_label_mask = Time()
                mask_cells_9_tiles_labels = sk_morphology.label(mask_cells_9_tiles)
                # display_mask(mask_cells_9_tiles_labels,title1= "labelisation cells dans les 9 tiles ("+str(tile_row)+"," +str(tile_col)+")", figsize = (10,10))
                t_label_mask_elapsed = t_label_mask.elapsed()

                t_label_mask_tot+= t_label_mask_elapsed


                #print("Il y a en tout dans les 9 tiles :", np.max(mask_cells_9_tiles_labels))
                t_filter_tile_center = Time()
                cells_centre_label = np.where(np.logical_and(mask_cells_9_tiles_labels,mask_tile_centre),mask_cells_9_tiles_labels,0)
                t_filter_tile_center_elapsed = t_filter_tile_center.elapsed()
                t_filter_tile_center_tot+=t_filter_tile_center_elapsed
                

                for label in np.unique(cells_centre_label)[1:]: #Pour chaque cell dont (x_tile,y_tile) est dans la tile 
                    # if show==0:
                    #     display_rgb(rgb_9_tiles, "rgb 9 tiles- ("+str(tile_row)+"," +str(tile_col)+")", figsize = (20,20))
                    #     display_mask(mask_tile,title1= "("+str(tile_row)+"," +str(tile_col)+")", figsize = (10,10))
                    #     display_mask(mask_cells_9_tiles,title1= "("+str(tile_row)+"," +str(tile_col)+")", figsize = (10,10))
                    #     show+=1

                    cells_filtered +=1
                    t_np_where_filter_cell = Time()
                    mask_cell_center = np.where(mask_cells_9_tiles_labels==label,1,0)
                    t_np_where_filter_cell_tot+=t_np_where_filter_cell.elapsed()
                    size = np.sum(mask_cell_center)
                    if size>dataset_config.cell_segmentation_param["min_cell_size"]:
                        #Si j filtre la taille a ce moment la alors il faut doubler la fonction V3 et supprimer le filtre car il met tellement de temps
                        xx,yy = np.where(mask_cell_center == 1)
                        x_tile_9_tiles = int(np.mean(xx))
                        y_tile_9_tiles = int(np.mean(yy))
                        x_tile,y_tile = int(x_tile_9_tiles-dataset_config.tile_height),int(y_tile_9_tiles-dataset_config.tile_width)
                        x_max = int(np.max(xx))
                        x_min = int(np.min(xx))
                        y_max = int(np.max(yy))
                        y_min = int(np.min(yy))
                        length_x = int(np.max(xx)-np.min(xx)+1)
                        length_y = int(np.max(yy)-np.min(yy)+1)
                        length_max_ = max(length_x,length_y)
                        cell_is_truncated_by_3_tiles = test_cell_truncated_by_3_tiles(x_min,x_max,y_min,y_max,n_tiles_border=3)

                        if (dataset_config.tile_height<=x_tile_9_tiles<2*dataset_config.tile_height and dataset_config.tile_width<=y_tile_9_tiles<2*dataset_config.tile_width) or cell_is_truncated_by_3_tiles : #Si la cellule a son centre dans la tile centrale
                            nb_cells_in_tile+=1
                            #print("Va ettre add car x_tile et y_tile dans 9 tiles : ", x_tile_9_tiles,y_tile_9_tiles)
                            #display_mask(mask_cell_center,title1="mask_cell_center de la cell "+str(idx_cell))

                            if cell_is_truncated_by_3_tiles: #Cas très très rare : la cellules touche un bord
                                idx_enorme_cluster +=1
                                # print(red("Cell is truncated by 3 tiles"))
                                # display_rgb(rgb_9_tiles, "rgb 9 tiles- ("+str(tile_row)+"," +str(tile_col)+")", figsize = (10,21))
                                # display_mask(mask_cells_9_tiles_labels,title1= "("+str(tile_row)+"," +str(tile_col)+")", figsize = (10,10))
                                cLUSTER_mask_in_tile = np.where(mask_tile_centre,mask_cell_center,0)
                                # display_mask(cLUSTER_mask_in_tile,title1= "cLUSTER_mask_in_tile de la cell "+str(idx_cell), figsize = (10,10))

                                cLUSTER_mask_in_tile_1_tile = cLUSTER_mask_in_tile[dataset_config.tile_height:-dataset_config.tile_height,dataset_config.tile_width:-dataset_config.tile_width]
                                # display_mask(cLUSTER_mask_in_tile_1_tile,title1= "cLUSTER_mask_in_tile_1_tile de la cell "+str(idx_cell), figsize = (10,10))
                                for crop_idx_row in range(4):
                                    for crop_idx_col in range(4):
                                        mask_crop = cLUSTER_mask_in_tile_1_tile[crop_idx_row*256:(crop_idx_row+1)*256,crop_idx_col*256:(crop_idx_col+1)*256]
                                        size = np.sum(mask_crop)
                                        if size>0: #J'ajoute ce crop comme new cellule appartenant à cluster 
                                            bonification_cluster_size_sup = True
                                            penalite_cluster_size_inf = False
                                            cells_classified +=1
                                            list_pred = [3,0,0,0,1,0,0]
                                            x_tile = crop_idx_row*dataset_config.crop_size+256//2
                                            y_tile = crop_idx_col*dataset_config.crop_size+256//2
                                            # display_mask(mask_crop,title1= "mask_crop de la cell "+str(idx_cell)+"_row_"+str(tile_row)+"_col_"+str(tile_col)+"_x_"+str(x_tile)+"_y_"+str(y_tile), figsize = (10,10))
                                            row_table_cells = [idx_cell,idx_tile, int(tile_row),int(tile_col),int(x_tile),int(y_tile),int(size),-1,-1,-1,-1,-1,bonification_cluster_size_sup,penalite_cluster_size_inf]+list_pred + [idx_enorme_cluster]
                                            cell_mask = 3*mask_crop

                                            path_crop = os.path.join(path_folder_slide_mask_cells,"cell_"+str(idx_cell)+"_row_"+str(tile_row)+"_col_"+str(tile_col)+"_x_"+str(x_tile)+"_y_"+str(y_tile)+".png")
                                            mask_to_save = (255-cell_mask)*(cell_mask>0)
                                            mask_to_save = mask_to_save.astype(np.uint8)
                                            
                                            mask_pil = np_to_pil(mask_to_save)
                                            mask_pil.save(path_crop) 
                                            table_cells.loc[idx_cell] = row_table_cells
                                            idx_cell+=1

                            else :
                                """ Sauvegarde du mask de la cellule """
                                if np.mod(length_max_,2) != 0:
                                    length_max_+=1
                                crop_shape = max(length_max_,dataset_config.crop_size) #Comme ca dans tous les cas c'est pair 
                                if length_max_ > dataset_config.crop_size:
                                    #J'avais un problème ici car avec les trop grandes composantes connexes (ce qui ne doit pas arriver normalement) la taille du truc faisait que je sortais du cadre) des 9 tiles 
                                    crop_mask = mask_cell_center[max(0,x_tile_9_tiles-crop_shape):min(mask_cell_center.shape[0],x_tile_9_tiles+crop_shape),max(0,y_tile_9_tiles-crop_shape):min(mask_cell_center.shape[1],y_tile_9_tiles+crop_shape)].astype(np.uint8)
                                else :
                                    crop_mask = mask_cell_center[x_tile_9_tiles-int(crop_shape/2):x_tile_9_tiles+int(crop_shape/2),y_tile_9_tiles-int(crop_shape/2):y_tile_9_tiles+int(crop_shape/2)].astype(np.uint8)
                                crop_shape=crop_mask.shape[0]
                                
                                crop_rgb_256 = rgb_9_tiles[x_tile_9_tiles-int(dataset_config.crop_size/2):x_tile_9_tiles+int(dataset_config.crop_size/2),y_tile_9_tiles-int(dataset_config.crop_size/2):y_tile_9_tiles+int(dataset_config.crop_size/2),:]
                                bonification_cluster_size_sup = (size > fixed_thresh_size_bonification_cluster)
                                penalite_cluster_size_inf= (thresh_cluster_size_min < size)

                                cells_classified +=1
                                t_compute_classification = Time()

                                cell_class_or_proba = model_classification.classify_cells(crop_rgb_256, mask_cell = np.copy(crop_mask), mode_prediction = "summed_proba", bonification_cluster_size_sup = bonification_cluster_size_sup, penalite_cluster_size_inf= penalite_cluster_size_inf,verbose = verbose) # Binary mask prediction of the crop (eventually several cells are on the crop) of dim [256x256x(decision+nb_labels)]
                                list_pred = []

                                for state_idx,state in enumerate(cell_label_names):
                                    if state_idx==0:
                                        list_pred.append(int(np.max(cell_class_or_proba[:,:,state_idx])))
                                    else : 
                                        list_pred.append(np.max(cell_class_or_proba[:,:,state_idx]))
                                
                                row_table_cells = [idx_cell,idx_tile, int(tile_row),int(tile_col),int(x_tile),int(y_tile),int(size),int(x_min),int(x_max),int(y_min),int(y_max),int(length_max_),bonification_cluster_size_sup,penalite_cluster_size_inf]+list_pred + [0]
                                cell_mask = list_pred[0]*crop_mask
                                t_compute_classification_tot += t_compute_classification.elapsed()
                                t_save_mask = Time()
                                path_crop = os.path.join(path_folder_slide_mask_cells,"cell_"+str(idx_cell)+"_row_"+str(tile_row)+"_col_"+str(tile_col)+"_x_"+str(x_tile)+"_y_"+str(y_tile)+".png")
                                #print("np.unique(cell_mask)",np.unique(cell_mask))
                                mask_to_save = (255-cell_mask)*(cell_mask>0)
                                #print("np.unique(mask_to_save)",np.unique(mask_to_save))
                                mask_pil = np_to_pil(mask_to_save)
                                #print(path_crop)
                                mask_pil.save(path_crop) 
                                t_save_mask_tot+=t_save_mask.elapsed()
                                table_cells.loc[idx_cell] = row_table_cells
                                idx_cell+=1
            
        #print("Go fin de boucle")
        #print("idx_tile : ",idx_tile," t_compµute_1_tile_tot AVANT AJOUT DE ",t_compute_1_tile_tot)
        t_compute_1_tile_elapsed = t_compute_1_tile.elapsed()
        t_compute_1_tile_tot +=t_compute_1_tile_elapsed
        #print("idx_tile : ", idx_tile ,"t_compute_1_tile_elapsed : ",t_compute_1_tile_elapsed," t_compute_1_tile_tot ",t_compute_1_tile_tot)

        #print("idx_tile : ",idx_tile," nb cells ", nb_cells_in_tile, " idx_tile_center",tile_center )
        #print("t_compute_1_tile.elapsed() : ",t_compute_1_tile_elapsed)
        if pourcentage_tissu > threshold_tissu : 
            df_time.loc[idx_tile] = [t_compute_1_tile_elapsed.seconds,t_read_region_elapsed.seconds,t_asarray_region_elapsed.microseconds,t_compute_mask_elapsed.seconds,t_label_mask_elapsed.microseconds,t_filter_tile_center_elapsed.microseconds,cells_filtered,nb_cells_in_tile,tile_row, tile_col,pourcentage_tissu]
        else : 
            df_time.loc[idx_tile] = [t_compute_1_tile_elapsed.seconds,0,0,0,0,0,cells_filtered,0,tile_row, tile_col,pourcentage_tissu]

        #print("ENTRE idx_tile : ", idx_tile ,"t_compute_1_tile_elapsed : ",t_compute_1_tile_elapsed," t_compute_1_tile_tot ",t_compute_1_tile_tot )

        ##df_time.loc[1000+idx_tile] = [t_compute_1_tile_tot.seconds,t_read_region_tot.seconds,t_asarray_region_tot.microseconds,t_compute_mask_tot.seconds,t_label_mask_tot.microseconds,t_filter_tile_center_tot.microseconds,cells_filtered,idx_cell,(tile_row, tile_col)]
        idx_tile+=1
            
    df_time.loc["Total"] = [t_compute_1_tile_tot.seconds,t_read_region_tot.seconds,t_asarray_region_tot.seconds,t_compute_mask_tot.seconds,t_label_mask_tot.seconds,t_filter_tile_center_tot.seconds,cells_filtered,idx_cell,-1,-1,200]

    print("%-20s | Time: %-14s " % ("TOT compute 1 tile", str(t_compute_1_tile_tot)))
    print("%-20s | Time: %-14s " % ("TOT crop rgb", str(t_read_region_tot)))
    print("%-20s | Time: %-14s " % ("TOT Convert to rgb", str(t_asarray_region_tot)))
    print("%-20s | Time: %-14s " % ("TOT create mask microglia rgb", str(t_compute_mask_tot)))
    print("%-20s | Time: %-14s " % ("TOT ccompute classification", str(t_compute_classification_tot)))

    """ Partie création GIF """
    # img_tiles[pixel_per_tiles*(tile_row-1-1):pixel_per_tiles*(tile_row+2-1),pixel_per_tiles*(tile_col-1-1):pixel_per_tiles*(tile_col+2-1)] -=20
    # img_tiles[pixel_per_tiles*(tile_row-1):pixel_per_tiles*(tile_row-1)+pixel_per_tiles,pixel_per_tiles*(tile_col-1):pixel_per_tiles*(tile_col-1)+pixel_per_tiles] -=50
    # im = Image.fromarray(np.uint8(cm.gist_earth(img_tiles)*255))
    # images_gif.append(im)

    table_cells[colnames_table_cells] = table_cells[colnames_table_cells].astype(int)
    path_csv = os.path.join(path_folder_slide_models_classif,"table_cells.csv")
    table_cells.to_csv(path_csv, sep = ";", index = False)

    df_time.to_csv(os.path.join(path_folder_slide_models_classif,"computation_time_per_tile.csv"), sep = ";", index = True)

    df_time_per_cells = pd.DataFrame({"Filter_1_cell": t_np_where_filter_cell_tot.seconds,"Compute_classification":t_compute_classification_tot.seconds,"Mask_saving":t_save_mask_tot.seconds,"cells_filtered" :cells_filtered, "cells_classified":cells_classified},index = ["Total"])
    df_time_per_cells.to_csv(os.path.join(path_folder_slide_models_classif,"computation_time_per_cells.csv"), sep = ";", index = True)
    
def classify_microglial_cells(dataset_config, model, crop_rgb, cell_mask , dict_cell, Model_segmentation = None  ,mode_prediction="summed_proba", bonification_cluster_size_sup = False, penalite_cluster_size_inf= False,verbose = 0):
    """
    Only for the cell present on mask -> compute model prediction and probabilities for each pixel of the cell to belong to state k 
    Input :
        - rgb image (256x256)
        - mask image (cell_size, cell_size)
        - classification model 
        - mode_prediction : one among [max_proba, summed_proba] 
    Output : 
        - enrich dict_cell with 
            -bonification_cluster_size_sup
            -penalite_cluster_size_inf
            -decision
            -proba_<class_k>
    """
    fixed_thresh_size_bonification_cluster = 12000
    thresh_cluster_size_min = 4000

    mode_prediction = dataset_config.classification_param["mode_prediction"]
    n_class = dataset_config.classification_param["n_class"]
    crop_size = dataset_config.crop_size

    dict_cell["bonification_cluster_size_sup"] = (dict_cell["size"] > fixed_thresh_size_bonification_cluster)
    dict_cell["penalite_cluster_size_inf"] = (thresh_cluster_size_min < dict_cell["size"])

    crop_probability_maps = model.predict(np.expand_dims(crop_rgb, axis = 0),verbose=0)
    crop_probability_maps = np.squeeze(crop_probability_maps)    #Remove single-dimensional entries from the shape of an array.
    
    # mask_prediction = np.zeros((mask_cell.shape[0],mask_cell.shape[1]))
    # probability_maps = np.zeros((mask_cell.shape[0],mask_cell.shape[1],n_class))

    crop_mask = cell_mask[int(cell_mask.shape[0]/2-crop_size/2):int(cell_mask.shape[0]/2+crop_size/2),int(cell_mask.shape[1]/2-crop_size/2):int(cell_mask.shape[1]/2+crop_size/2)]
    size_cell_crop = np.sum(crop_mask)

    probability_map_background = crop_mask*crop_probability_maps[:,:,0]
    probability_map_detected_cells = crop_mask*crop_probability_maps[:,:,-1]

    sum_background = np.sum(probability_map_background)
    sum_detected_cells = np.sum(probability_map_detected_cells)

    reallocation = (sum_background + sum_detected_cells)/(n_class-2)
    list_proba_classes = []
    for class_k, class_name in enumerate(dataset_config.cell_class_names_for_classification): 
        cell_probability_map_class_k = crop_mask*crop_probability_maps[:,:,class_k]  ###np.where(mask_composante_connexe == 1, probability_maps[:,:,label], 0)
        if mode_prediction == "max_proba":
            proba_class_k = np.max(cell_probability_map_class_k) #Le max de la proba sur la cellule 
        elif mode_prediction == "summed_proba":
            if class_k == 0 : #background
                proba_background = sum_background/size_cell_crop
                proba_class_k = sum_background/size_cell_crop
                if proba_background == 1 :
                    proba_background = 0 
                    proba_class_k = 0
            elif class_k ==3: # Label aggregated 
                if dict_cell["bonification_cluster_size_sup"] : 
                    proba_class_k = 1 
                else : 
                    proba_class_k = (np.sum(cell_probability_map_class_k)+reallocation)/size_cell_crop
            else : 
                if dict_cell["bonification_cluster_size_sup"] : 
                    proba_class_k = 0
                else : 
                    proba_class_k = (np.sum(cell_probability_map_class_k)+reallocation)/size_cell_crop
        else : 
            raise ValueError("mode_prediction should be max proba or summed_proba")
        dict_cell["proba_"+class_name] = proba_class_k
        list_proba_classes.append(proba_class_k)
        
        # probability_maps[:,:,class_k+1] +=mask_cell*proba_class_k  
    
    decision = np.argmax(list_proba_classes[1:-1])+1# [1:] to ex_tilelude background  and detected
    dict_cell["cell_type"] = decision
    # mask_prediction = mask_cell*decision  

    return dict_cell

def classify_cells_microglia_ihc(crop_rgb, model, mask_cell = None ,dict_cell=None, Model_segmentation = None  ,mode_prediction="summed_proba", bonification_cluster_size_sup = False, penalite_cluster_size_inf= False,verbose = 0, display = False, dataset_config=None):
    """
    Only for the cell present on mask -> compute model prediction and probabilities for each pixel of the cell to belong to state k 
    Input :
        - rgb image (256x256)
        - mask image (cell_size, cell_size)
        - classification model 
        - mode_prediction : one among [max_proba, summed_proba] 
    Output : 
        - cell_state_and_proba (cell_size,cell_size,n_states+1) # last dim is [decision, proba_background, proba_c1, ..., proba_cN]
    """
    DISPLAY_FIG = False
    check_proba = False 
    mode_prediction = dataset_config.classification_param["mode_prediction"]

    probability_maps = model.predict(np.expand_dims(crop_rgb, axis = 0),verbose=0)
    probability_maps = np.squeeze(probability_maps)    #Remove single-dimensional entries from the shape of an array.
    nb_states = probability_maps.shape[-1] #Bachground, C1,..,CN, Detected -> N+2 

    if mask_cell is not None : 
        # mask_cell = np.where(mask_cell>0,1,0)
        prediction_finale_and_probability = np.zeros((mask_cell.shape[0],mask_cell.shape[1],nb_states+1))
        list_proba_states = []
        mask_size = mask_cell.shape[0]
        mask_crop_256 = mask_cell[int(mask_size/2-dataset_config.crop_size/2):int(mask_size/2+dataset_config.crop_size/2),int(mask_size/2-dataset_config.crop_size/2):int(mask_size/2+dataset_config.crop_size/2)]
        cell_size_on_256 = np.sum(mask_crop_256)
        probability_map_background = mask_crop_256*probability_maps[:,:,0]
        probability_map_detected_cells = mask_crop_256*probability_maps[:,:,-1]
        sum_background = np.sum(probability_map_background)
        sum_detected_cells = np.sum(probability_map_detected_cells)
        reallocation = (sum_background + sum_detected_cells)/(nb_states-2)
        for label in range(nb_states): # Background C1, ..., CN 
            probability_map_label = mask_crop_256*probability_maps[:,:,label]  ###np.where(mask_composante_connexe == 1, probability_maps[:,:,label], 0)
            if mode_prediction == "max_proba":
                proba_class_k = np.max(probability_map_label) #Le max de la proba sur la cellule 
            if mode_prediction == "summed_proba":
                if label == 0 : 
                    proba_background = sum_background/cell_size_on_256
                    proba_class_k = sum_background/cell_size_on_256
                    if proba_background == 1 :
                        proba_background = 0 
                        proba_class_k = 0
                elif label ==3: # Label cluster 
                    if bonification_cluster_size_sup : 
                        proba_class_k = 1 
                    else : 
                        proba_class_k = (np.sum(probability_map_label)+reallocation)/cell_size_on_256
                else : 
                    if bonification_cluster_size_sup : 
                        proba_class_k = 0
                    else : 
                        proba_class_k = (np.sum(probability_map_label)+reallocation)/cell_size_on_256
            mask_state_k = mask_cell*proba_class_k  
            list_proba_states.append(proba_class_k)

            prediction_finale_and_probability[:,:,label+1] +=mask_state_k
        class_more_probable = np.argmax(list_proba_states[1:-1])+1# [1:] to ex_tilelude background  and detected 
        mask_pred_decision = np.where(mask_cell == 1, class_more_probable, 0)
        prediction_finale_and_probability[:,:,0] = mask_pred_decision

    return prediction_finale_and_probability


def see_available_models(training_set_config_name=None):
    if training_set_config_name is None:
        list_training_set = os.listdir(dataset_config.dir_models)
        list_training_set = [x for x in list_training_set if x!=".DS_Store"]
        for training_set_name in list_training_set:
            print("training_set_config_name : ",training_set_name)
    else :
        path_models = os.path.join(dataset_config.dir_models,training_set_config_name)
        info_models_path = os.path.join(path_models,"info_models.csv")
        df_models_comparaison = pd.read_csv(info_models_path,sep = ";")
        list_models = df_models_comparaison["model_name"].tolist()
        for i in range(len(list_models)):
            print(blue("Model n°"+str(i)+" : " + list_models[i]))
            print("global_f1",df_models_comparaison["global_f1"][i])



def crop_around_cell(img,x,y,crop_larger):
    """
    Crop a cell around its center of mass 
    """
    if len(img.shape) == 2 : #mask
        pad = int(crop_larger/2)
        img_padded = np.pad(img, ((pad,pad),(pad,pad)),"constant" ) #0 padding
        x+=pad
        y+=pad
        crop_cell = img_padded[x-int(crop_larger/2):x+int(crop_larger/2),y-int(crop_larger/2):y+int(crop_larger/2)]
    elif len(img.shape) == 3 : #rgb
        pad = int(crop_larger/2)
        img_padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)),"constant" ) #0 padding
        x+=pad
        y+=pad
        crop_cell = img_padded[x-int(crop_larger/2):x+int(crop_larger/2),y-int(crop_larger/2):y+int(crop_larger/2),:]
    return crop_cell

def check_already_classified_tiles(dataset_config,slide_num):
    """
    Check if tiles have already been classified
    Save all info about classified tiles in tiles_and_tissue_percentage.csv
    """
    path_df_classified_cells = os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(slide_num).zfill(3)),"slide_{}".format(str(slide_num).zfill(3))+"_cells.csv")

    if os.path.exists(path_df_classified_cells) : 
        print("tiles_and_tissue_percentage already exist")
        some_tiles_already_processed = True 
        df_cells = pd.read_csv(path_df_classified_cells,sep = ";")
        sub_df = df_cells[["tile_row","tile_col"]]
        # print("sub_df",sub_df)
        sub_df = sub_df.drop_duplicates()
        list_tiles_already_processed = sub_df[["tile_row","tile_col"]].values.tolist()
        print("Final list : ",list_tiles_already_processed)
        # list_tiles_already_processed = list(set(list_tiles_already_processed))
        return some_tiles_already_processed, list_tiles_already_processed
    else : 
        some_tiles_already_processed = False
        list_tiles_already_processed = None 
        return some_tiles_already_processed, list_tiles_already_processed


def get_tissue_percentage_tiles(slide_num,dataset_config, particular_roi=None,nuclei_segmentation_usage=False):
    """
    Get tiles (row, col) where there is enough tissue 

    Save df with info on tissu percent. If tile has been classified already, skip it because cells are already here
    """ 
    path_tissue_percentage_tiles = os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(slide_num).zfill(3)),"tiles_and_tissue_percentage.csv")
    if os.path.exists(path_tissue_percentage_tiles):
        print("path_tissue_percentage_tiles already exist")
        tissue_percentage_tiles = pd.read_csv(path_tissue_percentage_tiles,sep = ";")
        return tissue_percentage_tiles
    
    else : 
        tissue_percentage_tiles = pd.DataFrame(columns = ["tile_row","tile_col","categorie","tissue_percentage"])
       
        downscaled_tile_size = int(dataset_config.tile_width/dataset_config.preprocessing_config.scale_factor)
        n_tiles_row_slide,n_tiles_col_slide = slide.get_n_row_col_img(slide_num,dataset_config)
        if dataset_config.consider_image_with_channels:
            _, path_mask_tissu  = slide.get_filter_image_result(slide_num,thumbnail = False,channel_number = -1,dataset_config=dataset_config)
        else : 
            _, path_mask_tissu  = slide.get_filter_image_result(slide_num,thumbnail = False,channel_number = None,dataset_config=dataset_config)
        mask_tissu_filtered = Image.open(path_mask_tissu)
        mask_tissu_filtered_np = np.asarray(mask_tissu_filtered)
        mask_tissu = np.where(mask_tissu_filtered_np>0,1,0 )

        list_tiles = [(r,c) for r in range(1,n_tiles_row_slide+1) for c in range(1,n_tiles_col_slide+1)]
        for row, col in list_tiles :
            # print("[get_idx_tiles_with_enough_tissue] row, col -> ", row, col)
            tile_tissu = mask_tissu[(row-1)*downscaled_tile_size:row*downscaled_tile_size,(col-1)*downscaled_tile_size:col*downscaled_tile_size]
            pad_height = downscaled_tile_size-tile_tissu.shape[0]
            pad_width = downscaled_tile_size-tile_tissu.shape[1]
            tile_tissu_padded = np.pad(tile_tissu, ((0,pad_height),(0,pad_width)),"constant" ) #0 padding
            tissue_part = np.sum(tile_tissu_padded)*100/(tile_tissu_padded.shape[0]*tile_tissu_padded.shape[1])
            if row == 1 : 
                if col == 1 : 
                    categorie = "top-left"
                elif col == n_tiles_col_slide : 
                    categorie = "top-right"
                else : 
                    categorie = "top"
            elif row == n_tiles_row_slide : 
                if col == 1 : 
                    categorie = "bottom-left"
                elif col == n_tiles_col_slide : 
                    categorie = "bottom-right"
                else : 
                    categorie = "bottom"
            elif col == 1 : 
                categorie = "left"
            elif col == n_tiles_col_slide :
                categorie = "right"
            else : 
                categorie = "center"
            #Add the tile to df with tissue percent
            tissue_percentage_tiles.loc[len(tissue_percentage_tiles)] = [row,col,categorie,tissue_part]

    if nuclei_segmentation_usage : 
        dir = os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(slide_num).zfill(3))+"_nuclei")
    else : 
        dir = os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(slide_num).zfill(3)))
    mkdir_if_nexist(dir)
    tissue_percentage_tiles.to_csv(os.path.join(dir,"tiles_and_tissue_percentage.csv"),sep = ";",index = False)
    return tissue_percentage_tiles 

def add_cell_coord_info(dict_cell,mask_cell,dataset_config): 
    """
    Get info about cell position in the image and add it to dict_cell
    """
    row, col = dict_cell["tile_row"], dict_cell["tile_col"]
    border_size_during_segmentation = dataset_config.border_size_during_segmentation
    xx,yy = np.where(mask_cell)
    # dict_cell["tile_width"] = dataset_config.tile_width
    # dict_cell["tile_height"] = dataset_config.tile_height
    dict_cell["size"] = np.sum(mask_cell)
    dict_cell["x_tile_border"] = int(np.mean(xx))
    dict_cell["y_tile_border"] = int(np.mean(yy))
    dict_cell["x_tile"] = int(dict_cell["x_tile_border"]-border_size_during_segmentation)
    dict_cell["y_tile"] = int(dict_cell["y_tile_border"]-border_size_during_segmentation)
    dict_cell["x_img"] = int((row-1)*dataset_config.tile_height + dict_cell["x_tile"])
    dict_cell["y_img"] = int((col-1)*dataset_config.tile_width + dict_cell["y_tile"])
    # dict_cell["x_max"] = int(np.max(xx))
    # dict_cell["x_min"] = int(np.min(xx))
    # dict_cell["y_max"] = int(np.max(yy))
    # dict_cell["y_min"] = int(np.min(yy))
    length_x = int(np.max(xx)-np.min(xx)+1)
    length_y = int(np.max(yy)-np.min(yy)+1)
    length_max = max(length_x,length_y)
    length_max+=1 if np.mod(length_max,2) != 0 else length_max
    dict_cell["length_max"] = length_max
    check_out_of_borders = check_if_out_of_borders(dataset_config, int(np.min(xx)), int(np.max(xx)),int(np.min(yy)),int(np.max(yy)),(border_size_during_segmentation,border_size_during_segmentation))
    dict_cell["check_out_of_borders"] = check_out_of_borders
    dict_cell["check_in_centered_tile"] = 0 <= dict_cell["x_tile"] and dict_cell["x_tile"] < dataset_config.tile_height and 0 <= dict_cell["y_tile"] and dict_cell["y_tile"] < dataset_config.tile_width

    return dict_cell

def check_if_out_of_borders(dataset_config, x_min,x_max,y_min,y_max,borders):
    """
    Check if a cell is outside the borders of the tile 
    """
    height_border, width_border = borders
    if x_min ==0 or x_max == 2*height_border+dataset_config.tile_height : 
        return True 
    elif y_min == 0 or y_max == 2*width_border+dataset_config.tile_width :
        return True 
    else : 
        return False

def get_path_cell(slide_num,channel_number, row, col,x_tile,y_tile, dataset_config):
    """
    Get cell path
    """
    if dataset_config.data_type == "fluorescence":
        path_dir_cells = os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(slide_num).zfill(3)))
        path_mask_cell = os.path.join(path_dir_cells, "C_{}_r_{}_c_{}_x_{}_y_{}_cell_mask.png".format(channel_number,row,col,x_tile,y_tile))
    else : 
        path_dir_cells = os.path.join(dataset_config.dir_classified_img, "slide_{}".format(str(slide_num).zfill(3)))
        path_mask_cell = os.path.join(path_dir_cells, "r_{}_c_{}_x_{}_y_{}_cell_mask.png".format(row,col,x_tile,y_tile))
    os.makedirs(path_dir_cells, exist_ok=True)
    return path_mask_cell

def find_good_cell_segmentation_param(slide_num,dataset_config):
    if dataset_config.data_type == "fluorescence":
        if "image_"+str(slide_num) in list(dataset_config.cell_segmentation_param_by_cannnel.keys()):
            # print("Using specific segmentor for slide {}".format(slide_num))
            cell_segmentation_param = dataset_config.cell_segmentation_param_by_cannnel["image_"+str(slide_num)]
        else : 
            # print("Default segmentor for slide {}".format(slide_num))
            cell_segmentation_param = dataset_config.cell_segmentation_param_by_cannnel["default"]
    else :
        if "image_"+str(slide_num) in list(dataset_config.cell_segmentation_param.keys()):
            # print("Using specific segmentor for slide {}".format(slide_num))
            cell_segmentation_param = dataset_config.cell_segmentation_param["image_"+str(slide_num)]
        else : 
            # print("Default segmentor for slide {}".format(slide_num))
            cell_segmentation_param = dataset_config.cell_segmentation_param["default"]


    return cell_segmentation_param

def get_cells_fluorescence_from_tile(slide_num, row, col, tile_cat,dataset_config,id_cell_max,channels_cells_to_segment=None):
    """
    RQ : "coord_x_min","coord_x_max","coord_y_min","coord_y_max"used to compute length max 
    Get cells (mask + info) whose center of mass in inside the tile 

    3 cases : 
        - cell body is completly inside the tile(row,col)
        - cell body is inside (border+tile+border)
        - cell body is outside 
            2 cases : 
            -> center of mass is inside tile(row,col)
            -> center of mass is outside (border+tile+border) because large cell : calculs are done but not optimised
    """
    save_mask = True
    list_cells = []
    tile_dict = tiles.get_tile(dataset_config, slide_num, row, col, (dataset_config.border_size_during_segmentation,dataset_config.border_size_during_segmentation),channels_of_interest = channels_cells_to_segment)
    cell_segmentation_param = find_good_cell_segmentation_param(slide_num,dataset_config)
    segmentor = segmentation.ModelSegmentation(dataset_config=dataset_config,object_to_segment="cells",cell_segmentation_param=cell_segmentation_param)
    id_cell = 0
    for channel_number in tile_dict.keys():#Channels of interest already filtered in get_tile_czi
        segmented_cells_in_center = segmentor.segment_cells(tile_dict[channel_number],channel_number=channel_number)
        
        segmented_cells_in_center_label = sk_morphology.label(segmented_cells_in_center)
        for cell in np.unique(segmented_cells_in_center_label)[1:]: #0 == background 
            mask_cell = np.where(segmented_cells_in_center_label==cell,1,0).astype(bool)
            dict_cell = {
                # "slide_num" : slide_num,
                "channel_number" : channel_number,
                "cell_type" : channel_number, 
                "tile_row" : row,
                "tile_col" : col,
                "tile_cat" : tile_cat,
            }
            dict_cell = add_cell_coord_info(dict_cell,mask_cell,dataset_config) 
            if dict_cell["size"] > 50000 : #Check temporaire 
                display_mask(mask_cell, title='size > 50000')
                print("check_out_of_borders",dict_cell["check_out_of_borders"])
            if dict_cell["check_out_of_borders"] :  #Check temporaire 
                
                display_mask(mask_cell, title='Cell larger than borders')
            if not dict_cell["check_in_centered_tile"] :
                # print("Cell outside center tile but should have been excluded earlier ")
                continue
            id_cell += 1
            dict_cell["id_cell"] = id_cell_max + id_cell
            crop_cell = crop_around_cell(mask_cell,dict_cell["x_tile_border"],dict_cell["y_tile_border"],max(dict_cell["length_max"],256)) 
            # cell_mask = crop_around_cell(tile_mask_cell,dict_cell["x_tile_border"],dict_cell["y_tile_border"],max(dict_cell["length_max"],256)).astype(bool)

            if save_mask : 
                path_mask_cell = get_path_cell(slide_num,channel_number, row, col,dict_cell["x_tile"],dict_cell["y_tile"], dataset_config)
                crop_cell_pil = np_to_pil(crop_cell)
                crop_cell_pil.save(path_mask_cell)

            list_cells.append(dict_cell)
    return list_cells


def segment_classify_cells(slide_num, dataset_config,channels_cells_to_segment=None, particular_roi = None): 
    """
    Return dataframe with cells info 
     - In particular_roi if defined and other already classified tiles  
     - In the whole image otherwise 
    """
    threshold_tissue = dataset_config.threshold_tissue
    verbose = False
    path_table_cells = get_path_table_cells(slide_num,dataset_config,particular_roi)
    # print("segment_classify_cells", dataset_config.cell_segmentation_param_by_cannnel)
    # table_cells = pd.DataFrame(columns = dataset_config.colnames_table_cells_base)
    list_all_cells = []
    tissue_percentage_tiles = get_tissue_percentage_tiles(slide_num,dataset_config, particular_roi = particular_roi)
    n_tiles_with_tissue = len(tissue_percentage_tiles[tissue_percentage_tiles["tissue_percentage"]>threshold_tissue])
    print(blue("Total tiles with tissue :" ),n_tiles_with_tissue)
    id_cell_max = 0

    if particular_roi is not None :
        n_tiles_border_to_cover_roi_border_size = int(dataset_config.roi_border_size/dataset_config.tile_width)+1
        list_tiles_roi = tiles.get_tile_list_from_ROI(particular_roi[0]-n_tiles_border_to_cover_roi_border_size,particular_roi[1]-n_tiles_border_to_cover_roi_border_size,particular_roi[2]+n_tiles_border_to_cover_roi_border_size,particular_roi[3]+n_tiles_border_to_cover_roi_border_size)

    some_tiles_already_processed, list_tiles_already_processed = check_already_classified_tiles(dataset_config,slide_num)
    processed_tiles = 0 
    for tile_info in tissue_percentage_tiles.iterrows():
        row,col,categorie,tissue_part = tile_info[1]
        if tissue_part < (100-threshold_tissue) : 
            continue 
        tile_coord = [row,col]
        if particular_roi is not None :
            if tile_coord not in list_tiles_roi : 
                continue
        if some_tiles_already_processed : 
            if tile_coord in list_tiles_already_processed : 
                continue
        print(yellow("Segment and classify cells on tile {} ".format(tile_coord))) if verbose else None
        # if tile_coord != (17,37):
        #     continue
        # ID_CURRENT+=1
        if dataset_config.data_type == "fluorescence":
            list_cells_tile = get_cells_fluorescence_from_tile(slide_num, tile_coord[0], tile_coord[1], categorie, dataset_config,id_cell_max,channels_cells_to_segment=channels_cells_to_segment)
        elif dataset_config.data_type == "wsi":
            # table_cells_temp = segment_classify_cells_wsi_from_tile(slide_num, tile_coord, tile_cat, dataset_config)
            list_cells_tile = segment_classify_cells_wsi_from_tile(slide_num,tile_coord[0], tile_coord[1], categorie,dataset_config,id_cell_max,channels_cells_to_segment=None)
        else : 
            raise NotImplementedError("data_type not implemented")
        id_cell_max += len(list_cells_tile)
        list_all_cells+= list_cells_tile
        # table_cells = pd.concat([table_cells,table_cells_temp],axis = 0, ignore_index=True) 
        # if ID_CURRENT == N_MAX:
        #     break
        if processed_tiles%20 == 0 : 
            print("processed_tiles",processed_tiles)
            table_cells = pd.DataFrame(list_all_cells, columns = dataset_config.colnames_table_cells_base)
            if os.path.exists(path_table_cells):
                table_cells.to_csv(path_table_cells,mode='a',header=False,index=False,sep = ";")
            else :
                table_cells.to_csv(path_table_cells,index=False,sep = ";")
            list_all_cells = []
        processed_tiles+=1
    
    # print("path_table_cells",path_table_cells)
    table_cells = pd.DataFrame(list_all_cells, columns = dataset_config.colnames_table_cells_base)
    if os.path.exists(path_table_cells):
        table_cells.to_csv(path_table_cells,mode='a',header=False,index=False,sep = ";")
    else :
        table_cells.to_csv(path_table_cells,index=False,sep = ";")
    table_cells = pd.read_csv(path_table_cells,sep = ";")
    print(blue("Size of table_cells : {} in slide {}".format(table_cells.shape,slide_num ))) if verbose else None 
    return table_cells


def filter_table_cells_around_tile(table_cells,row_of_interest, col_of_interest, dataset_config,channels_list):
    """Filter table_slide with only the tiles of the roi"""
    border_size_during_segmentation = dataset_config.border_size_during_segmentation
    x_min = (row_of_interest-1)*dataset_config.tile_height-border_size_during_segmentation
    x_max = row_of_interest*dataset_config.tile_height+border_size_during_segmentation
    y_min = (col_of_interest-1)*dataset_config.tile_width - border_size_during_segmentation
    y_max = col_of_interest*dataset_config.tile_width + border_size_during_segmentation

    table_cells_in_tile_w_border = table_cells[(table_cells['x_img'] >= x_min) & (table_cells['x_img'] < x_max) & (table_cells['y_img'] >= y_min) & (table_cells['y_img'] < y_max)]

    # n_tiles_border_to_cover_roi_border_size = int(dataset_config.roi_border_size/dataset_config.tile_width)+1
    # row_origin, col_origin, row_end, col_end = row-n_tiles_border_to_cover_roi_border_size,col-n_tiles_border_to_cover_roi_border_size,row+n_tiles_border_to_cover_roi_border_size,col+n_tiles_border_to_cover_roi_border_size
    # list_tiles = [(r,c) for r in range(row_origin,row_end+1) for c in range(col_origin,col_end+1)]
    table_cells_in_tile_w_border = table_cells_in_tile_w_border[table_cells_in_tile_w_border["channel_number"].isin(channels_list)]

    return table_cells_in_tile_w_border

def _add_mask_cell_to_empty_mask(mask_tile_w_borders, mask_cell, row_of_interest, col_of_interest, row, col, x_tile, y_tile, length_max,dataset_config):
    """
    Add the mask of the cell in the mask 

    All commented code can be used to see what appens 
    """
    length_max = max(256,length_max)
    # print("length_max",length_max)
    border_size = dataset_config.border_size_during_segmentation
    tile_height = dataset_config.tile_height
    tile_width = dataset_config.tile_width

    # x_in_mask_tile = tile_height+border_size +(row-row_of_interest)*tile_height + x_tile 
    # y_in_mask_tile = border_size + (col-col_of_interest)*tile_width + y_tile
    x_in_mask_tile = border_size + (row-row_of_interest)*tile_height + x_tile 
    y_in_mask_tile = border_size + (col-col_of_interest)*tile_width + y_tile
    

    pad = int(length_max/2)
    mask_tile_w_borders_padded = np.pad(mask_tile_w_borders, ((pad,pad),(pad,pad)),"constant" ) #0 padding
    x_in_mask_tile+=pad
    y_in_mask_tile+=pad
    #Check if already a cell 
    # if row_of_interest == 1 and col_of_interest == 3 : 
    # if x_tile == 951 or x_tile == 971 : 

    #     print(green("x_tile"),x_tile)
    #     print(green("y_tile"),y_tile)
    #     display_mask(mask_tile_w_borders_padded[x_in_mask_tile-int(length_max/2):x_in_mask_tile+int(length_max/2),y_in_mask_tile-int(length_max/2):y_in_mask_tile+int(length_max/2)], title="place avant ajout du mask ")
        
    mask_tile_w_borders_padded[x_in_mask_tile-int(length_max/2):x_in_mask_tile+int(length_max/2),y_in_mask_tile-int(length_max/2):y_in_mask_tile+int(length_max/2)] += mask_cell
    mask_tile_w_borders = mask_tile_w_borders_padded[pad:-pad,pad:-pad]
    # if row_of_interest == 1 and col_of_interest == 3 : 
    # if x_tile == 951 or x_tile == 971 : 
    #     print(green("x_tile"),x_tile)
    #     print(green("y_tile"),y_tile)
    #     display_mask(mask_tile_w_borders_padded[x_in_mask_tile-int(length_max/2):x_in_mask_tile+int(length_max/2),y_in_mask_tile-int(length_max/2):y_in_mask_tile+int(length_max/2)], title="place apres ajout du mask ")
    #     display_mask(mask_cell, title="mask_cell ajouté a l'emplacement ")

    return mask_tile_w_borders

def find_cell_from_center_of_mass(cell_name,table_cells, table_cells_in_tile_w_border,channel,row_of_interest,col_of_interest,cell_from_1_channel,dataset_config):
    """ From cell_from_1_channel (mask) find the cell in table_cells_in_tile_w_border that has the closest center of mass and modify the value of  table_cells["used_to_build_{}".format(cell_name)"]"""
    dict_cell = {
        "cell_type" : channel, 
        "tile_row" : row_of_interest,
        "tile_col" : col_of_interest,
        "tile_cat" : "center",
    }
    dict_cell = add_cell_coord_info(dict_cell,cell_from_1_channel,dataset_config)
    x_tile = (dict_cell["x_tile"]%dataset_config.tile_height)
    y_tile = (dict_cell["y_tile"]%dataset_config.tile_width)

    radius = 2
    filtered_table_cell_channel = table_cells_in_tile_w_border[table_cells_in_tile_w_border["channel_number"]==channel]
    filtered_table_cell_channel = filtered_table_cell_channel[(filtered_table_cell_channel["x_tile"]>=x_tile-radius)&(filtered_table_cell_channel["x_tile"]<=x_tile+radius)]
    filtered_table_cell_channel = filtered_table_cell_channel[(filtered_table_cell_channel["y_tile"]>=y_tile-radius)&(filtered_table_cell_channel["y_tile"]<=y_tile+radius)]
    if len(filtered_table_cell_channel.index) == 0 : 
        print("No cell found for cell {} in channel {}".format(cell_name,channel))
        return table_cells
    idx_cell = filtered_table_cell_channel.index[0]
    table_cells.iloc[idx_cell,table_cells.columns.get_loc("used_to_build_{}".format(cell_name))] = True
    return table_cells

def modifiy_table_cells_to_mark_used_cells(cell_name, cell_with_identities, table_cells,table_cells_in_tile_w_border, channels_list, row_of_interest,col_of_interest,dataset_config):
    """
    Cells that have been used to create new celltype should receive value True in table_cells["used_to_build_{}".format(cell_name)"]
    it can happens that 2 cells from 1 channels are used and linked with the body of a cell from the other channel. Then the two cells receive True in the column
    """
    for channel in channels_list : 
        cell_from_1_channel = np.logical_or((cell_with_identities==np.sum(channels_list)),(cell_with_identities==channel))
        # display_mask(cell_from_this_channel, title="cell_from_this_channel -> "+str(channel))
        mask_with_labels = sk_morphology.label(cell_from_1_channel)
        if len(np.unique(mask_with_labels)) > 2 :
            for cell_id in np.unique(mask_with_labels)[1:] : 
                cell_from_1_channel = np.where(mask_with_labels==cell_id,1,0).astype(bool)
                table_cells = find_cell_from_center_of_mass(cell_name,table_cells, table_cells_in_tile_w_border,channel,row_of_interest,col_of_interest,cell_from_1_channel,dataset_config)
        else : 
            table_cells = find_cell_from_center_of_mass(cell_name,table_cells, table_cells_in_tile_w_border,channel,row_of_interest,col_of_interest,cell_from_1_channel,dataset_config)
    return table_cells


def reconstitute_mask_cells_tile_w_borders(table_cells_in_tile_w_border,row_of_interest,slide_num, col_of_interest, dataset_config):
    """Build mask with the cells in table_cells_in_tile_w_border
    Contains cells that have a part of body in the tile of interest and are not too close to the border
    """
    mask_tile_w_borders = np.zeros((2*dataset_config.border_size_during_segmentation+dataset_config.tile_width,2*dataset_config.border_size_during_segmentation+dataset_config.tile_width)).astype(np.uint8)
    for index, row_df in table_cells_in_tile_w_border.iterrows():
        channel_number, row, col,x_tile, y_tile , length_max = row_df["channel_number"], row_df["tile_row"], row_df["tile_col"], row_df["x_tile"], row_df["y_tile"], row_df["length_max"]
        path_cell = get_path_cell(slide_num,channel_number, row, col, x_tile, y_tile, dataset_config)
        mask_cell = channel_number*plt.imread(path_cell)
        mask_cell=mask_cell.astype(np.uint8)
        mask_tile_w_borders = _add_mask_cell_to_empty_mask(mask_tile_w_borders, mask_cell, row_of_interest, col_of_interest, row, col, x_tile, y_tile, length_max,dataset_config)
    mask_tile_w_borders_centered_cells = filter.filter_center_cells(mask_tile_w_borders, dataset_config.tile_width,preserve_identity=True)
    return mask_tile_w_borders_centered_cells

def build_dict_cell_from_mask(cell_name,cell_with_identities,channel_number_new_type,row_of_interest,col_of_interest,id_cell,dataset_config):
    """ Build dict of the cell and check if satisfy requirements, if not, return None"""
    cell_newtype = cell_with_identities.astype(bool)
    dict_cell = {
        "cell_type" : channel_number_new_type,
        "channel_number" : channel_number_new_type,
        "tile_row" : row_of_interest,
        "tile_col" : col_of_interest,
        "tile_cat" : "center",
    }
    dict_cell = add_cell_coord_info(dict_cell,cell_newtype,dataset_config)
    if dict_cell["size"] > 30000 : #Check temporaire 
        display_mask(cell_newtype, title='size > 50000')
        print(blue("TOO BIG"))
        return None
    if dict_cell["check_out_of_borders"] :  #Check temporaire 
        display_mask(cell_newtype, title='Cell larger than borders')
    if not dict_cell["check_in_centered_tile"] :
        return None
    dict_cell["id_cell"] = id_cell
    dict_cell["used_to_build_{}".format(cell_name)] = False 
    # info_np(crop_cell,"crop_cell")
    return dict_cell    

def get_cells_from_merge_channel_from_tile(slide_num,table_cells, row_of_interest, col_of_interest,id_cell_max, dataset_config, cells_from_multiple_channels):
    """ 
    For 1 tiles, take the neighbors and find cells that are superposed
    Build on the same scheme as get_cells_fluorescence_from_tile 

    cells_from_multiple_channels is a dict with key : cell_name, value = list of channels 
    table_cells is modified because some cells are used to build the new ones
    """
    verbose = True 
    save_mask = True
    list_cells = []
    id_cell = 0
    for cell_name, channels_list in cells_from_multiple_channels.items():
        channel_number_new_type = dataset_config.association_cell_name_channel_number[cell_name]
        table_cells_in_tile_w_border = filter_table_cells_around_tile(table_cells,row_of_interest,col_of_interest,dataset_config,channels_list)
        if len(table_cells_in_tile_w_border) == 0 :
            continue

        mask_tile_w_borders_centered_cells = reconstitute_mask_cells_tile_w_borders(table_cells_in_tile_w_border,row_of_interest,slide_num, col_of_interest, dataset_config)

        cells_numeroted = sk_morphology.label((mask_tile_w_borders_centered_cells>0).astype(bool))
        cells_from_overlap = np.where(mask_tile_w_borders_centered_cells==np.sum(channels_list),1,0).astype(bool)
        numeroted_overlaping_cells = np.where(cells_from_overlap,cells_numeroted,0)

        for cell_id in np.unique(numeroted_overlaping_cells)[1:]:
            cell_with_identities = np.where(cells_numeroted==cell_id, mask_tile_w_borders_centered_cells,0)
            if np.max(cell_with_identities) > np.sum(channels_list) :

                print(red("Ici une cell est de np max supperieur a la somme des cannaux superposées")) if verbose else None 
                continue 
            dict_cell = build_dict_cell_from_mask(cell_name,cell_with_identities,channel_number_new_type,row_of_interest,col_of_interest,id_cell_max+id_cell,dataset_config)
            if dict_cell is None : 
                continue 
            id_cell += 1
            table_cells = modifiy_table_cells_to_mark_used_cells(cell_name,cell_with_identities, table_cells,table_cells_in_tile_w_border, channels_list, row_of_interest,col_of_interest,dataset_config)
            list_cells.append(dict_cell)
            if save_mask : 
                crop_cell = crop_around_cell(cell_with_identities.astype(bool),dict_cell["x_tile_border"],dict_cell["y_tile_border"],max(dict_cell["length_max"],256)) 
                path_mask_cell = get_path_cell(slide_num,channel_number_new_type, row_of_interest, col_of_interest,dict_cell["x_tile"],dict_cell["y_tile"], dataset_config)
                crop_cell_pil = np_to_pil(crop_cell)
                crop_cell_pil.save(path_mask_cell)

    return table_cells, list_cells

def segment_cells_coming_from_several_channels(slide_num, dataset_config,cells_from_multiple_channels=None, particular_roi = None): 
    """
    Build on the same pattern as segment_classify_cells
    list_channels_to_merge = [[1,3]]
    Return dataframe with cells info 
    Save masks of the new cells 
    Si je veux ajouter + de 2 cellules : je dois considérer des nb premiers pour retrouver dans le mask la catégories des cellules d'origine (sinon 3 peut venir de 1+2 ou de la catégorie 3)
    """
    if cells_from_multiple_channels is None : 
        cells_from_multiple_channels = dataset_config.cells_from_multiple_channels
    list_all_cells = [] #contains dictionnaries of infos about a cell 
    path_classified_slide = get_path_table_cells(slide_num,dataset_config,particular_roi = particular_roi)
    if os.path.exists(path_classified_slide):
        table_cells = pd.read_csv(path_classified_slide,sep = ";")
        for cell_name in cells_from_multiple_channels.keys():
            if not "used_to_build_{}".format(cell_name) in table_cells.columns:
                table_cells["used_to_build_{}".format(cell_name)] = False
    else :
        raise NotImplementedError("path_classified_slide does not exist")
    if particular_roi is not None : 
        n_tiles_border_to_cover_roi_border_size = int(dataset_config.border_size_during_segmentation/dataset_config.tile_width)+1
        row_origin, col_origin, row_end, col_end = particular_roi[0]-n_tiles_border_to_cover_roi_border_size,particular_roi[1]-n_tiles_border_to_cover_roi_border_size,particular_roi[2]+n_tiles_border_to_cover_roi_border_size,particular_roi[3]+n_tiles_border_to_cover_roi_border_size
        list_tiles = [(r,c) for r in range(row_origin,row_end+1) for c in range(col_origin,col_end+1)]
    else : 
        n_tiles_row_slide,n_tiles_col_slide = slide.get_n_row_col_img(slide_num,dataset_config)
        list_tiles = [(r,c) for r in range(1,n_tiles_row_slide+1) for c in range(1,n_tiles_col_slide+1)]

    id_cell_max = table_cells["id_cell"].max()
    for tile_coord in tqdm(list_tiles):
        # print(yellow("{} is being processed ".format(tile_coord)))
        table_cells, list_cells_tile = get_cells_from_merge_channel_from_tile(slide_num,table_cells, tile_coord[0], tile_coord[1],id_cell_max, dataset_config,cells_from_multiple_channels=cells_from_multiple_channels)
        # print(blue("{} cells added".format(len(list_cells_tile))))
        id_cell_max+= len(list_cells_tile)
        list_all_cells+= list_cells_tile

    path_table_cells = get_path_table_cells(slide_num,dataset_config,particular_roi)
    table_cells_extended = pd.DataFrame(list_all_cells, columns = dataset_config.colnames_table_cells_base+["used_to_build_{}".format(cell_name) for cell_name in cells_from_multiple_channels.keys()])
    table_cells = pd.concat([table_cells,table_cells_extended],axis = 0, ignore_index=True)
    table_cells.to_csv(path_table_cells,index=False,sep = ";")
    print("njernf",path_table_cells)
    table_cells = pd.read_csv(path_table_cells,sep = ";")
    print(blue("Size of table_cells : {} in slide {}".format(table_cells.shape,slide_num ))) 
    return table_cells



class ModelClassification:
    """
    Information sur le model de classification des cellules microgliales 

    Si le model existe deja : je rentre le nom et il ira loader les poids 
    Si le nom de model donné en input n'existe pas dans le follder, les paramètres serviront à faire l'entrainement 
    
    Required key in classification_param : 
    - model_name : name of the classification model
    """
    # def __init__(self, model_name,dir_base_classif, training_set_config_name="prol_amoe_clust_phag_rami",verbose = 0):
    def __init__(self, classification_param,dataset_config=None,verbose=0):
        model, param_model = self.load_model_from_name(classification_param)
        self.model_name = classification_param["model_name"]
        self.param_model = param_model
        self.model = model
        self.dataset_config = dataset_config

    def classify_cells(self, crop_rgb, cell_mask, dict_cell, verbose=0):
        """
        Utilise le model pour prédire un crop rgb et renvoie une mask de taille [LxLx(1+nb_classes)] le 1 compte pour la decision 
        """
        if self.model_name == "best_model_microglia_IHC":
            return classify_microglial_cells(dataset_config=self.dataset_config,crop_rgb=crop_rgb,model=self.model,cell_mask = cell_mask, dict_cell=dict_cell, verbose = verbose)
        else : 
            # Classification modularity 
            raise NotImplementedError("model_name not implemented")

    def load_model_from_name(self,classification_param):
        """
        Return model, param_model 
        """
        
        model_name = classification_param["model_name"]
        if "microglia_IHC" in model_name :
            dir_base_classif = classification_param["dir_base_classif"]
            training_set_config_name = classification_param["training_set_config_name"]
            path_models = os.path.join(dir_base_classif,"models",training_set_config_name,model_name)
            # Param model loading 
            path_param_model = os.path.join(path_models,"dict_param_model.json")
            with open(path_param_model) as f:
                param_model = json.loads(f.read())
            #Model loading 
            path_weights_model = os.path.join(path_models,"weights","weights-best.hdf5")
            model = unet(param_model, pretrained_weights = None, verbose=False)
            model.load_weights(path_weights_model)
            return model, param_model

        else : 
            # Classification modularity 
            raise NotImplementedError("Model not implemented")