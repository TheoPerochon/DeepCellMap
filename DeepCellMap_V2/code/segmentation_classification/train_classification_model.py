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

from segmentation_classification import segmentation
from preprocessing import filter
from preprocessing import slide
from preprocessing import tiles
from utils.util import * 
#from python_files.const import *

# from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall 
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows
# Suppressing some warnings
import warnings
warnings.filterwarnings('ignore')


from keras import models
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.metrics import SparseCategoricalAccuracy,MeanIoU
from tensorflow.keras.optimizers import Adam 


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
# elif base_config.dataset_name == "covid_data_immunofluorescence":
#     dataset_config = FluorescenceCovidConfig()
# elif base_config.dataset_name == "ihc_pig":
#     dataset_config = PigIhc()
# else : 
#     raise Exception("Dataset config not found")

"""  1. Util functions  """


def _path_distinct_int(training_config_name, supress_old_folder = False):
    """
    Create training config folder and folders with rgb_distinct_int_distinct_labels and mask_distinct_int_distinct_labels
    """
    training_config_path = os.path.join(DIR_TRAINING_DATASET,training_config_name)
    dest_path_rgb = os.path.join(training_config_path,"rgb_distinct_int_distinct_labels")
    dest_path_mask = os.path.join(training_config_path,"mask_distinct_int_distinct_labels")
    if supress_old_folder : 
        if os.path.exists(training_config_path):
            shutil.rmtree(training_config_path)
        os.makedirs(dest_path_rgb)
        os.makedirs(dest_path_mask)
    return training_config_path, dest_path_rgb, dest_path_mask


""""  2. Data preparation  """

def attribute_distinct_int_to_distinct_labels(state_names, list_substates, training_config_name):
    """
    Attribute different int to masks according to cell class 
    Save new masks in rgb_distinct_int_distinct_labels and mask_distinct_int_distinct_labels
    """
    dest_path, dest_path_rgb, dest_path_mask = _path_distinct_int(training_config_name,supress_old_folder = True)
    n_already_in_dataset = 0
    state_int = 1
    df_stat_dict = dict()
    weights_states = []
    for state_idx, substates in enumerate(list_substates,1):
        state_name = state_names[state_idx]
        df_stat_dict[state_name] = dict({"associated_int" : state_int, "list_substates": [], "n_in_substates":[],"n_in_state":np.nan,"weiths_state":np.nan})
        nb_of_cells_under_label = 0
        substates = substates.split("+")
        for substate in substates:
            df_stat_dict[state_name]["list_substates"].append(substate)
            path_mask_classe = os.path.join(dataset_config.dir_training_set_cells,substate)
            path_mask_classe_rgb = os.path.join(path_mask_classe,"rgb")
            path_mask_classe_mask = os.path.join(path_mask_classe,"mask")
            liste_noms_rgb = [path_mask_classe_rgb+"/"+f for f in supprimer_DS_Store(os.listdir(path_mask_classe_rgb))]
            liste_noms_mask = [path_mask_classe_mask+"/"+f for f in supprimer_DS_Store(os.listdir(path_mask_classe_mask))]
            n_in_subgroup = len(liste_noms_rgb)
            for path_rgb, path_mask in zip(liste_noms_rgb,liste_noms_mask):
                destination_mask = os.path.join(dest_path_mask,os.path.split(path_mask)[1])
                destination_rgb = os.path.join(dest_path_rgb,os.path.split(path_rgb)[1])
                nb_of_cells_under_label+=1
                img_mask = plt.imread(path_mask)
                img_mask = np.asarray(img_mask)*255
                img_mask_np = np.where(img_mask>0,state_int,0)
                img_mask_np = img_mask_np.astype(np.uint8)
                im_pil = np_to_pil(img_mask_np)
                if os.path.exists(destination_mask):
                    print(destination_mask)
                    n_already_in_dataset +=1
                    n_in_subgroup -=1
                    nb_of_cells_under_label -=1
                shutil.copyfile(path_rgb, destination_rgb)
                im_pil.save(destination_mask)
            # print("already exists", n_already_in_dataset)
            # print(substate + " = " + str(n_in_subgroup) + " masks")
            df_stat_dict[state_name]["n_in_substates"].append(n_in_subgroup)
        df_stat_dict[state_name]["n_in_state"] = nb_of_cells_under_label
        print(state_name + " : " + str(nb_of_cells_under_label) + " sample")
        state_int+=1
        weights_states.append(nb_of_cells_under_label)
    weights_states = np.asarray(weights_states)
    for state_idx, substates in enumerate(list_substates,1):
        state_name = state_names[state_idx]
        df_stat_dict[state_name]["weiths_state"] = weights_states[state_idx-1]/weights_states.sum()
    colname_dataframe = ["n_state","state_names"]
    colname_info_labels_soft_strong = ["substates_in_"+state for state in state_names[1:]]
    colname_info_labels_number = ["n_"+state for state in state_names[1:]]
    colname_info_labels_proportion = ["proportion_"+state for state in state_names[1:]]
    colname_info_labels_int = ["associated_int_"+state for state in state_names[1:]]+["int_detected"]
    colname_df_path = ["dest_path_rgb", "dest_path_mask"]
    colname_df_models = colname_dataframe + colname_info_labels_soft_strong + colname_info_labels_number+ colname_info_labels_proportion + colname_info_labels_int + colname_df_path
    df_models = pd.DataFrame(columns = colname_df_models)
    print("Il y a en tout " + str(weights_states.sum()) + " masks et " + str(n_already_in_dataset) + " masks qui n'ont pas été ajouté car les cellulse étaient déjà présentes")
    liste_row = [len(list_substates), state_names[1:]] + [substates.split("+") for substates in list_substates] + [df_stat_dict[state_name]["n_in_state"] for state_name in state_names[1:]] + [df_stat_dict[state_name]["weiths_state"] for state_name in state_names[1:]] + [df_stat_dict[state_name]["associated_int"] for state_name in state_names[1:]]+ [state_int] + [dest_path_rgb, dest_path_mask]
    df_models.loc[1] = liste_row
    dest_path_csv = os.path.join(dest_path,"info_training_dataset.csv")
    df_models.to_csv(dest_path_csv, sep = ";", index = False)
    return df_models, df_stat_dict

def get_rgb_wsi_in_slide(slide_num, x,y, img_size = None):
    """Construit l'image RGB de la ROI 
    MARGE : 1 tile de chaque coté 
    """
    # img_size = dataset_config.roi_border_size
    path_slide = slide.get_training_slide_path(slide_num)
    # print(path_slide)
    s = slide.open_slide(path_slide) 
    # print("Slide oppened")

    rgb = s.read_region((int(y-img_size/2),int(x-img_size/2)),0,(img_size,img_size))
    rgb_til = rgb.convert("RGB")
    rgb_np = np.asarray(rgb_til)
    # print(rgb_np.shape)
    rgb_np = rgb_np.astype(np.float32)
    #display_rgb(rgb)
    rgb_np = (rgb_np/255).astype(np.float32)
    return rgb_np

def _put_cell_A_in_middle(mask_A,img_size):
    mask = np.zeros((img_size,img_size))
    mask[img_size//2-mask_A.shape[0]//2:img_size//2-mask_A.shape[0]//2+mask_A.shape[0],img_size//2-mask_A.shape[1]//2:img_size//2-mask_A.shape[1]//2+mask_A.shape[1]] = mask_A
    return mask

def add_detected_cells(param_training_set, verbose = 0):
    """
    Ajoute les cellules detectées par le modèle de segmentation au mask de la cellule A avec un label "Detected"
    Remarque : les coordonnées x,y sont les coordonnées dans la tile elargie de dataset_config.roi_border_size = 512 en prenant pour origine le coin en haut a gauche (comme ce que donne get coordinate cell de ROI )
    cellule A : cellule de la boucle principale 
    crop A : crop de taille 256x256 centré en le centre de masse de la cellule A 
    cellule B : cellule de la deuxième boucle 
    """
    state_names=param_training_set["state_names"]
    list_substates=param_training_set["list_substates"]
    training_set_config_name = param_training_set["training_set_config_name"]

    check = False 
    adding_detected_cells = True
    training_config_path, rgb_distinct_int, mask_distinct_int = _path_distinct_int(training_set_config_name)
    if adding_detected_cells :
        path_folder_adding_detected_cells = os.path.join(training_config_path,"adding_detected_cells")
        mkdir_if_nexist(path_folder_adding_detected_cells)
    path_rgb_completed = os.path.join(training_config_path,"rgb_completed")
    path_mask_completed = os.path.join(training_config_path,"mask_completed")
    mkdir_if_nexist(path_rgb_completed)
    mkdir_if_nexist(path_mask_completed)

    list_rgb_distinct_int = supprimer_DS_Store(os.listdir(rgb_distinct_int))
    list_rgb_distinct_int.sort()
    list_rgb_distinct_int = [ os.path.join(rgb_distinct_int,f) for f in list_rgb_distinct_int]
    list_mask_distinct_int = supprimer_DS_Store(os.listdir(mask_distinct_int))
    list_mask_distinct_int.sort()
    list_mask_distinct_int = [ os.path.join(mask_distinct_int,f) for f in list_mask_distinct_int]

    idx_cell = 0
    # liste_idx = list(np.random.randint(0,len(list_rgb_distinct_int),1))

    for filename_cell_A_rgb,filename_cell_A_mask in tqdm(zip(list_rgb_distinct_int,list_mask_distinct_int)): #Boucle sur chacune des cellules du training set
        
        # if idx_cell in liste_idx:
        #     idx_cell +=1
        #     display_sample = True
        # else:
        #     idx_cell+=1
        #     continue
        display_sample = False
        # print("path cellule A filename_cell_A_rgb ",filename_cell_A_rgb, "filename_cell_A_mask",filename_cell_A_mask)
        """ Pour de la visualisation """
        rgb_A = plt.imread(filename_cell_A_rgb) if display_sample else None 
        mask_A_256 = plt.imread(filename_cell_A_mask)
        mask_A_256 = np.asarray(mask_A_256)*255
        mask_A_256 = mask_A_256.astype(np.uint8)
        
        """ input filename """
        slide_num_A, tile_row_A, tile_col_A, xA_with_border, yA_with_border  = decompose_filename(filename_cell_A_rgb) 
        xA_in_slide = ((tile_row_A-1)*dataset_config.roi_border_size)+ xA_with_border
        yA_in_slide = ((tile_col_A-1)*dataset_config.roi_border_size)+ yA_with_border
        rgb_A_1024 = get_rgb_wsi_in_slide(slide_num_A, xA_in_slide,yA_in_slide, img_size = dataset_config.roi_border_size) #cell au centre de l'image RGB 
        mask_detected_A_1024 = segment_microglia(rgb_A_1024,model_segmentation_type = MODEL_SEGMENTATION_PARAMS[str(slide_num_A)]["model_segmentation_type"],dilation_radius = MODEL_SEGMENTATION_PARAMS[str(slide_num_A)]["dilation_radius"], min_cell_size = MODEL_SEGMENTATION_PARAMS[str(slide_num_A)]["min_cell_size"], verbose = 1)
        mask_detected_A_1024 = mask_detected_A_1024.astype(np.uint8)
        mask_detected_A_1024 = mask_detected_A_1024*TABLE_STATE_INT["Detected"]

        mask_cell_A_in_1024 = _put_cell_A_in_middle(mask_A_256,dataset_config.roi_border_size) 
        mask_detected_A_1024_distinct_int = sk_morphology.label(mask_detected_A_1024)
        detected_overlapping_with_cell_A = np.where(np.logical_and(mask_detected_A_1024_distinct_int, mask_cell_A_in_1024), mask_detected_A_1024_distinct_int, 0)
        for i in np.unique(detected_overlapping_with_cell_A):
            if i == 0:
                continue
            mask_detected_A_1024_distinct_int = np.where(mask_detected_A_1024_distinct_int==i,0,mask_detected_A_1024_distinct_int)
        mask_detected_A_1024_without_overlap = np.where(mask_detected_A_1024_distinct_int, mask_detected_A_1024, 0)
        union_mask_A_and_detected_cells = mask_detected_A_1024_without_overlap + mask_cell_A_in_1024
        liste_mask = [mask_detected_A_1024,union_mask_A_and_detected_cells] if display_sample else None 
        liste_mask_title = ["Detected cells","Final mask in tissue"] if display_sample else None 
        plot_several_mask_1_rgb(rgb_A_1024, liste_mask,liste_mask_title,path_save = os.path.join(path_folder_adding_detected_cells,os.path.split(filename_cell_A_mask)[1])) if display_sample else None 

        new_mask_A = union_mask_A_and_detected_cells[dataset_config.roi_border_size//2-128:dataset_config.roi_border_size//2+128,dataset_config.roi_border_size//2-128:dataset_config.roi_border_size//2+128]
        new_mask_A = new_mask_A.astype(np.uint8)

        # liste_mask_256 = [mask_A_256, new_mask_A] if display_sample else None 
        # liste_mask_title_256 = ["mask A 256"+os.path.split(filename_cell_A_mask)[1],"mask A 256 with detected cells"] if display_sample else None 
        # plot_several_mask_1_rgb(rgb_A, liste_mask_256,liste_mask_title_256) if display_sample else None 

        destination_mask = os.path.join(path_mask_completed,os.path.split(filename_cell_A_mask)[1])
        destination_rgb = os.path.join(path_rgb_completed,os.path.split(filename_cell_A_rgb)[1])
        
        shutil.copyfile(filename_cell_A_rgb, destination_rgb)
        im_pil = np_to_pil(new_mask_A)
        im_pil.save(destination_mask)
        idx_cell +=1 


def split_training_test_set(param_training_set):
    """
    training_config_name/rgb_cells_completed     ->   proportion_training % in training_config_name/rgb_train and (1-proportion_training) % in training_config_name/rgb_test 
    training_config_name/mask_cells_completed    ->   proportion_training % in training_config_name/mask_train and (1-proportion_training) % in training_config_name/mask_test 
    """
    training_set_config_name = param_training_set["training_set_config_name"]
    img_size = param_training_set["img_size"]
    proportion_test = param_training_set["proportion_test"]

    path_training_set_info = os.path.join(DIR_TRAINING_DATASET,training_set_config_name,"training_set_info.csv")
    training_set_info = pd.read_csv(path_training_set_info, sep = ";")
    
    df_spliting_training_test = pd.DataFrame(columns = ["path_cell", "test"])
    training_config_path = os.path.join(DIR_TRAINING_DATASET,training_set_config_name)
    path_rgb_source = os.path.join(training_config_path, "rgb_completed")
    path_mask_source = os.path.join(training_config_path, "mask_completed") 

    path_rgb_test = os.path.join(training_config_path, "rgb_test")
    path_mask_test = os.path.join(training_config_path, "mask_test") 
    mkdir_if_nexist(path_rgb_test)
    mkdir_if_nexist(path_mask_test)

    list_all_cells = os.listdir(path_mask_source)
    list_all_cells = [f[:25] for f in list_all_cells]
    n = len(list_all_cells)
    proportion_training = 1-proportion_test
    nb_training = int(n*proportion_training)
    nb_test = n-nb_training
    print("Total sample :", n)
    print("Training dataset : ",str(nb_training))
    print("Test dataset : ",str(nb_test))
    training_set_info.loc[0,"proportion_test"] = proportion_test
    training_set_info.loc[0,"n_training"] = nb_training
    training_set_info.loc[0,"n_test"] = nb_test


    list_test_cells = random.sample(set(list_all_cells), k=nb_test)
    list_path_test_dataset_mask = [os.path.join(path_mask_source,f+"_mask.png") for f in list_test_cells]
    list_path_test_dataset_rgb = [os.path.join(path_rgb_source,f+"_rgb.png") for f in list_test_cells]
    idx_df = 0
    idx_test_renumerate = 0 
    for idx_test_renumerate,(filename_rgb_source,filename_mask_source)  in enumerate(zip(list_path_test_dataset_rgb,list_path_test_dataset_mask)):
        # dest_rgb_test = os.path.join(path_rgb_test,"sample_"+str(idx_test_renumerate)+".png")
        # dest_mask_test = os.path.join(path_mask_test,"sample_"+str(idx_test_renumerate)+".png")
        dest_rgb_test = os.path.join(path_rgb_test,os.path.split(filename_rgb_source)[-1])
        dest_mask_test = os.path.join(path_mask_test,os.path.split(filename_mask_source)[-1])
        shutil.move(filename_rgb_source,dest_rgb_test)
        shutil.move(filename_mask_source,dest_mask_test)
        df_spliting_training_test.loc[idx_df] = [os.path.split(filename_rgb_source)[-1][:25],1]
        idx_df+=1
        idx_test_renumerate+=1

    path_rgb_train = os.path.join(training_config_path, "rgb_training")
    path_mask_train = os.path.join(training_config_path, "mask_training")
    os.rename(path_rgb_source,path_rgb_train)
    os.rename(path_mask_source,path_mask_train)
    for training_example in os.listdir(path_rgb_train):
    # for training_example in os.listdir(path_rgb_source):
        df_spliting_training_test.loc[idx_df] = [training_example[:25],0]
        idx_df+=1
    df_spliting_training_test.to_csv(os.path.join(training_config_path,"csv_train_test.csv"), sep = ";", index = False)
    training_set_info.to_csv(path_training_set_info, sep = ";", index = False)
    return df_spliting_training_test, path_rgb_test, path_mask_test, path_rgb_train, path_mask_train

""""  3.Get Training-Validation-Test sets  """

def get_training_validation_test_set(param_training_set,param_model):
    normalize =True
    img_size = param_training_set["img_size"]
    training_set_config_name = param_training_set["training_set_config_name"]
    batch_size = param_model["batch_size"]
    proportion_validation = param_model["proportion_validation"]
    normalise = param_model["normalise"]
    batch_size_test = param_model["batch_size_test"]
    
    training_config_path = os.path.join(DIR_TRAINING_DATASET,training_set_config_name)
    path_rgb_train = os.path.join(training_config_path, "rgb_training")
    path_mask_train = os.path.join(training_config_path, "mask_training")
    path_rgb_test = os.path.join(training_config_path, "rgb_test")
    path_mask_test = os.path.join(training_config_path, "mask_test") 

    # print(blue("Préparation du training set"))
    train_rgb = tf.keras.utils.image_dataset_from_directory(
      path_rgb_train,
      labels = None,
      validation_split=proportion_validation,
      subset="training",
      color_mode="rgb",
      seed=123,
      shuffle = True,
      image_size=(img_size, img_size),
      batch_size=batch_size)
    train_mask = tf.keras.utils.image_dataset_from_directory(
      path_mask_train,
      labels = None,
      validation_split=proportion_validation,
      subset="training",
      color_mode="grayscale",
      seed=123,
      shuffle = True,
      image_size=(img_size, img_size),
      batch_size=batch_size)
    train_dataset = tf.data.Dataset.zip((train_rgb, train_mask))
    if normalise :
        normalization_layer = layers.Rescaling(1./255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    # print(blue("Préparation du validation set"))
    val_rgb = tf.keras.utils.image_dataset_from_directory(
      path_rgb_train,
      labels = None,
      validation_split=proportion_validation,
      subset="validation",
      color_mode="rgb",
      seed=123,
      shuffle = True,
      image_size=(img_size, img_size),
      batch_size=batch_size)
    val_mask = tf.keras.utils.image_dataset_from_directory(
      path_mask_train,
      labels = None,
      validation_split=proportion_validation,
      subset="validation",
      color_mode="grayscale",
      seed=123,
      shuffle = True,
      image_size=(img_size, img_size),
      batch_size=batch_size)
    validation_data = tf.data.Dataset.zip((val_rgb, val_mask))
    validation_data = validation_data.shuffle(buffer_size=len(validation_data))
    if normalise :
        validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

    #Test 
    # print(blue("Préparation du test set"))
    test_rgb = tf.keras.utils.image_dataset_from_directory(
      path_rgb_test,
      labels = None,
      color_mode="rgb",
      seed=123,
      shuffle = True,
      image_size=(img_size, img_size),
      batch_size=batch_size_test)
    test_mask = tf.keras.utils.image_dataset_from_directory(
      path_mask_test,
      labels = None,
      color_mode="grayscale",
      seed=123,
      shuffle = True,
      image_size=(img_size, img_size),
      batch_size=batch_size_test)

    test_dataset = tf.data.Dataset.zip((test_rgb, test_mask))
    test_dataset = test_dataset.shuffle(buffer_size=len(test_dataset))
    if normalise : 

        test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_data, test_dataset

def gamma(img, mask):
    img = tf.image.adjust_gamma(img, 0.05)
    return img, mask

def hue(img, mask):
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask

def crop(img, mask):
    img = tf.image.central_crop(img, 0.7)
    img = tf.image.resize(img, (256,256))
    mask = tf.image.central_crop(mask, 0.7)
    mask = tf.image.resize(mask, (256,256))
    #mask = tf.cast(mask, tf.uint8)
    return img, mask

def flip_hori(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask

def flip_vert(img, mask):
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask

def rotate(img, mask):
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask

def brightness(img, mask):
    img = tf.image.random_brightness(img, 0.1)
    return img, mask

def contrast(img, mask):
    img = tf.image.random_contrast(img, 0.5, 1.5)
    return img, mask

def saturation(img, mask):
    img = tf.image.random_saturation(img, 0.5, 1.5)
    return img, mask

def visu_data_augment(train_dataset,liste_data_augment, path_save_example = None):
    """
    Save example of augmented data 
    Save info on training set dataframe 
    """
    nb_examples_to_save = 2
    if path_save_example is not None:
        path_training_set_info = os.path.join(DIR_TRAINING_DATASET,path_save_example)
        path_folder_data_augment = os.path.join(path_training_set_info,"data_augmentation")
        mkdir_if_nexist(path_folder_data_augment)
    idx = 0
    for images, mask in train_dataset.take(nb_examples_to_save):
        list_rgb = [np.asarray(images[0]).astype(np.float32)]
        liste_mask = [np.zeros(images[0].shape, dtype = np.uint8)]
        liste_mask_title = ["Img original"]
        if "rotate" in liste_data_augment:
            images_rotate, mask_rotate = rotate(images, mask)
            liste_mask.append(np.asarray(mask_rotate[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_rotate[0]).astype(np.float32))
            liste_mask_title.append("rotate")
        if "gamma" in liste_data_augment:
            images_gamma, mask_gamma = gamma(images, mask)
            liste_mask.append(np.asarray(mask_gamma[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_gamma[0]).astype(np.float32))
            liste_mask_title.append("gamma")
        if "hue" in liste_data_augment:
            images_hue, mask_hue = hue(images, mask)
            liste_mask.append(np.asarray(mask_hue[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_hue[0]).astype(np.float32))
            liste_mask_title.append("hue")
        if "crop" in liste_data_augment:
            images_crop, mask_crop = crop(images, mask)
            liste_mask.append(np.asarray(mask_crop[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_crop[0]).astype(np.float32))
            liste_mask_title.append("crop")
        if "flip_hori" in liste_data_augment:
            images_flip_hori, mask_flip_hori = flip_hori(images, mask)
            liste_mask.append(np.asarray(mask_flip_hori[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_flip_hori[0]).astype(np.float32))
            liste_mask_title.append("flip_hori")
        if "flip_vert" in liste_data_augment:
            images_flip_vert, mask_flip_vert = flip_vert(images, mask)
            liste_mask.append(np.asarray(mask_flip_vert[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_flip_vert[0]).astype(np.float32))
            liste_mask_title.append("flip_vert")
        if "brightness" in liste_data_augment:
            images_brightness, mask_brightness = brightness(images, mask)
            liste_mask.append(np.asarray(mask_brightness[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_brightness[0]).astype(np.float32))
            liste_mask_title.append("brightness")
        if "contrast" in liste_data_augment:
            images_contrast, mask_contrast = contrast(images, mask)
            liste_mask.append(np.asarray(mask_contrast[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_contrast[0]).astype(np.float32))
            liste_mask_title.append("contrast")
        if "saturation" in liste_data_augment:
            images_saturation, mask_saturation = saturation(images, mask)
            liste_mask.append(np.asarray(mask_saturation[0]).astype(np.uint8)[:,:,0])
            list_rgb.append(np.asarray(images_saturation[0]).astype(np.float32))
            liste_mask_title.append("saturation")

        path_save = os.path.join(path_folder_data_augment,"data_augment_{}.png".format(idx))
        if len(liste_data_augment) > 0 :

            plot_several_mask_several_rgb(list_rgb, liste_mask,liste_mask_title,path_save = path_save, display = False,figsize=(30, 7))
        idx+=1

def apply_data_augment_to_training_set(train_dataset,param_model):
    training_set_config_name = param_model["param_training_set"]["training_set_config_name"]
    liste_data_augment = param_model["liste_data_augment"]
    visu_data_augment(train_dataset,liste_data_augment, path_save_example = training_set_config_name)
    n_before_augment = len(train_dataset)
    
    if "rotate" in liste_data_augment:
        a = train_dataset.map(rotate)
    if "gamma" in liste_data_augment:
        b = train_dataset.map(gamma)
    if "hue" in liste_data_augment:
        c = train_dataset.map(hue)
    if "crop" in liste_data_augment:  
        d = train_dataset.map(crop)   
    if "flip_hori" in liste_data_augment:
        e = train_dataset.map(flip_hori)
    if "flip_vert" in liste_data_augment:
        f = train_dataset.map(flip_vert)
    if "brightness" in liste_data_augment:
        g = train_dataset.map(brightness)
    if "contrast" in liste_data_augment:
        h = train_dataset.map(contrast)
    if "saturation" in liste_data_augment:
        i = train_dataset.map(saturation)
    if "rotate" in liste_data_augment:
        train_dataset = train_dataset.concatenate(a)
    if "gamma" in liste_data_augment:
        train_dataset = train_dataset.concatenate(b)
    if "hue" in liste_data_augment:
        train_dataset = train_dataset.concatenate(c)
    if "crop" in liste_data_augment:  
        train_dataset = train_dataset.concatenate(d)
    if "flip_hori" in liste_data_augment:
        train_dataset = train_dataset.concatenate(e)
    if "flip_vert" in liste_data_augment:
        train_dataset = train_dataset.concatenate(f)
    if "brightness" in liste_data_augment:
        train_dataset = train_dataset.concatenate(g)
    if "contrast" in liste_data_augment:
        train_dataset = train_dataset.concatenate(h)
    if "saturation" in liste_data_augment:
        train_dataset = train_dataset.concatenate(i)

    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    n_after_augment = len(train_dataset)
    print("Nombre d'images dans le train_dataset avant augmentation : ", n_before_augment)
    print("Nombre d'images dans le train_dataset après augmentation : ", n_after_augment)
    param_model["n_before_augment"] = n_before_augment
    param_model["n_after_augment"] = n_after_augment
    return train_dataset, param_model

""""  4.1. Model definition  """

class UpdatedMeanIoU(MeanIoU):
    def __init__(self,
        y_true=None,
        y_pred=None,
        ignore_class=None,
        num_classes=None,
        name=None,
        dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes, ignore_class=ignore_class, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight) #https://stackoverflow.com/questions/43751455/supertype-obj-obj-must-be-an-instance-or-subtype-of-type

def unet(param_model, pretrained_weights = None,verbose=True ):
      img_size = param_model["param_training_set"]["img_size"]
      input_size = (img_size,img_size,3)
      pooling_steps = param_model["pooling_steps"]
      learning_rate = param_model["initial_learning_rate"]
      labels=param_model["param_training_set"]["num_classes"]

      inputs = Input(input_size)
      conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
      conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
      # Downsampling steps
      pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
      conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
      conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
      
      if pooling_steps > 1:
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

        if pooling_steps > 2:
          pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
          conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
          conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
          drop4 = Dropout(0.5)(conv4)

          if pooling_steps > 3:
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
            conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            #Upsampling steps
            up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
            merge6 = concatenate([drop4,up6], axis = 3)
            conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

      if pooling_steps > 2:
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        if pooling_steps > 3:
          up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

      if pooling_steps > 1:
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
        if pooling_steps > 2:
          up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation= 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation= 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

      if pooling_steps == 1:
        up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
      else:
        up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) #activation = 'relu'

      merge9 = concatenate([conv1,up9], axis = 3)
      conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9) #activation = 'relu'
      conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9) #activation = 'relu'
      conv9 = Conv2D(labels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9) #activation = 'relu'
      conv10 = Conv2D(labels, 1, activation = 'softmax')(conv9)

      model = Model(inputs = inputs, outputs = conv10)

    #   model.compile(optimizer = Adam(lr = learning_rate), loss = 'sparse_categorical_crossentropy',metrics=["accuracy"])
      
      model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'sparse_categorical_crossentropy',metrics=[SparseCategoricalAccuracy(),UpdatedMeanIoU(name='iou', ignore_class = 0,num_classes=labels)])
    #   model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'sparse_categorical_crossentropy',metrics=[SparseCategoricalAccuracy(),MeanIoU(name='iou',ignore_class = None, num_classes=labels)])


      if verbose:
        model.summary()

      if(pretrained_weights):
        model.load_weights(pretrained_weights);

      return model

def classify_cells(crop_rgb, model, mask_cell = None , Model_segmentation = None  ,mode_prediction="summed_proba", bonification_cluster_size_sup = False, penalite_cluster_size_inf= False,verbose = 0, display = False):
    '''
    Only for the cell present on mask -> compute model prediction and probabilities for each pixel of the cell to belong to state k 
    Input :
        - rgb image (256x256)
        - mask image (cell_size, cell_size)
        - classification model 
        - mode_prediction : one among [max_proba, summed_proba]
    Output : 
        - cell_state_and_proba (cell_size,cell_size,n_states+1) # last dim is [decision, proba_background, proba_c1, ..., proba_cN]
    '''
    DISPLAY_FIG = False
    check_proba = False 
    #patch = np.expand_dims(normalizePercentile(crop_rgb), axis = [0, -1])    #shape(x,y,canaux) -> shape(1,x,y,canaux,1)
    #patch = np.expand_dims(normalizePercentile(crop_rgb), axis = 0)    #shape(x,y,canaux) -> shape(1,x,y,canaux,1)

    #display_rgb(crop_rgb, titre="max dans model_classif = "+ str(np.max(crop_rgb)))
    ##### ATTENTION #### probability_maps = Model_classification.model.predict_on_batch(patch)

    probability_maps = model.predict(np.expand_dims(crop_rgb, axis = 0),verbose=0)
    #print("(probability_maps.shape",probability_maps.shape)
    probability_maps = np.squeeze(probability_maps)    #Remove single-dimensional entries from the shape of an array.
    #print("(probability_maps.shape after squeeze",probability_maps.shape)
    #print("probability_maps sape ",probability_maps.shape)
    nb_states = probability_maps.shape[-1]

    

    if mask_cell is not None : 
        # print("max du cell mask : ", np.max(mask_cell))
        mask_cell = np.where(mask_cell>0,1,0)
        prediction_finale_and_probability = np.zeros((mask_cell.shape[0],mask_cell.shape[1],nb_states+1))
        list_proba_states = []
        mask_size = mask_cell.shape[0]
        #display_mask(probability_maps[:,:,-1], title1 = "probability map ramified")
        #display_mask(mask_cell, title1 = "Entire cell - crop size "+str(mask_size)) if DISPLAY_FIG else None 
        mask_crop_256 = mask_cell[int(mask_size/2-dataset_config.crop_size/2):int(mask_size/2+dataset_config.crop_size/2),int(mask_size/2-dataset_config.crop_size/2):int(mask_size/2+dataset_config.crop_size/2)]
        cell_size_on_256 = np.sum(mask_crop_256)
        #display_mask(mask_crop_256,title1 = "cell - crop size ,max =1 normalement ") if DISPLAY_FIG else None 

        for label in range(nb_states): # Background C1, ..., CN 
            #print("La on est dans label : ", label)
            #display_mask(probability_maps[:,:,label],title1="P(pixels="+str(label)+")") if DISPLAY_FIG else None 
            probability_map_label = mask_crop_256*probability_maps[:,:,label]  ###np.where(mask_composante_connexe == 1, probability_maps[:,:,label], 0)
            #display_mask(probability_map_label, title1="P(pixels="+str(label)+").Mask") #if DISPLAY_FIG else None 
            # if label == 0 : 
            #     display_mask(mask_crop_256, title1="label 0 , crop mask")

            # display_mask(probability_maps[:,:,label], title1="label "+str(label) + " crop_probability_maps[:,:,label]")
            if mode_prediction == "max_proba":
                proba_state_k = np.max(probability_map_label) #Le max de la proba sur la cellule 

            if mode_prediction == "summed_proba":
                if label == 0 : 
                    
                    proba_background = np.sum(probability_map_label)/cell_size_on_256
                    proba_state_k = np.sum(probability_map_label)/cell_size_on_256
                    if proba_background == 1 :
                        print("Proba background == 1 ")
                        proba_background = 0 
                        proba_state_k = 0
                    #print("label = 0 et label" , label  ,"proba_background = ",proba_background)
                    #display_mask(probability_map_label)

                elif label ==3: # Label cluster 
                    if bonification_cluster_size_sup : 
                        #print("Ok vonificvation")
                        proba_state_k = 1 
                    elif penalite_cluster_size_inf :
                        proba_state_k = 0 
                    else : 
                        proba_state_k = np.sum(probability_map_label)/(cell_size_on_256*(1-proba_background+0.00000001))
                    #weight_state = np.sum(prediction[0,:,:,classe_i])/(size_cell*(1-weight_background+0.0000001))
                    #print("label 3 = ",label," et proba_state_k = ",proba_state_k )
                else : 
                    if bonification_cluster_size_sup : 
                        proba_state_k = 0
                    else : 
                        proba_state_k = np.sum(probability_map_label)/(cell_size_on_256*(1-proba_background)+0.00000001)
                        # print("label else = ",label)
                        # print("numerateur",np.sum(probability_map_label))
                        # print("dennominateur : cell size ",cell_size_on_256)
                        # print("dennominateur : cell size ",cell_size_on_256)
                        # print("denominateur (1-proba_background) : ",(1-proba_background))
                        # print("denominateur : ",(cell_size_on_256*(1-proba_background)+0.00000001))
                        # print("final : ", proba_state_k)
                        
                    #print("dans else : label = ",label," et proba_state_k = ",proba_state_k )
            # print("max de mask_cell", np.max(mask_cell) , " et taille du mask : ", mask_cell.shape)
            mask_state_k = mask_cell*proba_state_k  
            #display_mask(mask_state_k, title1 = "max(P(pixels="+str(label)+").Mask)") #if DISPLAY_FIG else None 
            list_proba_states.append(proba_state_k)
            prediction_finale_and_probability[:,:,label+1] +=mask_state_k
        #print("Liste proba : ",list_proba_states ) if check_proba else None 
        # print("list_proba_states : ",list_proba_states)
        # print("Class more probable : ", np.argmax(list_proba_states[1:-1])+1)
        class_more_probable = np.argmax(list_proba_states[1:-1])+1# [1:] to exclude background  and detected 
        #print("Label more probable state : " , class_more_probable)
        mask_pred_decision = np.where(mask_cell == 1, class_more_probable, 0)
        #display_mask(mask_pred_decision, title1 = "more probable label") if DISPLAY_FIG else None 
        prediction_finale_and_probability[:,:,0] = mask_pred_decision

    return prediction_finale_and_probability


""""  4.2. Callbacks definition """

class SampleImageCallback(Callback):

    def __init__(self, model, param_model, data_callback ,save=False):

        self.model = model
        self.list_rgb_callback = data_callback[0]
        self.list_GT_mask_callback = data_callback[1]
        self.model_path = param_model["model_path"]
        self.interval_epoch = param_model["interval_epoch_callback"]
        self.state_names = param_model["param_training_set"]["state_names_with_detected"]
        self.save = save

    def on_epoch_end(self, epoch, logs={}):
            if np.mod(epoch,self.interval_epoch) == 0:
                for idx in range(len(self.list_rgb_callback)):
                    sample_data = self.list_rgb_callback[idx]
                    print("sample_data shape : ", sample_data.shape)
                    print("sample_data dtyyoe : ", sample_data.dtype)
                    ground_truth = self.list_GT_mask_callback[idx]
                    sample_predict = self.model.predict(sample_data,verbose=0)
                    # f = display_prediction_22(rgb=sample_data[0,:,:,:,0],prediction=sample_predict,ground_true = ground_truth, state_names = self.state_names,epoch_number =epoch  )
                    f = display_prediction_22(rgb=sample_data[0,:,:,:],prediction=sample_predict,ground_true = ground_truth, state_names = self.state_names,epoch_number =epoch  )

                    #display_prediction(self.sample_data[0,:,:,:,0],prediction = sample_predict,class_name =self.state_names )
                    mkdir_if_nexist(self.model_path )
                    path_sample_check = os.path.join(self.model_path, "callback_probability_map_across_epoch")
                    mkdir_if_nexist(path_sample_check)
                    path_sample = os.path.join(path_sample_check, "sample_"+str(idx))
                    mkdir_if_nexist(path_sample)
                    path_sample_epoch = os.path.join(path_sample, "epoch_" + str(epoch+1).zfill(3) + '.png')
                    f.savefig(path_sample_epoch, facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1)
                    plt.close("all")
      
class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data,interval_epoch, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.interval_epoch = interval_epoch
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir
        self.path_roc_curves = os.path.join(self.image_dir,"callback_roc_curves")
        self.path_cm = os.path.join(self.image_dir,"callback_confusion_matrix")
        mkdir_if_nexist(self.path_roc_curves)
        mkdir_if_nexist(self.path_cm)

    def on_epoch_end(self, epoch, logs={}):
        if np.mod(epoch,self.interval_epoch) == 0:
            images, mask = list(self.validation_data.take(1))[0]
            images = np.asarray(images).astype(np.float32)
            mask = np.asarray(mask).astype(np.uint8)

            y_pred = np.asarray(self.model.predict(images,verbose=0))
            y_true = mask
            y_pred_class = np.argmax(y_pred, axis=-1)

            #Plot result prediction 
            list_rgb = [images[0,:,:,:]]
            liste_mask = [y_true[0,:,:,0],y_pred_class[0,:,:]]
            liste_mask_title = ["Ground Truth","Predicted"]
            plot_several_mask_1_rgb(list_rgb[0], liste_mask,liste_mask_title,path_save = os.path.join(self.path_cm,f'img_pred_epoch_{epoch}'), display = False,figsize=(8, 3))


            # plot and save confusion matrix
            fig, ax = plt.subplots(figsize=(16,12))
            plot_confusion_matrix(y_true.reshape(-1), y_pred_class.reshape(-1), ax=ax)
            fig.savefig(os.path.join(self.path_cm, f'confusion_matrix_epoch_{epoch}'), facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1)

        # plot and save roc curve
            fig, ax = plt.subplots(figsize=(16,12))
            
            # print("y_pred shape = ",y_pred.shape)

            y_pred = y_pred.reshape(256*256*y_pred.shape[0],y_pred.shape[-1])
            present_labels = list(np.unique(y_true))
            y_pred = y_pred[:,present_labels]
            plot_roc(y_true.reshape(256*256*y_true.shape[0],1), y_pred, ax=ax)
            fig.savefig(os.path.join(self.path_roc_curves, f'roc_curve_epoch_{epoch}'), facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1)

def prepare_callback_weight_saving(param_model, just_best = True):
    path_model = param_model["model_path"]
    if just_best:
        path_weights_model = os.path.join(path_model, "weights")
        mkdir_if_nexist(path_weights_model)
        fname = os.path.sep.join([path_weights_model,"weights-best.hdf5"])
    else :
        path_weights_model = os.path.join(path_model, "weights")
        mkdir_if_nexist(path_weights_model)
        fname = os.path.sep.join([path_weights_model,"weights-epoch-{epoch:03d}-val_loss-{val_loss:.4f}.hdf5"])

    model_checkpoint = ModelCheckpoint(fname, monitor='val_loss',mode="min",verbose=1, save_best_only=True)
    return model_checkpoint

def prepare_callback_cell(param_model, to_display = False):
    list_cell_name_callback = param_model["list_cell_name_callback"]
    param_training_set = param_model["param_training_set"]
    training_config_path = os.path.join(DIR_TRAINING_DATASET,param_training_set["training_set_config_name"])
    path_rgb_train = os.path.join(training_config_path, "rgb_training")
    path_mask_train = os.path.join(training_config_path, "mask_training")
    list_rgb_callback=[]
    list_GT_mask_callback = []
    for cell_name in list_cell_name_callback:
        path_rgb = os.path.join(path_rgb_train, "{}_rgb.png".format(cell_name))
        path_mask = os.path.join(path_mask_train, "{}_mask.png".format(cell_name))
        if os.path.exists(path_rgb):
            # rgb = plt.imread(path_rgb) dtype : uint8 et ça va pas 
            # mask = plt.imread(path_mask)

            rgb = io.imread(path_rgb)
            mask = io.imread(path_mask)
            if to_display:
                list_rgb_callback.append(rgb)
                list_GT_mask_callback.append(mask)
            else :
                # print("In prepare_callback_cell : type rgb = ",type(rgb))
                # print("In prepare_callback_cell : type data = ",rgb.dtype)
                # print("In prepare_callback_cell : type mask = ",type(mask))
                # print("max rgb = ",np.max(rgb))
                rgb_rescaled = rgb/255
                rgb_rescaled = rgb_rescaled.astype(np.float32)
                # print("After rescalling")
                # print("In prepare_callback_cell : type rgb = ",type(rgb_rescaled))
                # print("In prepare_callback_cell : type data = ",rgb_rescaled.dtype)

                # print("max rgb = ",np.max(rgb_rescaled))
                # batch_sample = np.expand_dims(rgb_rescaled, axis = [0, -1])
                batch_sample = np.expand_dims(rgb_rescaled, axis = [0])
                # print("batch_sample shape = ",batch_sample.shape)
                list_rgb_callback.append(batch_sample)
                list_GT_mask_callback.append(mask)
    if len(list_GT_mask_callback) == 0 :
        print(red("There is no cells to display in the callback"))
    return [list_rgb_callback, list_GT_mask_callback]

def plot_callback_cell(param_model):
    data_callback = prepare_callback_cell(param_model, to_display = True)
    list_rgb = data_callback[0]
    liste_mask = data_callback[1]
    liste_mask_title = param_model["param_training_set"]["state_names"][1:]
    plot_several_mask_several_rgb(list_rgb, liste_mask,liste_mask_title,path_save = None, display = True,figsize=(20, 7))

def prepare_callbacks(model,param_model, validation_data):
    model_path = param_model["model_path"]
    interval_epoch = param_model["interval_epoch_callback"]
    
    callbacks = []
    if "model_checkpoint" in param_model["list_callbacks"]:
        model_checkpoint = prepare_callback_weight_saving(param_model, just_best = True)
        callbacks.append(model_checkpoint)
    if "reduce_lr" in param_model["list_callbacks"]:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, mode='auto',patience=20, min_lr=0)
        callbacks.append(reduce_lr)
    if "performance_cbk" in param_model["list_callbacks"]:
        performance_cbk = PerformanceVisualizationCallback(model=model,validation_data=validation_data,interval_epoch=interval_epoch,image_dir=model_path)
        callbacks.append(performance_cbk)
    if "sample_img" in param_model["list_callbacks"]:
        data_callback = prepare_callback_cell(param_model, to_display = False)
        sample_img = SampleImageCallback(model, param_model, data_callback ,save = True )
        callbacks.append(sample_img)
    return callbacks

""""  5. Learning visualisation """

def display_save_loss_accuracy(history,param_model): 
    model_path = param_model["model_path"]
    lossData = pd.DataFrame(history.history) 
    loss_path = os.path.join(model_path,'training_evaluation.csv')
    lossData.to_csv(loss_path)
    COLORS_METRICS_CLASSIFICATION = ['rgba(204,51,0,1)','rgba(255,153,47,1)','rgba(34,12,234,1)','rgba(36,30,200,1)','rgba(12,186,12,1)','rgba(40,200,40,1)']
    y_loss = ["loss","val_loss"]
    y_accuracy = ["sparse_categorical_accuracy","val_sparse_categorical_accuracy","iou","val_iou"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
    
    fig.add_trace(go.Scatter(y=lossData["loss"],name = "loss",mode='lines',line={'dash': 'solid', 'width': 5},line_color=COLORS_METRICS_CLASSIFICATION[0],showlegend=True),row=1, col=1)
    fig.add_trace(go.Scatter(y=lossData["val_loss"],name = "val_loss",mode='lines',line={'dash': 'dash', 'width': 5},line_color=COLORS_METRICS_CLASSIFICATION[0],showlegend=True),row=1, col=1)
    fig.add_trace(go.Scatter(y=lossData["sparse_categorical_accuracy"],name = "sparse_categorical_accuracy", mode='lines',line={'dash': 'solid', 'width': 5},line_color=COLORS_METRICS_CLASSIFICATION[2], showlegend=True),row=1, col=2)
    fig.add_trace(go.Scatter(y=lossData["val_sparse_categorical_accuracy"],name = "val_sparse_categorical_accuracy", mode='lines',line={'dash': 'dash', 'width': 5},line_color=COLORS_METRICS_CLASSIFICATION[2], showlegend=True),row=1, col=2)
    fig.add_trace(go.Scatter(y=lossData["iou"],name = "iou",mode='lines',line={'dash': 'solid', 'width': 5},line_color=COLORS_METRICS_CLASSIFICATION[4], showlegend=True),row=1, col=2)
    fig.add_trace(go.Scatter(y=lossData["val_iou"],name = "val_iou", mode='lines',line={'dash': 'dash', 'width': 5},line_color=COLORS_METRICS_CLASSIFICATION[4], showlegend=True),row=1, col=2)
    fig.update_layout(title_text="<b>Loss & Accuracy across epochs",title_x=0.5,title_font=dict(size=30),
                        showlegend=True, width=1600,height=500,margin=dict(l=50,r=50,b=100,t=100,pad=4))
    fig.update_xaxes(title_text = "Epochs",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    fig.show()       
    fig.write_image(os.path.join(model_path, "loss_accuracy.png"))

def make_gif(frame_folder):
    path_to_save = os.path.dirname(frame_folder)
    filename = os.path.splitext(os.path.basename(frame_folder))[0]
    path_to_save = os.path.join(path_to_save,filename+".gif")
    path_img = glob.glob(f"{frame_folder}/*.png")
    path_img.sort()

    frames = [Image.open(image) for image in path_img]
    frame_one = frames[0]
    frame_one.save(path_to_save, format="GIF", append_images=frames,save_all=True, duration=300, loop=0)
    print(f"gif saved at {path_to_save}")

def make_gifs_from_sample_callback(param_model):
    path_img_sample_callback = os.path.join(param_model["model_path"],"callback_probability_map_across_epoch")
    samples_path = [os.path.join(path_img_sample_callback,sample) for sample in os.listdir(path_img_sample_callback) if sample != ".DS_Store"]
    for sample_path in samples_path:
        if os.path.isdir(sample_path):
            make_gif(sample_path)

""""  6. Model evaluation on test set """

def get_df_probabilities_on_test_set(test_dataset, param_model, best_weights = True):
    state_names_with_detected = param_model["param_training_set"]["state_names_with_detected"]
    model_path = param_model["model_path"]

    if best_weights : 
        model = unet(param_model, pretrained_weights = None, verbose=False)
        path_weights_model = os.path.join(model_path, "weights","weights-best.hdf5")
        model.load_weights(path_weights_model)
        # model =load_model(path_weights_model) # regarder : https://www.tensorflow.org/guide/keras/save_and_serialize?hl=fr#registering_the_custom_object

        
    
    path_test_set_dir = os.path.join(model_path, "results_on_test_set")
    mkdir_if_nexist(path_test_set_dir)

    colname_GT_pred = ["Ground_truth", "Prediction_decision"]
    colname_proba_states = ["proba_"+state for state in state_names_with_detected[1:]]
    colname_proba_states_without_detected = ["proba_wt_detected_"+state for state in state_names_with_detected[1:-1]]
    pred_on_each_cell = pd.DataFrame(columns = colname_GT_pred+colname_proba_states+colname_proba_states_without_detected)
    idx_df = 0

    cm_proba = np.zeros((5,5))
    cm = np.zeros((5,5))

    for rgb, mask in test_dataset.as_numpy_iterator():
        label_GT = list(np.unique(mask))
        if 6 in label_GT:
            label_GT.remove(6)
        label_GT = np.max(label_GT) #Label GT

        prediction = model.predict(rgb,verbose=0)
        mask_cell = np.where(mask[0,:,:,0] >0,1,0)

        cell_class_or_proba = classify_cells(rgb[0], model, mask_cell = mask_cell , Model_segmentation = None  ,mode_prediction="summed_proba", bonification_cluster_size_sup = False, penalite_cluster_size_inf= False,verbose = 0, display = False)
        cell_decision = cell_class_or_proba[:,:,0]
        cell_probas = cell_class_or_proba[:,:,2:]

        label_decision = int(np.max(cell_decision))
        # print("label gt ,",label_GT," label pred ",label_decision)
        cm[int(label_GT)-1,label_decision-1]+=1

        list_proba_states = [] #['Proliferative','Amoeboid','Cluster','Phagocytic','Ramified',"Detected"]
        for state_idx,state in enumerate(state_names_with_detected[1:]):
            
            proba_state = np.max(cell_probas[:,:,state_idx])
            list_proba_states.append(proba_state)
            # print("state : ",state," proba : ",proba_state)
            # print("P(cell = "+state+")=",proba_state)

        list_proba_states_normalised = []
        for i in range(1,len(state_names_with_detected)-1): #['Proliferative','Amoeboid','Cluster','Phagocytic','Ramified']
            proba_normalised_state_i = list_proba_states[i]/np.sum(list_proba_states[1:-1])
            cm_proba[int(label_GT)-1,i-1]+=proba_normalised_state_i
            list_proba_states_normalised.append(proba_normalised_state_i)
            # state = state_names_with_detected[i]
            # print("P_normalised(cell = "+state+")=",proba_normalised_state_i)

        pred_on_each_cell.loc[idx_df] = [label_GT,label_decision]+list_proba_states+list_proba_states_normalised
        idx_df+=1
        # print("In get_df_probabilities_on_test_set : type rgb = ",type(rgb))
        # print("In get_df_probabilities_on_test_set : type rgb = ",type(rgb[0]))
        # print("shape rgb = ",rgb.shape)
        # print("max rgb = ",np.max(rgb))
        # print("In prepare_callback_cell : type data = ",rgb.dtype)

        f = display_prediction_22(rgb=rgb[0],prediction=prediction,ground_true = mask[0],final_pred = np.argmax(prediction, axis = 3)[0], state_names = state_names_with_detected)
        pred_2 = np.copy(cell_class_or_proba)
        pred_2 = np.expand_dims(pred_2, axis = 0)

        g = display_prediction_23(rgb=rgb[0],prediction=pred_2[:,:,:,1:],ground_true=mask[0,:,:,0], final_pred = pred_2[0,:,:,0], mask_cell = mask_cell,compute_weiht = False, state_names = state_names_with_detected, mode_prediction = "summed_proba")

        filepath = os.path.join(path_test_set_dir, "sample_"+str(idx_df)+".png")
        f.savefig(filepath, facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1)
        filepath = os.path.join(path_test_set_dir, "sample_"+str(idx_df)+"_function_23.png")
        g.savefig(filepath, facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1)
        plt.close("All")
        i+=1
    pred_on_each_cell.to_csv(os.path.join(path_test_set_dir, "pred_on_each_cell.csv"),index=False)
    cm_df = pd.DataFrame(cm,columns = state_names_with_detected[1:-1],index = state_names_with_detected[1:-1])
    cm_proba_df = pd.DataFrame(cm_proba,columns = state_names_with_detected[1:-1],index = state_names_with_detected[1:-1])
    cm_df.to_csv(os.path.join(path_test_set_dir, "confusion_matrix.csv"))
    cm_proba_df.to_csv(os.path.join(path_test_set_dir, "confusion_matrix_proba.csv"))
    return pred_on_each_cell, cm, cm_proba

def plot_confusion_matrix_decision(pred_on_each_cell,param_model):
    global_f1 = param_model["global_f1"]
    model_path = param_model["model_path"]
    gt_np = np.array(pred_on_each_cell["Ground_truth"])
    pred_np = np.array(pred_on_each_cell["Prediction_decision"])
    plot_confusion_matrix(gt_np, pred_np)
    plt.savefig(os.path.join(model_path,"Confusion_matrix_"+str(global_f1)+".png"), facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1)
    plt.close("All")

def get_model_evaluation(test_dataset,param_model):
    
    pred_on_each_cell, cm, cm_proba = get_df_probabilities_on_test_set(test_dataset, param_model, best_weights = True)
    model_name = param_model["model_name"]
    state_names_with_detected = param_model["param_training_set"]["state_names_with_detected"] 
    model_path = param_model["model_path"]

    colname_model = ["Model", "global_F1"]
    colname_precision = ["Precision_"+state for state in state_names_with_detected[1:-1]]
    colname_recall = ["Recall_"+state for state in state_names_with_detected[1:-1]]
    colname_f1_score = ["F1_score_"+state for state in state_names_with_detected[1:-1]]
    df_model_performance_on_test_set = pd.DataFrame(columns = colname_model + colname_precision+colname_recall+colname_f1_score)

    weights = np.sum(cm,axis = 1)
    weights = weights/np.sum(weights)

    global_f1 = 0
    dict_model_performance_on_test_set = dict()
    for i in range(len(state_names_with_detected)-1-1): #['Proliferative','Amoeboid','Cluster','Phagocytic','Ramified']
        state = state_names_with_detected[i+1]
        # print("State : ",state)
        precision_i = cm[i,i]/np.sum(cm[:,i]) if np.sum(cm[:,i]) != 0 else 0
        recall_i = cm[i,i]/np.sum(cm[i,:]) if np.sum(cm[i,:]) != 0 else 0
        f1_score_i = 2*precision_i*recall_i/(precision_i+recall_i) if(precision_i+recall_i) != 0 else 0
        # print("Precision : ",precision_i," Recall : ",recall_i," F1_score : ",f1_score_i)
        dict_model_performance_on_test_set["Model"] = model_name
        dict_model_performance_on_test_set["Precision_"+state] = precision_i
        dict_model_performance_on_test_set["Recall_"+state] = recall_i
        dict_model_performance_on_test_set["F1_score_"+state] = f1_score_i
        global_f1+=f1_score_i*weights[i]

    dict_model_performance_on_test_set["global_F1"] = global_f1
    df_model_performance_on_test_set = df_model_performance_on_test_set.append(dict_model_performance_on_test_set,ignore_index=True)
    print(blue("F1-score : "+str(global_f1)))
    param_model["global_f1"] = global_f1
    df_model_performance_on_test_set.to_csv(os.path.join(model_path,"results_on_test_set","model_performance_on_test_set.csv"),index=False)

    plot_confusion_matrix_decision(pred_on_each_cell,param_model)
    return df_model_performance_on_test_set, param_model

""""  7. Model selection """

def create_load_models_info(param_model):
    training_set_config_name = param_model["param_training_set"]["training_set_config_name"]
    state_names_with_detected = param_model["param_training_set"]["state_names_with_detected"]

    training_set_info = pd.read_csv(os.path.join(DIR_TRAINING_DATASET,training_set_config_name,"training_set_info.csv"), sep = ";")
    
    model_path = os.path.join(dataset_config.dir_models,training_set_config_name)
    info_models_path = os.path.join(model_path,"info_models.csv")
    if os.path.exists(info_models_path):
        df_models_comparaison = pd.read_csv(info_models_path, sep = ";")
    else :
        colname_model = ["model_id","model_name", "global_f1"]
        colname_df = ["prop_training","prop_validation","normalisation","batch_size","img_size"]
        colname_hyperparameters = ["pooling_steps","epochs","optimizer","initial_learning_rate","callback_reduce_on_plateau","number_of_steps","loss","n_parameters_model"] 
        colname_precision = ["Precision_"+state for state in state_names_with_detected[1:-1]]
        colname_recall = ["Recall_"+state for state in state_names_with_detected[1:-1]]
        colname_f1_score = ["F1_score_"+state for state in state_names_with_detected[1:-1]]
        colname_models_info = colname_model + colname_df + colname_hyperparameters + colname_precision+colname_recall+colname_f1_score
        df_models_comparaison = pd.DataFrame(columns = colname_models_info)
        df_models_comparaison.to_csv(info_models_path, index = False, sep = ";")
    return df_models_comparaison, info_models_path

def fill_model_info(param_model,model,df_model_performance_on_test_set):
    training_set_config_name = param_model["param_training_set"]["training_set_config_name"]
    state_names_with_detected = param_model["param_training_set"]["state_names_with_detected"]
    df_models_comparaison, info_models_path = create_load_models_info(param_model)
    training_set_info = pd.read_csv(os.path.join(DIR_TRAINING_DATASET,training_set_config_name,"training_set_info.csv"), sep = ";")
    id_model_max = df_models_comparaison["model_id"].max()
    dict_models_info = dict()
    for col in training_set_info.columns:
        dict_models_info[col] = training_set_info[col].values[0]

    liste_data_augment = param_model["liste_data_augment"]
    liste_all_data_augment = ["rotate","gamma","hue","crop","flip_hori","flip_vert","brightness","contrast","saturation"]
    for method in liste_all_data_augment:
        if method in liste_data_augment:
            dict_models_info["augment_"+method] = True
        else:
            dict_models_info["augment_"+method] = False
    dict_models_info["model_id"] = id_model_max+1
    dict_models_info["n_train_before_augment"] = param_model["n_before_augment"]
    dict_models_info["n_train_after_augment"] = param_model["n_after_augment"]
    dict_models_info["model_name"] = param_model["model_name"]
    dict_models_info["global_f1"] = param_model["global_f1"]
    dict_models_info["proportion_training"] = param_model["proportion_training"]
    dict_models_info["proportion_validation"] = param_model["proportion_validation"]
    dict_models_info["normalise"] = param_model["normalise"]
    dict_models_info["batch_size"] = param_model["batch_size"]
    dict_models_info["img_size"] = param_model["img_size"]
    dict_models_info["pooling_steps"] = param_model["pooling_steps"]
    dict_models_info["epochs"] = param_model["number_of_epochs"]
    dict_models_info["optimizer"] = "Adam"
    dict_models_info["initial_learning_rate"] = param_model["initial_learning_rate"]
    dict_models_info["callback_reduce_on_plateau"] = ("reduce_lr" in param_model["list_callbacks"])
    dict_models_info["loss"] = "sparse_categorical_crossentropy"
    dict_models_info["n_parameters_model"] = model.count_params()

    for state in state_names_with_detected[1:-1]:
        dict_models_info["Precision_"+state] = df_model_performance_on_test_set.loc[0,"Precision_"+state]
        dict_models_info["Recall_"+state] = df_model_performance_on_test_set.loc[0,"Recall_"+state]
        dict_models_info["F1_score_"+state] = df_model_performance_on_test_set.loc[0,"F1_score_"+state]

    df_models_comparaison = df_models_comparaison.append(dict_models_info, ignore_index = True)
    df_models_comparaison.to_csv(info_models_path, index = False, sep = ";")
    return df_models_comparaison

def grid_search_model(grid_search_param, param_training_set):
    """
    Entraine plein de models 
    """ 
    list_img_size = grid_search_param["list_img_size"]
    list_epochs = grid_search_param["list_epochs"]
    list_batch_size = grid_search_param["list_batch_size"]
    list_pooling_steps = grid_search_param["list_pooling_steps"]
    list_proportion_validation = grid_search_param["list_proportion_validation"]
    liste_liste_data_augment = grid_search_param["liste_liste_data_augment"]

    training_set_config_name = param_training_set["training_set_config_name"]
    num_classes = param_training_set["num_classes"]
    state_names_with_detected = param_training_set["state_names_with_detected"]
    model_training_set_path = os.path.join(dataset_config.dir_models,training_set_config_name)

    experiment_nb = 0
    # Param qui ne varient pas pendant la grid search 
    normalise = True
    batch_size_test = 1
    interval_epoch_callback = 1
    initial_learning_rate = 0.0003
    list_callbacks = ["model_checkpoint","reduce_lr","performance_cbk","sample_img"]
    list_cell_name_callback = ["001-R026-C015-x0675-y0270","004-R041-C060-x0262-y0486","004-R027-C038-x0682-y0777","004-R040-C060-x0331-y0894","002-R014-C052-x0412-y0282"]

    model_id=1
    for idx_proportion_validation,proportion_validation in enumerate(list_proportion_validation):
        for idx_liste_data_augment, liste_data_augment in enumerate(liste_liste_data_augment):
            for idx_epoch, number_of_epochs in enumerate(list_epochs):
                for idx_img_size, img_size in enumerate(list_img_size):
                    for idx_batch_size, batch_size in enumerate(list_batch_size):
                        for idx_pooling_step, pooling_steps in enumerate(list_pooling_steps):
                            param_model = dict()
                            param_model["param_training_set"] = param_training_set
                            param_model["initial_learning_rate"] = initial_learning_rate
                            param_model["normalise"] = normalise
                            param_model["batch_size_test"] = batch_size_test
                            param_model["list_callbacks"] = list_callbacks
                            param_model["interval_epoch_callback"] = interval_epoch_callback
                            param_model["list_cell_name_callback"] = list_cell_name_callback

                            param_model["liste_data_augment"] = liste_data_augment
                            param_model["proportion_validation"] = proportion_validation
                            param_model["proportion_training"] = 1-proportion_validation
                            param_model["img_size"] = img_size
                            param_model["batch_size"] = batch_size
                            param_model["number_of_epochs"] = number_of_epochs
                            param_model["pooling_steps"] = pooling_steps

                            param_model["model_name"] = "epochs_"+str(param_model["number_of_epochs"])+"_pooling_steps_"+str(param_model["pooling_steps"])+"_batch_size_"+str(param_model["batch_size"])+"_img_size_"+str(param_model["img_size"])+"_normalize_"+str(param_model["normalise"])
                            param_model["model_path"] = os.path.join(dataset_config.dir_models,param_model["param_training_set"]["training_set_config_name"],param_model["model_name"])
                            mkdir_if_nexist(param_model["model_path"])
                            print(blue("Entrainement model : "+param_model["model_name"]))
                            train_dataset, validation_data, test_dataset = get_training_validation_test_set(param_training_set,param_model)
                            train_dataset, param_model = apply_data_augment_to_training_set(train_dataset,param_model)
                            model =unet(param_model, pretrained_weights = None, verbose=False)
                            im = plot_model(model, os.path.join(param_model["model_path"],"model_architecture.png"), show_shapes=True)
                            callbacks=prepare_callbacks(model,param_model, validation_data)

                            #Training 
                            history = model.fit(train_dataset,validation_data=validation_data,epochs=param_model["number_of_epochs"], callbacks=callbacks)
                            display_save_loss_accuracy(history,param_model)

                            df_model_performance_on_test_set, param_model = get_model_evaluation(test_dataset,param_model)

                            save_param_model(param_model)
                            make_gifs_from_sample_callback(param_model)
                            df_models_comparaison = fill_model_info(param_model,model,df_model_performance_on_test_set)
                            model_id +=1 

"""" 8. Load and use model """

def save_param_model(param_model):
    path_dict = os.path.join(param_model["model_path"],"dict_param_model.json")
    with open(path_dict, 'w') as f:
        f.write(json.dumps(param_model))
        # print("Saved at {}".format(path_dict))



"""" 9. Other """

def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)
    return x

# Simple normalization to min/max fir the Mask
def normalizeMinMax(x, dtype=np.float32):
  x = x.astype(dtype,copy=False)
  x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
  return x

def weighted_binary_crossentropy(class_weights):
  """
  Permet de calculer l'entropie croisée pondérée avec les poids des classes 
  """
  def _weighted_binary_crossentropy(y_true, y_pred):
    binary_crossentropy = keras.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
    weighted_binary_crossentropy = weight_vector * binary_crossentropy

    return keras.mean(weighted_binary_crossentropy)

  return _weighted_binary_crossentropy
