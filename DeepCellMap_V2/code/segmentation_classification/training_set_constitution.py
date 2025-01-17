
#Importation des librairies 
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import os
#Traitement des SCANS 
from PIL import Image

from segmentation_classification.region_of_interest import RegionOfInterest
from os import listdir
#from pigeon import annotate
import pigeon
import random
import skimage.morphology as sk_morphology 
from scipy.spatial import distance


from preprocessing import filter 
from preprocessing import slide 
from preprocessing import tiles 
from utils.util import *
from tqdm.notebook import tqdm
#from python_files.const import *
from config.html_generation_config import HtmlGenerationConfig
# from config.base_config import BaseConfig
# from config.datasets_config import *
# from config.html_generation_config import HtmlGenerationConfig

# base_config = BaseConfig()
# # Get the configuration
# if base_config.dataset_name == "ihc_microglia_fetal_human_brain_database":
#     dataset_config = IhcMicrogliaFetalHumanBrain()
# elif base_config.dataset_name == "cancer_data_immunofluorescence":
#     dataset_config = FluorescenceCancerConfig()



def info_training_set():
    """
    Permet de connaitre le nombre d'exempels dans chaque catégorie du dataset d'entrainement 
    """
    print("Nombre d'exemple dans le training dataset contenant toutes les catégories")
    # TODO : mettre les noms de dossier du dossier dir mask lavelise

    tot_masks = 0 
    for nom_categorie in dataset_config.cells_on_tissue["all_cells_list"]:
        path_categorie = os.path.join(dataset_config.dir_training_set_cells, nom_categorie)
        if not os.path.exists(path_categorie):
            os.makedirs(path_categorie)
        path_dir_categorie_mask = os.path.join(path_categorie,"mask")
        nb = len(os.listdir(path_dir_categorie_mask)) 
        print(nom_categorie + ": " + str(nb))
        tot_masks += nb
    print(str(tot_masks) + " cells are in the training dataset")

def show_exemples_categorie(categorie_name, how_many = 5):
    """
    Permet de visualiser how_many d'image RGB, MASK et EOSIN de la categorie categorie_name du training dataset 
    """

    path_dir_training_dataset = os.path.join(dataset_config.dir_training_set_cells,categorie_name)

    list_examples_rgb = [os.path.join(path_dir_training_dataset,"rgb",f) for f in os.listdir(os.path.join(path_dir_training_dataset,"rgb"))]
    list_examples_mask = [os.path.join(path_dir_training_dataset,"mask",f) for f in os.listdir(os.path.join(path_dir_training_dataset,"mask"))]
    nb_training_examples = len(list_examples_rgb)
    print("There is "+ str(nb_training_examples) + " examples in categorie "+categorie_name )

    examples_to_display = random.sample(set(np.arange(nb_training_examples)), k=min(nb_training_examples,how_many) )
    for idx in examples_to_display:
        path_rgb_example = list_examples_rgb[idx]
        path_mask_example = list_examples_mask[idx]
        dir, filename_rgb = os.path.split(path_rgb_example)
        dir, filename_mask = os.path.split(path_mask_example)

        rgb = Image.open(path_rgb_example)
        rgb = np.asarray(rgb)
        mask = Image.open(path_mask_example)
        mask = np.asarray(mask)
        eosin = rgb_to_eosin(rgb)

        display_rgb_mask_eosin(rgb,mask,eosin,filename_rgb,filename_mask,figsize = (17,8))

def _display_img_pigeon(mask_path):
    """
    Used in attribute_good_label
    Cette fonction recoit en entrée le nom du mask (dans le dataset d'entrainement) et display les images RGB, Mask, et Eosin 
    """
    dir, filename_mask = os.path.split(mask_path)

    filename_rgb = filename_mask[:-8]+"rgb.png"
    path_rgb_example = os.path.join(os.path.split(dir)[0],"rgb",filename_rgb)

    rgb = Image.open(path_rgb_example)
    rgb = np.asarray(rgb)
    mask = Image.open(mask_path)
    mask = np.asarray(mask)
    eosin = rgb_to_eosin(rgb)

    a = display_rgb_mask_eosin(rgb,mask,eosin,title1="RGB\n" + filename_mask[:-9],title2="Mask Microglia\n" + filename_mask[:-9],title3="Eosin\n" + filename_mask[:-9], figsize=(17,8))

    slide_num, tile_row, tile_col, x_coord, y_coord = decompose_filename(filename_mask)
    print("To see the entire tile: labeling.display_tile(" + str(slide_num) + "," + str(tile_row) + "," + str(tile_col) + ")")

def _display_new_mask_pigeon(mask_path):
    """
    Propose plusieurs nouveaux masks a choisir. Prend un nom de chemin de type training_dataset_all_categories/amoe_1/mask/021-R010-C009-x1021-y0990_mask.png
    """
    dir, filename_mask = os.path.split(mask_path)

    filename_rgb = filename_mask[:-8]+"rgb.png"
    path_rgb_example = os.path.join(os.path.split(dir)[0],"rgb",filename_rgb)

    rgb = Image.open(path_rgb_example)
    rgb = np.asarray(rgb)
    mask = Image.open(mask_path)
    mask = np.asarray(mask)
    eosin = rgb_to_eosin(rgb)
    
    list_new_masks = []

    new_mask = filter.segment_microglia_v3(rgb, min_cell_size = 500,dilation_radius = 2)
    new_mask = sk_morphology.label(new_mask)
    intersection = np.logical_and(mask,new_mask)
    number_good_cell = np.max(np.where(intersection,new_mask,0))
    list_new_masks.append(np.where(new_mask == number_good_cell,new_mask,0 ))

    new_mask = filter.segment_microglia_v3(rgb, min_cell_size = 500,dilation_radius = 3)
    new_mask = sk_morphology.label(new_mask)
    intersection = np.logical_and(mask,new_mask)
    number_good_cell = np.max(np.where(intersection,new_mask,0))
    list_new_masks.append(np.where(new_mask == number_good_cell,new_mask,0 ))

    for dilation_radius in range(1,5):
        new_mask = filter.filter_microglia(rgb, dilation_radius = dilation_radius, min_cell_size = 200)
        new_mask = sk_morphology.label(new_mask)
        intersection = np.logical_and(mask,new_mask)
        number_good_cell = np.max(np.where(intersection,new_mask,0))
        list_new_masks.append(np.where(new_mask == number_good_cell,new_mask,0 ))


    display_rgb_mask_mask_mask_mask_mask(rgb,eosin,list_new_masks[0],list_new_masks[1],list_new_masks[2],list_new_masks[3],list_new_masks[4],list_new_masks[5],"rgb","eosin","New method dil2","New method dil3","param = 1","param = 2","param = 3","param = 4", figsize = (19,10))
        
def _transform_and_replace_masks(annotations):
    """
    Used in move_annotation
    Transform the mask according to the list (output of annotate) and place masks and rgb to the directory "to_be_classified" 
    """
    mkdir_if_nexist(os.path.join(dataset_config.dir_training_set_cells, "to_be_classified"))
    path_dir_dest_rgb = os.path.join(dataset_config.dir_training_set_cells, "to_be_classified","rgb")
    path_dir_dest_mask = os.path.join(dataset_config.dir_training_set_cells, "to_be_classified","mask")
    mkdir_if_nexist(path_dir_dest_rgb)
    mkdir_if_nexist(path_dir_dest_mask)


    for idx_annotation in range(len(annotations)):
        path_mask, new_param = annotations[idx_annotation]

        dir, filename_mask = os.path.split(path_mask)
        dir = os.path.split(dir)[0]
        path_rgb = os.path.join(dir, 'rgb',filename_mask[:-8]+"rgb.png")
        rgb = Image.open(path_rgb)
        rgb = np.asarray(rgb)
        mask = Image.open(path_mask)
        mask = np.asarray(mask)

        if "new_method" in new_param:
            param_dil = int(new_param[-1])
            new_mask = filter.segment_microglia_v3(rgb, dilation_radius= param_dil)
            new_mask = sk_morphology.label(new_mask)
            intersection = np.logical_and(mask,new_mask)
            number_good_cell = np.max(np.where(intersection,new_mask,0))
            new_mask = np.where(new_mask == number_good_cell,new_mask,0 )
            new_mask = new_mask.astype(np.uint8)

        else : 
            new_param = int(new_param[-1])
            new_mask = filter.filter_microglia(rgb, dilation_radius = new_param, min_cell_size = 200)
            new_mask = sk_morphology.label(new_mask)
            intersection = np.logical_and(mask,new_mask)
            number_good_cell = np.max(np.where(intersection,new_mask,0))
            new_mask = np.where(new_mask == number_good_cell,new_mask,0 )
            new_mask=new_mask.astype(np.uint8)

        path_dest_rgb = os.path.join(path_dir_dest_rgb,filename_mask[:-8]+"rgb.png")
        path_dest_mask = os.path.join(path_dir_dest_mask,filename_mask)
        new_mask_img = Image.fromarray(new_mask * 255)
        new_mask_img.save(path_dest_mask)
        shutil.move(path_rgb, path_dest_rgb)
        os.remove(path_mask)

def attribute_good_label(nom_categorie,how_many = None):
    '''
    Cette fonction permet de gérer 3 cas de figure :
    - nom_categorie = "first" : permet de classifier pour la première fois des cellules qui ont été ajoutées à "to_be_classified" grace a la fonction find_cells_to_classify
    - nom_categorie = "bad_mask" : permet de modifier le mask des cellules placées dans cette catégorie et les renvoie dans la catégorie "to_be_classified"
    - nom_categorie = <label_name> permet de déplacer (éventuellement) les cellules déjà classées dans la catégorie <label_name> pour les mettre dans une autre catégorie 
    
    
    Renvoie le pairing {filename, 'catégorie'}
    '''

    path_masks_categorie = os.path.join(dataset_config.dir_training_set_cells, nom_categorie)
    list_masks = supprimer_DS_Store(os.listdir(os.path.join(path_masks_categorie,"mask")))
    list_masks.sort()
    list_path_masks = [os.path.join(path_masks_categorie,"mask",f) for f in list_masks ]
    print("There is " + str(len(list_path_masks)) + " cells in the categorie " + nom_categorie)
    if how_many is not None:
        list_path_masks = list_path_masks[0:min(len(list_path_masks),how_many)]
    print(str(len(list_path_masks)) + " of them will be classified")
    if nom_categorie == "bad_mask":
        annotations = pigeon.annotate(
            list_path_masks,
            options=["new_methode_dil2","new_methode_dil3","param = 1", "param = 2", "param = 3", "param = 4"],
            shuffle=False,
            display_fn=lambda filename_mask: _display_new_mask_pigeon(
                filename_mask))
        return annotations
    else:
        annotations = pigeon.annotate(list_path_masks,
                               options=dataset_config.cells_on_tissue["all_cells_list"],
                               shuffle=False,
                               display_fn=lambda filename_mask:
                               _display_img_pigeon(filename_mask))
    return annotations

def move_annotations(annotations):
    """
    Take the result of the function annotate and move masks and rgb images to the new label

    If annotate is new masks param, la fonction _transform_and_replace_masks est appelée, elle créer les nouveaux masks et les places dans la catégorie to_be_classified 
    """
    print("number of example to move : " + str(len(annotations)) )
    
    if "param" in annotations[0][1] or "new_method" in annotations[0][1]:
        return _transform_and_replace_masks(annotations)
    
    for idx in range(len(annotations)):
        path_mask, label = annotations[idx]
        dir, filename_mask = os.path.split(path_mask)
        dir = os.path.split(dir)[0]
        path_rgb = os.path.join(dir, 'rgb',filename_mask[:-8]+"rgb.png")

        dest_path_mask = os.path.join(dataset_config.dir_training_set_cells,label,'mask') 
        dest_path_rgb = os.path.join(dataset_config.dir_training_set_cells,label,'rgb')
        mkdir_if_nexist(dest_path_rgb)
        mkdir_if_nexist(dest_path_mask)

        dest_path_mask = os.path.join(dest_path_mask,filename_mask)
        dest_path_rgb = os.path.join(dest_path_rgb,filename_mask[:-8]+"rgb.png")

        shutil.move(path_mask, dest_path_mask)
        shutil.move(path_rgb, dest_path_rgb)

def _compute_rgb_mask_crop_unique_cell(RegionOfInterest,idx_cell):
    """
    Used in find_cells_to_classify
    Compute the rgb crop and microglial mask of cell of idx idx_cell. The original mask from regionofinterest.crop_mask_microglia_cell(x,y,crop_radius=size_big_crop) is processed
    so that it remains only the cell of the center 
    """
    x,y = RegionOfInterest.table_cells["x_coord"][idx_cell],RegionOfInterest.table_cells["y_coord"][idx_cell]
    length_max_cell = RegionOfInterest.table_cells['length_max'][idx_cell]
    size_big_crop = max(length_max_cell +100,dataset_config.crop_size)
    bigger_mask_crop = RegionOfInterest.crop_mask_microglia_cell(x,y,crop_radius=size_big_crop)

    bigger_mask_crop_label = sk_morphology.label(bigger_mask_crop)
    complete_cell_of_interest = np.zeros_like(bigger_mask_crop)

    for label in range(1,np.max(bigger_mask_crop_label)+1):
        cell_i_in_crop = np.where(bigger_mask_crop_label == label, 1,0).astype(np.uint8)
        center_x = int(np.mean(np.where(cell_i_in_crop == 1)[0]))
        center_y = int(np.mean(np.where(cell_i_in_crop == 1)[1])) 
        distance_to_the_center = distance.euclidean((center_x,center_y), ((int(size_big_crop/2),int(size_big_crop/2))))

        if distance_to_the_center == 0 :
            complete_cell_of_interest+= cell_i_in_crop
            break 
    mask_crop_unique_cell = complete_cell_of_interest[int(complete_cell_of_interest.shape[0]/2-dataset_config.crop_size/2):int(complete_cell_of_interest.shape[0]/2+dataset_config.crop_size/2),int(complete_cell_of_interest.shape[1]/2-dataset_config.crop_size/2):int(complete_cell_of_interest.shape[1]/2+dataset_config.crop_size/2)]
    rgb_crop = RegionOfInterest.crop_rgb_around_cell(x,y)

    return rgb_crop, mask_crop_unique_cell, x, y 

def _compute_filename_cell_rgb_mask(RegionOfInterest,slide_num,tile_row,tile_col,coord_x_in_tile,coord_y_in_tile):
    """
    Compute name of mask and rgb crop of a cell from slide_num,tile_row,tile_col,coord_x_in_tile,coord_y_in_tile. Ex : 006-R021-C035-x0609-y0565_mask.png and 006-R021-C035-x0609-y0565_rgb.png
    """
    filename_rgb = os.path.join(str(slide_num).zfill(3) + "-R" + str(tile_row).zfill(3) + "-C" +str(tile_col).zfill(3) + "-x" + str(coord_x_in_tile).zfill(4) + "-y" + str(coord_y_in_tile).zfill(4) + "_rgb.png")
    # filename_mask = os.path.join(str(slide_num).zfill(3) + "-R" + str(tile_row).zfill(3) + "-C" +str(tile_col).zfill(3) + "-x" + str(coord_x_in_tile).zfill(4) + "-y" + str(coord_y_in_tile).zfill(4) + "_mask_p" +str(RegionOfInterest.model_segmentation.dilation_radius)+".png")
    filename_mask = os.path.join(str(slide_num).zfill(3) + "-R" + str(tile_row).zfill(3) + "-C" +str(tile_col).zfill(3) + "-x" + str(coord_x_in_tile).zfill(4) + "-y" + str(coord_y_in_tile).zfill(4) + "_mask.png")

    return filename_rgb, filename_mask
    

def _compute_coords_in_tile_from_roi(RegionOfInterest, x_ROI_enlarged,y_ROI_enlarged):
    """
    Used in find_cells_to_classify
    Return tile_row,tile_col and x_coord_in_tile, y_coord_in_tile 
    from RegionOfInterest object and x,y coords of the cell in the ROI 
    """

    x= x_ROI_enlarged - dataset_config.roi_border_size
    y= y_ROI_enlarged - dataset_config.roi_border_size
    tile_row = RegionOfInterest.origin_row+ x//dataset_config.tile_height
    tile_col = RegionOfInterest.origin_col+ y//dataset_config.tile_width

    coord_x_in_tile = x % dataset_config.tile_height
    coord_y_in_tile = y % dataset_config.tile_width

    coord_x_in_tile_enlarged = coord_x_in_tile + dataset_config.roi_border_size
    coord_y_in_tile_enlarged = coord_y_in_tile + dataset_config.roi_border_size


    return tile_row,tile_col,coord_x_in_tile_enlarged,coord_y_in_tile_enlarged

def find_cells_to_classify(slide_num, origin_row, origin_col, end_row = None,end_col = None, tx_of_cells_to_classify = 0.3,verbose = 0):
    """
    Ajoute tx_of_cells_to_classify % des cellules de la ROI catactérisée par slide_num, origin_row, origin_col, end_row,end_col dans le directory 
    "to_be_classified/mask" et "to_be_classified/rgb". 

    """
    if end_row is None and end_col is None :
        end_row = origin_row
        end_col = origin_col
    mkdir_if_nexist(dataset_config.to_be_classified)
    mkdir_if_nexist(os.path.join(dataset_config.to_be_classified, "rgb"))
    mkdir_if_nexist(os.path.join(dataset_config.to_be_classified, "mask"))

    regionofinterest = RegionOfInterest(slide_num, origin_row, origin_col, end_row,end_col, get_images = False, verbose = verbose)
    regionofinterest.get_rgb_wsi(verbose = verbose)

    regionofinterest.get_table_cells(verbose = verbose)

    nb_cells_to_classify = int(regionofinterest.nb_cells*tx_of_cells_to_classify)
    print(str(nb_cells_to_classify) + " cells will be add to the directory"+dataset_config.to_be_classified)
    list_cells_to_be_classified = random.sample(set(np.arange(regionofinterest.nb_cells)), k=nb_cells_to_classify)
    print("Le paramètre de la segmentatio vaut p = ",regionofinterest.model_segmentation.dilation_radius)
    for idx_cell in tqdm(list_cells_to_be_classified):
        #print("Indice de la cellule en cours de traitement :", idx_cell) if verbose >=2 else None 
        rgb_crop, mask_crop_unique_cell, x, y = _compute_rgb_mask_crop_unique_cell(regionofinterest,idx_cell)
        tile_row,tile_col,coord_x_in_tile,coord_y_in_tile = _compute_coords_in_tile_from_roi(regionofinterest, x,y)

        filename_rgb, filename_mask = _compute_filename_cell_rgb_mask(regionofinterest,regionofinterest.slide_num,tile_row,tile_col,coord_x_in_tile,coord_y_in_tile)

        path_rgb = os.path.join(dataset_config.to_be_classified, "rgb",filename_rgb)
        path_mask = os.path.join(dataset_config.to_be_classified, "mask",filename_mask)

        display_rgb_mask(rgb_crop,mask_crop_unique_cell) if verbose >= 5 else None 

        rgb_img = np_to_pil(rgb_crop)
        rgb_img.save(path_rgb)

        mask_img = np_to_pil(mask_crop_unique_cell*255)
        mask_img.save(path_mask)  

def display_tile(slide_num, tile_row_number, tile_col_number, display = True):

    """
    A partir des infos de base d'une tile, renvoie le chemin vers le png du tile 
    Si Display = True, display la tile 
    """

    padded_sl_num = str(slide_num).zfill(3)
    dossier_tile_png = os.path.join(dataset_config.preprocessing_config.preprocessing_path["dir_tiles"] , padded_sl_num)

    liste_noms = [dossier_tile_png+"/"+f for f in listdir(dossier_tile_png)]

    prefix_nom_fichier = padded_sl_num+"-"+dataset_config.preprocessing_config.tile_suffix+"-r"+str(tile_row_number)+"-c"+str(tile_col_number) 
    
    for noms in liste_noms:
        if prefix_nom_fichier in noms: 
            path_image = noms 

    #print(prefix_nom_fichier)

    if display : 
        """
        ERROR si la tile n'est pas dans le dossier tile_png"""
        til_rgb = Image.open(path_image)
        til_rgb = np.asarray(til_rgb)  

        fig = plt.figure(figsize = (8,8))
        plt.imshow(til_rgb)

        ax = plt.gca()
        ax = plt.gca()

        # Major ticks
        ax.set_xticks(np.arange(0, dataset_config.tile_height, dataset_config.crop_size));
        ax.set_yticks(np.arange(0, dataset_config.tile_height, dataset_config.crop_size));

        # Gridlines based on minor ticks
        ax.grid(which='major', color='w', linestyle='-', linewidth=1)


