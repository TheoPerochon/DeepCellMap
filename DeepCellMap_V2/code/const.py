import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(ROOT_DIR)),"outputs","microglial_cells_ihc_dataset")
os.makedirs(BASE_DIR,exist_ok=True)
FONT_PATH = "/Library/Fonts/Arial Bold.ttf"
SUMMARY_TITLE_FONT_PATH = "/Library/Fonts/Courier New Bold.ttf"



TABLE_PHYSIOLOGICAL_PART_GROUP_FOR_COMPARAISON = dict({"striatum": 1, "ganglionic_eminence": 2, "neocortex": 3, "cortical_boundary": 4,"manually_segmented":5})
TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART = dict({"1": "striatum", "2": "ganglionic_eminence", "3": "neocortex", "4": "cortical_boundary", "5":"manually_segmented"})

SCALE_FACTOR_CELLPOSE_TISSUE_SEGMENTATION = 4 
SCALE_FACTOR_HANDLY_DEFINED_TISSUE_SEGMENTATION = 32
DISPLAY_FIG = False


def adaptative_color_lines_transparency(coef_alpha=0.5):
        return dict({"Proliferative":'rgba(173,40,125,'+str(coef_alpha)+')',"Amoeboid":'rgba(79,157,214,'+str(coef_alpha)+')',"Cluster":'rgba(58,180,15,'+str(coef_alpha)+')',"Phagocytic":'rgba(240,186,64,'+str(coef_alpha)+')',"Ramified":'rgba(35,32,179,'+str(coef_alpha)+')',"Detected":'rgba(0,0,0,'+str(coef_alpha)+')',"Detected_purple":'rgba(63,4,126,'+str(coef_alpha)+')'})

TABLE_STATE_INT = dict({"Background":0, "Proliferative":1, "Amoeboid":2, "Cluster":3, "Phagocytic":4, "Ramified":5, "Detected":6, "Detected_purple":7})
TABLE_INT_STATE = dict({0:"Background", 1:"Proliferative", 2:"Amoeboid", 3:"Cluster", 4:"Phagocytic", 5:"Ramified", 6:"Detected", 7:"Detected_purple"})
MODEL_SEGMENTATION_PARAMS = dict({"1":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "2":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":400}),
                                "3":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "4":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "5":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "6":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "7":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "8":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "9":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "10":dict({"model_segmentation_type":"rgb_threshold_based","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "11":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "12":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "13":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "14":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "15":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "16":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "17":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "18":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "19":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "20":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "21":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700}),
                                "22":dict({"model_segmentation_type":"otsu_based_dil2","radius_dilation_during_segmentation":2,"min_size_cell":700})})

MODEL_SEGMENTATION_PARAMS["1"]
### Variables de noms
TISSUE_EXTRACTION_ACCEPT_HOLES = True 
TRAIN_PREFIX = ""
TRAIN_SUFFIX = ""
SRC_TRAIN_EXT = "tif"
DEST_TRAIN_SUFFIX = ""  # Example: "train-"
DEST_TRAIN_EXT = "png"
FILTER_SUFFIX = ""  # Example: "filter-"
FILTER_RESULT_TEXT = "filtered"

PREDICTION_PREFIX = 'Predicted_'

TILE_SUMMARY_SUFFIX = "tile_summary"

TILE_DATA_SUFFIX = "tile_data"

TOP_TILES_SUFFIX = "region_of_interest"  

TILE_SUFFIX = "tile"
THUMBNAIL_EXT = "jpg"
DEST_CROP_EXT = "png"

SUFFIX_EOSIN = "_eosin.png"
SUFFIX_RGB = "_RGB.png"

DIR_FIG_NAME = "figures"
HEATMAP_FOLDER = "Density_heatmaps"
CELL_CLUSTERING_DIRECTORY = "DBSCAN"
NAME_FOLDER_MASK_CELLS = "mask_cells"
NAME_FOLDER_PRED_CELLS = "mask_pred_cells"

######## Data

SRC_TRAIN_DIR = os.path.join(
    BASE_DIR, "1_IHC_slides")  # répertoire contenant les slides/images
STATS_DIR = os.path.join(BASE_DIR, "training_slides_stats")

######## Pré processing

DIR_PRE_PROCESSING = os.path.join(BASE_DIR, "2_Image_pre_processing")

DIR_SCALED_IMAGES = os.path.join(DIR_PRE_PROCESSING, "downscaled_images")

FILTER_HTML_DIR = DIR_PRE_PROCESSING
TILE_SUMMARY_HTML_DIR = DIR_PRE_PROCESSING
DEST_TRAIN_DIR = os.path.join(DIR_SCALED_IMAGES, "training_" + DEST_TRAIN_EXT)
FILTER_DIR = os.path.join(DIR_SCALED_IMAGES, "filter_" + DEST_TRAIN_EXT)
FILTER_THUMBNAIL_DIR = os.path.join(DIR_SCALED_IMAGES,
                                    "filter_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_DIR = os.path.join(DIR_SCALED_IMAGES,
                                "tile_summary_" + DEST_TRAIN_EXT)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(
    DIR_SCALED_IMAGES, "tile_summary_on_original_" + DEST_TRAIN_EXT)
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(
    DIR_SCALED_IMAGES, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(
    DIR_SCALED_IMAGES, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)
DEST_TRAIN_THUMBNAIL_DIR = os.path.join(DIR_SCALED_IMAGES,
                                        "training_thumbnail_" + THUMBNAIL_EXT)
TOP_TILES_DIR = os.path.join(DIR_SCALED_IMAGES,
                             TOP_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
TOP_TILES_THUMBNAIL_DIR = os.path.join(
    DIR_SCALED_IMAGES, TOP_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(
    DIR_SCALED_IMAGES, TOP_TILES_SUFFIX + "_on_original_" + DEST_TRAIN_EXT)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(
    DIR_SCALED_IMAGES,
    TOP_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)

TILE_DIR = os.path.join(DIR_PRE_PROCESSING, "tiles_" + DEST_TRAIN_EXT)
TILE_DATA_DIR = os.path.join(DIR_PRE_PROCESSING, "tile_data")

CROP_DIR = os.path.join(DIR_PRE_PROCESSING, "crop_" + DEST_CROP_EXT)

######## Classification des cellules micro gliales

DIR_CLASSIFICATION_MICROGLIAL_CELLS = os.path.join(
    BASE_DIR, "3_Classification_microglial_cells")

DIR_CLASSIFIED_SLIDES = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,"classified_slides")
TRAINING_MICROGLIA_DETECTION = os.path.join(
    DIR_CLASSIFICATION_MICROGLIAL_CELLS, "training_set_microglia_screening")

DIR_CELLS_PER_SLIDES = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                    "cells_per_slide")

#DIR_MASK_LABELISEE = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS, "mask_labelisees")
DIR_MASK_LABELISEE = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                  "training_dataset_all_categories")
DIR_CELLS_TO_BE_CLASSIFIED = os.path.join(DIR_MASK_LABELISEE,
                                          "to_be_classified")

##PATCH_SOURCE = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,"dataset_training_patches","rgb")
##PATCH_TARGET = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,"dataset_training_patches","mask")

DIR_TRAINING_DATASET = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                    "training_dataset")

TRAINING_SOURCE = os.path.join(DIR_TRAINING_DATASET, "rgb")
TRAINING_MASK = os.path.join(DIR_TRAINING_DATASET, "mask")

DIR_QC_DATASET = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                              "QC_dataset")
DIR_QC_DATASET_RGB = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                  "QC_dataset", "rgb")
DIR_QC_DATASET_MASK = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                   "QC_dataset", "mask")

DIR_UNSEEN_DATA_SOURCE = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                      "unseen_data", "source")
DIR_UNSEEN_DATA_PREDICTION = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS,
                                          "unseen_data", "prediction")

DIR_MODEL = os.path.join(DIR_CLASSIFICATION_MICROGLIAL_CELLS, "models")

#### DIR_ANATOMICAL_REGION_SEGMENTATION

DIR_ANATOMICAL_REGION_SEGMENTATION = os.path.join(BASE_DIR,"4_Anatomical_region_segmentation")

####################### Results #############################


DIR_RESULTS = os.path.join(BASE_DIR,"6_Spatiotemporal_analysis")

DIR_RESULTS_PER_CELL_STATE = os.path.join(DIR_RESULTS,"Results_per_states")


######## Region of interest

DIR_REGION_OF_INTEREST = os.path.join(BASE_DIR, "5_Region_of_interest")
THRESH_MIN_TISSU = 0.05  # Il faut qu'il y ait au moins 5% de tissus dans la tile pour calculer le mask de la microglie
######## Paramètres preprocessing
COEF_CONVERSION_PX_MICROM = 0.45
COEF_SEUILLAGE_CANAL_ROUGE = 2

SCALE_FACTOR = 32

ROW_CROP_SIZE = 256
COL_CROP_SIZE = 256
SIZE_CROP = 256

#SIZE_BORDER_ROI = 512
SIZE_BORDER_ROI = 1024
DISTANCE_MAX_CENTER_OF_MASS_TO_ROI = 70

if SIZE_BORDER_ROI < DISTANCE_MAX_CENTER_OF_MASS_TO_ROI:
    raise ValueError(
        "SIZE_BORDER_ROI should be larger than DISTANCE_MAX_CENTER_OF_MASS_TO_ROI"
    )

JPEG_COMPRESSION_QUALITY = 75

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
# NUM_TOP_TILES = 100

TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

NUM_TILES_TO_DISPLAY_HTML = 50

## Affichage

NUMBER_OF_EXAMPLE_TO_DISPLAY = 3  #10

NB_OF_CSV_TO_SAVE = 30

DISPLAY_TILE_SUMMARY_LABELS = True
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330

THUMBNAIL_SIZE = 300
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True

TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = True

######## Paramètres segmentation

DEFAULT_PARAM_MODEL_SEGMENTATION = {
    "radius_dilation_during_segmentation": 2,
    "min_size_cell": 700
}

######### Paramètres du model de classification


DICT_MODEL_CLASSIFICATION = dict()
DICT_MODEL_CLASSIFICATION["best_model"] = dict({
    "path_folder":
    os.path.join("Prolif_Amoe_Clust_Phag_Rami",
                 "Model_epochs25_batch_size3_crop_size256_pooling_steps4"),
    "path_weights":os.path.join(DIR_MODEL,"Prolif_Amoe_Clust_Phag_Rami","Model_epochs25_batch_size3_crop_size256_pooling_steps4",'historic_weights','weights_last.hdf5'),
    "training_config_name" : "Prolif_Amoe_Clust_Phag_Rami",
    "state_subgroups":
    ["prolif_1+prolif_2", "amoe_1+amoe_2", "clust", "phag", "rami_1+rami_2"],
    "state_names": [
        'Background', 'Proliferative', 'Amoeboid', 'Cluster', "Phagocytic",
        "Ramified"],
    "unet_params" : {
    "number_of_epochs": 20,
    "normalize":True,
    "batch_size": 3,
    "pooling_steps": 4,
    "percentage_validation": 20,
    "initial_learning_rate": 0.0003,
    "patch_width": ROW_CROP_SIZE,
    "patch_height": ROW_CROP_SIZE,
    "number_of_steps": 0,
    }
})

CATEGORIES_MICROGLIAL_CELLS = [
    'amoe_HIGH', 'amoe_LOW', 'prolif_HIGH', 'prolif_LOW', 'rami_HIGH', 'rami_LOW', 'cluster','phag', 'other', 'bad_mask'
]
LABELS_MICROGLIAL_CELLS = ['Background', 'Proliferative', 'Amoeboid', 'Cluster',"Phagocytic","Ramified"]



DICT_PARAM_BY_SLIDE = dict()
DICT_PARAM_BY_SLIDE[1] = dict({
    "radius_dilation_during_segmentation": 2,
    "min_size_cell": 700
})
DICT_PARAM_BY_SLIDE[2] = dict({
    "radius_dilation_during_segmentation": 2,
    "min_size_cell": 700
})
DICT_PARAM_BY_SLIDE[3] = dict({
    "radius_dilation_during_segmentation": 2,
    "min_size_cell": 700
})

LIST_LEVELSETS = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]

TABLE_SLIDE_PCW = [
    "0", "17", "19", "20", "14", "10", "17", "6", "7", "8", "9", "20", "11",
    "xx", "10", "11", "15", "20", "16", "14", "13", "7.5", "12", "38", "23",
    "33", "12", "17", "19", "25", "3d", "1m"]
TABLE_SLIDE_PCW_AVANT_code_reproductibility_TOSUPPR = [
    "0", "5", "12", "4", "14", "10", "17", "6", "7", "8", "9", "20", "11",
    "xx", "10", "11", "15", "20", "16", "14", "13", "7.5", "12", "38", "23",
    "33", "12", "17", "19", "25", "3d", "1m"
]
CATEGOGY_NAMES = dict({
    "amoe_1": "Amoeboid",
    "amoe_2": "Amoeboid",
    'prolif_1': "Proliferative",
    "prolif_2": "Proliferative",
    'rami_1': "Ramified",
    'rami_2': "Ramified",
    'clust': "Cluster",
    'mig_tan': "Migratory",
    'mig_rad': "Migratory",
    'phag': "Phagocytic",
    'other': "Other",
    'bad_mask': "Bad_mask",
    "ball_tail": "Ball Tail",
    "rod": "Rod"
})

# STATISTIQUES

LIST_DEFAULT_RADIUS = [500, 1000, 2000, 3000, 4000]

## Segmentation pink-brown cells

DICT_ROI_MODEL_SEGMENTATION_PINK_BROWN = [[11, 27, 62, 27, 62],
                                          [11, 26, 72, 26, 72],
                                          [1, 27, 15, 27, 15],
                                          [6, 21, 33, 21, 33],
                                          [6, 17, 27, 17, 27]]

LIST_MEAN_CHANELS_SEG_PINK_BROWN_CELLS = [
    78.25242718446601, 59.898058252427184, 62.16019417475728
]
LIST_STD_CHANELS_SEG_PINK_BROWN_CELLS = [
    17.265611169612036, 15.694111693902409, 14.501709573178621
]

### DBSCAN VISUALISATION

COLOR_CONVEX_HULL = dict({
    'All_cells': "xkcd:british racing green",
    'Proliferative': "xkcd:bubble gum pink",
    'Amoeboid': "xkcd:sky",
    'Cluster':"xkcd:lightish green",
    'Phagocytic': 'xkcd:dandelion',
    'Ramified':'xkcd:primary blue'
})

MARKERS_CLUSTERED_CELLS = dict({
    'All_cells': "o",
    'Proliferative': "s",
    'Amoeboid': "X",
    'Cluster': "x",
    'Phagocytic': 'x',
    "Ramified" : 'x'})

### Couleuyrs liées à DBSCAN

COLORS_DBSCAN = [
    "xkcd:green", "xkcd:blue", "xkcd:brown", "xkcd:yellow", "xkcd:red",
    "xkcd:teal", "xkcd:orange", "xkcd:magenta", "xkcd:bright green",
    "xkcd:royal blue", "xkcd:hot pink", "xkcd:bright blue", "xkcd:gold",
    "xkcd:goldenrod", "xkcd:crimson", "xkcd:cerulean", "xkcd:fuchsia",
    "xkcd:aqua blue", "xkcd:dark rose", "xkcd:bright yellow", "xkcd:neon blue",
    "xkcd:yellowish", "xkcd:bright teal", "xkcd:amber", 'xkcd:black',
    'xkcd:very light green', 'xkcd:pastel pink', 'xkcd:bordeaux',
    'xkcd:very light blue', 'xkcd:dark mauve', 'xkcd:kermit green',
    'xkcd:ice blue', 'xkcd:light tan', 'xkcd:dirty green', 'xkcd:neon blue',
    'xkcd:wine red', 'xkcd:chocolate brown', 'xkcd:watermelon', 'black'
]
COLORS_DBSCAN = COLORS_DBSCAN * 30


### Tissue segmentation 

PHYSIOLOGICAL_PARTS = ["ganglionic_eminence", "Striatum", "Cortex_border", "Cortex"]


TABLE_GROUP_FOR_COMPARAISON_TO_PHYSIOLOGICAL_PART = dict({1: "striatum", 2: "ganglionic_eminence", 3: "neocortex", 4: "cortical_boundary"})


TEMPORAL_ANALYSIS_GET_RGB = True
TEMPORAL_ANALYSIS_background = False
TEMPORAL_ANALYSIS_display_convex_hull_clusters = False
TEMPORAL_ANALYSIS_save_figure_B_in_border_levelsets= False

AGGREGATION_METHOD_USE = "area_weighted"
LIST_METRIC_TO_CONVERT_IN_MICROMETER = ["coloc_delta_a","coloc_delta_a_proba","coloc_state_border_delta_a","coloc_state_border_delta_a_proba","neighbours_r_coloc","neighbours_mean_dist_first_B_around_A","neighbours_mean_dist_second_B_around_A","neighbours_mean_dist_third_B_around_A","neighbours_mean_A_first_neighbour","neighbours_mean_A_second_neighbour","neighbours_mean_A_third_neighbour"]


#### Tables conversions
DICT_PHYSIOLOGICAL_PART_MAX_SQUARE_SIZE = {"striatum":100,"ganglionic_eminence":100,"cortical_boundary":50,"neocortex":50,"manually_segmented":50}

TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME = dict({
    "striatum":"Striatum",
    "ganglionic_eminence":"Ganglionic eminence",
    "cortical_boundary":"Cortex border",
    "neocortex":"Cortex",
})

### Colors 

def adaptative_color_lines_transparency(coef_alpha=0.5):
        return dict({"Proliferative":'rgba(173,40,125,'+str(coef_alpha)+')',"Amoeboid":'rgba(79,157,214,'+str(coef_alpha)+')',"Cluster":'rgba(58,180,15,'+str(coef_alpha)+')',"Phagocytic":'rgba(240,186,64,'+str(coef_alpha)+')',"Ramified":'rgba(35,32,179,'+str(coef_alpha)+')',"Other":'rgba(63,4,126,'+str(coef_alpha)+')'})


color_discrete_map = dict({
        "Proliferative": [255, 255, 255],
        "Amoeboid": [23, 29, 71],
        "Cluster": [64, 140, 186],
        "Phagocytic": [188, 205, 210],
        "Ramified": [198, 107, 80]
    })

def adaptative_color_tissue_segmentation(coef_alpha=0.5):
        return dict({"ganglionic_eminence":'rgba(195,121,54,'+str(coef_alpha)+')',"striatum":'rgba(101,170,215,'+str(coef_alpha)+')',"cortical_boundary":'rgba(131,208,104,'+str(coef_alpha)+')',"neocortex":'rgba(228,99,160,'+str(coef_alpha)+')'})

line_dash_map=dict({
        "striatum": "solid",
        "ganglionic_eminence": "dash",
        "cortical_boundary": "dashdot",
        "neocortex": "dot"
    })

symbol_map = dict({
        "striatum": "star",
        "ganglionic_eminence": "star",
        "cortical_boundary": "star",
        "neocortex": "star"    })


