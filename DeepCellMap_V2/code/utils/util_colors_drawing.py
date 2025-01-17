#Utils for colors and drawings 

# Colors section is structured like this :
# 1) Dictionnaries of colors for statistics graphs 
# 2) Dictionnaries of colors for images and drawings
# 2) Functions to create dictionnaries of colors
# 3) Functions to create colormap
# 4) Functions to draw on images
from matplotlib import cm
import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt 
from simple_colors import *
from shapely.geometry import (
    MultiPoint,
    MultiPolygon,
    polygon,
    Point,
    LineString,
    Polygon,
)
from shapely import geometry
from rasterio import features, Affine
from matplotlib.colors import ListedColormap

"""
Usefull links
Couleurs matplotlib (étendues) 
https://xkcd.com/color/rgb/tile_coord
https://coolors.co/palettes/search
https://eltos.github.io/gradient/#1F4E5A-15:029C8E-FFDB69-85:FFA658-EA5F40%22%22%22

"""


COLORS_STATES_PLOTLY = [
    "rgb(0,0,0)",
    "rgb(173,40,125)",
    "rgb(79,157,214)",
    "rgb(58,180,15)",
    "rgb(240,186,64)",
    "rgb(35,32,179)",
]
# COLORS_STATES_PLOTLY = ['rgb(0,0,0)','rgb(35,32,179)','rgb(35,32,179)','rgb(35,32,179)','rgb(35,32,179)','rgb(35,32,179)' ]

COLORS_STATES_PLOTLY_TRANSPARENCY = [
    "rgba(0,0,0,0.5)",
    "rgba(173,40,125,0.5)",
    "rgba(79,157,213,0.5)",
    "rgba(58,180,15,0.5)",
    "rgba(240,186,64,0.5)",
    "rgba(35,32,179,0.5)",
]



dict_colors = dict({
    "Expo_colors_2" : dict({
        "proliferative": [0.176*255, 0.604*255, 0.318*255],
        "amoeboid": [0.490*255, 0.537*255, 0.082*255],
        "aggregated": [0.627*255, 0.459*255, 0.212*255],
        "phagocytic": [0.675*255, 0.388*255, 0.404*255],
        "ramified": [0.631*255, 0.361*255, 0.541*255],
        "other" : [63, 4, 126]
    }),
    "Expo_colors_3" : dict({
        "proliferative": [112, 214, 255],
        "amoeboid": [255, 112, 166],
        "aggregated": [255, 151, 112],
        "phagocytic": [255, 214, 112],
        "ramified": [233, 255, 112],
        "other" : [63, 4, 126]
    }),
    "Expo_colors_4" : dict({
        "proliferative": [255, 89, 94],
        "amoeboid": [255, 202, 58],
        "aggregated": [138, 201, 38],
        "phagocytic": [25, 130, 196],
        "ramified": [25, 130, 196],
        "other" : [63, 4, 126]
    }),
    "Expo_colors_5" : dict({
        "proliferative": [84, 13, 110],
        "amoeboid": [238, 66, 102],
        "aggregated": [255, 210, 63],
        "phagocytic": [59, 206, 172],
        "ramified": [238, 66, 102],
        "other" : [63, 4, 126]
    }),
    "Expo_colors_6" : dict({
        "proliferative": [211, 248, 226],
        "amoeboid": [228, 193, 249],
        "aggregated": [246, 148, 193],
        "phagocytic": [237, 231, 177],
        "ramified": [169, 222, 249],
        "other" : [63, 4, 126]
    }),
    "Expo_colors_7" : dict({
        "proliferative": [],
        "amoeboid": [],
        "aggregated": [],
        "phagocytic": [],
        "ramified": [],
        "other" : [63, 4, 126]
    }),
})


#Dictionaries creation 
def adaptative_color_lines_transparency(coef_alpha=0.5):
    """Vient de const.py"""
    return dict(
        {
            "Proliferative": "rgba(173,40,125," + str(coef_alpha) + ")",
            "Amoeboid": "rgba(79,157,214," + str(coef_alpha) + ")",
            "Cluster": "rgba(58,180,15," + str(coef_alpha) + ")",
            "Phagocytic": "rgba(240,186,64," + str(coef_alpha) + ")",
            "Ramified": "rgba(35,32,179," + str(coef_alpha) + ")",
            "Detected": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Detected_purple": "rgba(63,4,126," + str(coef_alpha) + ")",
            "iba1": "rgba(255,255,255," + str(coef_alpha) + ")",
            "blood_vessels": "rgba(58,180,15," + str(coef_alpha) + ")",
            "cd68":"rgba(255,0,0," + str(coef_alpha) + ")",
            "iba1_cd68":"rgba(240, 186, 64," + str(coef_alpha) + ")",
        }
    )
def adaptative_color_classification(coef_alpha=0.5):
    return dict(
        {
            "Proliferative": "rgba(173,40,125," + str(coef_alpha) + ")",
            "Amoeboid": "rgba(79,157,214," + str(coef_alpha) + ")",
            "Cluster": "rgba(58,180,15," + str(coef_alpha) + ")",
            "Phagocytic": "rgba(240,186,64," + str(coef_alpha) + ")",
            "Ramified": "rgba(35,32,179," + str(coef_alpha) + ")",
            "Detected": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Detected_purple": "rgba(63,4,126," + str(coef_alpha) + ")",
            "Both_bc_symetrie": "rgba(0,0,0," + str(coef_alpha) + ")",
        }
    )

def degrade_colors_neighbors_analysis():
    return dict(
        {
            "mean_dist_1_first_cell_around_A": "rgba(35,32,179," + str(1) + ")",
            "mean_dist_2_first_cell_around_A": "rgba(35,32,179," + str(0.75) + ")",
            "mean_dist_3_first_cell_around_A": "rgba(35,32,179," + str(0.5) + ")",
            "mean_dist_4_first_cell_around_A": "rgba(35,32,179," + str(0.3) + ")",
            "mean_dist_5_first_cell_around_A": "rgba(35,32,179," + str(0.2) + ")",
        }
    )


def seconde_line_colors(coef_alpha=0.5):
    return dict(
        {
            "Proliferative": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Amoeboid": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Cluster": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Phagocytic": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Ramified": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Detected": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Detected_purple": "rgba(63,4,126," + str(coef_alpha) + ")",
        }
    )



def from_color_list_to_dict(liste_color_from_eltos, coef_alpha=125):
    dict_colormap = dict()
    for i in range(len(liste_color_from_eltos)):
        dict_colormap[i] = (
            "rgba("
            + str(int(liste_color_from_eltos[i][1][0] * 255))
            + ","
            + str(int(liste_color_from_eltos[i][1][1] * 255))
            + ","
            + str(int(liste_color_from_eltos[i][1][2] * 255))
            + ","
            + str(coef_alpha)
            + ")"
        )
    return dict_colormap

LIST_NUCLEI_COLORS = [
    (0.000, (0.122, 0.306, 0.353)),
    (0.031, (0.008, 0.612, 0.557)),
    (0.063, (1.000, 0.859, 0.412)),
    (0.094, (1.000, 0.651, 0.345)),
    (0.125, (0.918, 0.373, 0.251)),
    (0.156, (1.000, 0.651, 0.345)),
    (0.188, (1.000, 0.859, 0.412)),
    (0.219, (0.008, 0.612, 0.557)),
    (0.250, (0.122, 0.306, 0.353)),
    (0.281, (0.008, 0.612, 0.557)),
    (0.313, (1.000, 0.859, 0.412)),
    (0.344, (1.000, 0.651, 0.345)),
    (0.375, (0.918, 0.373, 0.251)),
    (0.406, (1.000, 0.651, 0.345)),
    (0.438, (1.000, 0.859, 0.412)),
    (0.469, (0.008, 0.612, 0.557)),
    (0.500, (0.122, 0.306, 0.353)),
    (0.531, (0.008, 0.612, 0.557)),
    (0.563, (1.000, 0.859, 0.412)),
    (0.594, (1.000, 0.651, 0.345)),
    (0.625, (0.918, 0.373, 0.251)),
    (0.656, (1.000, 0.651, 0.345)),
    (0.688, (1.000, 0.859, 0.412)),
    (0.719, (0.008, 0.612, 0.557)),
    (0.750, (0.122, 0.306, 0.353)),
    (0.781, (0.008, 0.612, 0.557)),
    (0.813, (1.000, 0.859, 0.412)),
    (0.844, (1.000, 0.651, 0.345)),
    (0.875, (0.918, 0.373, 0.251)),
    (0.906, (1.000, 0.651, 0.345)),
    (0.938, (1.000, 0.859, 0.412)),
    (0.969, (0.008, 0.612, 0.557)),
    (1.000, (0.122, 0.306, 0.353)),
]
DICT_COLORMAP_CELLPOSE = from_color_list_to_dict(LIST_NUCLEI_COLORS, coef_alpha=255)

def adaptative_color_lines_transparency(coef_alpha=0.5):
    return dict(
        {
            "Proliferative": (173, 40, 125, coef_alpha),
            "Amoeboid": (79, 157, 214, coef_alpha),
            "Cluster": (58, 180, 15, coef_alpha),
            "Phagocytic": (240, 186, 64, coef_alpha),
            "Ramified": (35, 32, 179, coef_alpha),
            "Other": (63, 4, 126, coef_alpha),
        }
    )
def adaptative_color_lines_transparency(coef_alpha=0.5):
    """Vient de const.py"""
    return dict(
        {
            "Proliferative": "rgba(173,40,125," + str(coef_alpha) + ")",
            "Amoeboid": "rgba(79,157,214," + str(coef_alpha) + ")",
            "Cluster": "rgba(58,180,15," + str(coef_alpha) + ")",
            "Phagocytic": "rgba(240,186,64," + str(coef_alpha) + ")",
            "Ramified": "rgba(35,32,179," + str(coef_alpha) + ")",
            "Detected": "rgba(0,0,0," + str(coef_alpha) + ")",
            "Detected_purple": "rgba(63,4,126," + str(coef_alpha) + ")",
            "Other":  "rgba(63,4,126," + str(coef_alpha) + ")",
            "iba1": "rgba(255,255,255," + str(coef_alpha) + ")",
            "blood_vessels": "rgba(58,180,15," + str(coef_alpha) + ")",
            "cd68":"rgba(255,0,0," + str(coef_alpha) + ")",
            "iba1_cd68":"rgba(240, 186, 64," + str(coef_alpha) + ")",
        }
    )
def create_colors_cell_drawing_ihc_microglia(coef_alpha=125):
    """Used to color cell masks in drawings for microglia 
    """
    return dict(
        {
            "proliferative": (173, 40, 125, coef_alpha),
            "amoeboid": (79, 157, 214, coef_alpha),
            "aggregated": (58, 180, 15, coef_alpha),
            "phagocytic": (240, 186, 64, coef_alpha),
            "ramified": (35, 32, 179, coef_alpha),
            "other": (63, 4, 126, coef_alpha),
        }
    )

def create_colors(coef_alpha, dict_colors_cells): 
    dict_colors = dict()
    for cell_name in dict_colors_cells.keys():
        dict_colors[cell_name] = (
            int(dict_colors_cells[cell_name][0]),
            int(dict_colors_cells[cell_name][1]),
            int(dict_colors_cells[cell_name][2]),
            coef_alpha
        )
    return dict_colors


def create_colors_cell_drawing(coef_alpha=125):
    """Used to color cell masks in drawings 
    """
    return dict(
        {
            "iba1": (255, 255, 255, coef_alpha),#IBA1 -> White slide 2 
            "blood_vessels": (58, 180, 15, coef_alpha),
            "cd68": (255, 0, 0, coef_alpha),#Phagocytic sur image 2 (CD68 -> Red) 
            "iba1_cd68": (240, 186, 64, coef_alpha),#Combo RED + WHITE -> Orange
            "type_5": (35, 32, 179, coef_alpha),
            "type_6": (63, 4, 126, coef_alpha),
            "iba1": (255, 255, 255, coef_alpha), #IBA1 -> White slide 2 
            "blood_vessels": (58, 180, 15, coef_alpha), #Blood vessels on image 2 
            "cd68": (255, 0, 0, coef_alpha),#Phagocytic sur image 2 (CD68 -> Red) 
            "iba1_cd68": (240, 186, 64, coef_alpha),#Combo RED + WHITE -> Orange
            "from_channel_5": (35, 32, 179, coef_alpha),
            "from_channel_6": (63, 4, 126, coef_alpha),
            "proliferative": (173, 40, 125, coef_alpha),
            "amoeboid": (79, 157, 214, coef_alpha),
            "aggregated": (58, 180, 15, coef_alpha),
            "phagocytic": (240, 186, 64, coef_alpha),
            "ramified": (35, 32, 179, coef_alpha),
            "other": (63, 4, 126, coef_alpha),
            "iba1_tnem119": (63, 4, 126, coef_alpha),
            "iba1": (30, 255, 30, coef_alpha),#IBA1 -> White slide 2 #Covid : 155,255,255
            "tnem119": (255, 30, 30, coef_alpha),
        }
    )



def adaptative_color_center_of_mass(coef_alpha=125):
    return dict(
        {
            "Proliferative": (123, 17, 66, coef_alpha),
            "Amoeboid": (4, 51, 255, coef_alpha),
            "Cluster": (0, 165, 0, coef_alpha),
            "Phagocytic": (215, 114, 0, coef_alpha),
            "Ramified": (0, 0, 67, coef_alpha),
            "Other": (63, 4, 126, coef_alpha),
        }
    )

def adaptative_color_tissue_segmentation(coef_alpha=125):
    return dict(
        {
            "ganglionic_eminence": (195, 121, 54, coef_alpha),
            "striatum": (101, 170, 215, coef_alpha),
            "cortical_boundary": (131, 208, 104, coef_alpha),
            "neocortex": (228, 99, 160, int(coef_alpha // 2)),
            "manually_segmented": (123, 237, 7, int(coef_alpha // 2)),
        }
    )  # 64
def get_cmap_microglial_project():
    """
    Returns:
      A colormap for visualizing segmentation results.
    Je peux aller ici pour trouver de nouvelles couleurs : https://eltos.github.io/gradient/#1F4E5A-15:029C8E-FFDB69-85:FFA658-EA5F40
    """
    colormap = np.ones((256, 4))
    colormap[0, :] = [1, 1, 1, 1]  # Background
    colormap[1, :] = [0.776, 0.145, 0.557, 1]  # Proliferative  xkcd : sky blue
    colormap[2, :] = [0.271, 0.729, 0.988, 1]  # Amoeboid sky
    colormap[3, :] = [
        156 / 255,
        211 / 255,
        121 / 255,
        1,
    ]  # Clust  : 0.980, 0.859, 0.263 (pas mal) vert : 0.388, 0.773, 0.341 lightish green
    colormap[4, :] = [254 / 256, 223 / 256, 8 / 256, 1]  # Phag  dandelion
    colormap[5, :] = [0.055, 0.098, 0.816, 1]  # Ramified primary blue
    colormap[6, :] = [0.008, 0.612, 0.557, 1]
    colormap[7, :] = [0.918, 0.373, 0.251, 1]
    colormap[8, :] = [0.57647055, 0, 0.27107143, 1]
    colormap[9, :] = [0.514, 0.820, 0.333, 1]
    ### A partir de la les couleurs sont dédiés à des applications particulières
    colormap[10, :] = [
        0,
        0,
        0,
        1,
    ]  # Levelsets  Jaune : [0.980, 0.859, 0.263, 1]  Rouge =0.843, 0.149, 0.118 ; Vert = 0.125, 0.639, 0.541
    colormap[11, :] = [
        173 / 256,
        3 / 256,
        222 / 256,
        1,
    ]  # pop B dans la fonction de ripley

    # relatif to tissu slide mask
    colormap[100, :] = [1, 1, 1, 1]  # Pas de tissu
    colormap[101, :] = [0.000, 0.298, 0.494, 1]  # tissu
    colormap[102, :] = [1.000, 0.859, 0.412, 1]  # Slide border
    colormap[103, :] = [198 / 255, 188 / 255, 186 / 255, 1]  # Visited tissu
    colormap[104, :] = [
        0.835,
        0.553,
        0.306,
        1,
    ]  # Tissu more than thresh and cells have been saved
    # colormap[105, :] = [0.647, 0.686, 0.086, 1]

    return colormap


def get_cmap_tissu_slide():
    """
    Returns:
      A colormap for visualizing segmentation results.
    Je peux aller ici pour trouver de nouvelles couleurs : https://eltos.github.io/gradient/#1F4E5A-15:029C8E-FFDB69-85:FFA658-EA5F40
    """
    colormap = np.ones((256, 4))

    # relatif to tissu slide mask
    colormap[1, :] = [1, 1, 1, 1]  # Pas de tissu
    colormap[1, :] = [198 / 255, 188 / 255, 186 / 255, 1]  # tissu
    colormap[2, :] = [246 / 255, 255 / 255, 190 / 255, 1]  # Slide border
    colormap[3, :] = [246 / 255, 255 / 255, 190 / 255, 1]  # Visited tissu
    colormap[4, :] = [
        246 / 255,
        255 / 255,
        190 / 255,
        1,
    ]  # Tissu more than thresh and cells have been saved
    # colormap[105, :] = [0.647, 0.686, 0.086, 1]
    return colormap


def get_cmap_physiopart_segmentation():
    """
    Returns:
      A colormap for visualizing segmentation results.
    Je peux aller ici pour trouver de nouvelles couleurs : https://eltos.github.io/gradient/#1F4E5A-15:029C8E-FFDB69-85:FFA658-EA5F40
    """
    colormap = np.ones((256, 4))

    # relatif to tissu slide mask
    colormap[0, :] = [1, 1, 1, 1]  # Pas de tissu
    # colormap[1, :] = [231/255,170/255,199/255, 1] #Cortex

    colormap[1, :] = [200 / 255, 124 / 255, 56 / 255, 1]  # Galglionic eminence
    colormap[3, :] = [200 / 255, 124 / 255, 56 / 255, 1]  # Galglionic eminence
    # colormap[2, :] = [152/255,184/255,205/255, 1] #Striatum
    # colormap[4, :] = [183/255,219/255,170/255, 1] #Tissu more than thresh and cells have been saved
    # colormap[105, :] = [0.647, 0.686, 0.086, 1]
    return colormap

def get_cmap_model_segmentation():
    """"
    Code pour voir la colormap 
    liste = [i for i in range(20)]
    liste = np.asarray(liste)
    liste = liste.reshape((4,5))
    print(liste.shape)
    print(np.unique(liste))
    plt.imshow(liste, cmap = CMAP, vmin = 0, vmax = 256)
    plt.colorbar()

    Returns:
      A colormap for visualizing segmentation results.
    Je peux aller ici pour trouver de nouvelles couleurs : https://eltos.github.io/gradient/#1F4E5A-15:029C8E-FFDB69-85:FFA658-EA5F40
    """
    colormap = np.ones((256, 4))
    colormap[0, :] = [1, 1, 1, 1]
    colormap[1, :] = [0.776, 0.145, 0.557, 1]
    colormap[2, :] = [0.867, 0.518, 0.082, 1]
    colormap[3, :] = [0.514, 0.820, 0.333, 1]
    colormap[4, :] = [0.57647055, 0, 0.27107143, 1]
    colormap[5, :] = [0.980, 0.859, 0.263, 1]
    colormap[6, :] = [0.533, 0.208, 0.075, 1]
    colormap[7, :] = [0.918, 0.373, 0.251, 1]
    colormap[8, :] = [0.353, 0.282, 0.471, 1]
    colormap[9, :] = [1.000, 0.651, 0.345, 1]
    colormap[10, :] = [0.008, 0.612, 0.557, 1]
    colormap[11, :] = [0.545, 0.169, 0.345, 1]
    colormap[12, :] = [0.122, 0.306, 0.353, 1]
    colormap[13, :] = [0.243, 0.325, 0.420, 1]

    for i in range(14, 244, 13):
        colormap[i : i + 13, :] = colormap[1:14, :]

    colormap[248:256, :] = colormap[1:9, :]
    return colormap



COLOR_TISSUE_SEGMENTATION_DRAW = adaptative_color_tissue_segmentation(coef_alpha=150)
COLOR_CELLS_DRAW = create_colors_cell_drawing(coef_alpha=255)
COLOR_CELLS_CENTER_OF_MASS_DRAW = adaptative_color_center_of_mass(coef_alpha=255)


CMAP_PROBABILITY_MAP = ListedColormap(get_cmap_physiopart_segmentation())


LABELS_TISSU_SLIDE = dict(
    {
        0: "Without tissu",
        1: "Tissu",
        2: "Slide border",
        3: "Visited tiles",
        4: "%tissu more than thresh",
    }
)
# Heatmaps colormap
import matplotlib
coolwarm = matplotlib.colormaps.get_cmap("viridis")
newcolors = coolwarm(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors = np.vstack([white, newcolors])
# newcolors[0, :] = white
# newcolors[1:3, :] = tisssu



# CMAP_PROBABILITY_MAP = ListedColormap(newcolors)
CMAP_MODEL_SEG = ListedColormap(get_cmap_model_segmentation(), name="modelseg")
CMAP = ListedColormap(get_cmap_microglial_project(), name="microglial cells")
CMAP_TISSU_SLIDE = ListedColormap(get_cmap_tissu_slide(), name="tissu_slide")


def create_txt_pp_square_size(dict_physiological_part_size_max_square):
    """Util for filename creation to distinguish squares configurations"""
    txt_square_per_physiological_part = "_squares_size"
    for physiological_part in dict_physiological_part_size_max_square.keys():
        size_max_square = dict_physiological_part_size_max_square[physiological_part]
        txt_square_per_physiological_part += (
            "_" + physiological_part + "_" + str(size_max_square)
        )
    return txt_square_per_physiological_part

# Related to drawings 

def mask_to_polygons_layer(mask: np.array) -> Polygon:
    """
    https://www.kaggle.com/code/sohaibanwaar1203/polygons-and-masks-visualisation
    Converting mask to polygon object

    Input:
        mask: (np.array): Image like Mask [0,1] where all 1 are consider as masks

    Output:
        shapely.geometry.Polygon: Polygons
    """
    all_polygons = []
    for shape, value in features.shapes(
        mask.astype(np.int16), mask=(mask > 0), transform=Affine(1.0, 0, 0, 0, 1.0, 0)
    ):
        all_polygons.append(geometry.shape(shape))

    all_polygons = MultiPolygon(all_polygons)

    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.geom_type == "Polygon":
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def draw_center_of_mass(roi, img, idx_pop = "all",channels_of_interest = None):
    """
    Add center of mass 
    
    To be modified : unconsistent with both wsi and fluorescence 
    """
    border_size = 15
    if roi.dataset_config.data_type == "fluorescence":
        if channels_of_interest == None :
            channels_of_interest = roi.dataset_config.channels_of_interest
        table_cell = roi.table_cells_w_borders

        cells_coords = table_cell[["x_roi_w_borders", "y_roi_w_borders"]].values

    else : 

        table_cells_pop_A = roi.table_cells_extended[
            roi.table_cells_extended["Decision"] == idx_pop
        ]
        table_cells_pop_A.reset_index(drop=True, inplace=True)
        cells_coords = table_cells_pop_A[["x_roi_extended", "y_roi_extended"]].values


    for coord in cells_coords:
        # print(coord)
        circle_boundaries = [
            coord[1] - border_size / 2,
            coord[0] - border_size / 2,
            coord[1] + border_size / 2,
            coord[0] + border_size / 2,
        ]

        img.ellipse(
            circle_boundaries,
            fill=(255, 255, 255, 200),
            outline="black",
            width=5,
        )

        # img.ellipse(circle_boundaries, fill="black", outline=None, width=1)

    return img

def draw_roi_delimiter(roi, img):
    """
    Ajoute le contour de la ROI dans l'image des levelsets
    """
    border_size = roi.dataset_config.roi_border_size
    tl = (border_size, border_size)
    tr = (border_size + roi.roi_shape[1], border_size)
    br = (border_size + roi.roi_shape[1], border_size + roi.roi_shape[0])
    bl = (border_size, border_size + roi.roi_shape[0])

    img.line([tl, tr, br, bl, tl], width=20, fill="red")
    return img

def draw_anatomical_part_mask(roi, img):
    poly_tissue = mask_to_polygons_layer(roi.mask_roi_extended)
    poly_cells = [list(poly.exterior.coords) for poly in list(poly_tissue.geoms)]
    # for points in poly_cells:
    #     img.polygon(points,fill = (0,0,0,125), outline ="red")
    if len(poly_cells) > 0:
        # img.polygon(poly_cells[0],fill = (0,0,0,20), outline ="black",width = 15) marche pas avec roi stylé
        img.polygon(poly_cells[0], fill=(0, 0, 0, 40), outline="black")

    # img.line(poly_cells[0],fill = "blue", width=25)
    return img

def with_tiles_delimitations(roi, img):
    """Use dataset_config.tile_size to draw tiles delimitations
    """
    #TODO 
    pass 

## Vient de la classification


def add_crop_line_on_img(img, mask):
    """
    Display function
    -------
    Add crop contour to image
    """
    border_size = (mask.shape[0] - 256) / 2
    tl = (border_size, border_size)
    tr = (border_size + 256, border_size)
    br = (border_size + 256, border_size + 256)
    bl = (border_size, border_size + 256)

    img.line([tl, tr, br, bl, tl], width=6)
    return img


def add_mask_to_img(img, mask):
    """
    Display function
    -------
    RGB -> RGB + MASK CELL with color coded by class
    """
    distinct_int = list(np.unique(mask))
    for label in distinct_int:
        if label != 0:
            state_name = TABLE_INT_STATE[label]
            mask_label = np.where(mask == label, 1, 0)
            poly_cells = mask_to_polygons_layer(mask_label)
            poly_cells = [list(poly.exterior.coords) for poly in list(poly_cells.geoms)]
            for points in poly_cells:
                # img.polygon(points,fill = adaptative_color_lines_transparency(190)[state_name], outline ="black")
                img.polygon(
                    points,
                    fill=adaptative_color_classification(190)[state_name],
                    outline="black",
                )

                # idisplay_segmented_cellsmg.polygon(points,fill = "black", outline ="black")
    return img

def draw_cells_on_img(roi, img, cells_of_interest = None, cell_type_filter = "all", coef_alpha = 255 ,color_name="dataset_config"):
    if color_name == "dataset_config":
        colors = create_colors(coef_alpha, roi.dataset_config.mapping_cells_colors)
    else : 
        colors = create_colors(coef_alpha, dict_colors[color_name])
    if roi.dataset_config.data_type == "fluorescence":
        # colors = create_colors_cell_drawing(coef_alpha=180)
        if cells_of_interest == None :

            cells_of_interest = roi.dataset_config.cell_class_names
        if cell_type_filter == "all":
            cells_of_interest = roi.dataset_config.cell_class_names
        for cell_name in cells_of_interest:
            if cell_name in cells_of_interest:
                mask_cells = roi.masks_cells_w_borders[cell_name]
                mask_cells_poly = mask_to_polygons_layer(mask_cells)
                mask_cells_poly = [list(poly.exterior.coords) for poly in list(mask_cells_poly.geoms)]
                for points in mask_cells_poly:
                    img.polygon(points, fill=colors[cell_name], outline="blue")
    else : 
        if cell_type_filter == "all":
            cell_type_filter = roi.dataset_config.cell_class_names
        else : 
            cell_type_filter = [cell_type_filter]
    
        # colors = create_colors_cell_drawing_ihc_microglia(coef_alpha=255)
        cell_class_names = roi.dataset_config.cell_class_names
        for idx_cell_type, cell_type in enumerate(cell_class_names,1):
            if cell_type in cell_type_filter:
                mask_cells_class_k = (roi.masks_cells_w_borders == idx_cell_type)
                poly_cells = mask_to_polygons_layer(mask_cells_class_k)
                poly_cells = [list(poly.exterior.coords) for poly in list(poly_cells.geoms)]
                for points in poly_cells:
                    img.polygon(points, fill=colors[cell_type], outline="blue")
    return img

def draw_nuclei_on_img(img, mask):
    """
    draw_cells_on_img
    ex add_mask_nuclei_to_img_CELLPOSE
    Display function
    -------
    RGB -> RGB + MASK CELL with color coded by class 
    """
    import skimage.morphology as sk_morphology
    mask_label = sk_morphology.label(mask)
    uniques_values = list(np.unique(mask_label))
    nb_cells = len(uniques_values) - 1
    print("nb_cells" ,nb_cells)
    for i in uniques_values:
        if i == 0:
            continue
        mask_i = (mask_label == i).astype(int)
        poly_cells = mask_to_polygons_layer(mask_i)
        poly_cells = [
            list(poly.exterior.coords) for poly in list(poly_cells.geoms)
        ]
        for points in poly_cells:
            img.polygon(points,
                        fill=DICT_COLORMAP_CELLPOSE[int(
                            i % len(DICT_COLORMAP_CELLPOSE))],
                        outline="black")
    # print("Number of cells : ",nb_cells)
    return img, nb_cells