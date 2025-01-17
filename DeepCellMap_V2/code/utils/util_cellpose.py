import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from shapely.geometry import Polygon, MultiPolygon  #MultiPoint,polygon, Point, LineString,
from rasterio import features, Affine
from shapely import geometry
import pandas as pd
from preprocessing import slide
from utils.util import *
# from cellpose.io import imread
from cellpose import models
from cellpose.io import imread
from simple_colors import *
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm.notebook import tqdm


from tqdm.notebook import tqdm
from simple_colors import *
from skimage.filters import threshold_multiotsu
from skimage.filters import threshold_otsu

def from_list_to_dict(liste_color_from_eltos, coef_alpha=125):
    dict_colormap = dict()
    for i in range(len(liste_color_from_eltos)):
        dict_colormap[i] = 'rgba(' + str(
            int(liste_color_from_eltos[i][1][0] * 255)) + ',' + str(
                int(liste_color_from_eltos[i][1][1] * 255)) + "," + str(
                    int(liste_color_from_eltos[i][1][2] *
                        255)) + "," + str(coef_alpha) + ')'
    return dict_colormap


LIST_NUCLEI_COLORS = [(0.000, (0.122, 0.306, 0.353)),
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
                      (1.000, (0.122, 0.306, 0.353))]

DICT_COLORMAP_CELLPOSE = from_list_to_dict(LIST_NUCLEI_COLORS, coef_alpha=255)

## Test cellpose param

def test_cellpose_on_several_images(hyperparam_cellpose,
                               filepath_examples,
                               save_fig=False):
    if save_fig:
        path_save_segmentation_choice = os.path.join(
            dataset_config.dir_output_dataset, "segmentation_nuclei_cellpose")
        mkdir_if_nexist(path_save_segmentation_choice)
    imgs = [plt.imread(f) for f in filepath_examples]
    imgs_512 = [f[:512, :512, :] for f in imgs]
    exp_number = 1
    dict_correspondance_exp_model_param = dict()
    for model_type in hyperparam_cellpose["model_type"]:
        model = models.Cellpose(model_type=model_type)
        for channels in hyperparam_cellpose["channels"]:
            for normalisation in hyperparam_cellpose["normalisation"]:
                for net_avg in hyperparam_cellpose["net_avg"]:
                    for diameter in hyperparam_cellpose["diameter"]:

                        name_model = "e_" + str(exp_number).zfill(
                            2) + "_" + model_type + "_channels_" + str(
                                channels) + "_normalisation_" + str(
                                    normalisation) + "_net_avg_" + str(
                                        net_avg) + "_diameter_" + str(diameter)
                        print(name_model)

                        masks, flows, styles, diam = model.eval(
                            imgs_512,
                            diameter=diameter,
                            channels=channels,
                            normalize=normalisation,
                            net_avg=net_avg)
                        dict_correspondance_exp_model_param[exp_number] = dict(
                            {
                                "name_model": name_model,
                                "model_type": model_type,
                                "channels": channels,
                                "normalisation": normalisation,
                                "net_avg": net_avg,
                                "diameter": diameter,
                                "diam_after_eval": diam
                            })
                        if diameter is not None:
                            diam = [diam] * len(imgs_512)
                        liste_mask_title = [
                            "e : " + str(exp_number) + " - diam :" +
                            str(np.round(diam[k], 0))
                            for k in range(len(imgs_512))
                        ]
                        path_save = os.path.join(path_save_segmentation_choice,
                                                 name_model + ".png")
                        plot_several_mask_several_rgb_CELLPOSE(
                            imgs_512,
                            masks,
                            liste_mask_title,
                            path_save=path_save,
                            display=False,
                            figsize=(30, 10))
                        exp_number += 1

    print("There is " + str(exp_number) + " experiments")
    return dict_correspondance_exp_model_param

def test_best_cellpose_on_test_set(param_cellpose, n_images_to_test = 2):
    """ Test best cellpose on the test set """
    model_type = param_cellpose["model_type"]
    diameter = param_cellpose["diameter"]
    channels = param_cellpose["channels"]
    normalisation = param_cellpose["normalisation"]
    net_avg = param_cellpose["net_avg"]
    path_examples_cellpose_imgs = os.path.join(dataset_config.dir_base_anatomical_region_segmentation, "cellpose_model_selection", "test_data")
    imgs = [imread(os.path.join(path_examples_cellpose_imgs,f)) for f in os.listdir(path_examples_cellpose_imgs) if f.endswith(".png")]
    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diam = model.eval(imgs[:n_images_to_test], diameter=diameter, channels=channels,normalize=normalisation, net_avg = net_avg)
    path_save_segmentation_best = os.path.join(dataset_config.dir_base_anatomical_region_segmentation,"cellpose_model_selection","results_on_test_set")
    print(blue("See results here : \n"+path_save_segmentation_best) )
    mkdir_if_nexist(path_save_segmentation_best)
    for idx_img in range(n_images_to_test):
        path_save = os.path.join(path_save_segmentation_best,"cellpose_"+str(idx_img)+".png")
        img = imgs[idx_img]
        mask = masks[idx_img]
        nb_cells = display_nuclei_on_tile_CELLPOSE(img,mask,path_save)



def plot_several_mask_several_rgb_CELLPOSE(list_rgb,
                                           liste_mask,
                                           liste_mask_title,
                                           path_save=None,
                                           display=False,
                                           figsize=(13, 5)):
    """
    Display function
    -------
    Display several masks on the same rgb image
    """
    if len(liste_mask) > 1:

        fig, axes = plt.subplots(1,
                                 len(liste_mask),
                                 figsize=figsize,
                                 sharex=True,
                                 sharey=True)
        ax = axes.ravel()
        for i, mask in enumerate(liste_mask):

            img_pil = np_to_pil(list_rgb[i])
            img = ImageDraw.Draw(img_pil, "RGBA")
            im, nb_cells = add_mask_nuclei_to_img_CELLPOSE(img, mask)
            ax[i].imshow(img_pil)
            ax[i].set_title(liste_mask_title[i] + "\n n:" + str(nb_cells))
            ax[i].set_axis_off()
        # plt.autoscale(tight=True)
        if display:
            plt.show()
        if path_save is not None:
            fig.savefig(path_save,
                        facecolor='white',
                        dpi="figure",
                        bbox_inches='tight',
                        pad_inches=0.1)
        plt.close("All")
    else:
        img_pil = np_to_pil(list_rgb[0])
        img = ImageDraw.Draw(img_pil, "RGBA")
        im = add_mask_nuclei_to_img_CELLPOSE(img, liste_mask[0])
        plt.imshow(img_pil)
        plt.axis('off')
        if display:
            plt.show()
        if path_save is not None:
            fig.savefig(path_save,
                        facecolor='white',
                        dpi="figure",
                        bbox_inches='tight',
                        pad_inches=0.1)
        plt.close("All")


def draw_roi_delimiter_CELLPOSE(img):
    """Ajoute le contour de la ROI dans l'image des levelsets"""

    tl = (256, 0)
    tr = (256, 1024)
    img.line([tl, tr], width=2)
    tl = (512, 0)
    tr = (512, 1024)
    img.line([tl, tr], width=2)
    tl = (3 * 256, 0)
    tr = (3 * 256, 1024)
    img.line([tl, tr], width=2)
    tl = (0, 256)
    tr = (1024, 256)
    img.line([tl, tr], width=2)
    tl = (0, 512)
    tr = (1024, 512)
    img.line([tl, tr], width=2)
    tl = (0, 3 * 256)
    tr = (1024, 3 * 256)
    img.line([tl, tr], width=2)
    return img


def display_nuclei_on_tile_CELLPOSE(img_rgb_np,
                                    mask,
                                    path_save=None,
                                    figsize=(16, 8),
                                    display=False):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    img_pil_rgb = np_to_pil(img_rgb_np)
    img_rgb = ImageDraw.Draw(img_pil_rgb, "RGBA")
    img_rgb = draw_roi_delimiter_CELLPOSE(img_rgb)

    ax[0].imshow(img_pil_rgb)
    ax[0].set_title("RGB image")
    ax[0].set_axis_off()

    img_pil = np_to_pil(img_rgb_np)
    img = ImageDraw.Draw(img_pil, "RGBA")
    img, nb_cells = add_mask_nuclei_to_img_CELLPOSE(img, mask)
    img = draw_roi_delimiter_CELLPOSE(img)
    ax[1].imshow(img_pil)
    ax[1].set_title("Cellpose segmentation - N nuclei = " + str(nb_cells))
    ax[1].set_axis_off()
    fig.savefig(path_save, facecolor='white',dpi = "figure",bbox_inches='tight',pad_inches = 0.1) if path_save is not None else None
    if display:
        plt.show()
    else:
        plt.close("All")
    return nb_cells

def segment_nuclei_on_slide(slide_num, param_cellpose, verbose = 0):
    """
    Applique Cellpose a chaque tile de l'image (celles contenant au moins 5% de tissue. Et enregistre dans un dataframe le nombre de nuclei dans chaque crops (la tile est divisé en 4x4 crops de taille 256x256))
    """
    model_type = param_cellpose["model_type"]
    diameter = param_cellpose["diameter"]
    channels = param_cellpose["channels"]
    normalisation = param_cellpose["normalisation"]
    net_avg = param_cellpose["net_avg"]

    model = models.Cellpose(model_type=model_type)

    # path_save_segmentation_best = os.path.join(dataset_config.dir_base_anatomical_region_segmentation,"best_model")
    # mkdir_if_nexist(path_save_segmentation_best)
    # path_save_segmentation_best_tiles = os.path.join(path_save_segmentation_best,"tiles")
    # mkdir_if_nexist(path_save_segmentation_best_tiles)

    t = Time()
    t_compute_1_tile_tot = t.elapsed()
    threshold_tissu = 5

    """Time dataframe"""
    colnames_time_df = ["Compute_1_tile"]
    df_time = pd.DataFrame(columns = colnames_time_df)
    df_time.loc[0] = [t_compute_1_tile_tot.seconds]


    idx_tile = 1

    df_n_nuclei = pd.DataFrame()
    """To exclude tiles without tissue"""

    path_filtered_image, path_mask_tissue = slide.get_filter_image_result(slide_num)
    mask_tissue_filtered = Image.open(path_mask_tissue)
    mask_tissue_filtered_np = np.asarray(mask_tissue_filtered)

    mask_tissu = np.copy(mask_tissue_filtered_np)
    mask_tissu = np.where(mask_tissu>0,1,0 )

    pcw = dataset_config.mapping_img_number[slide_num]
    name_path = "slide_"+str(slide_num)+"_"+str(pcw)+"_pcw_new"
    path_slide = os.path.join(dataset_config.dir_base_anatomical_region_segmentation,name_path)

    path_folder_nuclei_segmentation = os.path.join(path_slide,"cellpose_nuclei_segmentation")
    path_folder_nuclei_segmentation_fig_callback = os.path.join(path_folder_nuclei_segmentation,"Callbacks")
    mkdir_if_nexist(path_folder_nuclei_segmentation)
    mkdir_if_nexist(path_folder_nuclei_segmentation_fig_callback)


    path_slide = slide.get_training_slide_path(slide_num)
    s = slide.open_slide(path_slide) 
    tot_row_tiles = s.dimensions[1]//dataset_config.tile_height+1
    tot_col_tiles = s.dimensions[0]//dataset_config.tile_height+1 
    liste_tiles_slide = [(r,c) for r in range(1,tot_row_tiles) for c in range(1,tot_col_tiles)]

    print("# tiles col: ",tot_col_tiles )
    print("# tiles row : ", tot_row_tiles)

    nuclei_density_crops = np.zeros((tot_row_tiles*4,tot_col_tiles*4))
    print("shape nuclei_density : ", nuclei_density_crops.shape)
    nuclei_density_tiles = np.zeros((tot_row_tiles,tot_col_tiles))
    print("shape nuclei_density : ", nuclei_density_tiles.shape)

    tile_with_tissue = 0
    for tile_row, tile_col in tqdm(liste_tiles_slide):
        t_compute_1_tile = Time()

        mask_tile = mask_tissue_filtered_np[(tile_row-1)*32:tile_row*32,(tile_col-1)*32:tile_col*32]
        pourcentage_tissue = np.sum(mask_tile)*100/(32*32*255)

        if pourcentage_tissue > threshold_tissu : 
            tile_with_tissue+=1
            # if tile_with_tissue > 10 : 
            #     break
            coord_origin_row = (tile_row-1)*dataset_config.tile_height
            coord_origin_col = (tile_col-1)*dataset_config.tile_width
            weight = dataset_config.tile_width
            height = dataset_config.tile_height

            rgb = s.read_region((coord_origin_col,coord_origin_row),0,(weight,height))

            rgb_tile = np.asarray(rgb)
            rgb_tile = rgb_tile[:,:,:3]
            # print("Should be int8 until 255 : ",rgb_tile.dtype, " min : ",np.min(rgb_tile)," max : ",np.max(rgb_tile))
            
            mask_tile, flows, styles, diam = model.eval([rgb_tile], diameter=diameter, channels=channels,normalize=normalisation, net_avg = net_avg)
            mask_tile = mask_tile[0]

            df_n_nuclei_tile = pd.DataFrame()
            nuclei_density_tiles[tile_row-1,tile_col-1] = mask_tile.max()-1
            for crop_idx_row in range(4):
                for crop_idx_col in range(4):
                    dict_tile = dict()
                    dict_tile["tile_row"] = tile_row
                    dict_tile["tile_col"] = tile_col
                    dict_tile["nb_nuclei_tile"] = mask_tile.max()-1
                    dict_tile["crop_row_in_tile"] = crop_idx_row+1
                    dict_tile["crop_col_in_tile"] = crop_idx_col+1
                    mask_crop = mask_tile[crop_idx_row*256:(crop_idx_row+1)*256,crop_idx_col*256:(crop_idx_col+1)*256]
                    nb_nuclei_crop = len(np.unique(mask_crop))-1
                    dict_tile["nb_nuclei_crop"] = nb_nuclei_crop
                    nuclei_density_crops[(tile_row-1)*4+crop_idx_row,(tile_col-1)*4+crop_idx_col] = nb_nuclei_crop

                    dict_tile["crop_row"] = (tile_row-1)*4+dict_tile["crop_row_in_tile"]
                    dict_tile["crop_col"] = (tile_col-1)*4+dict_tile["crop_col_in_tile"]

                    df_n_nuclei_tile = pd.concat([df_n_nuclei_tile,pd.DataFrame(dict_tile,index=[0])],axis=0)
                    


            if idx_tile%500 == 0:
                print("idx_tile = ",idx_tile," tile_row = ",tile_row," tile_col = ",tile_col," nb_nuclei_tile = ",dict_tile["nb_nuclei_tile"]," nb_nuclei_crop = ",dict_tile["nb_nuclei_crop"])
                path_save = os.path.join(path_folder_nuclei_segmentation_fig_callback,"cellpose_tile_row_"+str(tile_row).zfill(3)+"_tile_col_"+str(tile_col).zfill(3)+".png")
                display_nuclei_on_tile_CELLPOSE(rgb_tile,mask_tile,path_save, display = False) 
                df_n_nuclei.to_csv(os.path.join(path_folder_nuclei_segmentation,"df_n_nuclei.csv"),index=False, sep = ";")

            #   print("idx_tile : ",idx_tile," t_compµute_1_tile_tot AVANT AJOUT DE ",t_compute_1_tile_tot)
            t_compute_1_tile_elapsed = t_compute_1_tile.elapsed()

            df_n_nuclei_tile["computation_time"] = t_compute_1_tile_elapsed.seconds
            df_n_nuclei = pd.concat([df_n_nuclei,df_n_nuclei_tile],axis=0)
        idx_tile+=1
        
    df_n_nuclei.to_csv(os.path.join(path_folder_nuclei_segmentation,"df_n_nuclei_final.csv"),index=False, sep = ";")
    np.save(os.path.join(path_folder_nuclei_segmentation,"nuclei_density_tiles.npy"), nuclei_density_tiles)
    nuclei_density_crops_pil = np_to_pil(nuclei_density_crops)
    nuclei_density_crops_pil.save(os.path.join(path_folder_nuclei_segmentation,"nuclei_density_crops_pil.png"))
    return df_n_nuclei, nuclei_density_tiles, nuclei_density_crops 


def load_nuclei_segmentation_results_1_slide(slide_num):

    # sur les tiles direct :
    pcw = dataset_config.mapping_img_number[slide_num]
    name_path = "slide_" + str(slide_num) + "_" + str(pcw) + "_pcw"

    path_slide = os.path.join(dataset_config.dir_base_anatomical_region_segmentation, name_path)

    path_folder_nuclei_segmentation = os.path.join(path_slide,
                                                   "cellpose_nuclei_segmentation")
    path_folder_nuclei_segmentation_fig_callback = os.path.join(
        path_folder_nuclei_segmentation, "Callbacks")
    path_df = os.path.join(path_folder_nuclei_segmentation,
                           "df_n_nuclei_final.csv")
    path_nuclei_tiles = os.path.join(path_folder_nuclei_segmentation,
                                     "nuclei_density_tiles.png")
    path_nuclei_crops = os.path.join(path_folder_nuclei_segmentation,
                                     "nuclei_density_crops_pil.png")

    # nuclei_density_tiles = plt.imread(path_nuclei_tiles) #TODOO de commenter puis commenter cellui du dessous puis une fois que j'ai tout changé chez 6,11,28 à l'ens je peux décommenter
    nuclei_density_tiles = np.load(
        os.path.join(path_folder_nuclei_segmentation,
                     "nuclei_density_tiles.npy"))

    df_n_nuclei = pd.read_csv(path_df, sep=";")

    nuclei_density_crops = plt.imread(path_nuclei_crops)
    nuclei_density_crops = nuclei_density_crops * 255
    df_n_nuclei["crop_coord_row"] = (df_n_nuclei["tile_row"]-1)*4+df_n_nuclei["crop_row"] #Keep for 6,11,28
    df_n_nuclei["crop_coord_col"] = (df_n_nuclei["tile_col"]-1)*4+df_n_nuclei["crop_col"] #Keep for 6,11,28

    mask_tissue = np.zeros(nuclei_density_crops.shape)
    # print("mask_tissue shape",mask_tissue.shape)
    for row in df_n_nuclei.iterrows():
        row = row[1]
        crop_row = row["crop_coord_row"]-1 #Keep for 6,11,28
        crop_col = row["crop_coord_col"]-1 #Keep for 6,11,28
        # crop_row = row["crop_row"] - 1
        # crop_col = row["crop_col"] - 1
        if row["nb_nuclei_crop"] > 0:
            mask_tissue[crop_row, crop_col] = 1

    return df_n_nuclei, nuclei_density_tiles, nuclei_density_crops, mask_tissue


def display_result_cellpose(nuclei_density_tiles, nuclei_density_crops,
                            mask_tissue):
    fig, axes = plt.subplots(ncols=3, figsize=(30, 15))
    ax = axes.ravel()
    ax[0].imshow(mask_tissue, cmap="jet")
    ax[0].set_title("Mask tissue in slide")
    ax[0].axis('off')

    ax[1].imshow(nuclei_density_tiles, cmap="coolwarm")
    ax[1].set_title("Nuclei density in tiles")
    ax[1].axis('off')

    ax[2].imshow(nuclei_density_crops, cmap="coolwarm")
    ax[2].set_title("Nuclei density in crops")
    ax[2].axis('off')
    plt.show()


# TABLE_SLIDE_FLIPUP_OR_NOT =dict({'6':dict({"flipud_mask":True,"fliplr_mask":False,"flipud":False,"fliplr":False}),'28':dict({"flipud_mask":False,"fliplr_mask":True,"flipud":True,"fliplr":True}),'11':dict({"flipud_mask":False,"fliplr_mask":True,"flipud":True,"fliplr":True})})
TABLE_SLIDE_FLIPUP_OR_NOT = dict({
    '1':
    dict({
        "flipud_mask": False,
        "fliplr_mask": False,
        "flipud": False,
        "fliplr": False
    }),
    '2':
    dict({
        "flipud_mask": False,
        "fliplr_mask": False,
        "flipud": False,
        "fliplr": False
    }),
    '3':
    dict({
        "flipud_mask": False,
        "fliplr_mask": False,
        "flipud": False,
        "fliplr": False
    })
    })


def load_nuclei_density(slide_nums):
    """ Load the nuclei density for the slides in slide_nums"""
    dict_nuclei_density = dict()
    for slide_num in slide_nums:
        print(blue("Loading slide " + str(slide_num) + " ..."))
        df_n_nuclei, nuclei_density_tiles, nuclei_density_crops, mask_tissue = load_nuclei_segmentation_results_1_slide(
            slide_num)
        rgb = plt.imread(slide.get_training_image_path(slide_num))

        if TABLE_SLIDE_FLIPUP_OR_NOT[str(slide_num)]["flipud_mask"]:
            nuclei_density_tiles = np.flipud(nuclei_density_tiles)
            nuclei_density_crops = np.flipud(nuclei_density_crops)
            mask_tissue = np.flipud(mask_tissue)
        if TABLE_SLIDE_FLIPUP_OR_NOT[str(slide_num)]["fliplr_mask"]:
            nuclei_density_tiles = np.fliplr(nuclei_density_tiles)
            nuclei_density_crops = np.fliplr(nuclei_density_crops)
            mask_tissue = np.fliplr(mask_tissue)
        if TABLE_SLIDE_FLIPUP_OR_NOT[str(slide_num)]["flipud"]:
            rgb = np.flipud(rgb)
        if TABLE_SLIDE_FLIPUP_OR_NOT[str(slide_num)]["fliplr"]:
            rgb = np.fliplr(rgb)

        pcw = dataset_config.mapping_img_number[slide_num]
        dict_nuclei_density[slide_num] = dict({
            "pcw": pcw,
            "df_n_nuclei": df_n_nuclei,
            "nuclei_density_tiles": nuclei_density_tiles,
            "nuclei_density_crops": nuclei_density_crops,
            "mask_tissue": mask_tissue,
            "rgb": rgb
        })
    return dict_nuclei_density

def display_computation_time(dict_nuclei_density):
    fig = make_subplots(len(dict_nuclei_density),
                        1,
                        subplot_titles=[
                            "Slide " + str(dict_nuclei_density[k]["pcw"]) +
                            " pcw" for k in list(dict_nuclei_density.keys())
                        ],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.06)
    for slide_num_idx, slide_num in enumerate(list(
            dict_nuclei_density.keys())):
        fig.add_trace(px.histogram(dict_nuclei_density[slide_num]
                                   ["df_n_nuclei"].query("crop_row == 1"),
                                   x="computation_time").data[0],
                      row=slide_num_idx + 1,
                      col=1)
    fig.update_xaxes(title_text="Computation time (s)"
                     )  #side top permet de placer les labels en haut
    fig.update_yaxes(title_text="Number of crops")
    fig.update_layout(
        title_text="<b>Nuclei segmentation<br> Histogram computation time </b>",
        title_x=0.5,
        title_font=dict(size=30),
        coloraxis=dict(colorscale='balance',
                       colorbar_thickness=25,
                       colorbar_x=-0.16),
        coloraxis2=dict(
            colorscale='curl',
            colorbar_thickness=25,
        ),
        showlegend=True,
        width=1250,
        height=1100,
        margin=dict(l=50, r=50, b=50, t=170, pad=4))

    fig.show()


def display_rgb_nuclei_density(dict_nuclei_density):
    fig, axes = plt.subplots(nrows=len(dict_nuclei_density),
                             ncols=2,
                             figsize=(20, 7 * len(dict_nuclei_density)))
    ax = axes.ravel()
    for slide_num_idx, slide_num in enumerate(list(
            dict_nuclei_density.keys())):
        ax[2 * slide_num_idx].imshow(
            np.flipud(dict_nuclei_density[slide_num]["rgb"]))
        ax[2 * slide_num_idx].set_title(
            "RGB  " + str(dict_nuclei_density[slide_num]["pcw"]) + " pcw")
        ax[2 * slide_num_idx].axis('off')

        # mask_1 = np.where(nuclei_density_crops_mask_tissue_1>0,nuclei_density_crops_1,-5 )
        ax[2 * slide_num_idx + 1].imshow(np.flipud(
            dict_nuclei_density[slide_num]["nuclei_density_crops"]),
                                         cmap="coolwarm")
        ax[2 * slide_num_idx +
           1].set_title("Nuclei density " +
                        str(dict_nuclei_density[slide_num]["pcw"]) + " pcw")
        ax[2 * slide_num_idx + 1].axis('off')
    # fig.suptitle('RGB - Cell density', fontsize=16)
    plt.show()


def apply_multi_otsu_thresholding(dict_nuclei_density,display_fig = True):
    print(blue("Outputs here -> \n"+dataset_config.dir_base_anatomical_region_segmentation))
    for n_regions in [3,4]:
        for slide_num_idx, slide_num in enumerate(list(dict_nuclei_density.keys())):

            image = dict_nuclei_density[slide_num]["nuclei_density_crops"]
            thresholds = threshold_multiotsu(image, classes=n_regions)
            print("slide = "+str(slide_num_idx) + " : multi Otsu thresholding with " + str(n_regions)+ " regions and -> thresholds = ", thresholds)
            regions = np.digitize(image, bins=thresholds)

            fig = make_subplots(1, 3,subplot_titles=["Nuclei density", "Histogram with thresholds", "Img discretisation in "+str(n_regions)+ " regions"],vertical_spacing = 0.1,horizontal_spacing = 0.06)
            fig.add_trace(px.imshow(image, color_continuous_scale='RdBu_r').data[0], row=1, col=1)

            fig.add_trace(px.histogram(dict_nuclei_density[slide_num]["df_n_nuclei"], x="nb_nuclei_crop",  color_discrete_sequence = ['rgba(23, 29, 71,1)']).data[0], row=1, col=2)

            #add vertical threshold 
            for thresh in thresholds:
                fig.add_shape(type="line",x0=thresh,y0=0,x1=thresh,y1=800,line=dict(color="black",width=2,dash="dot"),row=1,col=2)

            mask = np.where(dict_nuclei_density[slide_num]["mask_tissue"]>0,regions,-1 )

            rgb = mask_to_rgb_img(mask,DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE[str(n_regions)])
            fig.add_trace(px.imshow(np.flipud(rgb)).data[0], row=1, col=3)
            fig.update_layout(title_text="<b>"+str(dict_nuclei_density[slide_num]["pcw"])+" pcw - Multi-otsu thresholding </b> In "+str(n_regions)+ " regions",title_x=0.5,title_font=dict(size=30),
                                    coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.1),
                                    coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                                    showlegend=True,
                                    width=1800, #Taille de la figure 
                                    height=600,
                                    margin=dict(l=50,r=50,b=50,t=170, pad=4))
            fig.update_xaxes(visible=True)
            fig.update_yaxes(visible=True)


            path_multi_otsu_thresh = os.path.join(dataset_config.dir_base_anatomical_region_segmentation,"slide_"+str(slide_num)+"_"+str(dict_nuclei_density[slide_num]["pcw"])+"_pcw" ,"multi_otsu_"+str(dict_nuclei_density[slide_num]["pcw"])+"_pcw_"+str(n_regions)+"_regions.png")
            fig.write_image(path_multi_otsu_thresh)
            fig.show() if display_fig else None 

            dict_nuclei_density[slide_num]["post_otsu_binarized_images_"+str(n_regions)+"_regions"] = []
            for idx, thresh in enumerate(thresholds,1):
                dict_nuclei_density[slide_num]["post_otsu_binarized_images_"+str(n_regions)+"_regions"].append(np.where(regions == idx,idx,0))

            fig = make_subplots(1, n_regions-1,subplot_titles = ["Region : "+str(k) for k in range(n_regions-1)],vertical_spacing = 0.1,horizontal_spacing = 0.06)
            for idx_part, img_part in enumerate(dict_nuclei_density[slide_num]["post_otsu_binarized_images_"+str(n_regions)+"_regions"],1):
                mask = np.where(dict_nuclei_density[slide_num]["mask_tissue"]>0,img_part,-1 )
                rgb = mask_to_rgb_img(mask,DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE[str(n_regions)])
                fig.add_trace(px.imshow(np.flipud(rgb)).data[0], row=1, col=idx_part)

            fig.update_layout(title_text="<b>"+str(dict_nuclei_density[slide_num]["pcw"])+" pcw - Multi-otsu thresholding </b> In "+str(n_regions)+ " regions",title_x=0.5,title_font=dict(size=30),coloraxis_showscale=False,
                                coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.1),
                                showlegend=False,
                                width=(n_regions-1)*600, #Taille de la figure 
                                height=600,
                                margin=dict(l=50,r=50,b=50,t=170, pad=4))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

            fig.show() if display_fig else None 

def display_histogram_nuclei_per_crops(dict_nuclei_density):
    fig = make_subplots(len(dict_nuclei_density),
                        1,
                        subplot_titles=[
                            "Slide " + str(dict_nuclei_density[k]["pcw"]) +
                            " pcw" for k in list(dict_nuclei_density.keys())
                        ],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.06)
    for slide_num_idx, slide_num in enumerate(list(
            dict_nuclei_density.keys())):
        fig.add_trace(
            px.histogram(dict_nuclei_density[slide_num]["df_n_nuclei"],
                         x="nb_nuclei_crop").data[0],
            row=slide_num_idx + 1,
            col=1)
    fig.update_xaxes(title_text="Number of nuclei"
                     )  #side top permet de placer les labels en haut
    fig.update_yaxes(title_text="Number of crops")
    fig.update_layout(
        title_text=
        "<b>Nuclei segmentation<br> Histogram nuclei density (nb per crops) </b>",
        title_x=0.5,
        title_font=dict(size=30),
        coloraxis=dict(colorscale='balance',
                       colorbar_thickness=25,
                       colorbar_x=-0.16),
        coloraxis2=dict(
            colorscale='curl',
            colorbar_thickness=25,
        ),
        showlegend=True,
        width=1250,
        height=1100,
        margin=dict(l=50, r=50, b=50, t=170, pad=4))

    fig.show()


DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE = dict({
    "3":
    dict({
        "-1": [255, 255, 255],
        "0": [23, 29, 71],
        "1": [57, 135, 186],
        "2": [185, 77, 58]
    }),
    "4":
    dict({
        "-1": [255, 255, 255],
        "0": [23, 29, 71],
        "1": [64, 140, 186],
        "2": [188, 205, 210],
        "3": [198, 107, 80]
    })
})

def mask_to_rgb_img(mask,DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE):
    """ Creer une image RGB a partir d'un mask ou chaque valeur du mask a la couleur RGB specifiée dans le dictionnaire DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE
    """
    rgb = np.zeros((mask.shape[0],mask.shape[1],3))
    for key in DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE.keys():
        rgb[np.where(mask == int(key))] = DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE[key]
    return rgb.astype(np.uint8)

threshold_list = [29,59]
def build_map_color(DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE,threshold_list):
    """ Creer un dictionnaire qui associe une valeur(entre 1 et 255) a une couleur et change de couleur a chaque threshold. 
    """
    map_color = dict()
    n_thresh = len(threshold_list)
    idx = 0
    for idx_thresh, thresh in enumerate(threshold_list):
        # print(idx_thresh, thresh)
        while idx <= thresh :
            map_color[str(idx)] = DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE[str(idx_thresh)]
            idx += 1
    while idx <= 255:
        map_color[str(idx)] = DICT_COLORS_OTSU_THRESHOLDING_CELLPOSE[str(n_thresh)]
        idx += 1
    return map_color

