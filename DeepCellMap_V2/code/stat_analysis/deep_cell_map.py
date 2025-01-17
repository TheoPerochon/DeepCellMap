### Fichier contenant les fonctions de l'analyse temporelle des images

# Standard library imports
import shutil
import json
# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns

from scipy.stats import norm
import pandas as pd
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
import plotly.express as px
from math import ceil
from simple_colors import *
from ast import literal_eval

# Project-specific imports
from preprocessing import slide
from utils.util import *
# from segmentation_classification.region_of_interest import RegionOfInterest, get_roi_path
# from config.const_roi import *
from segmentation_classification.segmentation import ModelSegmentation
from segmentation_classification.classification import ModelClassification


# from stat_analysis import colocalisation_analysis,neighbours_analysis,dbscan

""" Informations on slides """


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
            "n_tiles",
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


# Fonctions auxiliaires


def load_table_cells_slide_CHANGED(slide_num, model_segmentation, model_classification):
    """
    Return table_cells_slide en fonction du model de segmentation et du model de classification.
    """

    path_slide_seg_classif = os.path.join(
        dataset_config.dir_classified_img,
        "slide_" + str(slide_num) + "_" + str(dataset_config.mapping_img_number[slide_num]) + "_pcw",
        "segmentation_"
        + model_segmentation.model_segmentation_type
        + "_min_cell_size_"
        + str(model_segmentation.min_cell_size),
        "classification_" + model_classification.param_model["model_name"],
    )
    path_table_cells_slide = os.path.join(path_slide_seg_classif, "table_cells.csv")
    table_cells_slide = pd.read_csv(path_table_cells_slide, sep=";")
    return table_cells_slide, path_table_cells_slide


def load_time_per_tile_slide_CHANGED(slide_num, model_segmentation, model_classification):
    """
    Not in generalised version 
    Util for slide info (py function?)- Return time_per_tile_slide en fonction du model de segmentation et du model de classification.
    """
    path_slide_seg_classif = os.path.join(
        dataset_config.dir_classified_img,
        "slide_" + str(slide_num) + "_" + str(dataset_config.mapping_img_number[slide_num]) + "_pcw",
        "segmentation_"
        + model_segmentation.model_segmentation_type
        + "_min_cell_size_"
        + str(model_segmentation.min_cell_size),
        "classification_" + model_classification.param_model["model_name"],
    )
    path_time_per_tile_slide = os.path.join(
        path_slide_seg_classif, "computation_time_per_tile.csv"
    )
    time_per_tile_slide = pd.read_csv(path_time_per_tile_slide, sep=";")
    return time_per_tile_slide, path_time_per_tile_slide

def _get_slide_shape_CHANGED(slide_num):
    """Util for slide info (py function?)- Get slide shape from the name of the downsized image"""
    padded_sl_num = str(slide_num).zfill(3)
    # training_img_path = slide.get_training_image_path(slide_num)
    training_img_path = get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number)
    large_w, large_h, small_w, small_h = slide.parse_dimensions_from_image_filename(
        training_img_path
    )
    return int(large_w), int(large_h)


def get_tissue_percent_stats_CHANGED(time_per_tile_slide):
    """Get some stats on tissue percentage"""
    nb_tiles_tissue_percent_100 = (
        len(time_per_tile_slide[time_per_tile_slide["tissue_percent"] == 100])
        / len(time_per_tile_slide)
        * 100
    )
    nb_tiles_tissue_percent_sup_75 = (
        len(time_per_tile_slide[time_per_tile_slide["tissue_percent"] >= 75])
        / len(time_per_tile_slide)
        * 100
    )
    nb_tiles_tissue_percent_sup_5 = (
        len(time_per_tile_slide[time_per_tile_slide["tissue_percent"] >= 5])
        / len(time_per_tile_slide)
        * 100
    )

    tissue_percent_stat = [
        nb_tiles_tissue_percent_100,
        nb_tiles_tissue_percent_sup_75,
        nb_tiles_tissue_percent_sup_5,
    ]
    return tissue_percent_stat


def _get_qte_size_from_table_cells_CHANGED(
    table_cells, time_per_tile_slide, model_classification
):
    """
    Util for slide info (py function?)-  Remplie les colonnes
    """
    # colname_decision = "summed_proba_Decision"+"_"+model_classification.param_model["model_name"]
    colname_decision = "Decision"

    dict_qte_size = dict()
    dict_qte_size["n_cells"] = table_cells.shape[0]
    dict_qte_size["mean_n_cells_per_tile"] = np.mean(
        time_per_tile_slide[time_per_tile_slide["nb_cells_tile"] != 0]["nb_cells_tile"]
    )
    dict_qte_size["std_n_cells_per_tile"] = np.std(
        time_per_tile_slide[time_per_tile_slide["nb_cells_tile"] != 0]["nb_cells_tile"]
    )

    dict_qte_size["mean_cell_size"] = table_cells["size"].mean()
    dict_qte_size["std_cell_size"] = table_cells["size"].std()
    for state_idx, state in enumerate(
        model_classification.param_model["param_training_set"]["state_names"][1:], 1
    ):
        # colname_proba = "summed_proba_"+state+"_"+model_classification.param_model["model_name"]
        colname_proba = "proba_" + state

        dict_qte_size["n_" + state] = len(
            table_cells[table_cells[colname_decision] == state_idx]
        )
        dict_qte_size["n_" + state + "_proba"] = table_cells[colname_proba].sum()
        dict_qte_size["mean_size_" + state] = table_cells[
            table_cells[colname_decision] == state_idx
        ]["size"].mean()
        dict_qte_size["std_size_" + state] = table_cells[
            table_cells[colname_decision] == state_idx
        ]["size"].std()
    return dict_qte_size


# n_cells,n_cells_per_state,n_cells_per_state_proba, mean_size_per_state,std_size_per_state = _get_qte_size_from_table_cells(table_cells_slide,time_per_tile_slide,model_classification)


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


""" ROI """


def create_load_roi_df(roi):
    """
    Get roi info
    """
    path_roi_info = os.path.join(roi.path_roi, "roi_info.csv")
    if os.path.exists(path_roi_info):
        df_roi_info = pd.read_csv(path_roi_info, sep=";")
        return df_roi_info, path_roi_info
    else:
        print("Construction du dataframe lié à la ROI")
        df_slides, path_temporal_analysis_csv = create_load_df_slides()
        colnames_slide = list(df_slides.columns)
        colnames_features_roi = [
            "physiological_part",
            "group_for_comparison",
            "roi_loc",
            "roi_shape",
            "area_roi",
            "area_tissue_roi",
            "area_physiological_part_roi",
            "fraction_tissue_roi",
            "fraction_physiological_part_roi",
            "fraction_tot_tissue_in_roi",
        ]
        colname_model_segmentation_classification = [
            "model_segmentation_roi",
            "model_classification_roi",
        ]
        colnames_features_nuclei_density = [
            "n_nuclei_in_roi",
            "mean_nuclei_density_roi",
            "std_nuclei_density_roi",
            "ratio_nuclei_density_roi_vs_slide",
        ]
        colnames_ncells = [
            "n_cells_roi",
            "fraction_tot_cells_in_roi",
            "mean_n_cells_per_tile_roi",
            "std_n_cells_per_tile_roi",
        ] + [
            "n_" + state + "_roi"
            for state in roi.model_classification.param_model["param_training_set"][
                "state_names"
            ][1:]
        ]
        colnames_ncells_proba_conservation = [
            "n_" + state + "_proba_roi"
            for state in roi.model_classification.param_model["param_training_set"][
                "state_names"
            ][1:]
        ]
        colnames_ncells_tx = [
            "fraction_tot_" + state + "_cells_in_roi"
            for state in roi.model_classification.param_model["param_training_set"][
                "state_names"
            ][1:]
        ]
        colnames_ncells_tx_proba = [
            "fraction_tot_" + state + "_cells_proba_in_roi"
            for state in roi.model_classification.param_model["param_training_set"][
                "state_names"
            ][1:]
        ]
        colnames_size = (
            ["mean_cell_size_roi"]
            + [
                "mean_size_" + state + "_roi"
                for state in roi.model_classification.param_model["param_training_set"][
                    "state_names"
                ][1:]
            ]
            + ["std_cell_size_roi"]
            + [
                "std_size_" + state + "_roi"
                for state in roi.model_classification.param_model["param_training_set"][
                    "state_names"
                ][1:]
            ]
        )
        colnames = (
            colnames_slide
            + colnames_features_roi
            + colname_model_segmentation_classification
            + colnames_features_nuclei_density
            + colnames_ncells
            + colnames_ncells_proba_conservation
            + colnames_ncells_tx
            + colnames_ncells_tx_proba
            + colnames_size
        )
        df_roi_info = pd.DataFrame(columns=colnames)
        df_roi_info.to_csv(path_roi_info, sep=";", index=False)
        return df_roi_info, path_roi_info


def add_roi_results_to_roi_df(roi):
    """
    Complète la partie ROI du dataframe sur les slides
    Option : df_slides = df_slides.sort_values('pcw')
    """
    df_slides, path_temporal_analysis_csv = create_load_df_slides()
    row_slide = df_slides[df_slides["slide_num"] == roi.slide_num]
    df_roi_info, path_roi_info = create_load_roi_df(roi)
    if df_roi_info.shape[0] > 0:
        None
    roi.get_time_per_tile_roi()

    dict_qte_size = _get_qte_size_from_table_cells(
        roi.table_cells, roi.time_per_tile_roi, roi.model_classification
    )
    state_names = roi.model_classification.param_model["param_training_set"][
        "state_names"
    ][1:]

    dict_info_roi = dict()
    for col in row_slide.columns:
        dict_info_roi[col] = row_slide[col].values[0]
    # tissue_percent_roi = roi.time_per_tile_roi["tissu_percent"].mean()
    slide_shape = _get_slide_shape(roi.slide_num)
    dict_info_roi["physiological_part"] = roi.physiological_part
    dict_info_roi["group_for_comparison"] = roi.group_for_comparison
    dict_info_roi["roi_loc"] = (
        roi.origin_row,
        roi.origin_col,
        roi.end_row,
        roi.end_col,
    )
    dict_info_roi["roi_shape"] = roi.shape
    dict_info_roi["area_roi"] = roi.shape[0] * roi.shape[1]
    dict_info_roi["area_tissue_roi"] = np.sum(
        roi.mask_tissue_roi
    )  # devrait etre pareil que roi.time_per_tile_roi["tissu_percent"].sum()*(dataset_config.tile_height**2)
    dict_info_roi["area_physiological_part_roi"] = np.sum(roi.mask_physiopart_roi)
    dict_info_roi["fraction_tissue_roi"] = (
        dict_info_roi["area_tissue_roi"] / dict_info_roi["area_roi"]
    )
    dict_info_roi["fraction_physiological_part_roi"] = (
        dict_info_roi["area_physiological_part_roi"] / dict_info_roi["area_roi"]
    )
    dict_info_roi["fraction_tot_tissue_in_roi"] = (
        dict_info_roi["area_tissue_roi"] / row_slide["area_tissue_slide"].values[0]
    )
    dict_info_roi[
        "model_segmentation_roi"
    ] = roi.model_segmentation.model_segmentation_type
    dict_info_roi["model_classification_roi"] = roi.model_classification.param_model[
        "model_name"
    ]

    dict_info_roi["n_nuclei_in_roi"] = 2
    dict_info_roi["mean_nuclei_density_roi"] = 2
    dict_info_roi["std_nuclei_density_roi"] = 2
    dict_info_roi["ratio_nuclei_density_roi_vs_slide"] = (
        dict_info_roi["mean_nuclei_density_roi"]
        / row_slide["mean_nuclei_density_slide"].values[0]
    )
    dict_info_roi["n_cells_roi"] = dict_qte_size["n_cells"]
    dict_info_roi["fraction_tot_cells_in_roi"] = (
        dict_info_roi["n_cells_roi"] / row_slide["n_cells_slide"].values[0]
    )
    dict_info_roi["mean_n_cells_per_tile_roi"] = dict_qte_size["mean_n_cells_per_tile"]
    dict_info_roi["std_n_cells_per_tile_roi"] = dict_qte_size["std_n_cells_per_tile"]
    for state in state_names:
        dict_info_roi["n_" + state + "_roi"] = dict_qte_size["n_" + state]
        dict_info_roi["n_" + state + "_proba_roi"] = dict_qte_size[
            "n_" + state + "_proba"
        ]
        dict_info_roi["fraction_tot_" + state + "_cells_in_roi"] = (
            dict_info_roi["n_" + state + "_roi"]
            / row_slide["n_" + state + "_slide"].values[0]
        )
        dict_info_roi["fraction_tot_" + state + "_cells_proba_in_roi"] = (
            dict_info_roi["n_" + state + "_proba_roi"]
            / row_slide["n_" + state + "_proba_slide"].values[0]
        )
    dict_info_roi["mean_cell_size_roi"] = dict_qte_size["mean_cell_size"]
    dict_info_roi["std_cell_size_roi"] = dict_qte_size["std_cell_size"]
    for state in state_names:
        dict_info_roi["mean_size_" + state + "_roi"] = dict_qte_size[
            "mean_size_" + state
        ]
        dict_info_roi["std_size_" + state + "_roi"] = dict_qte_size["std_size_" + state]

    df_roi_info = df_roi_info.append(dict_info_roi, ignore_index=True)
    df_roi_info.to_csv(path_roi_info, sep=";", index=False)

    return df_roi_info


""" Include State-State colocalisation"""


def load_coloc_results(
    roi,
    list_levelsets,
    compute_coloc_if_nexist=True,
    save_fig_B_cells_in_A_levelsets=False,
):
    """
    Return table_cells_slide en fonction du model de segmentation et du model de classification.
    """
    verbose = 0
    path_coloc = os.path.join(roi.path_classification, "2_State_state_colocalisation")
    path_coloc_results = os.path.join(
        path_coloc,
        colocalisation_analysis.build_path_levelsets(list_levelsets),
        "df_colocalisation_global_idx.csv",
    )
    if os.path.exists(path_coloc_results):
        coloc_results = pd.read_csv(path_coloc_results, sep=";")
    else:
        if compute_coloc_if_nexist:
            print("Go calculer la coloc dans load_coloc_results")
            # print("Go calculer la coloc ")
            colocalanysis_roi = colocalisation_analysis.Coloc_analysis(
                roi,
                list_levelsets=list_levelsets,
                save_fig_B_cells_in_A_levelsets=save_fig_B_cells_in_A_levelsets,
                save_fig_zscore=True,
                save_csv=True,
                save_fig_features_omega_level=False,
                display=False,
                verbose=1,
            )
            coloc_results = pd.read_csv(path_coloc_results, sep=";")
        else:
            raise NameError("La colocalisation n'a pas été calculé avec ces levelsets")
    return coloc_results, path_coloc_results


def add_coloc_results_to_roi_df(
    roi,
    list_levelsets,
    compute_coloc_if_nexist=True,
    save_fig_B_cells_in_A_levelsets=False,
):
    """Ajoute les informations de la colocalisation au dataframe d'une ROI
    Pop_A Pop_B fait qu'il y a 5x4 = 20 colonnes pour chaque jeu de levelsets
    """
    coloc_results, path_coloc_results = load_coloc_results(
        roi,
        list_levelsets,
        compute_coloc_if_nexist=compute_coloc_if_nexist,
        save_fig_B_cells_in_A_levelsets=save_fig_B_cells_in_A_levelsets,
    )

    path_roi_coloc = os.path.join(roi.path_roi, "roi_coloc.csv")
    df_roi_info, path_roi_info = create_load_roi_df(roi)
    pop_AB_liste = [
        (state_A, state_B)
        for state_A in roi.model_classification.param_model["param_training_set"][
            "state_names"
        ][1:]
        for state_B in roi.model_classification.param_model["param_training_set"][
            "state_names"
        ][1:]
        if state_B != state_A
    ]

    columns_coloc = [
        "Accumulation_Index_ACI",
        "Significant_Accumulation_Index_SAI",
        "Association_Index_ASI",
        "p_value",
        "delta_a",
    ]
    columns_commun_val = ["n_else_than_A", "n_B_proba", "n_B"]
    colnames_features_slide_roi_coloc = list(df_roi_info.columns)

    if os.path.exists(path_roi_coloc):
        df_roi_coloc = pd.read_csv(path_roi_coloc, sep=";")
        if (
            len(
                df_roi_coloc[
                    (
                        df_roi_coloc["model_segmentation_roi"]
                        == roi.model_segmentation.model_segmentation_type
                    )
                    & (
                        df_roi_coloc["model_classification_roi"]
                        == roi.model_classification.param_model["model_name"]
                    )
                    & (
                        df_roi_coloc["levelsets_state_state"]
                        == ("_").join(list(map(str, list_levelsets)))
                    )
                ]
            )
            > 0
        ):
            return df_roi_coloc
    else:
        df_roi_coloc = pd.DataFrame(
            columns=colnames_features_slide_roi_coloc
        )  # On met deja les colonnes du df  slide + roi

    for pop_A, pop_B in pop_AB_liste:
        row_slide_roi_pop_A_pop_B = df_roi_info[
            np.logical_and(
                df_roi_info["model_segmentation_roi"]
                == roi.model_segmentation.model_segmentation_type,
                df_roi_info["model_classification_roi"]
                == roi.model_classification.param_model["model_name"],
            )
        ]
        row_slide_roi_pop_A_pop_B["pop_A"] = pop_A
        row_slide_roi_pop_A_pop_B["pop_B"] = pop_B
        row_slide_roi_pop_A_pop_B["levelsets_state_state"] = ("_").join(
            list(map(str, list_levelsets))
        )
        for col in columns_commun_val:
            val = coloc_results[
                (coloc_results["pop_B"] == pop_B) & (coloc_results["proba"] == 0)
            ][col].values[0]
            row_slide_roi_pop_A_pop_B["coloc_" + col] = val
        for col in columns_coloc:
            colname_ = "coloc_" + col
            colname_proba = "coloc_" + col + "_proba"
            if (
                len(
                    coloc_results[
                        (coloc_results["pop_A"] == pop_A)
                        & (coloc_results["pop_B"] == pop_B)
                        & (coloc_results["proba"] == 0)
                    ][col]
                )
                == 0
            ):
                val = np.nan
                val_proba = np.nan
            else:
                val = coloc_results[
                    (coloc_results["pop_A"] == pop_A)
                    & (coloc_results["pop_B"] == pop_B)
                    & (coloc_results["proba"] == 0)
                ][col].values[0]
                val_proba = coloc_results[
                    (coloc_results["pop_A"] == pop_A)
                    & (coloc_results["pop_B"] == pop_B)
                    & (coloc_results["proba"] == 1)
                ][col].values[0]
            row_slide_roi_pop_A_pop_B[colname_] = val
            row_slide_roi_pop_A_pop_B[colname_proba] = val_proba
            # print(col +" val : ",val )
            # print(col +" val proba : ",val_proba )
        row_slide_roi_pop_A_pop_B = pd.DataFrame(row_slide_roi_pop_A_pop_B)
        df_roi_coloc = pd.concat([df_roi_coloc, row_slide_roi_pop_A_pop_B], axis=0)
        # df_roi_coloc=df_roi_coloc.append(row_slide_roi_pop_A_pop_B, ignore_index=True)
    df_roi_coloc.to_csv(path_roi_coloc, sep=";", index=False)
    return df_roi_coloc


"""  Include info on dbscan """


def get_dbscan_results(
    roi,
    param_experiences_pts_removing,
    min_sample=4,
    compute_dbscan_if_nexist=True,
    background=False,
    display_convex_hull_clusters=False,
):
    """
    Return table_cells_slide en fonction du model de segmentation et du model de classification.
    """
    path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
    path_save_clustering_ms = os.path.join(
        path_save_clustering, "min_sample_" + str(min_sample).zfill(2)
    )
    path_dbscan_results = os.path.join(
        path_save_clustering_ms, "dbscan_metrics_analysis.csv"
    )

    if os.path.exists(path_dbscan_results):
        dbscan_results = pd.read_csv(path_dbscan_results, sep=";")
        if min_sample in dbscan_results["min_sample_A"].values:
            return dbscan_results, path_dbscan_results
        else:
            if compute_dbscan_if_nexist:
                dbscan.compute_dbscan_on_roi(
                    roi,
                    param_experiences_pts_removing=None,
                    min_sample=min_sample,
                    liste_eps=None,
                    save_data=True,
                    save_fig=True,
                    display_fig=False,
                    figsize=(15, 12),
                    background=background,
                    display_convex_hull_clusters=display_convex_hull_clusters,
                    create_gif_dbscan=False,
                )
            else:
                raise ValueError("DBSCAN not computed for this ROI")
    else:
        if compute_dbscan_if_nexist:
            dbscan.compute_dbscan_on_roi(
                roi,
                param_experiences_pts_removing=None,
                min_sample=min_sample,
                liste_eps=None,
                save_data=True,
                save_fig=True,
                display_fig=False,
                figsize=(10, 10),
                background=background,
                display_convex_hull_clusters=display_convex_hull_clusters,
                create_gif_dbscan=False,
            )
        else:
            raise ValueError("DBSCAN not computed for this ROI")
    dbscan_results = pd.read_csv(path_dbscan_results, sep=";")
    return dbscan_results, path_dbscan_results


def add_dbscan_results_to_roi_df(
    roi,
    param_experiences_pts_removing,
    min_sample=4,
    compute_dbscan_if_nexist=True,
    background=False,
    display_convex_hull_clusters=False,
):
    dbscan_results, path_dbscan_results = get_dbscan_results(
        roi,
        param_experiences_pts_removing,
        min_sample=min_sample,
        compute_dbscan_if_nexist=True,
        background=background,
        display_convex_hull_clusters=display_convex_hull_clusters,
    )

    path_roi_coloc = os.path.join(roi.path_roi, "roi_coloc.csv")
    roi_coloc_df = pd.read_csv(path_roi_coloc, sep=";")
    columns_dbscan = [
        "epsilon_A",
        "min_sample_A",
        "epsilon_B",
        "min_sample_B",
        "n_clusters_A",
        "n_clustered_cells_A",
        "n_isolated_A",
        "n_clusters_B",
        "n_clustered_cells_B",
        "n_isolated_B",
        "iou",
        "i_A",
        "i_B",
        "fraction_A_in_B_clusters",
        "fraction_B_in_A_clusters",
        "fraction_clusterised_A",
        "fraction_clusterised_B",
        "fraction_clustered_A_in_intersect",
        "fraction_clustered_B_in_intersect",
        "threshold_conserved_area",
    ]
    colnames_features_slide_roi_coloc_dbscan = list(roi_coloc_df.columns) + [
        "dbscan_" + col for col in columns_dbscan
    ]
    df_roi_coloc_dbscan = pd.DataFrame(columns=colnames_features_slide_roi_coloc_dbscan)

    for (
        idx_row,
        row,
    ) in (
        roi_coloc_df.iterrows()
    ):  # Je prend ligne a ligne le df slide-roi-coloc et j'ajoute les colonnes puis je concatene au df final
        pop_A = row["pop_A"]
        pop_B = row["pop_B"]
        for col in columns_dbscan:
            # if col in ['fraction_clustered_A_in_intersection_wt_B','fraction_clustered_B_in_intersection_wt_A']:
            #     # print("Todo : "+col+" n'est pas calculé dans le fichier result dbscan ")
            #     val = 777
            # else :
            val = dbscan_results[
                (dbscan_results["pop_A"] == pop_A) & (dbscan_results["pop_B"] == pop_B)
            ][col].values[0]
            colname_ = "dbscan_" + col
            row[colname_] = val

        df_roi_coloc_dbscan = pd.concat([df_roi_coloc_dbscan, row.to_frame().T], axis=0)
        # df_roi_coloc=df_roi_coloc.append(row_slide_roi_pop_A_pop_B, ignore_index=True)

    path_roi_coloc_dbscan = os.path.join(roi.path_roi, "roi_coloc_dbscan.csv")
    df_roi_coloc_dbscan.to_csv(path_roi_coloc_dbscan, sep=";", index=False)
    return df_roi_coloc_dbscan


""" Include State - Border colocalisation  """


def load_coloc_state_border_results(
    roi,
    list_levelsets,
    compute_coloc_if_nexist=True,
    save_figure_B_in_border_levelsets=False,
    display=False,
):
    """
    Return table_cells_slide en fonction du model de segmentation et du model de classification.
    """
    verbose = 0
    path_coloc = os.path.join(roi.path_classification, "3_Cell_border_colocalisation")
    path_bcoloc_results = os.path.join(
        path_coloc,
        colocalisation_analysis.build_path_levelsets(list_levelsets),
        "Cell_border_colocalisation.csv",
    )
    if os.path.exists(path_bcoloc_results):
        bcoloc_results = pd.read_csv(path_bcoloc_results, sep=";")
    else:
        if compute_coloc_if_nexist:
            colocalisation_analysis.Coloc_analysis_border(
                roi,
                list_levelsets=list_levelsets,
                save_figure_B_in_border_levelsets=save_figure_B_in_border_levelsets,
                save_fig_zscore=True,
                save_csv=True,
                save_fig_features_omega_level=False,
                display=display,
                verbose=1,
            )
            bcoloc_results = pd.read_csv(path_bcoloc_results, sep=";")
        else:
            raise NameError("La colocalisation n'a pas été calculé avec ces levelsets")
    return bcoloc_results, path_bcoloc_results


def add_coloc_state_border_results_to_roi_df(
    roi,
    list_levelsets,
    compute_coloc_if_nexist=True,
    save_figure_B_in_border_levelsets=False,
    display=False,
):
    """ """
    bcoloc_results, path_bcoloc_results = load_coloc_state_border_results(
        roi,
        list_levelsets,
        compute_coloc_if_nexist=compute_coloc_if_nexist,
        save_figure_B_in_border_levelsets=save_figure_B_in_border_levelsets,
        display=display,
    )

    path_roi_coloc_dbscan = os.path.join(roi.path_roi, "roi_coloc_dbscan.csv")
    roi_coloc_dbscan_df = pd.read_csv(path_roi_coloc_dbscan, sep=";")

    columns_bcoloc = [
        "coloc_state_border_accumulation_index_ACI",
        "coloc_state_border_significant_accumulation_index_SAI",
        "coloc_state_border_association_index_ASI",
        "coloc_state_border_delta_a",
        "coloc_state_border_p_value",
    ]

    columns_commun_val = [
        "coloc_state_border_n_cells_roi",
        "coloc_state_border_n_B_proba",
        "coloc_state_border_n_B",
    ]
    colnames_features_slide_roi_coloc_dbscan_bcoloc = list(
        roi_coloc_dbscan_df.columns
    )  # +["coloc_state_border_levelsets"]+ [col for col in columns_bcoloc]
    df_roi_coloc_dbscan_bcoloc = pd.DataFrame(
        columns=colnames_features_slide_roi_coloc_dbscan_bcoloc
    )
    for idx_row, row in roi_coloc_dbscan_df.iterrows():
        pop_A = row["pop_A"]
        pop_B = row["pop_B"]

        row["coloc_state_border_levelsets"] = ("_").join(list(map(str, list_levelsets)))
        for col in columns_commun_val:
            val = bcoloc_results[
                (bcoloc_results["pop_B"] == pop_B) & (bcoloc_results["proba"] == 0)
            ][col].values[0]
            row[col] = val

        for col in columns_bcoloc:
            colname_ = col
            colname_proba = col + "_proba"
            val = bcoloc_results[
                (bcoloc_results["pop_B"] == pop_B) & (bcoloc_results["proba"] == 0)
            ][col].values[0]
            val_proba = bcoloc_results[
                (bcoloc_results["pop_B"] == pop_B) & (bcoloc_results["proba"] == 1)
            ][col].values[0]
            row[colname_] = val
            row[colname_proba] = val_proba
        df_roi_coloc_dbscan_bcoloc = pd.concat(
            [df_roi_coloc_dbscan_bcoloc, row.to_frame().T], axis=0
        )
    path_roi_coloc_dbscan_bcoloc = os.path.join(
        roi.path_roi, "roi_coloc_dbscan_bcoloc.csv"
    )
    df_roi_coloc_dbscan_bcoloc.to_csv(
        path_roi_coloc_dbscan_bcoloc, sep=";", index=False
    )
    return df_roi_coloc_dbscan_bcoloc


""" Include Neighbours analysis  """


def load_neighbours_analysis_results(roi, list_levelsets, compute_if_nexist=True):
    """ """
    verbose = 0
    path_neighbours = os.path.join(roi.path_classification, "5_Neighbours_analysis")
    path_neighbours_results = os.path.join(
        path_neighbours, "df_neighbours_analysis.csv"
    )
    if os.path.exists(path_neighbours_results):
        neighbours_results = pd.read_csv(path_neighbours_results, sep=";")
    else:
        if compute_if_nexist:
            print("La neighbours analysis va être calculée")
            df_neighbours_analysis = neighbours_analysis.stats_neighbors_analysis_A_B(
                roi, list_levelsets, save_csv=True
            )
            neighbours_results = pd.read_csv(path_neighbours_results, sep=";")
        else:
            raise NameError("Neighbours analysis n'a pas été calculé")
    return neighbours_results, path_neighbours_results


def add_neighbours_results_to_roi_df(roi, list_levelsets, compute_if_nexist=True):
    """ """
    neighbours_results, path_neighbours_results = load_neighbours_analysis_results(
        roi, list_levelsets, compute_if_nexist=compute_if_nexist
    )
    # previous df
    path_roi_coloc_dbscan_bcoloc = os.path.join(
        roi.path_roi, "roi_coloc_dbscan_bcoloc.csv"
    )
    roi_coloc_dbscan_bcoloc_df = pd.read_csv(path_roi_coloc_dbscan_bcoloc, sep=";")

    # New features
    columns_first_B_around_A = [
        "neighbours_mean_dist_first_B_around_A",
        "neighbours_mean_dist_second_B_around_A",
        "neighbours_mean_dist_third_B_around_A",
        "neighbours_std_dist_first_B_around_A",
        "neighbours_std_dist_second_B_around_A",
        "neighbours_std_dist_third_B_around_A",
    ]
    columns_rate_A_balls_containing_1_B = [
        "neighbours_rate_balls_r_coloc_containing_1_B",
        "neighbours_r_coloc",
    ]
    columns_rate_B_first = [
        "neighbours_rate_B_first_neighbour"
    ]  # ,"rate_B_second_neighbour","rate_B_third_neighbour"]
    columns_names_A_neighbours = [
        "neighbours_mean_A_first_neighbour",
        "neighbours_mean_A_second_neighbour",
        "neighbours_mean_A_third_neighbour",
        "neighbours_std_A_first_neighbour",
        "neighbours_std_A_second_neighbour",
        "neighbours_std_A_third_neighbour",
    ]
    columns_neighbours = (
        columns_first_B_around_A
        + columns_rate_A_balls_containing_1_B
        + columns_rate_B_first
        + columns_names_A_neighbours
    )

    # colnames_features_slide_DeepCellMap_roi = list(roi_coloc_dbscan_bcoloc_df.columns) + [col for col in columns_neighbours]
    deepcellmap_results_on_roi = pd.DataFrame(
        columns=list(roi_coloc_dbscan_bcoloc_df.columns)
    )
    for (
        idx_row,
        row,
    ) in roi_coloc_dbscan_bcoloc_df.iterrows():  # for each row of the previous df
        pop_A = row["pop_A"]
        pop_B = row["pop_B"]
        for col in columns_neighbours:
            colname = col
            val = neighbours_results[
                (neighbours_results["pop_A"] == pop_A)
                & (neighbours_results["pop_B"] == pop_B)
            ][col].values[0]
            row[colname] = val

        deepcellmap_results_on_roi = pd.concat(
            [deepcellmap_results_on_roi, row.to_frame().T], axis=0, ignore_index=True
        )

    path_DeepCellMap_roi = os.path.join(roi.path_roi, "DeepCellMap_roi.csv")
    deepcellmap_results_on_roi.to_csv(path_DeepCellMap_roi, sep=";", index=False)
    return deepcellmap_results_on_roi


""" Gather informations """


def apply_DeepCellMap_to_roi(
    roi_info,
    param_config_rois_reconstruction,
    param_colocalisation,
    param_dbscan_cluster_analysis
    ):
    """Create df_roi_coloc_dbscan for a ROI
    SAVING : ROI folder
    """
    save_img_classified_cells = True 
    model_name = param_config_rois_reconstruction["model_name"]
    training_set_config_name = param_config_rois_reconstruction["training_set_config_name"]
    save_table_cells = param_config_rois_reconstruction["save_table_cells"]
    save_img_classified_cells = param_config_rois_reconstruction["param_fig_classified_cells"]["save_img_classified_cells"]
    verbose = param_config_rois_reconstruction["verbose"]
    get_rgb_wsi = param_config_rois_reconstruction["get_rgb_wsi"]
    save_table_cells = param_config_rois_reconstruction["save_table_cells"]
    save_img_classified_cells = param_config_rois_reconstruction["param_fig_classified_cells"]["save_img_classified_cells"]
    display_fig = param_config_rois_reconstruction["param_fig_classified_cells"]["display_fig"]
    figsize = param_config_rois_reconstruction["param_fig_classified_cells"]["figsize"]
    with_roi_delimiter = param_config_rois_reconstruction["param_fig_classified_cells"]["with_roi_delimiter"]
    with_center_of_mass = param_config_rois_reconstruction["param_fig_classified_cells"]["with_center_of_mass"]
    with_background = param_config_rois_reconstruction["param_fig_classified_cells"]["with_background"]
    display_example_each_states = param_config_rois_reconstruction["display_example_each_states"]
    nb_cells_per_state_to_display = param_config_rois_reconstruction["nb_cells_per_state_to_display"]

    list_levelsets = param_colocalisation["list_levelsets"]
    save_fig_B_cells_in_A_levelsets = param_colocalisation["save_fig_B_cells_in_A_levelsets"]
    min_sample = param_dbscan_cluster_analysis["min_sample"]
    param_experiences_pts_removing = param_dbscan_cluster_analysis["param_experiences_pts_removing"]
    save_figure_dbscan = param_dbscan_cluster_analysis["save_figure_dbscan"]

    display_convex_hull_clusters = param_dbscan_cluster_analysis["display_convex_hull_clusters"]
    save_figure_B_in_border_levelsets = param_colocalisation["save_figure_B_in_border_levelsets"]

    background = with_background

    path_roi = get_roi_path(roi_info)
    roi = RegionOfInterest(roi_info,
        model_name=model_name,
        training_set_config_name=training_set_config_name,
        get_rgb_wsi=get_rgb_wsi,
        save_table_cells=save_table_cells,
        save_img_classified_cells=False,
        display_rect_roi_in_tissue = False,
        verbose=verbose)
    roi.create_fig_cells_on_tissue(
        with_background=True,
        save_fig=True,
        display_fig=display_fig,
        figsize=figsize,
        with_roi_delimiter=with_roi_delimiter,
        with_center_of_mass=with_center_of_mass,
        save_to_commun_path=None,
    ) if save_img_classified_cells else None
    print(yellow("Generation of examples of cells of each morphology"))
    roi.display_example_each_states(
        save_fig=True, nb_cells_per_state_to_display=nb_cells_per_state_to_display
    ) if display_example_each_states else None 

    df_slides = add_slide_to_slides_df(
        roi.slide_num, roi.model_segmentation, roi.model_classification
    )
    df_roi_info = add_roi_results_to_roi_df(roi)
    print(yellow("Computing cell-cell colocalisation"))
    df_roi_coloc = add_coloc_results_to_roi_df(
        roi,
        list_levelsets,
        compute_coloc_if_nexist=True,
        save_fig_B_cells_in_A_levelsets=save_fig_B_cells_in_A_levelsets,
    )
    print(yellow("Computing dbscan-based clusters analysis results "))
    df_roi_coloc_dbscan = add_dbscan_results_to_roi_df(
        roi,
        param_experiences_pts_removing,
        min_sample=min_sample,
        compute_dbscan_if_nexist=True,
        background=background,
        display_convex_hull_clusters=display_convex_hull_clusters,
    )
    print(yellow("Computing Cell-Region's border results"))
    df_roi_coloc_dbscan_bcoloc = add_coloc_state_border_results_to_roi_df(
        roi,
        list_levelsets,
        compute_coloc_if_nexist=True,
        save_figure_B_in_border_levelsets=save_figure_B_in_border_levelsets,
    )
    print(yellow("Computing neighbors analysis results"))
    deepcellmap_results_on_roi = add_neighbours_results_to_roi_df(
        roi, list_levelsets, compute_if_nexist=True
    )
    # else :
    #    print("Les calculs ont déjà été fait")

def apply_DeepCellMap_to_roi_GV(
    Roi,
    param_config_rois_reconstruction,
    param_colocalisation,
    param_dbscan_cluster_analysis
    ):
    """Create df_roi_coloc_dbscan for a ROI
    SAVING : ROI folder
    """
    model_name = param_config_rois_reconstruction["model_name"]
    training_set_config_name = param_config_rois_reconstruction["training_set_config_name"]
    save_table_cells = param_config_rois_reconstruction["save_table_cells"]
    save_img_classified_cells = param_config_rois_reconstruction["param_fig_classified_cells"]["save_img_classified_cells"]
    verbose = param_config_rois_reconstruction["verbose"]
    get_rgb_wsi = param_config_rois_reconstruction["get_rgb_wsi"]
    save_table_cells = param_config_rois_reconstruction["save_table_cells"]
    save_img_classified_cells = param_config_rois_reconstruction["param_fig_classified_cells"]["save_img_classified_cells"]
    display_fig = param_config_rois_reconstruction["param_fig_classified_cells"]["display_fig"]
    figsize = param_config_rois_reconstruction["param_fig_classified_cells"]["figsize"]
    with_roi_delimiter = param_config_rois_reconstruction["param_fig_classified_cells"]["with_roi_delimiter"]
    with_center_of_mass = param_config_rois_reconstruction["param_fig_classified_cells"]["with_center_of_mass"]
    with_background = param_config_rois_reconstruction["param_fig_classified_cells"]["with_background"]
    display_example_each_states = param_config_rois_reconstruction["display_example_each_states"]
    nb_cells_per_state_to_display = param_config_rois_reconstruction["nb_cells_per_state_to_display"]

    list_levelsets = param_colocalisation["list_levelsets"]
    save_fig_B_cells_in_A_levelsets = param_colocalisation["save_fig_B_cells_in_A_levelsets"]
    min_sample = param_dbscan_cluster_analysis["min_sample"]
    param_experiences_pts_removing = param_dbscan_cluster_analysis["param_experiences_pts_removing"]
    save_figure_dbscan = param_dbscan_cluster_analysis["save_figure_dbscan"]

    display_convex_hull_clusters = param_dbscan_cluster_analysis["display_convex_hull_clusters"]
    save_figure_B_in_border_levelsets = param_colocalisation["save_figure_B_in_border_levelsets"]

    background = with_background

    path_roi = get_roi_path(roi_info)
    roi = RegionOfInterest(roi_info,
        model_name=model_name,
        training_set_config_name=training_set_config_name,
        get_rgb_wsi=get_rgb_wsi,
        save_table_cells=save_table_cells,
        save_img_classified_cells=False,
        display_rect_roi_in_tissue = False,
        verbose=verbose)
    roi.create_fig_cells_on_tissue(
        with_background=True,
        save_fig=True,
        display_fig=display_fig,
        figsize=figsize,
        with_roi_delimiter=with_roi_delimiter,
        with_center_of_mass=with_center_of_mass,
        save_to_commun_path=None,
    ) if save_img_classified_cells else None
    print(yellow("Generation of examples of cells of each morphology"))
    roi.display_example_each_states(
        save_fig=True, nb_cells_per_state_to_display=nb_cells_per_state_to_display
    ) if display_example_each_states else None 

    df_slides = add_slide_to_slides_df(
        roi.slide_num, roi.model_segmentation, roi.model_classification
    )
    df_roi_info = add_roi_results_to_roi_df(roi)
    print(yellow("Computing cell-cell colocalisation"))
    df_roi_coloc = add_coloc_results_to_roi_df(
        roi,
        list_levelsets,
        compute_coloc_if_nexist=True,
        save_fig_B_cells_in_A_levelsets=save_fig_B_cells_in_A_levelsets,
    )
    print(yellow("Computing dbscan-based clusters analysis results "))
    df_roi_coloc_dbscan = add_dbscan_results_to_roi_df(
        roi,
        param_experiences_pts_removing,
        min_sample=min_sample,
        compute_dbscan_if_nexist=True,
        background=background,
        display_convex_hull_clusters=display_convex_hull_clusters,
    )
    print(yellow("Computing Cell-Region's border results"))
    df_roi_coloc_dbscan_bcoloc = add_coloc_state_border_results_to_roi_df(
        roi,
        list_levelsets,
        compute_coloc_if_nexist=True,
        save_figure_B_in_border_levelsets=save_figure_B_in_border_levelsets,
    )
    print(yellow("Computing neighbors analysis results"))
    deepcellmap_results_on_roi = add_neighbours_results_to_roi_df(
        roi, list_levelsets, compute_if_nexist=True
    )
    # else :
    #    print("Les calculs ont déjà été fait")

def reconstruct_and_visualise_roi(roi_info, param_config_rois_reconstruction):
    """Reconstruct and save :
    - 1.Position roi in tissue
    - 2.RGB with prediction
    - 4.Example of cells for each state
    """
    ## Parameters
    model_name = param_config_rois_reconstruction["model_name"]
    training_set_config_name = param_config_rois_reconstruction[
        "training_set_config_name"
    ]
    get_rgb_wsi = param_config_rois_reconstruction["get_rgb_wsi"]
    save_table_cells = param_config_rois_reconstruction["save_table_cells"]

    save_img_classified_cells = param_config_rois_reconstruction[
        "param_fig_classified_cells"
    ]["save_img_classified_cells"]
    param_fig_classified_cells = param_config_rois_reconstruction[
        "param_fig_classified_cells"
    ]
    display_fig = param_config_rois_reconstruction["param_fig_classified_cells"][
        "display_fig"
    ]
    figsize = param_config_rois_reconstruction["param_fig_classified_cells"]["figsize"]
    with_roi_delimiter = param_config_rois_reconstruction["param_fig_classified_cells"][
        "with_roi_delimiter"
    ]
    with_center_of_mass = param_config_rois_reconstruction[
        "param_fig_classified_cells"
    ]["with_center_of_mass"]
    with_background = param_config_rois_reconstruction["param_fig_classified_cells"][
        "with_background"
    ]
    display_example_each_states = param_config_rois_reconstruction["display_example_each_states"]
    nb_cells_per_state_to_display = param_config_rois_reconstruction[
        "nb_cells_per_state_to_display"
    ]
    verbose = param_config_rois_reconstruction["verbose"]

    roi = RegionOfInterest(
        roi_info,
        model_name=model_name,
        training_set_config_name=training_set_config_name,
        get_rgb_wsi=get_rgb_wsi,
        save_table_cells=save_table_cells,
        save_img_classified_cells=False,
        verbose=verbose,
    )
    roi.create_fig_cells_on_tissue(
        with_background=True,
        save_fig=True,
        display_fig=display_fig,
        figsize=figsize,
        with_roi_delimiter=with_roi_delimiter,
        with_center_of_mass=with_center_of_mass,
        save_to_commun_path=None,
    ) if save_img_classified_cells else None
    roi.display_example_each_states(
        save_fig=True, nb_cells_per_state_to_display=nb_cells_per_state_to_display
    ) if display_example_each_states else None 


def reconstruct_and_visualise_roi_list(
    liste_roi_to_add, dict_roi, param_config_rois_reconstruction
):
    """
    Apply reconstruct_and_visualise_roi to a list of ROIs
    """
    for nb_subregion, data_name in enumerate(liste_roi_to_add):
        print(
            "Computing ROI n°"
            + str(nb_subregion + 1)
            + "/"
            + str(len(liste_roi_to_add))
        )
        roi_info = dict_roi[data_name]
        reconstruct_and_visualise_roi(
            roi_info, param_config_rois_reconstruction=param_config_rois_reconstruction
        )

def apply_DeepCellMap_to_roi_list(
    liste_roi_to_add,
    dict_roi,
    param_config_rois_reconstruction,
    param_colocalisation,
    param_dbscan_cluster_analysis,
):
    """Compute df with all the columns on a ROI (slide, roi, coloc, dbscan)
    SAVING : ROIs folder
    """
    for nb_subregion, data_name in enumerate(liste_roi_to_add):
        print("Computing DeepCellMap on subregion n°"+ str(nb_subregion + 1)+ "/"+ str(len(liste_roi_to_add)))
        roi_info = dict_roi[data_name]
        apply_DeepCellMap_to_roi(
            roi_info,
            param_config_rois_reconstruction,
            param_colocalisation,
            param_dbscan_cluster_analysis
        )


def _add_nb_group_comparison(roi_analysis, df):
    """
    Dans chaque groupes de comparaison il peut y avoir plusieurs régions de la même slide. Donc il faut que j’attribue un identifiant à chaque roi ajouté au groupe.
    C’est fait automatiquement lorsque ROI est ajoutée au dataframe_ROIs.
    Si une roi appartient à plusieurs groupes, ses lignes sont dupliquées autant de fois qu'il y a de groupes.
    """
    list_group_comparaison = df["group_for_comparison"].values[0]
    list_group_comparaison = literal_eval(list_group_comparaison)
    group_comparaison = int(list_group_comparaison[0])
    max_id_in_group = roi_analysis.query("group_for_comparison == @group_comparaison")[
        "id_in_group"
    ].max()
    max_id_in_group = -1 if np.isnan(max_id_in_group) else max_id_in_group
    max_id = roi_analysis["id"].max()
    df["id_in_group"] = int(max_id_in_group + 1)
    df["id"] = int(max_id + 1) if not np.isnan(max_id) else 0
    df["group_for_comparison"] = group_comparaison
    df_duplicated = df.copy()  # Boucle sur tous les groupes de comparaison de la ROI
    for group in list_group_comparaison[1:]:
        max_id_in_group = roi_analysis.query("group_for_comparison == @group")[
            "id_in_group"
        ].max()
        max_id_in_group = 0 if np.isnan(max_id_in_group) else max_id_in_group
        max_id = df_duplicated["id"].max()
        df["id_in_group"] = int(max_id_in_group + 1)
        df["id"] = int(max_id + 1) if not np.isnan(max_id) else 0
        df["group_for_comparison"] = group
        df_duplicated = pd.concat([df_duplicated, df], axis=0)
    return df_duplicated


def load_enrich_roi_analysis(liste_roi_to_add, dict_roi):
    """
    Construit le dataframe contenant toutses les informations sur plusieurs ROIs handly defined
    Args : plusieurs coordonnées de ROI et leurs mode
    Return : dataframe contenant toutes les informations sur les ROIs
    """
    physiological_part = dict_roi[liste_roi_to_add[0]]["physiological_part"]
    group_for_comparison = dict_roi[liste_roi_to_add[0]]["group_for_comparison"]
    if physiological_part is None:
        physiological_part = "group_for_comparison_" + str(group_for_comparison[0])
    path_temporal_analysis_folder = os.path.join(
        dataset_config.dir_base_stat_analysis, physiological_part
    )
    mkdir_if_nexist(path_temporal_analysis_folder)
    path_temporal_analysis_csv = os.path.join(
        path_temporal_analysis_folder, "roi_analysis.csv"
    )
    if os.path.exists(path_temporal_analysis_csv):
        roi_analysis = pd.read_csv(path_temporal_analysis_csv, sep=";")
    else:  # Si le fichier n'existe pas, on le crée en prenant le df de result d'une ROI et on creer un df avec les memes colonnes (on ne fait rien du premier df pour l'instant)
        roi_info = dict_roi[liste_roi_to_add[0]]
        path_roi = get_roi_path(roi_info)
        path_all_results_on_roi = os.path.join(path_roi, "DeepCellMap_roi.csv")
        df_all_results_on_roi = pd.read_csv(path_all_results_on_roi, sep=";")
        df_all_results_on_roi["id_in_group"] = 0
        df_all_results_on_roi["id"] = 0
        roi_analysis = pd.DataFrame(columns=df_all_results_on_roi.columns)

    for data_name in liste_roi_to_add:  # Boucle sur les noms de roi a ajouter a roi_analysis
        path_roi = get_roi_path(dict_roi[data_name])
        path_all_results_on_roi = os.path.join(path_roi, "DeepCellMap_roi.csv")
        df_all_results_on_roi = pd.read_csv(path_all_results_on_roi, sep=";")
        df_all_results_on_roi = _add_nb_group_comparison(
            roi_analysis, df_all_results_on_roi
        )

        roi_analysis = pd.concat([roi_analysis, df_all_results_on_roi], axis=0)
    # roi_analysis = roi_analysis.drop_duplicates(subset=[colname for colname in list(roi_analysis.columns) if colname not in ["id_in_group", "id"]]
    # )
    roi_analysis = roi_analysis.drop_duplicates(subset=[colname for colname in list(roi_analysis.columns) if colname not in ["id","id_in_group"]])
    roi_analysis.to_csv(path_temporal_analysis_csv, sep=";", index=False)
    return roi_analysis, path_temporal_analysis_csv


def load_enrich_roi_analysis_from_subregions(liste_roi_to_add, dict_physio_slide_num):
    """
    Builds the dataframe containing all the information on several ROIs

    Args: several ROI coordinates and their modes
    Return: dataframe containing all the information on the ROIs    
    """

    physiological_part = dict_physio_slide_num["0"]["physiological_part"]
    group_for_comparison = dict_physio_slide_num["0"]["group_for_comparison"]
    slide_num = dict_physio_slide_num["0"]["slide_num"]
    print("physiological_part : ", physiological_part)
    path_temporal_analysis_folder = os.path.join(
        dataset_config.dir_base_stat_analysis, physiological_part
    )
    mkdir_if_nexist(path_temporal_analysis_folder)
    size_max_subregion = dict_physio_slide_num["0"]["folder_name"][-3:]
    path_temporal_analysis_csv = os.path.join(
        path_temporal_analysis_folder,
        "roi_analysis_size_max_subregion_" + str(size_max_subregion) + ".csv",
    )

    if os.path.exists(path_temporal_analysis_csv):
        roi_analysis = pd.read_csv(path_temporal_analysis_csv, sep=";")
        roi_info = dict_physio_slide_num[liste_roi_to_add[0]]
        path_roi = get_roi_path(roi_info)
        pathdir = os.path.dirname(path_roi)
        path_all_results_on_roi = os.path.join(pathdir, "df_physiological_part.csv")
        df_all_results_on_roi = pd.read_csv(path_all_results_on_roi, sep=";")
        df_all_results_on_roi = _add_nb_group_comparison(
            roi_analysis, df_all_results_on_roi
        )
        roi_analysis = pd.concat([roi_analysis, df_all_results_on_roi], axis=0)
        roi_analysis["group_for_comparison"] = roi_analysis[
            "group_for_comparison"
        ].astype(int)

    else:  # Si le fichier n'existe pas, on le crée en prenant le df de result d'une ROI et on creer un df avec les memes colonnes (on ne fait rien du premier df pour l'instant)
        roi_info = dict_physio_slide_num[liste_roi_to_add[0]]
        path_roi = get_roi_path(roi_info)
        pathdir = os.path.dirname(path_roi)

        path_all_results_on_roi = os.path.join(pathdir, "df_physiological_part.csv")
        df_all_results_on_roi = pd.read_csv(path_all_results_on_roi, sep=";")
        df_all_results_on_roi["id_in_group"] = 0
        df_all_results_on_roi["id"] = 0
        roi_analysis = pd.DataFrame(columns=df_all_results_on_roi.columns)
        df_all_results_on_roi = _add_nb_group_comparison(
            roi_analysis, df_all_results_on_roi
        )
        roi_analysis = df_all_results_on_roi
        roi_analysis["group_for_comparison"] = roi_analysis[
            "group_for_comparison"
        ].astype(int)

    roi_analysis = roi_analysis.reset_index(drop=True)
    roi_analysis = roi_analysis.sort_values("pcw")
    roi_analysis = roi_analysis.drop_duplicates(
        subset=[
            colname
            for colname in list(roi_analysis.columns)
            if colname not in ["id_in_group", "id"]
        ]
    )
    roi_analysis.to_csv(path_temporal_analysis_csv, sep=";", index=False)
    return roi_analysis, path_temporal_analysis_csv


def load_roi_analysis(physiological_part, size_max_subregion):
    path_temporal_analysis_folder = os.path.join(
        dataset_config.dir_base_stat_analysis, physiological_part
    )
    path_temporal_analysis_csv = os.path.join(
        path_temporal_analysis_folder,
        "roi_analysis_size_max_subregion_" + str(size_max_subregion).zfill(3) + ".csv",
    )
    roi_analysis = pd.read_csv(path_temporal_analysis_csv, sep=";")
    print("Le dataframe est de taille ", roi_analysis.shape)
    return roi_analysis, path_temporal_analysis_csv


# Create subregions


def add_tissue_segmentation_to_draw(img, region_name, mask):
    """Ajoute les cellules d'une sous population de cellules a une image"""

    poly_cells = mask_to_polygons_layer(mask)
    poly_cells = [list(poly.exterior.coords) for poly in list(poly_cells.geoms)]
    for points in poly_cells:
        if region_name in COLOR_TISSUE_SEGMENTATION_DRAW.keys():
            color = COLOR_TISSUE_SEGMENTATION_DRAW[region_name]
        else:
            color = COLOR_TISSUE_SEGMENTATION_DRAW["manually_segmented"]
        img.polygon(points, fill=color, outline="black")
        # img.polygon(points,fill = COLOR_CELLS_DRAW["Ramified"], outline ="blue")
    return img


# def _add_roi_in_tissue_img(img, origin_row, origin_col, end_row, end_col, roi=False):
#     """tl = (y,x)"""
#     tl = ((origin_col) * 32, (origin_row) * 32)  # (x,y)
#     tr = ((end_col + 1) * 32, (origin_row) * 32)  # (x,y)
#     br = ((end_col + 1) * 32, (end_row + 1) * 32)  # (x,y)
#     bl = ((origin_col) * 32, (end_row + 1) * 32)  # (x,y)
#     # print("tl",tl,"tr",tr,"br",br,"bl",bl)
#     img.line(
#         [tl, tr, br, bl, tl], fill=(246, 255, 190, 255), width=16
#     ) if roi else img.line([tl, tr, br, bl, tl], fill="green", width=7)
#     return img


def draw_roi_delimiter(roi, img):
    """Ajoute le contour de la ROI dans l'image des levelsets"""

    border_size = dataset_config.roi_border_size
    tl = (border_size, border_size)
    tr = (border_size + roi.shape[1], border_size)
    br = (border_size + roi.shape[1], border_size + roi.shape[0])
    bl = (border_size, border_size + roi.shape[0])

    img.line([tl, tr, br, bl, tl], width=20, fill="red")
    return img


def find_best_subregion_size_for_subdivision(
    x_min, x_max, y_min, y_max, size_max_subregion=20
):
    """Find a subdivision of the rectangle (x_min,x_max,y_min,y_max) into squares of max size size_max_subregion"""
    verbose = 0

    L_r = x_max - x_min + 1
    print(red("Lr = " + str(L_r))) if verbose else None
    N_subregion_row = 1
    # Nr = L_r//dr+1 if L_r%dr != 0 else L_r//dr
    Size_subregion_row = L_r / N_subregion_row
    while Size_subregion_row > size_max_subregion:
        N_subregion_row += 1
        # Nr = L_r//dr+1 if L_r%dr != 0 else L_r//dr
        Size_subregion_row = L_r / N_subregion_row

    print(black("Finnaly")) if verbose else None
    print(blue("N_subregion_row  =" + str(N_subregion_row))) if verbose else None
    print(" Size_subregion_row = ", Size_subregion_row) if verbose else None
    Size_subregion_row_int = ceil(Size_subregion_row)
    print(" Size_subregion_row entier = ", Size_subregion_row_int) if verbose else None
    taille_tile_row = N_subregion_row * Size_subregion_row_int
    supplement_r = taille_tile_row - L_r
    recul_r = supplement_r // 2
    L_c = y_max - y_min + 1
    print(red("Lc = " + str(L_c))) if verbose else None
    N_subregion_col = 1
    # Nc = L_c//dc+1 if L_c%dc != 0 else L_c//dc
    Size_subregion_col = L_c / N_subregion_col
    while Size_subregion_col > size_max_subregion:
        N_subregion_col += 1
        # Nc = L_c//dc+1 if L_c%dc!=+ 0 else L_c//dc
        Size_subregion_col = L_c / N_subregion_col

    print(black("Finally")) if verbose else None
    print(blue("N_subregion_col  =" + str(N_subregion_col))) if verbose else None
    print(" Size_subregion_col = ", Size_subregion_col) if verbose else None
    Size_subregion_col_int = ceil(Size_subregion_col)
    print(" Size_subregion_col entier = ", Size_subregion_col_int) if verbose else None
    taille_tile_col = N_subregion_col * Size_subregion_col_int
    supplement_c = taille_tile_col - L_c
    recul_c = supplement_c // 2
    return (
        N_subregion_row,
        Size_subregion_row_int,
        recul_r,
        N_subregion_col,
        Size_subregion_col_int,
        recul_c,
    )


def create_coord_from_subregion_tissue_decomposition(
    N_subregion_row,
    Size_subregion_row_int,
    recul_r,
    N_subregion_col,
    Size_subregion_col_int,
    recul_c,
    x_origin_roi,
    y_origin_roi,
    max_tile_row,
    max_tile_col,
):
    """
    N_subregion_row/N_subregion_col : umber of subregion per row
    Size_subregion_row_int/Size_subregion_col_int : row size of the subregions
    recul_r/recul_c : ce qu'il faut reculler sur les lignes
    x_origin_roi/ y_origin_roi : origin of the roi
    max_tile_row/max_tile_col : max tile row and col
    """
    list_coord_roi = []
    number_subregion = 1
    for idx_subregion_row in range(N_subregion_row):
        for idx_subregion_col in range(N_subregion_col):
            origin_row = idx_subregion_row * (Size_subregion_row_int) - recul_r
            origin_col = idx_subregion_col * (Size_subregion_col_int) - recul_c
            end_row = origin_row + Size_subregion_row_int - 1
            end_col = origin_col + Size_subregion_col_int - 1

            origin_row = origin_row + x_origin_roi
            origin_col = origin_col + y_origin_roi
            end_row = end_row + x_origin_roi
            end_col = end_col + y_origin_roi

            origin_row = max(1, origin_row)
            end_row = min(max_tile_row - 1, end_row)
            origin_col = max(1, origin_col)
            end_col = min(max_tile_col - 1, end_col)
            list_coord_roi.append([origin_row, origin_col, end_row, end_col])

            # list_coord_roi.append([origin_row+x_origin_roi, origin_col+y_origin_roi, end_row+x_origin_roi, end_col+y_origin_roi])

            # print("subregion n°",number_subregion," : ",origin_row,origin_col,end_row,end_col)
            number_subregion += 1
    return list_coord_roi

def create_subregions_if_anatomical_region_too_large(
    slide_num,
    physiological_part,
    size_max_subregion,
    display_fig=True,
    path_save=None,
    scale_factor=None,
    config_dataset = None):
    """
    Create subregions (dict of rois) of the anatomical region if too large. Physiological part masks should exists
    """
    verbose = 0
    # rgb = plt.imread(slide.get_training_image_path(slide_num))
    rgb = plt.imread(slide.get_downscaled_paths("dir_downscaled_img",slide_num,channel_number = channel_number))
    
    rgb_pil = np_to_pil(rgb)
    path_physio_part_mask = os.path.join(
        dataset_config.dir_base_anatomical_region_segmentation,
        "slide_" + str(slide_num) + "_" + str(dataset_config.mapping_img_number[slide_num]) + "_pcw",
        "mask_" + physiological_part + ".png",
    )
    mask_physiopart_slide = Image.open(path_physio_part_mask)
    mask_physiopart_slide = np.array(mask_physiopart_slide)

    # Info on mask
    xx, yy = np.where(mask_physiopart_slide != 0)
    x_min = int(np.min(xx) / scale_factor) - 1
    x_max = int(np.max(xx) / scale_factor) + 1
    y_min = int(np.min(yy) / scale_factor) - 1
    y_max = int(np.max(yy) / scale_factor) + 1

    print("Shape de mask_physiopart_slide", mask_physiopart_slide.shape) if verbose else None
    # display_mask(mask_physiopart_slide)
    max_tile_row = int(mask_physiopart_slide.shape[0] / scale_factor)
    max_tile_col = int(mask_physiopart_slide.shape[1] / scale_factor)
    print("max_tile_row = ", max_tile_row, " max_tile_col = ", max_tile_col) if verbose else None
    # Pour corriger si on n'est trop près du bord
    x_min = max(1, x_min)
    x_max = min(max_tile_row - 1, x_max)
    y_min = max(1, y_min)
    y_max = min(max_tile_col - 1, y_max)
    print("x_min = ", x_min, " x_max = ", x_max, " y_min = ", y_min, " y_max = ", y_max) if verbose else None
    (
        N_subregion_row,
        Size_subregion_row_int,
        recul_r,
        N_subregion_col,
        Size_subregion_col_int,
        recul_c,
    ) = find_best_subregion_size_for_subdivision(
        x_min, x_max, y_min, y_max, size_max_subregion=size_max_subregion
    )

    list_coord_subregions = create_coord_from_subregion_tissue_decomposition(
        N_subregion_row,
        Size_subregion_row_int,
        recul_r,
        N_subregion_col,
        Size_subregion_col_int,
        recul_c,
        x_origin_roi=x_min,
        y_origin_roi=y_min,
        max_tile_row=max_tile_row,
        max_tile_col=max_tile_col,
    )
    # Ajout des carrees a l'image

    if display_fig:
        mask_physiopart_slide_pil = np_to_pil(mask_physiopart_slide)
        mask_physiopart_slide_pil_resized = mask_physiopart_slide_pil.resize(
            (rgb_pil.size[0], rgb_pil.size[1])
        )

        mask_physiopart_slide_pil_resized_np = np.asarray(
            mask_physiopart_slide_pil_resized
        )

        # on dessine
        img = ImageDraw.Draw(rgb_pil, "RGBA")
        img = add_tissue_segmentation_to_draw(img, physiological_part, mask_physiopart_slide_pil_resized_np)
        print(yellow("Coordinate ROI :"
                + str(x_min)
                + ","
                + str(y_min)
                + ","
                + str(x_max)
                + ","
                + str(y_max)
            )
        ) if verbose else None
        img = _add_roi_in_tissue_img(img, x_min, y_min, x_max, y_max, roi=True,border_size =dataset_config.tile_height )
        for number_subregion, coord_roi in enumerate(list_coord_subregions[:]):
            print(
                red("Coordinates subregion n°:" + str(number_subregion))
            ) if verbose else None
            origin_row, origin_col, end_row, end_col = coord_roi
            print(origin_row, origin_col, end_row, end_col) if verbose else None
            img = _add_roi_in_tissue_img(img, origin_row, origin_col, end_row, end_col,border_size =dataset_config.tile_height )
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        plt.imshow(rgb_pil.transpose(Image.FLIP_TOP_BOTTOM))
        # plt.grid()
        plt.title("slide "+ str(slide_num)+ " - "+ str(dataset_config.mapping_img_number[slide_num])+ " pcw - ROI shape : ("+ str(Size_subregion_row_int)+ " tiles x "+ str(Size_subregion_col_int)+ " tiles)",fontsize=20,)
        if path_save is not None:
            path_results_region = os.path.join(dataset_config.dir_base_stat_analysis,physiological_part)
            mkdir_if_nexist(path_results_region)
            path_save_results = os.path.join(path_results_region,"slide_"+ str(slide_num)+ "_"+ str(dataset_config.mapping_img_number[slide_num])+ "_pcw_anatomical_region_subdivision.png")
            path_save = os.path.join(path_save,"Anatomical_region_subdivision.png")
            fig.savefig(
                path_save,
                facecolor="white",
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            fig.savefig(
                path_save_results,
                facecolor="white",
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.1,
            )

        plt.show()
    return list_coord_subregions

def create_subregions_if_anatomical_region_too_large_all(
    liste_physiological_part,
    list_slides,
    display_fig=False,
    scale_factor=None,
):
    """Compute create_subregions_if_anatomical_region_too_large for a list of slides and physiological part"""
    verbose = 0

    dict_subregions = dict()
    for physiological_part in liste_physiological_part:
        size_max_subregion = dataset_config.physiological_regions_max_square_size[physiological_part]

        if physiological_part not in list(dataset_config.physiological_regions_max_square_size.keys()):
            group_for_comparison = (len(dataset_config.physiological_regions_max_square_size) + 1)
            dataset_config.physiological_regions_group_for_comparaison[physiological_part] = group_for_comparison
        else:
            group_for_comparison = dataset_config.physiological_regions_group_for_comparaison[physiological_part]
        path_physio_part = os.path.join(dataset_config.dir_base_stat_analysis, physiological_part)
        print("path_physio_part",path_physio_part)
        mkdir_if_nexist(path_physio_part)
        print(blue("Physiological_part :"+ physiological_part+ " group_for_comparison :"+ str(group_for_comparison)))
        dict_subregions[physiological_part] = dict()
        for slide_num in list_slides:
            ############################ Paths ############################
            name_slide = ("slide_"+ str(slide_num)+ "_"+ str(dataset_config.mapping_img_number[slide_num])+ "_pcw")
            print(blue(name_slide)) if verbose else None
            folder_name = (physiological_part+ "_max_subregion_"+ str(size_max_subregion).zfill(3))
            path_region_roi = os.path.join(dataset_config.dir_base_roi, name_slide, folder_name)
            mkdir_if_nexist(path_region_roi)
            path_dict = os.path.join(path_region_roi, "dict_rois_info.json")
            ############################ Creation dictionaries ###########################
            dict_subregions[physiological_part][slide_num] = dict()
            list_coord_subregions = create_subregions_if_anatomical_region_too_large(slide_num,physiological_part,size_max_subregion,display_fig=display_fig,path_save=path_region_roi,scale_factor=scale_factor,)
            idx_roi = 0
            for coord_subregion in list_coord_subregions:
                dict_subregions[physiological_part][slide_num][idx_roi] = dict()
                dict_subregions[physiological_part][slide_num][idx_roi]["physiological_part"] = physiological_part
                dict_subregions[physiological_part][slide_num][idx_roi]["folder_name"] = folder_name
                dict_subregions[physiological_part][slide_num][idx_roi][ "slide_num"] = slide_num
                dict_subregions[physiological_part][slide_num][idx_roi]["origin_row"] = coord_subregion[0]
                dict_subregions[physiological_part][slide_num][idx_roi]["origin_col"] = coord_subregion[1]
                dict_subregions[physiological_part][slide_num][idx_roi]["end_row"] = coord_subregion[2]
                dict_subregions[physiological_part][slide_num][idx_roi]["end_col"] = coord_subregion[3]
                dict_subregions[physiological_part][slide_num][idx_roi]["subregion_size_if_subregion_roi"] = None
                dict_subregions[physiological_part][slide_num][idx_roi][ "group_for_comparison"] = [group_for_comparison]
                idx_roi += 1
            with open(path_dict, "w") as f:
                f.write(json.dumps(dict_subregions[physiological_part][slide_num]))
    return dict_subregions

def post_process_aggregated_df(physiological_part_info_subregions):
    """Ajoute les colonnes rouge dans features_analysis"""
    area_physiological_part_slide = 0
    n_cells_tot_physiological_part = 0
    for idx_subregion in (
        physiological_part_info_subregions["id_subregion"].unique().tolist()
    ):
        roi_analysis_1_subregion = physiological_part_info_subregions[
            physiological_part_info_subregions["id_subregion"] == idx_subregion
        ]
        area_physiological_part_slide += roi_analysis_1_subregion[
            "area_physiological_part_roi"
        ].iloc[0]
        roi_analysis_1_subregion = physiological_part_info_subregions[
            physiological_part_info_subregions["id_subregion"] == idx_subregion
        ]
        n_cells_tot_physiological_part += roi_analysis_1_subregion["n_cells_roi"].iloc[
            0
        ]

    physiological_part_info_subregions[
        "area_physiological_part_slide"
    ] = area_physiological_part_slide
    physiological_part_info_subregions["fraction_physiological_part_in_tissue"] = (
        physiological_part_info_subregions["area_physiological_part_slide"]
        / physiological_part_info_subregions["area_tissue_slide"]
    )
    physiological_part_info_subregions[
        "n_cells_tot_physiological_part"
    ] = n_cells_tot_physiological_part

    for idx_subregion in (
        physiological_part_info_subregions["id_subregion"].unique().tolist()
    ):
        physiological_part_info_subregions.loc[
            physiological_part_info_subregions["id_subregion"] == idx_subregion,
            "fraction_tot_physiological_part_in_roi",
        ] = (
            physiological_part_info_subregions[
                physiological_part_info_subregions["id_subregion"] == idx_subregion
            ]["area_physiological_part_roi"].iloc[0]
            / physiological_part_info_subregions["area_physiological_part_slide"].iloc[
                0
            ]
        )
        physiological_part_info_subregions.loc[
            physiological_part_info_subregions["id_subregion"] == idx_subregion,
            "fraction_n_cells_tot_physiological_part_in_ROI",
        ] = (
            physiological_part_info_subregions[
                physiological_part_info_subregions["id_subregion"] == idx_subregion
            ]["n_cells_roi"].iloc[0]
            / physiological_part_info_subregions["n_cells_tot_physiological_part"].iloc[
                0
            ]
        )

    return physiological_part_info_subregions


def concat_all_subregions(liste_roi_to_add, dict_roi):
    """Concatenate results dataframe of all the subregions in a single dataframe"""
    data_name = liste_roi_to_add[0]
    roi_info = dict_roi[data_name]
    folder_name = roi_info["folder_name"]
    physiological_part = roi_info["physiological_part"]

    size_max_subregion = int(folder_name[-2:])
    path_roi = get_roi_path(dict_roi[data_name])
    path_all_results_on_roi = os.path.join(path_roi, "DeepCellMap_roi.csv")
    df_all_results_on_roi = pd.read_csv(path_all_results_on_roi, sep=";")
    df_all_results_on_roi["id_subregion"] = 0
    df_all_results_on_roi["size_max_subregion"] = size_max_subregion
    physiological_part_info_subregions = pd.DataFrame(
        columns=df_all_results_on_roi.columns
    )
    for idx_subregion, data_name in enumerate(
        liste_roi_to_add
    ):  # Boucle sur les noms de roi a ajouter a roi_analysis
        # print(blue("Loop sur les noms de roi a ajouter a physiological_part_info_subregions :"+str(data_name), "bold"))
        path_roi = get_roi_path(dict_roi[data_name])
        path_all_results_on_roi = os.path.join(path_roi, "DeepCellMap_roi.csv")
        df_all_results_on_roi = pd.read_csv(path_all_results_on_roi, sep=";")

        df_all_results_on_roi["id_subregion"] = idx_subregion
        df_all_results_on_roi["size_max_subregion"] = size_max_subregion
        physiological_part_info_subregions = pd.concat(
            [physiological_part_info_subregions, df_all_results_on_roi], axis=0
        )
    physiological_part_info_subregions = post_process_aggregated_df(
        physiological_part_info_subregions
    )
    path_physiological_part_info_subregions = os.path.join(
        os.path.dirname(path_roi), "physiological_part_info_subregions.csv"
    )


    print(
        blue(
            "Save physiological_part_info_subregions :"
            + str(path_physiological_part_info_subregions),
            "bold",
        )
    )
    physiological_part_info_subregions.to_csv(
        path_physiological_part_info_subregions, sep=";", index=False
    )
    # print("savevfizeu : ",path_save_temporal_analysis)
    # physiological_part_info_subregions.to_csv(path_save_temporal_analysis, sep = ";", index = False)

    return physiological_part_info_subregions


def load_physiological_part_info_subregions(liste_roi_to_add, dict_roi):
    data_name = liste_roi_to_add[0]
    path_roi = get_roi_path(dict_roi[data_name])
    path_physiological_part_info_subregions = os.path.join(
        os.path.dirname(path_roi), "physiological_part_info_subregions.csv"
    )
    physiological_part_info_subregions = pd.read_csv(
        path_physiological_part_info_subregions, sep=";"
    )
    return physiological_part_info_subregions


def get_dict_all_info(liste_physiological_part, list_slides):
    dict_subregions = dict()
    for physiological_part in liste_physiological_part:
        size_max_subregion = DICT_PHYSIOLOGICAL_PART_MAX_SQUARE_SIZE[physiological_part]
        dict_subregions[physiological_part] = dict()
        for slide_num in list_slides:
            name_slide = (
                "slide_"
                + str(slide_num)
                + "_"
                + str(dataset_config.mapping_img_number[slide_num])
                + "_pcw"
            )
            path_region = os.path.join(
                dataset_config.dir_base_roi,
                name_slide,
                physiological_part
                + "_max_subregion_"
                + str(size_max_subregion).zfill(3),
            )
            path_dict = os.path.join(path_region, "dict_rois_info.json")
            with open(path_dict, "r") as f:
                dict_subregions[physiological_part][slide_num] = json.loads(f.read())
    return dict_subregions


def compute_pipeline_from_physiological_part_slide_list(
    liste_physiological_part,
    dict_subregions,
    param_config_rois_reconstruction,
    param_colocalisation,
    param_dbscan_cluster_analysis
):
    """Compute the pipeliene to the different subregions of the same physiological part and slide"""
    for physiological_part in liste_physiological_part:
        print(red("DeepCellMap on :" + physiological_part, "bold"))
        for slide_num in list(dict_subregions[physiological_part].keys())[:]:
            print(red("Slide :" + str(slide_num), "bold"))
            liste_roi_to_add = list(
                dict_subregions[physiological_part][slide_num].keys()
            )[:]
            print("liste_roi_to_add")
            apply_DeepCellMap_to_roi_list(
                liste_roi_to_add,
                dict_subregions[physiological_part][slide_num],
                param_config_rois_reconstruction,
                param_colocalisation,
                param_dbscan_cluster_analysis
            )


####################### Merge region subregions ############################################################################


def specific_functions_to_aggregate_subregions(feature, df_A_B, results_A_B):
    if feature in [
        "n_A_proba_slide",
        "fraction_tot_A_cells_in_roi",
        "fraction_tot_A_cells_proba_in_roi",
        "n_cells_A_proba_roi",
        "n_cells_A_roi",
        "n_A_cells_slide",
        "n_A_cells_proba_slide",
        "mean_size_A_cells_slide",
        "std_size_A_cells_slide",
        "n_A_proba_slide",
    ]:
        return feature, np.nan
    if (feature == "roi_loc") or (feature == "roi_shape"):
        min_row = np.inf
        min_col = np.inf
        max_row = -1
        max_col = -1
        for idx_subregion in df_A_B["id_subregion"].unique().tolist():
            df_A_B_subregion = df_A_B[df_A_B["id_subregion"] == idx_subregion][
                "roi_loc"
            ]
            coord = _eval(
                df_A_B[df_A_B["id_subregion"] == idx_subregion]["roi_loc"].iloc[0]
            )
            origin_row, origin_col, end_row, end_col = coord
            min_row = min(min_row, origin_row)
            min_col = min(min_col, origin_col)
            max_row = max(max_row, end_row)
            max_col = max(max_col, end_col)
        if feature == "roi_loc":
            return feature, str([min_row, min_col, max_row, max_col])
        if feature == "roi_shape":
            return feature, str(
                [
                    (max_row - min_row) * dataset_config.tile_height,
                    (max_col - min_col) * dataset_config.tile_height,
                ]
            )
    if feature.find("fraction_tot_") != -1:
        if feature == "fraction_tot_cells_in_roi":
            n_cells_slide = results_A_B["n_cells_slide"]
            n_cells_roi = results_A_B["n_cells_roi"]
            return feature, n_cells_roi / n_cells_slide
        # if feature == "fraction_tot_cells_in_roi":
        if "proba" in feature:
            label_state = feature[13 : feature.find("_cells_proba_in_roi")]
            n_cells_slide = results_A_B["n_" + label_state + "_proba_slide"]
            n_cells_roi = results_A_B["n_" + label_state + "_proba_roi"]
        else:
            label_state = feature[13 : feature.find("_cells_in_roi")]
            n_cells_slide = results_A_B["n_" + label_state + "_slide"]
            n_cells_roi = results_A_B["n_" + label_state + "_roi"]
        return feature, n_cells_roi / n_cells_slide


def find_feature_name_in_physiological_df(feature):
    path_features_info = os.path.join(dataset_config.dir_output_dataset, "features_metadata.csv")
    df_features_analysis = pd.read_csv(path_features_info, sep=";")
    if (
        df_features_analysis[df_features_analysis["feature_name"] == feature][
            "aggregation_same_for_all"
        ].values[0]
        == True
    ):
        return feature
    if (
        df_features_analysis[df_features_analysis["feature_name"] == feature][
            "aggregation_wheighted_sum_by_physiopart_fraction"
        ].values[0]
        == True
    ):
        return feature + "_weighted_sum_by_area_fraction"
    if (
        df_features_analysis[df_features_analysis["feature_name"] == feature][
            "aggregation_wheighted_sum_by_cell_fraction"
        ].values[0]
        == True
    ):
        return feature + "_weighted_sum_by_cell_fraction"
    if (
        df_features_analysis[df_features_analysis["feature_name"] == feature][
            "aggregation_by_sum"
        ].values[0]
        == True
    ):
        return feature


def aggregate_metrics_in_one_df(liste_roi_to_add, dict_roi, verbose=0):
    path_roi = get_roi_path(roi_info=dict_roi[liste_roi_to_add[0]])
    physiological_part_info_subregions = load_physiological_part_info_subregions(
        liste_roi_to_add, dict_roi
    )
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    pop_AB_liste = [
        (state_A, state_B)
        for state_A in labels
        for state_B in labels
        if state_B != state_A
    ]
    path_features_info = os.path.join(dataset_config.dir_output_dataset, "features_metadata.csv")
    df_features_analysis = pd.read_csv(path_features_info, sep=";")
    list_features = list(df_features_analysis["feature_name"])
    list_col_to_exclude = [
        "id_subregion",
        "fraction_tot_physiological_part_in_roi",
        "fraction_n_cells_tot_physiological_part_in_ROI",
        "fraction_tissue_roi",
        "fraction_physiological_part_roi",
        "fraction_tot_tissue_in_roi",
    ]
    df_reconstruction = pd.DataFrame()
    first = True
    for pop_A, pop_B in pop_AB_liste:
        df_A_B = physiological_part_info_subregions[
            (physiological_part_info_subregions["pop_A"] == pop_A)
            & (physiological_part_info_subregions["pop_B"] == pop_B)
        ]

        results_A_B = dict()
        results_A_B["pop_A"] = pop_A
        results_A_B["pop_B"] = pop_B
        for feature in list_features:
            if feature not in list_col_to_exclude:
                feature_conserved_in_new_df = False
                # stay the same for all the subregions
                if (
                    df_features_analysis[
                        df_features_analysis["feature_name"] == feature
                    ]["aggregation_same_for_all"].values[0]
                    == True
                ):
                    feature_conserved_in_new_df = True
                    results_A_B[feature] = df_A_B[feature].iloc[0]
                    print(
                        blue(feature + " -> same value : " + str(results_A_B[feature]))
                    ) if verbose and first else None
                # agregate by area ratio
                if (
                    df_features_analysis[
                        df_features_analysis["feature_name"] == feature
                    ]["aggregation_wheighted_sum_by_physiopart_fraction"].values[0]
                    == True
                ):
                    feature_conserved_in_new_df = True
                    val_feature = 0
                    feature_name = feature + "_weighted_sum_by_area_fraction"
                    print(
                        blue(
                            feature
                            + " -> weighted sum by physio part fraction in subregion"
                        )
                    ) if verbose and first else None
                    for idx_subregion in df_A_B["id_subregion"].unique().tolist():
                        df_A_B_subregion = df_A_B[
                            df_A_B["id_subregion"] == idx_subregion
                        ]
                        print(
                            df_A_B_subregion[feature].iloc[0]
                        ) if verbose and first else None
                        if (
                            df_A_B_subregion[feature].iloc[0] != 0
                            and not np.isnan(df_A_B_subregion[feature].iloc[0])
                            and (df_A_B_subregion[feature].iloc[0] != np.inf)
                        ):
                            val_feature += (
                                df_A_B_subregion[
                                    "fraction_tot_physiological_part_in_roi"
                                ].iloc[0]
                                * df_A_B_subregion[feature].iloc[0]
                            )
                    results_A_B[feature_name] = val_feature
                    print(
                        blue(
                            feature
                            + " -> "
                            + feature_name
                            + " : weighted sum by physio part fraction in subregion : "
                            + str(results_A_B[feature_name])
                        )
                    ) if verbose and first else None
                # agregate by cells ratio
                if (
                    df_features_analysis[
                        df_features_analysis["feature_name"] == feature
                    ]["aggregation_wheighted_sum_by_cell_fraction"].values[0]
                    == True
                ):
                    feature_conserved_in_new_df = True
                    val_feature = 0
                    feature_name = feature + "_weighted_sum_by_cell_fraction"
                    print(
                        blue(
                            feature
                            + " -> "
                            + feature_name
                            + " : weighted sum by cell fraction in subregion : "
                        )
                    ) if verbose and first else None
                    for idx_subregion in df_A_B["id_subregion"].unique().tolist():
                        df_A_B_subregion = df_A_B[
                            df_A_B["id_subregion"] == idx_subregion
                        ]
                        print(
                            df_A_B_subregion[feature].iloc[0]
                        ) if verbose and first else None
                        if (
                            df_A_B_subregion[feature].iloc[0] != 0
                            and not np.isnan(df_A_B_subregion[feature].iloc[0])
                            and (df_A_B_subregion[feature].iloc[0] != np.inf)
                        ):
                            val_feature += (
                                df_A_B_subregion[
                                    "fraction_n_cells_tot_physiological_part_in_ROI"
                                ].iloc[0]
                                * df_A_B_subregion[feature].iloc[0]
                            )
                    results_A_B[feature_name] = val_feature
                    print(
                        blue(
                            feature
                            + " -> "
                            + feature_name
                            + " : weighted sum by cell fraction in subregion : "
                            + str(results_A_B[feature_name])
                        )
                    ) if verbose and first else None

                # agregate by sum
                if (
                    df_features_analysis[
                        df_features_analysis["feature_name"] == feature
                    ]["aggregation_by_sum"].values[0]
                    == True
                ):
                    feature_conserved_in_new_df = True
                    print(
                        blue(feature + " -> aggregation_by_sum")
                    ) if verbose and first else None
                    val_feature = 0
                    feature_name = feature
                    for idx_subregion in df_A_B["id_subregion"].unique().tolist():
                        df_A_B_subregion = df_A_B[
                            df_A_B["id_subregion"] == idx_subregion
                        ]

                        if (
                            df_A_B_subregion[feature].iloc[0] != 0
                            and not np.isnan(df_A_B_subregion[feature].iloc[0])
                            and (df_A_B_subregion[feature].iloc[0] != np.inf)
                        ):
                            val_feature += df_A_B_subregion[feature].iloc[0]

                    results_A_B[feature_name] = val_feature
                    print(
                        blue(
                            feature
                            + " -> "
                            + feature_name
                            + " : aggregation_by_sum : "
                            + str(results_A_B[feature_name])
                        )
                    ) if verbose and first else None

                if (
                    df_features_analysis[
                        df_features_analysis["feature_name"] == feature
                    ]["aggregation_specific_function"].values[0]
                    == True
                ):
                    feature_conserved_in_new_df = True
                    print(
                        blue(feature + " -> specific function ->")
                    ) if verbose and first else None
                    colname_feature, val = specific_functions_to_aggregate_subregions(
                        feature, df_A_B, results_A_B
                    )
                    print(
                        blue(
                            feature
                            + " -> "
                            + colname_feature
                            + " specific function ->"
                            + str(val)
                        )
                    ) if verbose and first else None
                    results_A_B[colname_feature] = val

                if not feature_conserved_in_new_df:
                    print(
                        black("Feature not conserved in new df : " + feature)
                    ) if verbose and first else None
        first = False
        df_reconstruction = df_reconstruction.append(results_A_B, ignore_index=True)
    # save
    path_physiological_part_info_subregions = os.path.join(
        os.path.dirname(path_roi), "df_physiological_part.csv"
    )
    df_reconstruction.to_csv(
        path_physiological_part_info_subregions, sep=";", index=False
    )
    print(
        blue(
            "Save physiological_part_info_subregions :"
            + str(path_physiological_part_info_subregions),
            "bold",
        )
    )
    return df_reconstruction


####################################################" Concat Time-Regions in one DF ########################################################"


# Ajout des colonnes de cells a
def post_process_n_cells_state(df_temporal_analysis):
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    n_A_cells_slide = []
    n_A_cells_proba_slide = []
    mean_size_A_cells_slide = []
    std_size_A_cells_slide = []
    n_cells_A_roi = []
    n_cells_A_proba_roi = []
    fraction_tot_A_cells_in_roi = []
    fraction_tot_A_cells_proba_in_roi = []
    for row in df_deep_cell_map.iterrows():
        pop_A_row = row[1]["pop_A"]

        n_A_cells_slide_val = row[1]["n_" + pop_A_row + "_slide"]
        n_A_proba_cells_slide_val = row[1]["n_" + pop_A_row + "_proba_slide"]
        mean_size_A_cells_slide_val = row[1]["mean_size_" + pop_A_row + "_slide"]
        std_size_A_cells_slide_val = row[1]["std_size_" + pop_A_row + "_slide"]

        n_cells_A_roi_val = row[1]["n_" + pop_A_row + "_roi"]
        n_cells_A_proba_roi_val = row[1]["n_" + pop_A_row + "_proba_roi"]

        fraction_tot_A_cells_in_roi_val = row[1][
            "fraction_tot_" + pop_A_row + "_cells_in_roi"
        ]
        fraction_tot_A_cells_proba_in_roi_val = row[1][
            "fraction_tot_" + pop_A_row + "_cells_proba_in_roi"
        ]
        n_A_cells_slide.append(n_A_cells_slide_val)
        n_A_cells_proba_slide.append(n_A_proba_cells_slide_val)
        mean_size_A_cells_slide.append(mean_size_A_cells_slide_val)
        std_size_A_cells_slide.append(std_size_A_cells_slide_val)
        fraction_tot_A_cells_in_roi.append(fraction_tot_A_cells_in_roi_val)
        fraction_tot_A_cells_proba_in_roi.append(fraction_tot_A_cells_proba_in_roi_val)

        n_cells_A_roi.append(n_cells_A_roi_val)
        n_cells_A_proba_roi.append(n_cells_A_proba_roi_val)
    df_temporal_analysis["n_A_cells_slide"] = n_A_cells_slide
    df_temporal_analysis["n_A_cells_proba_slide"] = n_A_cells_proba_slide
    df_temporal_analysis["mean_size_A_cells_slide"] = mean_size_A_cells_slide
    df_temporal_analysis["std_size_A_cells_slide"] = std_size_A_cells_slide
    df_temporal_analysis["n_cells_A_roi"] = n_cells_A_roi
    df_temporal_analysis["n_cells_A_proba_roi"] = n_cells_A_proba_roi
    df_temporal_analysis["fraction_tot_A_cells_in_roi"] = fraction_tot_A_cells_in_roi
    df_temporal_analysis[
        "fraction_tot_A_cells_proba_in_roi"
    ] = fraction_tot_A_cells_proba_in_roi
    return df_temporal_analysis


def merge_all_spatiotemporal_data(dict_physiological_part_size_max_subregion):
    spatiotemporal_dataframe = pd.DataFrame()
    for physiological_part in dict_physiological_part_size_max_subregion.keys():
        if physiological_part == "manually_segmented":
            continue 
        size_max_subregion = dict_physiological_part_size_max_subregion[
            physiological_part
        ]
        print(red("Figures :" + physiological_part, "bold"))
        roi_analysis, path_temporal_analysis_csv = load_roi_analysis(
            physiological_part, size_max_subregion
        )
        spatiotemporal_dataframe = pd.concat(
            [spatiotemporal_dataframe, roi_analysis], axis=0, ignore_index=True
        )
    spatiotemporal_dataframe = post_process_n_cells_state(spatiotemporal_dataframe)
    path_temporal_analysis_csv = os.path.join(
        dataset_config.dir_base_stat_analysis, "spatiotemporal_analysis.csv"
    )
    spatiotemporal_dataframe.to_csv(path_temporal_analysis_csv, sep=";", index=False)
    return spatiotemporal_dataframe


def load_spatio_temporal_dataframe():
    path_temporal_analysis_csv = os.path.join(
        dataset_config.dir_base_stat_analysis, "spatiotemporal_analysis.csv"
    )
    spatiotemporal_dataframe = pd.read_csv(path_temporal_analysis_csv, sep=";")
    spatiotemporal_dataframe["n_cells_by_area_roi"] = spatiotemporal_dataframe["n_cells_roi"]/spatiotemporal_dataframe["area_roi"]
    spatiotemporal_dataframe["n_A_cells_by_area_roi_proba"] = spatiotemporal_dataframe["n_cells_A_proba_roi"]/spatiotemporal_dataframe["area_roi"]
    spatiotemporal_dataframe["n_A_cells_by_area_roi"] = spatiotemporal_dataframe["n_cells_A_roi"]/spatiotemporal_dataframe["area_roi"]
    return spatiotemporal_dataframe


def merge_results_several_entire_images(image_list, dataset_config):
    """ Take the results from several images and merge them together
    """
    for idx_slide, slide_num in enumerate(image_list):

        path_roi = create_path_roi(dataset_config,slide_num,origin_row=None,origin_col=None,end_row=None,end_col=None,entire_image=True)
        path_df = os.path.join(path_roi,"results_roi.csv")
        df = pd.read_csv(path_df, sep=";")
        if idx_slide == 0: 
            df_concatenated = df
        else:
            #test if dataframes have the same columns 
            for column_name in df.columns:
                if column_name not in df_concatenated.columns:
                    print("column_name",column_name)
                    # df_concatenated[column_name] = np.nan
            df_concatenated = pd.concat([df_concatenated, df])
    df_concatenated["group"] = df_concatenated["slide_num"].apply(lambda x: dataset_config.mapping_img_disease[x])

    path_entire_image_stats = os.path.join(dataset_config.dir_base_stat_analysis,"results_entire_images.csv")
    df_concatenated.to_csv(path_entire_image_stats, sep=";", index=False)
    return df_concatenated