#All the displaying functions are in this file

# Libraries importation

import shutil
import numpy as np 
import matplotlib.pyplot as plt

#import seaborn as sns

from scipy.stats import norm

import pandas as pd 
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt


#from preprocessing import filter
from preprocessing import slide
#from preprocessing import tiles
from utils.util import *
#from python_files import spatial_statistics
#from segmentation_classification import classification
#from python_files.const import *
from segmentation_classification.region_of_interest import RegionOfInterest, get_roi_path
#from python_files.const_roi import *
# from utils.util import *
# from python_files import colocalisation_analysis
# from python_files import dbscan
# from python_files import deep_cell_map
from utils.util_colors_drawing import *

from simple_colors import *
from ast import literal_eval 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import plotly.graph_objects as po

TITLE_SIZE = 20
SIZE_LINE =5
WIDTH_FIG = 800
HEIGHT_FIG = 700
# AGGREGATION_METHOD_USE = "_agregated_by_cell_fraction"
def find_feature_name_in_physiological_df(feature):

    path_features_info = os.path.join(dataset_config.dir_output_dataset,"features_metadata.csv")
    df_features_analysis = pd.read_csv(path_features_info, sep = ";")
    if df_features_analysis[df_features_analysis["feature_name"] == feature]["aggregation_same_for_all"].values[0] == True:
        return feature
    if df_features_analysis[df_features_analysis["feature_name"] == feature]["aggregation_wheighted_sum_by_physiopart_fraction"].values[0] == True:

        return feature+"_weighted_sum_by_area_fraction"
    if df_features_analysis[df_features_analysis["feature_name"] == feature]["aggregation_wheighted_sum_by_cell_fraction"].values[0] == True:

        return feature+"_weighted_sum_by_cell_fraction"
    if df_features_analysis[df_features_analysis["feature_name"] == feature]["aggregation_by_sum"].values[0] == True:
        return feature
    else : 
        return feature
    



DICT_QUANTITIES_SIZE_RESULTS_VISUALSIAITON = dict({
    'fraction_tot_A_cells_in_roi': # OKKKkKKKKK display_all_A_results
    dict({
        "title_figure":
        "<b>Cells quantities - Fraction of all <A> cells in the physiological part (ROI) <br>",
        "y_axis_name": "Fraction all <A> in ROI",
        "range_y" : "0_1",
        "metric_subgroup_folder_name": "1_Cells_quantities_and_sizes",
        "metric_folder_name": "fraction_tot_A_cells_in_rois",
        "filename": "fraction_tot_A_cells_in_roi_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        "associated_metric":"n_cells_A_roi", #Apres n_clusterised_cells_A
        'y2_axis_name' : "N <A> in ROI",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"<b>Fraction all <A> in ROI",
        "legend_name_y2":"<b>N <A> in ROI",
        "a_bellow":False,
        "proba_metric_name":"fraction_tot_A_cells_proba_in_roi",
        "proba_associated_metric":"n_cells_A_proba_roi"
    }),
    'n_cells_A_roi':
    dict({
        "title_figure":
        "<b>Cells quantities - Number of <A> cells in the physiological part (ROI) <br>",
        "y_axis_name": "# of <A>",
        "range_y" : "range_all_by_max",
        "metric_subgroup_folder_name": "1_Cells_quantities_and_sizes",
        "metric_folder_name": "n_cells_in_regions",
        "filename": "n_A_cells_per_regions",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        'y2_axis_name' : "N <A> in ROI",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"<b># <A> in ROI",
        "legend_name_y2":"<b>N <A> in ROI",
        "a_bellow":False,
        "proba_metric_name":"n_cells_A_proba_roi",
    }),
    'n_A_cells_by_area_roi':
    dict({
        "title_figure":
        "<b>Cells quantities - Number of <A> cells in the physiological part divided by area (ROI) <br>",
        "y_axis_name": "# of <A>/area",
        "range_y" : "range_all_by_max",
        "metric_subgroup_folder_name": "1_Cells_quantities_and_sizes",
        "metric_folder_name": "n_cells_by_area_in_regions",
        "filename": "n_A_cells_by_area_per_regions",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        'y2_axis_name' : "N <A> in ROI by area",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"<b># <A> in ROI/area",
        "legend_name_y2":"<b>N <A> in ROI/area",
        "a_bellow":False,
        "proba_metric_name":"n_A_cells_by_area_roi_proba",
    }),
    'n_cells_by_area_roi':
    dict({
        "title_figure":
        "<b>Cells quantities - Number of cells in the physiological part divided by area (ROI) <br>",
        "y_axis_name": "# of cells/area",
        "range_y" : "range_all_by_max",
        "metric_subgroup_folder_name": "1_Cells_quantities_and_sizes",
        "metric_folder_name": "n_cells_by_area_in_regions",
        "filename": "n_total_cells_by_area_per_regions",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        'y2_axis_name' : "N cells in ROI by area",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"<b># cells in ROI/area",
        "legend_name_y2":"<b>N cells in ROI/area",
        "a_bellow":False,
        "proba_metric_name":"n_A_cells_by_area_roi_proba",
    }),
})

DICT_COLOC_RESULTS_VISUALSIAITON = dict({
    'coloc_delta_a': # display_1_A_1_B
    dict({
        "title_figure":
        "<b>State vs State colocalisation : Mean distance [µm] & Colocalisation frequency (ASI)<br>",
        "y_axis_name": "Distance [µm]",
        "metric_subgroup_folder_name": "2_State_state_colocalisation",
        "metric_folder_name": "mean_distance_colocalisation",
        "filename": "State_vs_state_colocalisation_distance",
        "State_A_result":False,
        "associated_metric":"coloc_Association_Index_ASI",
        "y2_axis_name":"Clustering frequency (ASI)",
        "range_y" : "range_all_by_max",
        "range_y2" : "0_1",
        "proba_metric_name":"coloc_delta_a_proba",
        "proba_associated_metric":"coloc_Association_Index_ASI_proba"
    }),
    'coloc_Association_Index_ASI': #display_1_A_1_B
    dict({
        "title_figure":
        "<b>State vs State colocalisation : ASI - Clustering frequency<br>",
        "y_axis_name":"ASI",
        "metric_subgroup_folder_name": "2_State_state_colocalisation",
        "metric_folder_name": "Clustering_frequency",
        "range_y" : "0_1",
        "filename": "ASI",
        "State_A_result":False,
        "proba_metric_name":"coloc_Association_Index_ASI_proba"
    }),
    'coloc_p_value': #compare_pop_A_vs_pop_B_all_regions_all_time 
    dict({
        "title_figure":
        "<b>State vs State colocalisation : p-value<br>",
        "y_axis_name":"p value",
        "metric_subgroup_folder_name": "2_State_state_colocalisation",
        "metric_folder_name": "p_value",
        "range_y" : "other",
        "filename": "p_value",
        "State_A_result":False,
        "proba_metric_name":"coloc_p_value_proba"
    }),
}) 

DICT_BORDERCOLOC_RESULTS_VISUALSIAITON = dict({
    'coloc_state_border_delta_a': ################"" ok A_cells_results_across_times_for_all_regions
    dict({
        "title_figure":
        "<b>State vs Border colocalisation : Mean distance [µm]",
        "y_axis_name": "Distance (µm)",
        "legend_name_y" : "Distance [µm]",
        "metric_subgroup_folder_name": "3_Cell_border_colocalisation",
        "metric_folder_name": "mean_distance_colocalisation",
        "filename": "State_vs_border_colocalisation_distance_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_B",
        "associated_metric":"coloc_state_border_association_index_ASI",
        "y2_axis_name":"Clustering frequency (ASI)",
        "legend_name_y2" : "Coloc freq",
        'range_y': "range_all_by_max",
        'range_y2': "0_1",
        "proba_metric_name":"coloc_state_border_delta_a",
        "proba_associated_metric":"coloc_state_border_association_index_ASI_proba",
        "a_bellow" : False
    }),
    'coloc_state_border_association_index_ASI': ################"" ok A_cells_results_across_times_for_all_regions
    dict({
        "title_figure":
        "<b>State vs Border Colocalisation - ASI - Clustering frequency<br>",
        "y_axis_name": "Clustering frequency (ASI)",
        "legend_name_y" : "TODO",
        "metric_subgroup_folder_name": "3_Cell_border_colocalisation",
        "metric_folder_name": "Clustering_frequency",
        "filename": "State_border_colocalisation_distance_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_B",
        # "associated_metric":"coloc_state_border_association_index_ASI",
        "y2_axis_name":"Clustering frequency (ASI)",
        "legend_name_y2" : "Coloc freq",
        'range_y': "0_1",
        'range_y2': "0_1",
        "proba_metric_name":"coloc_state_border_association_index_ASI_proba",
        "proba_associated_metric":"coloc_state_border_association_index_ASI_proba",
        "a_bellow" : False
    }),
    'coloc_state_border_p_value':
    dict({
        "title_figure":
        "<b>State vs Border colocalisation : p-value<br>",
        "y_axis_name":"p value",
        "legend_name_y" : "p_value",
        "metric_subgroup_folder_name": "3_Cell_border_colocalisation",
        "metric_folder_name": "p_value",
        "filename": "state_border_coloc_p_value_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_B",
        "proba_metric_name":"coloc_state_border_p_value_proba",
        "y2_axis_name":"p-value",
        "legend_name_y2" : "TODO",
        'range_y': "0_1",
        'range_y2': "0_1",
        "proba_metric_name":"coloc_state_border_p_value_proba",
        "proba_associated_metric":"coloc_state_border_association_index_ASI_proba",
        "a_bellow" : False
    }),
}) 



DICT_DBSCAN_RESULTS_VISUALSIAITON = dict({
    #POP A RESULTS 
    'dbscan_fraction_clusterised_A': #OKKKKK  display_ALL_A_results
    dict({
        "title_figure":
        "<b>DBSCAN - Fraction clustered <A> cells <br>",
        "y_axis_name":"Fraction clustered <A>",
        "range_y" : "0_1",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "fraction_clustered_cells",
        "filename": "fraction_clusterised_cells_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        "associated_metric":"dbscan_n_clustered_cells_A", #Apres n_clusterised_cells_A
        'y2_axis_name' : "Number of <A> clusters",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"Fraction clustered <A>",
        "legend_name_y2":"Number of <A> clusters",
        "a_bellow":False
    }),
    'dbscan_n_clustered_cells_A': #OKKKKK  display_ALL_A_results
    dict({
        "title_figure":
        "<b>DBSCAN - Number of clustered cells per states in the different regions across time<br>",
        "y_axis_name":"N clustered cells",
        "range_y" : "range_all_by_max",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "n_clustered_cells",
        "filename": "n_clustered_cells_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"N clustered cells",
        "legend_name_y2":"N clustered cells", 
        "a_bellow":False
    }),
    'dbscan_n_clusters_A': #OKKKKK  display_ALL_A_results
    dict({
        "title_figure":
        "<b>DBSCAN - Number of clusters per states in the different regions across time <br>",
        "y_axis_name":"N clusters",
        "range_y" : "range_all_by_max",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "n_clusters",
        "filename": "n_clusters_",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        "range_y2" : "range_all_by_max",
        "legend_name_y":"N clusteres ",
        "legend_name_y2":"N clusters ",
        "a_bellow":False
    }),
    ################ A vs B ################
    'dbscan_iou': #OKKKKK display_1_A_1_B_iou
    dict({
        "title_figure":
        "<b>DBSCAN - Intersection over Union of the convex hulls <br>",
        "figure_subtitle_metric": "dbscan_min_sample_A",
        "y_axis_name":"area ratio",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "Intersection_over_union_convex_hulls",
        "filename": "DBSCAN_IoU_",
        "State_A_result":False,
        "associated_metric":"dbscan_i_A",
        "legend_name_y":"&#8745; area<br>----------<br>&#x222A; area",
        "legend_name_y2":"&#8745; area<br>----------<br> <A> area",
        "range_y":"range_all_by_max",
        "associated_metric_2":"dbscan_i_B",
        "legend_name_y3":"&#8745; area<br>----------<br> <B> area ",
        "range_0_1_y3":False,
        "a_bellow":False
    }),
    'dbscan_fraction_B_in_A_clusters': #OKKKKK c
    dict({
        "title_figure":
        "<b>DBSCAN - Fraction of B in A Clusters <br>",
        "figure_subtitle_metric": "dbscan_min_sample_A",
        "y_axis_name":"Cells ratio",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "Fract_B_in_A_clusters",
        "filename": "DBSCAN_fract_A_in_B_clusters_",
        "State_A_result":False,
        "associated_metric":"dbscan_fraction_A_in_B_clusters",
        "legend_name_y":"<B>  &#8712;<br> <A> <br>Clusters",
        "legend_name_y2":"<A>  &#8712;<br> <B> <br>Clusters",
        "range_y":"range_all_by_max",
        "a_bellow":False
    }),
    ########## ici comment je represente les clusters de A et B dans A-B
    'dbscan_fraction_clustered_B_in_intersect': #OKKKKK 
    dict({
        "title_figure":
        "<b>DBSCAN - Fraction Clustered B in A-B clusters intersection <br>",
        "figure_subtitle_metric": "dbscan_min_sample_A",
        "y_axis_name":"Fraction of cells",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "Fract_clustered_B_in_A_clusters",
        "filename": "DBSCAN_fract_clusterised_A_in_B_clusters_",
        "State_A_result":False,
        "associated_metric":"dbscan_fraction_clustered_A_in_intersect",
        "legend_name_y":"<B> in intersecting <br>A-B clusters",
        "legend_name_y2":"<A> in intersecting <br>A-B clusters",
        "range_y":"range_all_by_max",
        "a_bellow":False
    }),
    ### A RESULTS 
    'dbscan_epsilon_A BSN TRUC POP A ':
    dict({
        "title_figure":
        "<b>DBSCAN -Epsilon A <br>",
        "y_axis_name":"Epsilon_A",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "Clustering_frequency",
        "filename": "epsilon_A_",
        "range_y" : "Other",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A"
    }),
    'dbscan_epsilon_A BSN TRUC POP A ':
    dict({
        "title_figure":
        "<b>DBSCAN -Epsilon A <br>",
        "y_axis_name":"Epsilon_A",
        "metric_subgroup_folder_name": "3_DBSCAN",
        "metric_folder_name": "Clustering_frequency",
        "filename": "epsilon_A_",
        "range_y" : "Other",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A"
    }),
})
DICT_NEIGHBORS_ANALYSIS_RESULTS_VISUALSIAITON = dict({
    'neighbours_mean_dist_first_B_around_A': #OKKKKK display_1_A_ALL_B and display_1_A_1_B_3_first_B_around_A (3) 
    dict({
        "title_figure":
        "<b>Neighbors analysis - Distance of the first B around A [µm]<br>",
        "y_axis_name":"Distance [µm]",
        "metric_subgroup_folder_name": "5_Neighbours_analysis",
        "metric_folder_name": "first_B_around_A",
        "filename": "first_B_around_A_",
        "State_A_result":False,
        "associated_metric":"neighbours_mean_dist_second_B_around_A",
        "legend_name_y":"Dist 1st ",
        "legend_name_y2":"Dist 2nd ",
        "range_y":"range_all_by_max",
        "associated_metric_2":"neighbours_mean_dist_third_B_around_A",
        "legend_name_y3":"Dist 3rd ",
        "range_0_1_y3":False,
        "a_bellow":True
    }),
    'neighbours_r_coloc':###TODOO
    dict({
        "title_figure":
        "<b>Radius R applied to define <A>-balls <br> R = max(Mean colocalisation distance <A> vs all B)",
        "y_axis_name":"Radius [µm]",
        "metric_folder_name": "radius_r_A_balls",
        "metric_subgroup_folder_name" : "5_Neighbours_analysis",
        "filename": "r_coloc_applied_in_neighbors_analysis",
        "range_y":"range_all_by_max",
        "State_A_result":True
    }),
    'neighbours_rate_balls_r_coloc_containing_1_B':######################## OKKK display_1_A_ALL_B  --- display_1_A_1_B
    dict({
        "title_figure":
        "<b>Fraction of <A> balls(r) having at least 1 <B> inside<br> R = max(Mean colocalisation distance <A> vs all B)",
        "y_axis_name":"Fraction <A> balls",
        "metric_folder_name": "fraction_balls_habing_1_B_inside",
        "metric_subgroup_folder_name" : "5_Neighbours_analysis",
        "filename": "fraction_balls_having_1_B_inside_",
        "range_y":"0_1",
        "State_A_result":False
    }),
    'neighbours_rate_B_first_neighbour': ######################## OKKK display_1_A_ALL_B --- display_1_A_1_B
    dict({
        "title_figure":
        "<b>Neighbours analysis - Fraction of A cells having B as first neighbor<br>",
        "y_axis_name":"Fraction <A> cells",
        "metric_subgroup_folder_name" : "5_Neighbours_analysis",
        "metric_folder_name": "fraction_B_first_neighbors",
        "filename": "fraction_B_being_first_neighbor_of_A_",
        "State_A_result":False,
        "range_y":"0_1",

    }),
    'neighbours_mean_A_first_neighbour':
    dict({
        "title_figure":
        "<b>Neighbours analysis - Distance first neighbor",
        "y_axis_name":"Distance (µm)",
        "range_y":"range_all_by_max",
        "metric_subgroup_folder_name" : "5_Neighbours_analysis",
        "metric_folder_name": "dist_first_A_neighbors",
        "filename": "dist_first_A_neighbors",
        "State_A_result":True,
        "State_A_result_pop_of_interest" : "pop_A",
        "associated_metric":["neighbours_mean_A_second_neighbour","neighbours_mean_A_third_neighbour"]

    }),
}) 





def df_without_AB_comparaison(roi_analysis, liste_columns_to_keep):
    """
    Si je veux filtrer un group de comparaison : df_without_AB_pop.query("group_for_comparison == @group_comparaison")"""

    # df_without_AB_pop = df_without_AB_comparaison(roi_analysis)
    liste_col_to_keep = liste_columns_to_keep+["group_for_comparison", "id_in_group"]

    df_without_AB_pop = roi_analysis[liste_col_to_keep].copy()
    df_without_AB_pop.drop_duplicates(inplace = True)
    return  df_without_AB_pop


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Display ROIs in the same tissue """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def _add_roi_in_tissue_img(img,origin_row, origin_col, end_row, end_col, border_size,roi = False):
    """ tl = (y,x)
    Go ajouter le parametrage par dataset_config"""
    tl = ((origin_col-1)*32, (origin_row-1)*32)
    tr = (end_col*32, (origin_row-1)*32)
    br = (end_col*32,end_row*32)
    bl = ((origin_col-1)*32, end_row*32)
    # img.line([tl, tr,br,  bl, tl],fill =(246,255,190,255), width=10)
    img.line(
        [tl, tr, br, bl, tl], fill=(246, 255, 190, 255), width=16
    ) if roi else img.line([tl, tr, br, bl, tl], fill="green", width=7)
    return img

def display_several_roi_in_tissue(roi_analysis, group_for_comparison,save_fig = False,display=False,figsize = (20,15), dataset_config = None):
    """ 
    Pour l'instant toutes les ROIS sont dans le meme tissue
    """
    df_without_AB_pop = df_without_AB_comparaison(roi_analysis)
    df_group_comparaison = df_without_AB_pop.query("group_for_comparison == @group_for_comparison")    
    slide_num = df_group_comparaison["slide_num"].values[0]

    #Une fonction pour ce bloc ? 
    path_filtered_image, path_mask_tissu = slide.get_filter_image_result(slide_num)
    mask_tissu_filtered = Image.open(path_mask_tissu)
    mask_tissu_filtered_np = np.asarray(mask_tissu_filtered)
    mask_tissu = np.copy(mask_tissu_filtered_np)
    mask_tissu = np.where(mask_tissu>0,1,0).astype(np.uint8)
    mask_tissu = np.dstack([mask_tissu*198,mask_tissu*188,mask_tissu*186])
    mask_tissu = np.where(mask_tissu==0,255,mask_tissu)
    mask_tissu_pil = np_to_pil(mask_tissu)
    img = ImageDraw.Draw(mask_tissu_pil,"RGBA") 
    dict_text = dict()
    for idx, row in df_group_comparaison.iterrows():
        id_in_group = row["id_in_group"]
        coords_roi = literal_eval(row["roi_loc"])
        origin_row, origin_col, end_row, end_col =coords_roi 
        img = _add_roi_in_tissue_img(img,origin_row, origin_col, end_row, end_col, border_size = dataset_config.tile_height )
        dict_text["roi_"+str(id_in_group)] = dict({"x": (origin_row+1.7)*32, "y":(origin_col-0.8)*32, "s":"ROI  "+str(id_in_group)})
        
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path_if_save = os.path.join(path,"ROIs_in_tissue"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")

        fig = plt.figure(figsize=figsize, tight_layout = True)
        plt.imshow(mask_tissu_pil)
        text_kwargs = dict(fontsize=20, color=(0, 77/255, 128/255, 1))
        for key,item in dict_text.items():
            plt.text(item["y"],item["x"], item["s"],**text_kwargs) #, bbox=dict(fill=False, edgecolor='white', linewidth = 3)
        fig.savefig(path_if_save)
        plt.title("Group of comparaison "+str(group_for_comparison),fontsize=20)
        plt.show() if display else None 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Cell quantities """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def fig_n_cells_rois(roi_analysis, save_fig = False,display=False,display_one_row = False):
    '''
    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]

    if display_one_row:
        return _stat_n_cells_from_group_comparaison_one_row(roi_analysis, group_for_comparison = group_for_comparison , save_fig = save_fig, display=display)
    group_for_comparison = int(group_for_comparison)
    liste_nb_cell_roi = ["n_cells_roi",'n_Proliferative_roi', 'n_Amoeboid_roi', 'n_Cluster_roi', 'n_Phagocytic_roi', 'n_Ramified_roi']
    liste_nb_cell_roi_proba = ['n_cells_roi','n_Proliferative_proba_roi', 'n_Amoeboid_proba_roi', 'n_Cluster_proba_roi', 'n_Phagocytic_proba_roi', 'n_Ramified_proba_roi']

    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]

    # df_without_AB_pop = df_without_AB_comparaison(roi_analysis)
    liste_col_to_keep = liste_nb_cell_roi+liste_nb_cell_roi_proba+["pcw","group_for_comparison", "id_in_group"]

    df_without_AB_pop = roi_analysis[liste_col_to_keep].copy()
    df_without_AB_pop.drop_duplicates(inplace = True)
    df_without_AB_pop = df_without_AB_pop.query("group_for_comparison == @group_for_comparison")
    # print(df_without_AB_pop)
    fig = make_subplots(2, 3,subplot_titles=["#"+ c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    for i in range(1,len(labels)+1):
        if i ==1 : 
            continue
        fig.add_trace(po.Bar(
        name =labels[i-1],
        x = df_without_AB_pop[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"),
        y = df_without_AB_pop[liste_nb_cell_roi[i-1]],
        text= df_without_AB_pop[liste_nb_cell_roi[i-1]],
        marker_color = COLORS_STATES_PLOTLY[i-1]
    ), row=(i-1)//3+1, col=(i-1)%3+1)
        fig.add_trace(po.Bar(
        name = labels[i-1]+'(proba)',
        x = df_without_AB_pop[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"),
        y = df_without_AB_pop[liste_nb_cell_roi_proba[i-1]],
        text= df_without_AB_pop[liste_nb_cell_roi_proba[i-1]].apply(lambda x: np.round(x,1)),
        marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i-1]
        ),row=(i-1)//3+1, col=(i-1)%3+1)
    fig.update_traces( textangle=0, textposition="outside", cliponaxis=False)

    fig.update_layout(
       title_text="<b>Cell numbers in "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
        showlegend=True,
        width=1600,
        height=800,margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ))
    
    fig.update_yaxes(matches='y')
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "1_Cells_quantities_and_sizes")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        filepath = os.path.join(pathfile,"n_cells_by_states.png")
        fig.write_image(filepath)
    
def _stat_n_cells_from_group_comparaison_one_row(roi_analysis, group_for_comparison = 2 , save_fig = False, display=False):
    '''
    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    liste_nb_cell_roi = ["n_cells_roi",'n_Proliferative_roi', 'n_Amoeboid_roi', 'n_Cluster_roi', 'n_Phagocytic_roi', 'n_Ramified_roi']
    liste_nb_cell_roi_proba = ['n_cells_roi','n_Proliferative_proba_roi', 'n_Amoeboid_proba_roi', 'n_Cluster_proba_roi', 'n_Phagocytic_proba_roi', 'n_Ramified_proba_roi']

    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]

    df_without_AB_pop = df_without_AB_comparaison(roi_analysis)
    df_without_AB_pop = df_without_AB_pop.query("group_for_comparison == @group_for_comparison")

    
    fig = make_subplots(1, len(labels),subplot_titles=labels,horizontal_spacing = 0.03)
    for i in range(1,len(labels)+1):
        fig.add_trace(po.Bar(
        name =labels[i-1],
        x = df_without_AB_pop[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"),
        y = df_without_AB_pop[liste_nb_cell_roi[i-1]],
        text= df_without_AB_pop[liste_nb_cell_roi[i-1]],
        marker_color = COLORS_STATES_PLOTLY[i-1]
    ), 1,i)
        fig.add_trace(po.Bar(
        name = labels[i-1]+'(proba)',
        x = df_without_AB_pop[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"),
        y = df_without_AB_pop[liste_nb_cell_roi_proba[i-1]],
        text= df_without_AB_pop[liste_nb_cell_roi_proba[i-1]].apply(lambda x: np.round(x,1)),
        marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i-1]
        ),1,i)
    fig.update_traces( textangle=0, textposition="outside", cliponaxis=False)

    fig.update_layout(
        title="Number of cells in different ROIs",
        yaxis_title="Cell number",
        showlegend=True,
        width=1800,
        height=450
    )
    fig.update_yaxes(matches='y')

    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "1_Cells_quantities_and_sizes")
        mkdir_if_nexist(pathfile)
        filepath = os.path.join(pathfile,"n_cells_by_states_one_row.png")
        fig.write_image(filepath)
    
""" Statistique sur la taille des cellules pour 1 groupe de comparaison"""

def stat_size_cells_from_group_comparaison(roi_analysis, save_fig = False,display=False):
    '''
   Liens utilisés pour créer le graphique : 
   -https://plotly.com/python/line-charts/

    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"

    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    flag_from_square = ("size_max_square" in roi_analysis.columns)

    mean_size_states_roi = ['mean_cell_size_roi', 'mean_size_Proliferative_roi', 'mean_size_Amoeboid_roi', 'mean_size_Cluster_roi', 'mean_size_Phagocytic_roi', 'mean_size_Ramified_roi']
    std_size_state_roi = ['std_cell_size_roi', 'std_size_Proliferative_roi', 'std_size_Amoeboid_roi', 'std_size_Cluster_roi', 'std_size_Phagocytic_roi', 'std_size_Ramified_roi']
    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    if flag_from_square:
        mean_size_states_roi = [find_feature_name_in_physiological_df(feature) for feature in mean_size_states_roi]
        std_size_state_roi = [find_feature_name_in_physiological_df(feature) for feature in std_size_state_roi]

    df_without_AB_pop = df_without_AB_comparaison(roi_analysis,mean_size_states_roi+std_size_state_roi+["pcw","group_for_comparison", "id_in_group"])
    group_for_comparison = int(group_for_comparison)
    df_without_AB_pop = df_without_AB_pop.query("group_for_comparison == @group_for_comparison")
    df_without_AB_pop[mean_size_states_roi] = df_without_AB_pop[mean_size_states_roi].fillna(0)
    df_without_AB_pop[std_size_state_roi] = df_without_AB_pop[std_size_state_roi].fillna(0)
    
    df_without_AB_pop[mean_size_states_roi] = df_without_AB_pop[mean_size_states_roi]*dataset_config.conversion_px_micro_meter**2
    df_without_AB_pop[std_size_state_roi] = df_without_AB_pop[std_size_state_roi]*dataset_config.conversion_px_micro_meter**2

    fig = make_subplots(1,len(labels),subplot_titles=["Size "+ c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    for i in range(1,len(labels)+1):
        x = list(df_without_AB_pop[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")) 
        x_rev = x[::-1]
        size_upper = list(df_without_AB_pop[mean_size_states_roi[i-1]].values + df_without_AB_pop[std_size_state_roi[i-1]].values)
        size_lower = list(df_without_AB_pop[mean_size_states_roi[i-1]].values - df_without_AB_pop[std_size_state_roi[i-1]])
        size_lower = size_lower[::-1]
        fig.add_trace(go.Scatter(
        name = labels[i-1],
        x=x+x_rev,
        y=size_upper+size_lower, connectgaps=True,
        fill='toself',
        fillcolor=COLORS_STATES_PLOTLY_TRANSPARENCY[i-1],
        line_color=COLORS_STATES_PLOTLY[i-1],
        showlegend=False,
    ),1,i)
        fig.add_trace(go.Scatter(
            x=x,
            y=df_without_AB_pop[mean_size_states_roi[i-1]],
            line_color=COLORS_STATES_PLOTLY[i-1],
            name=labels[i-1],
        ),1,i)

    fig.update_layout(title_text="<b>Cell size in in "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        showlegend=True,width=1800,height=600,margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ))

    fig.update_yaxes(side="left",title_text = "Size µm²",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    # fig.update_yaxes(matches='y')
    fig.update_traces(mode='lines')
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "1_Cells_quantities_and_sizes")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        filepath = os.path.join(pathfile,"size_cells_by_states.png")
        fig.write_image(filepath)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Stats on slide """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def stat_n_cells_slides(save_fig = False,display=False):
    '''
    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''

    liste_nb_cells_slide = ['n_cells_slide', 'n_Proliferative_slide', 'n_Amoeboid_slide', 'n_Cluster_slide', 'n_Phagocytic_slide', 'n_Ramified_slide']
    liste_nb_cells_slide_proba = ['n_cells_slide','n_Proliferative_proba_slide', 'n_Amoeboid_proba_slide', 'n_Cluster_proba_slide', 'n_Phagocytic_proba_slide', 'n_Ramified_proba_slide']

    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]


    df_slides, path_temporal_analysis_csv = deep_cell_map.create_load_df_slides()


    fig = make_subplots(2, 3,subplot_titles=["#"+ c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    for i in range(1,len(labels)+1):
        fig.add_trace(po.Bar(
        name =labels[i-1],
        x = df_slides["pcw"].apply(lambda x: str(x)+" pcw"),
        y = df_slides[liste_nb_cells_slide[i-1]],
        text= df_slides[liste_nb_cells_slide[i-1]],
        marker_color = COLORS_STATES_PLOTLY[i-1]
    ), row=(i-1)//3+1, col=(i-1)%3+1)
        fig.add_trace(po.Bar(
        name = labels[i-1]+'(proba)',
        x = df_slides["pcw"].apply(lambda x: str(x)+" pcw"),
        y = df_slides[liste_nb_cells_slide_proba[i-1]],
        text= df_slides[liste_nb_cells_slide_proba[i-1]].apply(lambda x: np.round(x,1)),
        marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i-1]
        ),row=(i-1)//3+1, col=(i-1)%3+1)
    fig.update_traces( textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(title_text="<b>Cell number in slides across time",title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )
    # fig.update_yaxes(matches='y')
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path =os.path.join(path_temporal_analysis_folder, "Inter_slides_analysis")
        pathfile = os.path.join(path, "1_Cells_quantities_and_sizes")
        mkdir_if_nexist(pathfile)
        filepath = os.path.join(pathfile,"n_cells_by_states.png")
        fig.write_image(filepath)

def stat_n_cells_slides_only_one(save_fig = False,display=False, proba = True):
    '''
    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    suffix = "_proba" if proba else "_decision"
    liste_nb_cells_slide = ['n_cells_slide', 'n_Proliferative_slide', 'n_Amoeboid_slide', 'n_Cluster_slide', 'n_Phagocytic_slide', 'n_Ramified_slide']
    liste_nb_cells_slide_proba = ['n_cells_slide','n_Proliferative_proba_slide', 'n_Amoeboid_proba_slide', 'n_Cluster_proba_slide', 'n_Phagocytic_proba_slide', 'n_Ramified_proba_slide']

    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]


    df_slides, path_temporal_analysis_csv = deep_cell_map.create_load_df_slides()


    fig = make_subplots(2, 3,subplot_titles=["#"+ c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    for i in range(1,len(labels)+1):
        if proba: 
            fig.add_trace(po.Bar(
            name = labels[i-1]+'(proba)',
            x = df_slides["pcw"].apply(lambda x: str(x)+" pcw"),
            y = df_slides[liste_nb_cells_slide_proba[i-1]],
            text= df_slides[liste_nb_cells_slide_proba[i-1]].apply(lambda x: np.round(x,1)),
            marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i-1]
            ),row=(i-1)//3+1, col=(i-1)%3+1)
        else : 
            fig.add_trace(po.Bar(
            name =labels[i-1],
            x = df_slides["pcw"].apply(lambda x: str(x)+" pcw"),
            y = df_slides[liste_nb_cells_slide[i-1]],
            text= df_slides[liste_nb_cells_slide[i-1]],
            marker_color = COLORS_STATES_PLOTLY[i-1]
        ), row=(i-1)//3+1, col=(i-1)%3+1)
    fig.update_traces( textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(title_text="<b>Cell number in slides across time",title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )
    # fig.update_yaxes(matches='y')
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path =os.path.join(path_temporal_analysis_folder, "Inter_slides_analysis")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "1_Cells_quantities_and_sizes")
        mkdir_if_nexist(pathfile)
        filepath = os.path.join(pathfile,"n_cells_by_states.png")
        fig.write_image(filepath)

def stat_size_cells_slides(save_fig = False, display=False):
    '''
   Liens utilisés pour créer le graphique : 
   -https://plotly.com/python/line-charts/

    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    same_y_axix = False
    mean_size_states_slides = ['mean_cell_size_slide', 'mean_size_Proliferative_slide', 'mean_size_Amoeboid_slide', 'mean_size_Cluster_slide', 'mean_size_Phagocytic_slide', 'mean_size_Ramified_slide']
    std_size_state_slides = ['std_size_slide', 'std_cell_size_Proliferative_slide', 'std_cell_size_Amoeboid_slide', 'std_cell_size_Cluster_slide', 'std_cell_size_Phagocytic_slide', 'std_cell_size_Ramified_slide']
    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    df_slides, path_temporal_analysis_csv = deep_cell_map.create_load_df_slides()

    df_slides[mean_size_states_slides] = df_slides[mean_size_states_slides]*dataset_config.conversion_px_micro_meter**2
    df_slides[std_size_state_slides] = df_slides[std_size_state_slides]*dataset_config.conversion_px_micro_meter**2

    fig = make_subplots(1,len(labels),subplot_titles=["Size "+ c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    for i in range(1,len(labels)+1):
        x = list(df_slides["pcw"].apply(lambda x: str(x)+" pcw"))
        x_rev = x[::-1]
        size_upper = list(df_slides[mean_size_states_slides[i-1]].values + df_slides[std_size_state_slides[i-1]].values)
        size_lower = list(df_slides[mean_size_states_slides[i-1]].values - df_slides[std_size_state_slides[i-1]])
        size_lower = size_lower[::-1]
        fig.add_trace(go.Scatter(
        name = labels[i-1],
        x=x+x_rev,
        y=size_upper+size_lower, connectgaps=True,
        fill='toself',
        fillcolor=COLORS_STATES_PLOTLY_TRANSPARENCY[i-1],
        line_color=COLORS_STATES_PLOTLY[i-1],
        showlegend=False,
    ),1,i)
        fig.add_trace(go.Scatter(
            x=x,
            y=df_slides[mean_size_states_slides[i-1]],
            line_color=COLORS_STATES_PLOTLY[i-1],
            name=labels[i-1],
        ),1,i)

    # fig.update_traces( textangle=90, textposition="outside", cliponaxis=False)
    fig.update_layout(title_text="<b>Cell size in slides across time",title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )
    fig.update_yaxes(side="left",title_text = "Size µm²",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    fig.update_yaxes(matches='y') if same_y_axix else None 
    fig.update_traces(mode='lines')
    if display : 
        fig.show()
    if save_fig: 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path =os.path.join(path_temporal_analysis_folder, "inter_slides_analysis")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"Cell_size_by_states"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def _preprocess_before_heatmap(roi_analysis,colname_to_play_z="coloc_delta_a",round_z=-1, convert_micro_m = False):
    import math
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    group_for_comparison = int(group_for_comparison)
    """Return a matrix 5x5 to plot any metric of colocalisation between 5 states """
    df_group_of_comparaison = roi_analysis.copy()
    df_group_of_comparaison = df_group_of_comparaison.query("group_for_comparison == @group_for_comparison")
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    dict_matrix_pet_id_in_group_comparaison=dict()
    for id_roi in list(df_group_of_comparaison["id_in_group"].unique()):
        df = df_group_of_comparaison.query("id_in_group == @id_roi")
        dict_matrix_pet_id_in_group_comparaison[str(id_roi)] = np.zeros((5,5))
        for idx_A, state_A in enumerate(labels):
            for idx_B, state_B in enumerate(labels):
                if idx_B != idx_A : 
                    z = df[(df["pop_A"] == state_A) & (df["pop_B"] == state_B)][colname_to_play_z].values[0]
                    z = z*dataset_config.conversion_px_micro_meter if convert_micro_m else z
                    if round_z != -1:
                        z = round(z,round_z) if z != math.inf else np.nan
                    dict_matrix_pet_id_in_group_comparaison[str(id_roi)][idx_A,idx_B] = z
    return dict_matrix_pet_id_in_group_comparaison

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" State-state colocalisation  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
WIDTH_FIG_TEMPORAL_ANALYSIS = 1600
HEIGHT_FIG_TEMPORAL_ANALYSIS = 700

def stat_heatmap_colocalisation_delta_a(roi_analysis,proba = False, save_fig = True,display=True):
    # if len(roi_analysis["pcw"].unique()) > 0:
    #     x_axis_column = "pcw"
    # else : 
    #     x_axis_column = "id_in_group"
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]

    colname_to_play_z = find_feature_name_in_physiological_df("coloc_delta_a") if not proba else find_feature_name_in_physiological_df("coloc_delta_a_proba")
    title_metric = " (considering proba) " if proba else ""
    suffix_filename = "_proba" if proba else ""
    round_z = 0
    convert_micro_m = True
    
    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z = round_z,convert_micro_m=convert_micro_m)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())

    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)

    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"), #ici np.flipud est le seul moyen d'inversé le y-axis selon https://github.com/plotly/plotly.py/issues/413
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>State-state colocalisation : mean distance "+title_metric +"<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=290, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="Mean <br>distance<br>colocalisation<br>(<span>&#181;</span>m)"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path_heatmaps = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path_heatmaps)
        filepath = os.path.join(path_heatmaps,"Coloc_mean_dist_AB"+suffix_filename+""+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_line_colocalisation_delta_a(roi_analysis, save_fig=False, display = True): 
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"

    y_name = ["coloc_delta_a","coloc_delta_a_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a")] = df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a")]*dataset_config.conversion_px_micro_meter
    df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a_proba")] = df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a_proba")]*dataset_config.conversion_px_micro_meter
    fig_stat = px.line(df_group_of_comparaison, x=df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=1100, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>State A/B colocalisation : Mean distance <br> "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df("coloc_delta_a"):'Decision', find_feature_name_in_physiological_df("coloc_delta_a_proba"): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 4
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"
    df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a")] = df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a")]/dataset_config.conversion_px_micro_meter
    df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a_proba")] = df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a_proba")]/dataset_config.conversion_px_micro_meter
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_coloc_mean_dist_AB"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()


def stat_line_colocalisation_delta_a_V1(roi_analysis, save_fig=False, display = True): 
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"

    y_name = ["coloc_delta_a","coloc_delta_a_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a")] = df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a")]*dataset_config.conversion_px_micro_meter
    df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a_proba")] = df_group_of_comparaison[find_feature_name_in_physiological_df("coloc_delta_a_proba")]*dataset_config.conversion_px_micro_meter
    for pop_B in df_group_of_comparaison["pop_B"].unique():
        fig_stat = px.line(df_group_of_comparaison, x=df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"), y=y_name, facet_row="pop_A",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=2000, width= 800)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>State A/B colocalisation : Mean distance <br> "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df("coloc_delta_a"):'Decision', find_feature_name_in_physiological_df("coloc_delta_a_proba"): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    # for i in range(20):
    #     fig_stat.data[i].line.width = 4
    #     if i//4 ==0: 
    #         fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
    #         fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
    #         fig_stat.data[20+i].line.dash = "dot"
    #     if i//4 ==1: 
    #         fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
    #         fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
    #         fig_stat.data[20+i].line.dash = "dot"     
    #     if i//4 ==2: 
    #         fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
    #         fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
    #         fig_stat.data[20+i].line.dash = "dot"
    #     if i//4 ==3:
    #         fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
    #         fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
    #         fig_stat.data[20+i].line.dash = "dot"
    #     if i//4 ==4:
    #         fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
    #         fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
    #         fig_stat.data[20+i].line.dash = "dot"
    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_coloc_mean_dist_AB"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def _stat_line_colocalisation_delta_a_one_line(roi_analysis,proba = False,save_fig=False): 
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    y_name = "coloc_delta_a" if not proba else "coloc_delta_a_proba"
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    title_metric = " (considering proba) " if proba else ""
    suffix_filename = "_proba" if proba else ""
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=df_group_of_comparaison[y_name].apply(lambda x: x*dataset_config.conversion_px_micro_meter), facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},title="Analysis dist<br>",markers=True, height=1300, width= 1600)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="State "+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))



    fig_stat.update_layout(title_text="<b>Mean distance co localisation "+title_metric+"<br> "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )


    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 4
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]

    if save_fig:
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "coloc_mean_dist_AB_visutype_2"+suffix_filename+""+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    else : 
        fig_stat.show()
    return fig_stat 

def _stat_line_colocalisation_delta_a_one_line_V2(roi_analysis,save_fig = False,display=False):
    '''
   Liens utilisés pour créer le graphique : 
   -https://plotly.com/python/line-charts/

    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    colors = ["aliceblue", "black", "firebrick", "blue",
            "blueviolet", "hotpink", "darkturquoise", "cadetblue",
            "fuchsia", "greenyellow", "coral", "cornflowerblue",
            "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
            "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
            "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
            "darkorchid", "darkred", "darksalmon", "darkseagreen",
            "darkslateblue", "darkslategray", "darkslategrey",
            "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
            "dimgray", "dimgrey", "dodgerblue", "firebrick",
            "floralwhite", "forestgreen", "fuchsia", "gainsboro",
            "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
            "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
            "ivory", "khaki", "lavender", "lavenderblush", "lawngreen"]
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    fig = make_subplots(2, 3,subplot_titles=["Dist "+ c +" to others" for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())

    show_legend = True
    for idx_A, state_A in enumerate(labels,1):
        for idx_plot, indice_roi in enumerate(indices_roi,1):
            x = [c for c in labels if c != state_A]
            dist = [df_group_of_comparaison[(df_group_of_comparaison["pop_A"] == state_A) & (df_group_of_comparaison["pop_B"] == state_B) & (df_group_of_comparaison["id_in_group"] == indice_roi)]["coloc_delta_a"].values[0] for state_B in x]
            fig.add_trace(go.Scatter(
            name = 'ROI '+str(indice_roi),
            x=x,
            y=dist,mode='lines',marker={'size':10},
            showlegend=show_legend
        ),row=(idx_A-1)//3+1, col=(idx_A-1)%3+1)
        show_legend = False

    # fig.update_traces( textangle=90, textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="Cell size in ROIs of group "+str(group_for_comparison),
        showlegend=True,
        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
    fig.update_layout(title_text="<b>Mean distance co localisation <br> "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    # line_color=colors[idx_plot],connectgaps=False
    fig.update_yaxes(matches='y')
    # fig.update_traces(mode='lines')
    fig.update_layout(legend=dict(yanchor="top", xanchor="left",x=0.7,y=0.45),legend_title_text='ROIs')
 
    fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"coloc_mean_dist_AB_visutype_3"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_heatmap_colocalisation_p_value(roi_analysis,proba = False,save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("coloc_p_value") if not proba else find_feature_name_in_physiological_df("coloc_p_value_proba")
    title_metric = " (proba) " if proba else ""
    suffix_filename = "_proba" if proba else ""
    round_z = -1

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")

    n_roi = len(df_group_of_comparaison["id_in_group"].unique())

    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)

    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>Colocalisation : p-values "+title_metric+" <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=270, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="p value"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_coloc_p_value_AB"+suffix_filename+""+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_line_colocalisation_p_value(roi_analysis,convert_to_microm=False, save_fig=False, display = False): 
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    
    y_name = ["coloc_p_value","coloc_p_value_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")

    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},title="Analysis p value<br>",markers=True, height=1100, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="State A/B colocalisation : P-value <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('coloc_p_value'):'Decision', find_feature_name_in_physiological_df('coloc_p_value_proba'): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "p-value",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 4
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"
    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_p_value"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_heatmap_colocalisation_association_index(roi_analysis,proba=False, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("coloc_Association_Index_ASI") if not proba else find_feature_name_in_physiological_df("coloc_Association_Index_ASI_proba")
    title_metric = " (considering proba) " if proba else ""
    suffix_filename = "_proba" if proba else ""
    round_z = 2

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>State A/B colocalisation : Associacion Index (ASI)"+title_metric+" <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=270, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="ASI"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_coloc_ASI_AB"+suffix_filename+""+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_line_colocalisation_ASI(roi_analysis,convert_to_microm=False, save_fig=False, display = False): 
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ["coloc_Association_Index_ASI","coloc_Association_Index_ASI_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    # y_name = ["coloc_Association_Index_ASI","coloc_Association_Index_ASI_proba","coloc_Accumulation_Index_ACI","coloc_Accumulation_Index_ACI_proba"]

    dataset_config.conversion_px_micro_meter = 0.45
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    if convert_to_microm : 
        df_group_of_comparaison["FEATURE"] = df_group_of_comparaison["FEATURE"]*dataset_config.conversion_px_micro_meter
        df_group_of_comparaison["FEATURE_proba"] = df_group_of_comparaison["FEATURE_proba"]*dataset_config.conversion_px_micro_meter
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},title="Analysis FEATUREt<br>",markers=True, height=1100, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_traces(line_color='purple')
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="State A/B colocalisation : ASI <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )
    newnames = {find_feature_name_in_physiological_df('coloc_Association_Index_ASI'):'ASI decision', find_feature_name_in_physiological_df('coloc_Association_Index_ASI_proba'): 'ASI proba'}
    # newnames = {'coloc_Association_Index_ASI':'ASI decision', 'coloc_Association_Index_ASI_proba': 'ASI proba',"coloc_Accumulation_Index_ACI":"decision","coloc_Accumulation_Index_ACI_proba":"ASI proba"}

    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "ASI",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 1
        fig_stat.data[i+20].line.width = 1

        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"


        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     

        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"

        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"

        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_ASI"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_colocalisation_ASI_SAI(roi_analysis,convert_to_microm=False, save_fig=False, display = False): 
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ["coloc_Association_Index_ASI","coloc_Association_Index_ASI_proba","coloc_Significant_Accumulation_Index_SAI","coloc_Significant_Accumulation_Index_SAI_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    # y_name = ["coloc_Association_Index_ASI","coloc_Association_Index_ASI_proba","coloc_Accumulation_Index_ACI","coloc_Accumulation_Index_ACI_proba"]


    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=1100, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Colocalisation : Association (ASI) & Significant Accumulation (SAI) Indexes<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )
    newnames = {find_feature_name_in_physiological_df('coloc_Association_Index_ASI'):'ASI decision', find_feature_name_in_physiological_df('coloc_Association_Index_ASI_proba'): 'ASI proba',find_feature_name_in_physiological_df("coloc_Significant_Accumulation_Index_SAI"):"ASI decision",find_feature_name_in_physiological_df("coloc_Significant_Accumulation_Index_SAI_proba"):"ASI proba"}
    # newnames = {'coloc_Association_Index_ASI':'ASI decision', 'coloc_Association_Index_ASI_proba': 'ASI proba',"coloc_Accumulation_Index_ACI":"decision","coloc_Accumulation_Index_ACI_proba":"ASI proba"}

    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "ASI/SAI",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 1
        fig_stat.data[i+20].line.width = 1
        fig_stat.data[i+40].line.width = 1
        fig_stat.data[i+60].line.width = 1
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[40+i].line.dash = "dash"
            fig_stat.data[60+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[60+i].line.dash = "longdashdot"

        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     
            fig_stat.data[40+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[40+i].line.dash = "dash"
            fig_stat.data[60+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[60+i].line.dash = "longdashdot"
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[40+i].line.dash = "dash"
            fig_stat.data[60+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[60+i].line.dash = "longdashdot"
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[40+i].line.dash = "dash"
            fig_stat.data[60+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[60+i].line.dash = "longdashdot"
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[40+i].line.dash = "dash"
            fig_stat.data[60+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[60+i].line.dash = "longdashdot"
    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"2_State_state_colocalisation")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_ASI_SAI"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" DBSCAN  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def stat_heatmap_IoU(roi_analysis, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("dbscan_iou")
    round_z = False

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B ") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>DBSCAN -IoU convex hulls A and B<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=270, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="IoU"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_IoU"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_heatmap_IuA(roi_analysis, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("dbscan_i_A")
    round_z = False

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>DBSCAN - IoA : Intersection/A_area (convex hulls A and B)<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=250, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="IoA"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_IoA"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_line_dbscan_iou_ioa(roi_analysis,convert_to_microm=False, save_fig=False, display = False): 
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ["dbscan_iou","dbscan_i_A"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]

    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},title="Analysis Intersect<br>",markers=True, height=1100, width= 1400)
                              
    fig_stat.update_yaxes(matches=None,showticklabels=True)

    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>DBSCAN : IoU and IoA<br> "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('dbscan_iou'):'IoU', find_feature_name_in_physiological_df('dbscan_i_A'): 'IoA'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "IoU & IoA",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 4
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"
    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_IoU_IoA"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_heatmap_rate_A_in_B(roi_analysis, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("dbscan_fraction_A_in_B_clusters")
    round_z = False

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>DBSCAN - Rate A cells in B territories (B convex hulls)<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=250, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="Rate A cells <br>in B territories"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_rate_A_cells_in_B_territories"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_heatmap_fraction_clustered_A_in_intersect(roi_analysis, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("dbscan_fraction_clustered_A_in_intersect")
    print(colname_to_play_z)
    round_z = True

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    df_group_of_comparaison = df_group_of_comparaison.sort_values(by="pcw")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>DBSCAN - Rate clustered A cells in intersected convex hulls<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=250, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="Rate A cells <br>in B territories"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_TOCHANGEdbscan_fraction_clustered_A_in_intersect"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_heatmap_fraction_clustered_A_in_intersect_UNE_PCW(roi_analysis, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("dbscan_fraction_clustered_A_in_intersect")
    print(colname_to_play_z)
    round_z = True

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    df_group_of_comparaison = df_group_of_comparaison.sort_values(by="pcw")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(1)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower")
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # for idx_plot, indices_roi in enumerate(indices_roi,1):
    #     fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
    #                     x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
    #                     y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

    #     fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
    #     fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>DBSCAN - Rate clustered A cells in intersected convex hulls<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=250, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="Rate A cells <br>in B territories"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"heatmap_TOCHANGEdbscan_fraction_clustered_A_in_intersect"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_line_rate_A_in_B(roi_analysis, save_fig=False, display = False): 
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ["dbscan_fraction_A_in_B_clusters"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=1100, width= 1400)
            
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)
    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))
    fig_stat.update_layout(title_text="<b>DBSCAN - Rate state A (SA) cells in state B (SB) territories (B convex hulls)<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )
    # newnames = {'dbscan_fraction_A_in_B_clusters':'ASI proba'}
    # newnames = {'coloc_Association_Index_ASI':'ASI decision', 'coloc_Association_Index_ASI_proba': 'ASI proba',"coloc_Accumulation_Index_ACI":"decision","coloc_Accumulation_Index_ACI_proba":"ASI proba"}

    # fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "Rate A in B",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 3
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"3_DBSCAN")
        mkdir_if_nexist(path)
        pathfile = os.path.join(path, "lines_rate_A_in_B_territories"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def fig_clustered_cells_vs_other(roi_analysis, save_fig = False,display=False):
    '''
    Display les nombres de cellules (decision et proba) pour 1 groupe de comparaison
    '''
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]

    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified","Proliferative"]
    roi_analysis = roi_analysis.query("group_for_comparison == @group_for_comparison")


    fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
    for i in range(1,len(labels)):
        roi_analysis_pop_A = roi_analysis.query("pop_A == @labels[@i-1] & pop_B == @labels[@i]")
        col_nb_cells = "n_"+labels[i-1]+"_roi"
        fig.add_trace(po.Bar(
        name ="n " + labels[i-1] +" clustered",
        x = roi_analysis_pop_A[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"),
        y = roi_analysis_pop_A[col_nb_cells]-roi_analysis_pop_A["dbscan_n_isolated_A"],
        text= roi_analysis_pop_A[col_nb_cells]-roi_analysis_pop_A["dbscan_n_isolated_A"],
        marker_color = COLORS_STATES_PLOTLY[i]
    ), row=(i-1)//3+1, col=(i-1)%3+1)
        fig.add_trace(po.Bar(
        name = "n " + labels[i-1] +" isolated",
        x = roi_analysis_pop_A[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw"),
        y = roi_analysis_pop_A["dbscan_n_isolated_A"],
        text= roi_analysis_pop_A["dbscan_n_isolated_A"],
        marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i]
        ),row=(i-1)//3+1, col=(i-1)%3+1)
    fig.update_traces( textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
       title_text="<b>DBSCAN - Clusterisation profil of each states - "+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
        showlegend=True,
        width=1600,
        height=800,margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ))
    # fig.update_layout(barmode='stack')
    fig.update_yaxes(matches='y')
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "3_DBSCAN")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        filepath = os.path.join(pathfile,"clustered_vs_isolated.png")
        fig.write_image(filepath)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" State - Border colocalisation   """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def stat_line_bcoloc_delta_a(roi_analysis, save_fig=False, display = False): 
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ["coloc_state_border_delta_a","coloc_state_border_delta_a_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    dataset_config.conversion_px_micro_meter = 0.45
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    for feature in y_name :
        df_group_of_comparaison[feature] = df_group_of_comparaison[feature]*dataset_config.conversion_px_micro_meter
    # df_group_of_comparaison["coloc_state_border_delta_a"] = df_group_of_comparaison["coloc_state_border_delta_a"]*dataset_config.conversion_px_micro_meter
    # df_group_of_comparaison["coloc_state_border_delta_a_proba"] = df_group_of_comparaison["coloc_state_border_delta_a_proba"]*dataset_config.conversion_px_micro_meter
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_col="pop_B",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Cell-Border colocalisation : Mean distance <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('coloc_state_border_delta_a'):'Decision', find_feature_name_in_physiological_df('coloc_state_border_delta_a_proba'): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(5):
        fig_stat.data[i].line.width = 4
        fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.dash = "dot"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "3_Cell_border_colocalisation")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "lines_bcoloc_mean_dist_B"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()


def stat_line_bcoloc_p_value(roi_analysis, save_fig=False, display = False): 
    '''

    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ['coloc_state_border_p_value','coloc_state_border_p_value_proba']
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_col="pop_B",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1400)

    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Cell-Border Colocalisation : P-value <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('coloc_state_border_p_value'):'Decision', find_feature_name_in_physiological_df('coloc_state_border_p_value_proba'): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "p-value",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(5):
        fig_stat.data[i].line.width = 4
        fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.dash = "dot"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "3_Cell_border_colocalisation")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "lines_p_value_bcoloc"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_bcoloc_ASI(roi_analysis, save_fig=False, display = False): 
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ['coloc_state_border_association_index_ASI','coloc_state_border_association_index_ASI_proba']
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")

    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_col="pop_B",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>ASI Cell-Border colocalisation <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('coloc_state_border_association_index_ASI'):'Decision', find_feature_name_in_physiological_df('coloc_state_border_association_index_ASI_proba'): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "ASI",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(5):
        fig_stat.data[i].line.width = 4
        fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.dash = "dot"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "3_Cell_border_colocalisation")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "lines_ASI_bcoloc"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()


def stat_bar_bcoloc_p_value(roi_analysis, save_fig=False): 
    '''

    '''
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    y_name = ['coloc_state_border_p_value','coloc_state_border_p_value_proba']
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    df_group_of_comparaison = df_group_of_comparaison[y_name+[x_axis_column,"pop_B"]]
 
    df_group_of_comparaison.drop_duplicates(inplace = True)

    # fig_stat = px.line(df_group_of_comparaison, x=df_group_of_comparaison["id_in_group"].apply(lambda x: "ROI "+str(x)), y=y_name, facet_col="pop_B",facet_col_spacing=0.05, 
    #                             category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1400)
    fig_stat = px.bar(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name,barmode="group", facet_col="pop_B",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())}, height=600, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Cell-Border Colocalisation : P-value <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('coloc_state_border_p_value'):'Decision', find_feature_name_in_physiological_df('coloc_state_border_p_value_proba'): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "p-value",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(5):

        fig_stat.data[i].marker.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].marker.color = COLORS_STATES_PLOTLY_TRANSPARENCY[i+1]


    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        # path =os.path.join(path_temporal_analysis_folder, "group_"+str(group_for_comparison))
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "3_Cell_border_colocalisation")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "bar_p_value_bcoloc"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    else : 
        fig_stat.show()

def stat_line_bcoloc_SAI(roi_analysis, save_fig=False, display = False):  
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    y_name = ['coloc_state_border_significant_accumulation_index_SAI','coloc_state_border_significant_accumulation_index_SAI_proba']
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")

    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_col="pop_B",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>ASI Cell-Border colocalisation <br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('coloc_state_border_significant_accumulation_index_SAI'):'Decision', find_feature_name_in_physiological_df('coloc_state_border_significant_accumulation_index_SAI_proba'): 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "SAI",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(5):
        fig_stat.data[i].line.width = 4
        fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.color = COLORS_STATES_PLOTLY[i+1]
        fig_stat.data[5+i].line.dash = "dot"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "3_Cell_border_colocalisation")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "lines_SAI_bcoloc"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Neighbours analysis  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def stat_heatmap_neighbours_rate_balls_r_coloc_containing_1_B(roi_analysis, save_fig = True,display=True):
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    colname_to_play_z = find_feature_name_in_physiological_df("neighbours_rate_balls_r_coloc_containing_1_B")
    print(colname_to_play_z)
    round_z = True

    dict_matrix_pet_id_in_group_comparaison = _preprocess_before_heatmap(roi_analysis,colname_to_play_z=colname_to_play_z,round_z=round_z)
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    n_roi = len(df_group_of_comparaison["id_in_group"].unique())
    indices_roi = list(df_group_of_comparaison["id_in_group"].unique())
    
    fig = make_subplots(n_roi//3, 3, subplot_titles=[str(int(c)) + " pcw" for c in list(roi_analysis["pcw"].unique())],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    # fig = make_subplots(n_roi//2, 2, subplot_titles=["ROI "+ c for c in dict_matrix_pet_id_in_group_comparaison.keys()],vertical_spacing = 0.15,horizontal_spacing = 0.15)
    for idx_plot, indices_roi in enumerate(indices_roi,1):
        fig.add_trace(px.imshow(np.flipud(dict_matrix_pet_id_in_group_comparaison[str(indices_roi)]),labels=dict(x="State B", y="State A"),
                        x=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"],
                        y=["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"][::-1], text_auto=True, aspect="auto",origin = "lower").data[0], row=(idx_plot-1)//3+1, col=(idx_plot-1)%3+1)

        fig.update_xaxes(side="top",title_text = "State B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="State A")
    
    for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=20)  
        annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
    fig.update_layout(title_text="<b>Neighbours analysis  - Fraction of ball(r_coloc) having at least 1 B inside<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                        coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                        showlegend=True,
                        width=WIDTH_FIG_TEMPORAL_ANALYSIS, #Taille de la figure 
                        height=HEIGHT_FIG_TEMPORAL_ANALYSIS,
                        margin=dict(l=50,r=50,b=50,t=250, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        coloraxis_colorbar=dict(title="Fract ball type 1 <br> having B inside"))
    if display : 
        fig.show()
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        mkdir_if_nexist(path)
        path = os.path.join(path,"5_Neighbours_analysis")
        mkdir_if_nexist(path)
        filepath = os.path.join(path,"neighbours_rate_balls_r_coloc_containing_1_B_"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig.write_image(filepath)

def stat_line_neighbours_mean_dist_first_B_around_A(roi_analysis, save_fig = False, display=False):
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    y = ['neighbours_mean_dist_first_B_around_A','neighbours_mean_dist_second_B_around_A','neighbours_mean_dist_third_B_around_A']
    y = [ find_feature_name_in_physiological_df(feature) for feature in y]
    dataset_config.conversion_px_micro_meter = 0.45
    for feature in y :
        df_group_of_comparaison[feature] = df_group_of_comparaison[feature]*dataset_config.conversion_px_micro_meter

    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y, facet_row="pop_A",facet_col="pop_B", facet_row_spacing=0.06, facet_col_spacing=0.05, category_orders={"pop_A": list(df_group_of_comparaison["pop_A"].unique()),
                              "pop_B": list(df_group_of_comparaison["pop_A"].unique())},markers=True,title = "Succes", height=1000, width= 1300)
                              
    fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    newnames = {find_feature_name_in_physiological_df('neighbours_mean_dist_first_B_around_A'):'First', find_feature_name_in_physiological_df('neighbours_mean_dist_second_B_around_A'): 'Second',find_feature_name_in_physiological_df("neighbours_mean_dist_third_B_around_A"):"Third"}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    fig_stat.update_layout(title_text="<b>Neighbours analysis - Mean dist firsts B around A Cells<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=280),
                        )
                
    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+":<br>"+a.text.split("=")[-1]))
    
    for annotation in fig_stat['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=15)  
        annotation['yshift'] =1
    for i in range(20):
        fig_stat.data[i].line.width = 1
        fig_stat.data[i+20].line.width = 1
        fig_stat.data[i+40].line.width = 1

        if i//4 ==0: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==1: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"     
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==2: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==3:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==4:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "5_Neighbours_analysis")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "mean_dist_firsts_B_around_A"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_neighbours_std_dist_first_B_around_A(roi_analysis, save_fig = False, display=False):
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    y = ['neighbours_std_dist_first_B_around_A','neighbours_std_dist_second_B_around_A','neighbours_std_dist_third_B_around_A']
    y = [ find_feature_name_in_physiological_df(feature) for feature in y]
    dataset_config.conversion_px_micro_meter = 0.45
    for feature in y :
        df_group_of_comparaison[feature] = df_group_of_comparaison[feature]*dataset_config.conversion_px_micro_meter

    # df_group_of_comparaison["neighbours_std_dist_first_B_around_A"] = df_group_of_comparaison["neighbours_std_dist_first_B_around_A"]*dataset_config.conversion_px_micro_meter
    # df_group_of_comparaison["neighbours_std_dist_second_B_around_A"] = df_group_of_comparaison["neighbours_std_dist_second_B_around_A"]*dataset_config.conversion_px_micro_meter
    # df_group_of_comparaison["neighbours_std_dist_third_B_around_A"] = df_group_of_comparaison["neighbours_std_dist_third_B_around_A"]*dataset_config.conversion_px_micro_meter
    
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y, facet_row="pop_A",facet_col="pop_B", facet_row_spacing=0.06, facet_col_spacing=0.05, category_orders={"pop_A": list(df_group_of_comparaison["pop_A"].unique()),
                              "pop_B": list(df_group_of_comparaison["pop_A"].unique())},markers=True,title = "Succes", height=1000, width= 1300)
                              
    fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    newnames = {find_feature_name_in_physiological_df('neighbours_std_dist_first_B_around_A'):'First', find_feature_name_in_physiological_df('neighbours_std_dist_second_B_around_A'): 'Second',find_feature_name_in_physiological_df("neighbours_std_dist_third_B_around_A"):"Third"}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    fig_stat.update_layout(title_text="<b>Neighbours analysis - Std dist firsts B around A Cells<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=280),
                        )
                
    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+":<br>"+a.text.split("=")[-1]))
    
    for annotation in fig_stat['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=15)  
        annotation['yshift'] =1
    for i in range(20):
        fig_stat.data[i].line.width = 1
        fig_stat.data[i+20].line.width = 1
        fig_stat.data[i+40].line.width = 1

        if i//4 ==0: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==1: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"     
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==2: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==3:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

        if i//4 ==4:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            fig_stat.data[20+i].line.dash = "dot"
            fig_stat.data[40+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
            fig_stat.data[40+i].line.dash = "dash"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "5_Neighbours_analysis")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "std_dist_firsts_B_around_A"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_neighbours_rate_at_least_1_B_in_different_BALLS(roi_analysis, save_fig = False, display=False):
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    y = ['neighbours_rate_balls_r_coloc_containing_1_B']
    y = [ find_feature_name_in_physiological_df(feature) for feature in y]
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y, facet_row="pop_A",facet_col="pop_B", facet_row_spacing=0.06, facet_col_spacing=0.05, category_orders={"pop_A": list(df_group_of_comparaison["pop_A"].unique()),
                              "pop_B": list(df_group_of_comparaison["pop_A"].unique())},markers=True,title = "Succes", height=1000, width= 1300)
                              
    fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    newnames = {find_feature_name_in_physiological_df('neighbours_rate_balls_r_coloc_containing_1_B'):'Rate'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    fig_stat.update_layout(title_text="<b>Neighbours analysis -Rate balls(radius) having at least 1 B cells<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=280),
                        )
                
    fig_stat.update_yaxes(side="left",title_text = "Rate",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+":<br>"+a.text.split("=")[-1]))
    
    for annotation in fig_stat['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=15)  
        annotation['yshift'] =1
    for i in range(20):
        fig_stat.data[i].line.width = 1
        # fig_stat.data[i+20].line.width = 1


        if i//4 ==0: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            # fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            # fig_stat.data[20+i].line.dash = "dot"

        if i//4 ==1: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            # fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            # fig_stat.data[20+i].line.dash = "dot"     
        if i//4 ==2: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            # fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            # fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==3:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            # fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            # fig_stat.data[20+i].line.dash = "dot"

        if i//4 ==4:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
            # fig_stat.data[20+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
            # fig_stat.data[20+i].line.dash = "dot"
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "5_Neighbours_analysis")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "Rate_at_least_1_B_in_different_balls"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_neighbours_rate_B_first(roi_analysis, save_fig = False, display=False):
    '''
    ATT : il faudrait une matrice vs ici !!! c'est pour ça qu'on arrive pas a 100
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    y = ['neighbours_rate_B_first_neighbour']
    y = [ find_feature_name_in_physiological_df(feature) for feature in y]
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y, facet_row="pop_A",facet_col="pop_B", facet_row_spacing=0.06, facet_col_spacing=0.05, category_orders={"pop_A": list(df_group_of_comparaison["pop_A"].unique()),
                              "pop_B": list(df_group_of_comparaison["pop_A"].unique())},markers=True,title = "Succes", height=1000, width= 1300)
                              
    fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    newnames = {find_feature_name_in_physiological_df('neighbours_rate_B_first_neighbour'):'Rate B first'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    fig_stat.update_layout(title_text="<b>Neighbours analysis - Rate B first neighbours of A<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=280),
                        )
                
    fig_stat.update_yaxes(side="left",title_text = "Rate",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+":<br>"+a.text.split("=")[-1]))
    
    for annotation in fig_stat['layout']['annotations']: #Le style des titres du plot est est modifié ici 
        annotation['font'] = dict(size=15)  
        annotation['yshift'] =1
    for i in range(20):
        fig_stat.data[i].line.width = 1
        if i//4 ==0: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
        if i//4 ==1: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]  
        if i//4 ==2: 
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
        if i//4 ==3:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
        if i//4 ==4:
            lab = labels[i//4]
            fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]

    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "5_Neighbours_analysis")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "Rate_B_first_neighbours_of_A"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_neighbours_mean_dist_first_A_neighbours(roi_analysis, save_fig=False, display = False):  
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    # pop_B = "Amoeboid"
    # df_group_of_comparaison = roi_analysis.query("pop_B == @pop_B")
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    y = ['neighbours_mean_A_first_neighbour','neighbours_mean_A_second_neighbour','neighbours_mean_A_third_neighbour']
    y = [find_feature_name_in_physiological_df(feature) for feature in y]
    dataset_config.conversion_px_micro_meter = 0.45
    for coll in y :
        df_group_of_comparaison[coll] = df_group_of_comparaison[coll]*dataset_config.conversion_px_micro_meter

    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y, facet_col="pop_A",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1300)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Neighbours analysis -Mean dist first cell around A<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('neighbours_mean_A_first_neighbour'):'First', find_feature_name_in_physiological_df('neighbours_mean_A_second_neighbour'): 'Second',find_feature_name_in_physiological_df("neighbours_mean_A_third_neighbour"):"Third"}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    
    for i in range(5):
        fig_stat.data[i].line.width = 4
        lab = labels[i]
        fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
        fig_stat.data[5+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
        # fig_stat.data[5+i].line.dash = "dot"
        fig_stat.data[10+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
        # fig_stat.data[10+i].line.dash = "dot"

    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "5_Neighbours_analysis")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "mean_dist_firsts_A_neighbours"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_neighbours_std_dist_first_A_neighbours(roi_analysis, save_fig=False, display = False):  
    '''
    Il faut que j'arrive a mettre les delta a 
    Creer la le plot du Z score en fonction des levelsets 
    '''
    group_for_comparison = roi_analysis["group_for_comparison"].iloc[0]
    if len(roi_analysis["pcw"].unique()) > 0:
        x_axis_column = "pcw"
    else : 
        x_axis_column = "id_in_group"
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    # pop_B = "Amoeboid"
    # df_group_of_comparaison = roi_analysis.query("pop_B == @pop_B")
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    y = ['neighbours_std_A_first_neighbour','neighbours_std_A_second_neighbour','neighbours_std_A_third_neighbour']
    y = [find_feature_name_in_physiological_df(feature) for feature in y]
    dataset_config.conversion_px_micro_meter = 0.45
    for coll in y :
        df_group_of_comparaison[coll] = df_group_of_comparaison[coll]*dataset_config.conversion_px_micro_meter


    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y, facet_col="pop_A",facet_col_spacing=0.05, 
                                category_orders={"pop_B": list(roi_analysis["pop_A"].unique())},markers=True, height=600, width= 1300)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Neighbours analysis -Std dist first cell around A<br>"+roi_analysis["physiological_part"].iloc[0],title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {find_feature_name_in_physiological_df('neighbours_std_A_first_neighbour'):'First', find_feature_name_in_physiological_df('neighbours_std_A_second_neighbour'): 'Second',find_feature_name_in_physiological_df("neighbours_std_A_third_neighbour"):"Third"}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    fig_stat.update_yaxes(side="left",title_text = "Distance (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    
    for i in range(5):
        fig_stat.data[i].line.width = 4
        lab = labels[i]
        fig_stat.data[i].line.color = adaptative_color_lines_transparency(coef_alpha=0.9)[lab]
        fig_stat.data[5+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.7)[lab]
        # fig_stat.data[5+i].line.dash = "dot"
        fig_stat.data[10+i].line.color = adaptative_color_lines_transparency(coef_alpha=0.5)[lab]
        # fig_stat.data[10+i].line.dash = "dot"

    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "5_Neighbours_analysis")
        mkdir_if_nexist(path)
        mkdir_if_nexist(pathfile)
        pathfile = os.path.join(pathfile, "std_dist_firsts_B_around_A"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Récapitule le code  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Prend un dictionnaire de noms de roi et extrait les stats sur les groupes de comparaison
def extract_stats_on_group_of_comparaison(liste_roi_to_add, dict_roi, model_name="best_model", group_for_comparison = 1 , list_levelsets =None ,liste_min_samples=4, save_fig = True, save_img_classified_cells=False, save_fig_B_cells_in_A_levelsets=False,save_figure_dbscan=False):
    """
    Calcul les informations sur les ROIs contenue dans liste_roi_to_add et plot les résultats dans un dossier de "Temporal_analysis"
    """
    #Calcul sur les ROIs
    deep_cell_map.apply_DeepCellMap_to_roi_list(liste_roi_to_add,model_name="best_model",list_levelsets = list_levelsets ,dict_roi = dict_roi,save_fig_B_cells_in_A_levelsets = save_fig_B_cells_in_A_levelsets,save_img_classified_cells = save_img_classified_cells,save_figure_dbscan=False)
    roi_analysis, path_temporal_analysis_csv = deep_cell_map.load_enrich_roi_analysis(liste_roi_to_add,dict_roi = dict_roi)
    #Display stats functions 

    print(Red("Everithing will be saved at : "+os.path.join(dataset_config.dir_base_roi,"Temporal_analysis","group_"+str(group_for_comparison))))
    display_several_roi_in_tissue(roi_analysis, group_for_comparison,save_fig = save_fig, display=False,figsize = (20,15))
    stat_n_cells_from_group_comparaison(roi_analysis, group_for_comparison = group_for_comparison , save_fig = save_fig,display=False, display_one_row = False)
    idea_visu_to_explore(roi_analysis)
    stat_size_cells_from_group_comparaison(roi_analysis, group_for_comparison = group_for_comparison , save_fig = save_fig, display=False)

def extract_stats_on_all_slides():
    stat_n_cells_slides(save_fig = True, display=True)
    stat_size_cells_slides(save_fig = True, display=True)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Code générique pour nouvelles features a visualiser  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def stat_line_colocalisation_VALEUR(roi_analysis,convert_to_microm=False, save_fig=False, display = False): 

    y_name = ["FEATURE","FEATURE_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    dataset_config.conversion_px_micro_meter = 0.45
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    if convert_to_microm : 
        df_group_of_comparaison["FEATURE"] = df_group_of_comparaison["FEATURE"]*dataset_config.conversion_px_micro_meter
        df_group_of_comparaison["FEATURE_proba"] = df_group_of_comparaison["FEATURE_proba"]*dataset_config.conversion_px_micro_meter
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},title="Analysis FEATUREt<br>",markers=True, height=1100, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_traces(line_color='purple')
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="State "+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>FEATURE <br> FEATURE details ",title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )

    newnames = {'FEATURE':'Decision', 'FEATURE_proba': 'Proba'}
    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "FEATURE (µm)",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 4
        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     
        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"
        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"
    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "lines_FEATURE_proba"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()

def stat_line_FEATURE(roi_analysis,convert_to_microm=False, save_fig=False, display = False): 

    y_name = ["coloc_Association_Index_ASI","coloc_Association_Index_ASI_proba"]
    y_name = [ find_feature_name_in_physiological_df(feature) for feature in y_name]
    # y_name = ["coloc_Association_Index_ASI","coloc_Association_Index_ASI_proba","coloc_Accumulation_Index_ACI","coloc_Accumulation_Index_ACI_proba"]

    dataset_config.conversion_px_micro_meter = 0.45
    df_group_of_comparaison = roi_analysis.query("group_for_comparison == @group_for_comparison")
    if convert_to_microm : 
        df_group_of_comparaison["FEATURE"] = df_group_of_comparaison["FEATURE"]*dataset_config.conversion_px_micro_meter
        df_group_of_comparaison["FEATURE_proba"] = df_group_of_comparaison["FEATURE_proba"]*dataset_config.conversion_px_micro_meter
    fig_stat = px.line(df_group_of_comparaison, x = list(df_group_of_comparaison[x_axis_column].apply(lambda x: "ROI "+str(x) if x_axis_column == "id_in_group" else str(int(x))+" pcw")), y=y_name, facet_row="pop_A",facet_col="pop_B",facet_row_spacing=0.05, facet_col_spacing=0.05, category_orders={"pop_A": list(roi_analysis["pop_A"].unique()),
                              "pop_B": list(roi_analysis["pop_A"].unique())},title="Analysis FEATUREt<br>",markers=True, height=1100, width= 1400)
                              
    # fig_stat.update_yaxes(matches=None,showticklabels=True)
    fig_stat.update_traces(line_color='purple')
    fig_stat.update_xaxes(matches=None,showticklabels=True)

    fig_stat.for_each_annotation(lambda a: a.update(text="S"+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))
    fig_stat.update_annotations(font=dict(size=16))

    fig_stat.update_layout(title_text="<b>Colocalisation : Association Index (ASI) <br>State B (SB) is in State A (SA) levelsets</b>",title_x=0.5, title_font=dict(size=TITLE_SIZE),
                        margin=dict(t=200),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        )
    newnames = {'coloc_Association_Index_ASI':'ASI decision', 'coloc_Association_Index_ASI_proba': 'ASI proba'}
    # newnames = {'coloc_Association_Index_ASI':'ASI decision', 'coloc_Association_Index_ASI_proba': 'ASI proba',"coloc_Accumulation_Index_ACI":"decision","coloc_Accumulation_Index_ACI_proba":"ASI proba"}

    fig_stat.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig_stat.update_yaxes(side="left",title_text = "ASI",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    for i in range(20):
        fig_stat.data[i].line.width = 1
        fig_stat.data[i+20].line.width = 1

        if i//4 ==0: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[1]
            fig_stat.data[20+i].line.dash = "dot"


        if i//4 ==1: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[2]
            fig_stat.data[20+i].line.dash = "dot"     

        if i//4 ==2: 
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[3]
            fig_stat.data[20+i].line.dash = "dot"

        if i//4 ==3:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[4]
            fig_stat.data[20+i].line.dash = "dot"

        if i//4 ==4:
            fig_stat.data[i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.color = COLORS_STATES_PLOTLY[5]
            fig_stat.data[20+i].line.dash = "dot"

    
    if save_fig : 
        path_temporal_analysis_folder = os.path.join(dataset_config.dir_base_roi,"Temporal_analysis")
        path = os.path.join(path_temporal_analysis_folder,TABLE_GROUP_FOR_COMPARAISON_PHYSIOLOGICAL_PART[str(group_for_comparison)]) 
        pathfile = os.path.join(path, "lines_ASI"+AGGREGATION_METHOD_USE+"_size_max_square_"+str(int(roi_analysis["size_max_square"].iloc[0]))+".png")
        fig_stat.write_image(pathfile)
    if display : 
        fig_stat.show()



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Idea type viusalisation to explore """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def idea_visu_to_explore(roi_analysis):
    liste_nb_cell_roi = ["n_cells_roi",'n_Proliferative_roi', 'n_Amoeboid_roi', 'n_Cluster_roi', 'n_Phagocytic_roi', 'n_Ramified_roi']
    liste_nb_cell_roi_proba = ["n_cells_roi",'n_Proliferative_proba_roi', 'n_Amoeboid_proba_roi', 'n_Cluster_proba_roi', 'n_Phagocytic_proba_roi', 'n_Ramified_proba_roi']


    colors = ['rgb(0,0,0)','rgb(173,40,125)','rgb(79,157,213)','rgb(58,180,15)','rgb(240,186,64)','rgb(35,32,179)' ]
    colors_with_transparency = ['rgba(0,0,0,0.5)','rgba(173,40,125,0.5)','rgba(79,157,213,0.5)','rgba(58,180,15,0.5)','rgba(240,186,64,0.5)','rgba(35,32,179,0.5)']
    labels = ["All cells","Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    groupes = ["ROI1", "ROI2","ROI3", "ROI4"]
    x = labels
    fig = go.Figure()
    nb_row = 0
    for idx_color, row in df_without_AB_comparaison(roi_analysis).iterrows():
        fig.add_trace(po.Bar(
        name = 'ROI :'+str(row["id_in_group"]),
        x = x,
        y = row[liste_nb_cell_roi],
        text= row[liste_nb_cell_roi].apply(lambda x: "G"+str(nb_row+1)+" " +str(x)),
        texttemplate="G"+str(nb_row+1)+" <br>%{y}",
        marker_color = colors
    ))
        
        fig.add_trace(po.Bar(
        name = 'ROI :'+str(row["id_in_group"]) + "- proba",
        x = x,
        y = row[liste_nb_cell_roi_proba],
        text= row[liste_nb_cell_roi_proba].apply(lambda x: np.round(x,1)),
        marker_color = colors_with_transparency
        ))
        nb_row+=1
    fig.update_traces( textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="Number of cells in different ROIs",
        xaxis_title="Microglial states",
        yaxis_title="Cell number"
    )
    fig.update_layout(showlegend=False)
    fig.show()


def display_ALL_A_results(df_temporal_analysis,metric_name,proba,with_associated_metric, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ 
    display_4_regions_alltime_A_paired_results
    
    Permet de visualiser metric_name dans chaque regions (1 subplots/region) pour chaque temps (4 subplots) pour les populations pop_A et pop_B
    2 y_axis acceptées

    dbscan_fraction_clusterised_A
    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]

    metric_code = dict_param_metric_visualisation[metric_name]["proba_metric_name"] if proba else metric_name
    proba_title_txt = " (proba)" if proba else ""
    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"]
    title_figure = title_figure.replace("<A>","A") + proba_title_txt
    
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"].replace("<A>","A")
    range_y = dict_param_metric_visualisation[metric_name]["range_y"]
    #If double y axis
    if with_associated_metric and "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["proba_associated_metric"] if proba else dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [ find_feature_name_in_physiological_df(associated_metric)]
        y2_axis_name = dict_param_metric_visualisation[metric_name]["y2_axis_name"].replace("<A>","A")
        range_y2 = dict_param_metric_visualisation[metric_name]["range_y2"]
        double = True
    else :
        double = False
    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name = find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe

    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter

    # df_temporal_analysis_pop_A_pop_B = df_temporal_analysis[df_temporal_analysis["pop_A"] == pop_A]
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.12,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])

    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=0.5)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    fig.update_layout(plot_bgcolor = 'white',paper_bgcolor = "white",)
    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        df_temporal_analysis_physiological_part = df_temporal_analysis[df_temporal_analysis["physiological_part"] == physiological_part]
        for pop_A in labels:
            pop_B = "Amoeboid" if pop_A != "Amoeboid" else "Ramified"
            df_temporal_analysis_pop_A_pop_B = df_temporal_analysis_physiological_part[(df_temporal_analysis_physiological_part["pop_A"] == pop_A) & (df_temporal_analysis_physiological_part["pop_B"] == pop_B)]
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B, x=df_temporal_analysis_pop_A_pop_B["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"),y=y_name,color="pop_A",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(px.scatter(df_temporal_analysis_pop_A_pop_B, x=df_temporal_analysis_pop_A_pop_B["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_A",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None
        i+=1
    
    #Custom y axis 
    # fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True,rangemode="tozero")
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    #Custom x axis 
    # fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)[pop_A]) 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))

    #Custom y2 axis if exists 
    fig.update_yaxes(side="right",title_text = "<b>"+y2_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True, secondary_y=True)  if double else None

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))

    #Custom lines
    fig.update_traces(textposition=['top right', 'top center', 'top left'],textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_A])

    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'markers+lines+text':
            trace.mode = 'text+lines'
        if legend_showed==0:
            trace.name = y_axis_name
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                trace.name = y2_axis_name
            else :
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    fig.update_traces(line=dict(width=SIZE_LINE)) #Size line (16 for fig 6)
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    
    #Displaying and saving 
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    
    if dict_param_visualisation_saving["save_fig"]:
        txt_with_associated_metric = "_wt_associated_metric" if with_associated_metric else ""
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        text_proba = "proba" if proba else ""
        path_fig = os.path.join(path_results_pop_A_metric_folder,dict_param_metric_visualisation[metric_name]["filename"]+text_proba+txt_with_associated_metric+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None 
    return fig


def display_A_results_b_coloc(df_temporal_analysis,metric_name,proba,with_associated_metric, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ 
    Particulière pour gérer l'inversion pop_B -> pop_A
    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 

    metric_code = dict_param_metric_visualisation[metric_name]["proba_metric_name"] if proba else metric_name
    proba_title_txt = " (proba)" if proba else ""
    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"] +proba_title_txt 
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"]
    legend_name_y = dict_param_metric_visualisation[metric_name]["legend_name_y"]
    a_bellow = dict_param_metric_visualisation[metric_name]["a_bellow"]
    State_A_result = dict_param_metric_visualisation[metric_name]["State_A_result"]
    State_A_result_pop_of_interest = dict_param_metric_visualisation[metric_name]["State_A_result_pop_of_interest"]
    range_y = dict_param_metric_visualisation[metric_name]["range_y"]
    #If double y axis
    triple = False
    if with_associated_metric and "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [  find_feature_name_in_physiological_df(associated_metric)]
        legend_name_y2 = dict_param_metric_visualisation[metric_name]["legend_name_y2"]
        range_y2 = dict_param_metric_visualisation[metric_name]["range_y2"]
        double = True
        if  "associated_metric_2" in list(dict_param_metric_visualisation[metric_name].keys()) :
            associated_metric_2 = dict_param_metric_visualisation[metric_name]["associated_metric_2"]
            y_name_2 = y_name_2 + [ find_feature_name_in_physiological_df(associated_metric_2)]
            legend_name_y3 = dict_param_metric_visualisation[metric_name]["legend_name_y3"]
            triple = True
        else :
            triple = False
    else :
        double = False

    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name =  find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en micro meter")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter
    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    df_pop_A = dict()
    for pop_i in labels:
        other_pop = "pop_A" if State_A_result_pop_of_interest == "pop_B" else "pop_B"
        pop_j = "Amoeboid" if pop_i != "Amoeboid" else "Ramified"
        df_pop_A[pop_i] = df_temporal_analysis[(df_temporal_analysis[State_A_result_pop_of_interest] == pop_i) & (df_temporal_analysis[other_pop] == pop_j)]
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=0.5)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    # fig.update_layout(plot_bgcolor = 'grey',paper_bgcolor = "white",)

    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        for pop_i in labels:
            df_A_Physiopart = df_pop_A[pop_i][df_pop_A[pop_i]["physiological_part"] == physiological_part]
            fig.add_trace(px.line(df_A_Physiopart, x=df_A_Physiopart["pcw"].apply(lambda x:"<b>Border <br>"+str(int(x))+" pcw <b>"), y=y_name,color=State_A_result_pop_of_interest,color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(px.scatter(df_A_Physiopart, x=df_A_Physiopart["pcw"].apply(lambda x:"<b>Border <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None
        i+=1

    #Custom y axis 
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True,rangemode="tozero")
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    #Custom x axis 
    fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)["Both_bc_symetrie"]) 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))
    fig.update_layout(plot_bgcolor = 'white',paper_bgcolor = "white",)

    #Custom y2 axis if exists 
    fig.update_yaxes(side="right",title_text = "<b>"+legend_name_y2,title_standoff=0,tickfont=dict(size=15),showticklabels=True, secondary_y=True)  if double else None
    # if range_y == "0_1":
    #     fig.update_yaxes(range=[0,1]) 
    # if range_y == "range_all_by_max" : 
    #     max = 0
    #     for t in fig.data:
    #         if t.y.max() > max : 
    #             max = t.y.max()
    #     fig.update_yaxes(range=[0,max]) 

    # if double: 
    #     print("On rentre ds double")
    #     if range_y2 == "0_1":
    #         print("Okferfrefkk")
    #         fig.update_yaxes(secondary_y=True,range=[0,1]) 
    #     elif range_y2 == "range_all_by_max" : 
    #         max = 0
    #         for t in fig.data:
    #             if t.y.max() > max : 
    #                 max = t.y.max()
    #         fig.update_yaxes(range=[0,max]) 
    #     else:
    #         idx_y = 2
    #         for idx_row in range(2):
    #             for idx_col in range(2):
    #                 fig.update_yaxes(matches = "y"+str(idx_y),row=idx_row+1, col=idx_col+1) if double else None
    #                 idx_y +=2

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))
    #Custom lines
    # fig.update_traces(textposition='top center',textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_B],)
    #Remove marker of px.line 
    #Legend
    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'markers+lines+text':
            trace.mode = 'text+lines'
        if legend_showed==0:
            trace.name = legend_name_y.replace("<A>",State_A_result_pop_of_interest)
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                trace.name = legend_name_y2.replace("<A>",State_A_result_pop_of_interest)
            else :
                trace.showlegend = False
            legend_showed+=1
        elif legend_showed==2:
            if triple:
                trace.name = legend_name_y3.replace("<A>",State_A_result_pop_of_interest)
            else : 
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(legend = dict(font = dict(size = 20)),legend_title = dict(font = dict(family = "Courier", size = 30, color = "blue")))
    fig.update_traces(line=dict(width=SIZE_LINE))
    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    #Displaying and saving
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    if dict_param_visualisation_saving["save_fig"]:

        txt_with_associated_metric = "_wt_associated_metric" if with_associated_metric else ""
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        text_proba = "_proba_" if proba else ""
        path_fig = os.path.join(path_results_pop_A_metric_folder,dict_param_metric_visualisation[metric_name]["filename"]+"_A_"+State_A_result_pop_of_interest+"_B_"+txt_with_associated_metric+text_proba+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None
    return fig


def display_1_A_ALL_B(df_temporal_analysis,pop_A,metric_name,proba,with_associated_metric, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ Permet de visualiser metric_name dans chaque regions (1 subplots/region) pour chaque temps (4 subplots) pour les populations pop_A et pop_B


    neighbours_rate_B_first_neighbour
    neighbours_rate_balls_r_coloc_containing_1_B

    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 
    with_y2_as_txt = False
    metric_code = dict_param_metric_visualisation[metric_name]["proba_metric_name"] if proba else metric_name
    proba_title_txt = " (proba)" if proba else ""
    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"].replace("<A>",pop_A) +proba_title_txt
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"].replace("<A>",pop_A) 
    range_y = dict_param_metric_visualisation[metric_name]["range_y"]
    #If double y axis
    if with_associated_metric and "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["proba_associated_metric"] if proba else dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [  find_feature_name_in_physiological_df(associated_metric)]
        y2_axis_name = dict_param_metric_visualisation[metric_name]["y2_axis_name"]
        range_y2 = dict_param_metric_visualisation[metric_name]["range_y2"]
        double = True
    else :
        double = False

    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name =  find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter
    df_temporal_analysis_pop_A = df_temporal_analysis[df_temporal_analysis["pop_A"] == pop_A]
    
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=0.5)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    # fig.update_layout(plot_bgcolor = 'grey',paper_bgcolor = "white",)

    labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        df_temporal_analysis_pop_A_pop_B_physiological_part = df_temporal_analysis_pop_A[df_temporal_analysis_pop_A["physiological_part"] == physiological_part]
        for pop_B in labels:
            # print("Pop B : "+pop_B)
            if pop_B != pop_A :

                df_B_PP = df_temporal_analysis_pop_A_pop_B_physiological_part[df_temporal_analysis_pop_A_pop_B_physiological_part["pop_B"] == pop_B]
                # fig.add_trace(px.line(df_B_PP, x=df_B_PP["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), text = df_B_PP["pop_B"].apply(lambda x:"<b>"+x),y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
                #Without text
                if not with_y2_as_txt:
                    fig.add_trace(px.line(df_B_PP, x=df_B_PP["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"),y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
                    fig.add_trace(px.scatter(df_B_PP, x=df_B_PP["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None

                #With text
                else: 
                    fig.add_trace(px.line(df_B_PP, x=df_B_PP["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), text = df_B_PP[y_name_2[0]].apply(lambda x:np.round(x,2)),y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
        i+=1
        
    #Custom y axis 
        print("custom y")
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=40),showticklabels=True)#,rangemode="tozero")
    #Custom x axis 
    fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)[pop_A]) 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))

    #Custom y2 axis if exists 
    fig.update_yaxes(side="right",title_text = "<b>"+y2_axis_name,title_standoff=0,tickfont=dict(size=20),showticklabels=True, secondary_y=True)  if double else None

    # #range y axis
    # if range_y == "0_1":
    #     fig.update_yaxes(range=[0,1])
    # if range_y == "range_all_by_max" :
    #     max = 0
    #     for t in fig.data:
    #         if t.y.max() > max : 
    #             max = t.y.max()
    #     fig.update_yaxes(range=[0,max])
    # if double: 
    #     if range_y2 == "0_1":
    #         fig.update_yaxes(secondary_y=True,range=[0,1]) 
    #     elif range_y2 == "range_all_by_max" : 
    #         max = 0
    #         for t in fig.data:
    #             if t.y.max() > max : 
    #                 max = t.y.max()
    #         fig.update_yaxes(secondary_y=True,range=[0,max]) 
    #     else:
    #         idx_y = 2
    #         for idx_row in range(2):
    #             for idx_col in range(2):
    #                 fig.update_yaxes(matches = "y"+str(idx_y),row=idx_row+1, col=idx_col+1) if double else None
    #                 idx_y +=2

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))
    fig.update_layout(plot_bgcolor = 'white',paper_bgcolor = "white",)

    #Custom lines
    # liste_ = [adaptative_color_classification(coef_alpha=1)[pop_i] for pop_i in labels if pop_i != pop_A]
    fig.update_traces(textposition=['bottom right', 'bottom center', 'bottom left'],textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)["Both_bc_symetrie"])
    #Remove marker of px.line 
    #Legend
    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'markers+lines+text':
            trace.mode = 'text+lines'
        if legend_showed==0:
            trace.name = y_axis_name
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                trace.name = y2_axis_name 
                if with_y2_as_txt : 
                    trace.mode = 'text' 
                    # trace.showlegend = False
            else :
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    fig.update_traces(line=dict(width=SIZE_LINE))
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    
    #Displaying and saving 
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    
    if dict_param_visualisation_saving["save_fig"]:
        path_results_pop_A = os.path.join(dataset_config.dir_base_stat_analysis_PER_CELL_STATE,"A_"+pop_A)
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        text_proba = "_proba_" if proba else ""
        path_fig = os.path.join(path_results_pop_A_metric_folder,dict_param_metric_visualisation[metric_name]["filename"]+"_A_"+pop_A+text_proba+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None 
    return fig

def display_1_A_1_B(df_temporal_analysis,pop_A,pop_B,metric_name,proba, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ 
    1 subplot = 1 region 
    1 pop_A _ 1 pop_B 

    Metrics : 

    #2 A-B coloc 
    - coloc_p_value
    - coloc_Association_Index_ASI 
    -coloc_delta_a"

    neighbours_rate_B_first_neighbour
    neighbours_rate_balls_r_coloc_containing_1_B

    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 

    metric_code = dict_param_metric_visualisation[metric_name]["proba_metric_name"] if proba else metric_name
    proba_title_txt = " (proba)" if proba else ""
    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"].replace("<A>",pop_A).replace("<B>", pop_B) +proba_title_txt
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"].replace("<A>",pop_A).replace("<B>", pop_B) 
    range_y = dict_param_metric_visualisation[metric_name]["range_y"]
    #If double y axis
    if "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["proba_associated_metric"] if proba else dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [  find_feature_name_in_physiological_df(associated_metric)]
        y2_axis_name = dict_param_metric_visualisation[metric_name]["y2_axis_name"]
        range_y2 = dict_param_metric_visualisation[metric_name]["range_y2"]
        double = True
    else :
        double = False

    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name =  find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en micro meter")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter
    df_temporal_analysis_pop_A_pop_B = df_temporal_analysis[(df_temporal_analysis["pop_A"] == pop_A) & (df_temporal_analysis["pop_B"] == pop_B)]
    
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    fig.update_traces(line=dict(width=SIZE_LINE))
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=100)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    # fig.update_layout(plot_bgcolor = 'grey',paper_bgcolor = "white",)


    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        df_temporal_analysis_pop_A_pop_B_physiological_part = df_temporal_analysis_pop_A_pop_B[df_temporal_analysis_pop_A_pop_B["physiological_part"] == physiological_part]
        fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), text = ["","<b>"+pop_B,""],y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
        fig.add_trace(px.scatter(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None
        i+=1
    
    #Custom y axis 
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True,rangemode="tozero")
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    #Custom x axis 
    fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)[pop_A]) 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))

    #Custom y2 axis if exists 
    fig.update_yaxes(side="right",title_text = "<b>"+y2_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True, secondary_y=True)  if double else None
    fig.update_layout(plot_bgcolor = 'white',paper_bgcolor = "white",)

    #range y axis
    # if range_y == "0_1":
    #     fig.update_yaxes(range=[0,1])
    # if range_y == "range_all_by_max" :
    #     max = 0
    #     for t in fig.data:
    #         if t.y.max() > max : 
    #             max = t.y.max()
    #     fig.update_yaxes(range=[0,max])
    # if double: 
    #     if range_y2 == "0_1":
    #         fig.update_yaxes(secondary_y=True,range=[0,1]) 
    #     elif range_y2 == "range_all_by_max" : 
    #         max = 0
    #         for t in fig.data:
    #             if t.y.max() > max : 
    #                 max = t.y.max()
    #         print("Max value = ",max)
    #         fig.update_yaxes(range=[0,max]) 
    #     else:
    #         idx_y = 2
    #         for idx_row in range(2):
    #             for idx_col in range(2):
    #                 fig.update_yaxes(matches = "y"+str(idx_y),row=idx_row+1, col=idx_col+1) if double else None
    #                 idx_y +=2

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))

    #Custom lines
    fig.update_traces(textposition=['bottom right', 'bottom center', 'bottom left'],textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_B])
    #Remove marker of px.line 
    #Legend
    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'markers+lines+text':
            trace.mode = 'text+lines'
        if legend_showed==0:
            trace.name = y_axis_name
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                trace.name = y2_axis_name 
            else :
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    
    #Displaying and saving 
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    
    if dict_param_visualisation_saving["save_fig"]:
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        path_results_pop_A_metric_folder_1_A_1_B = os.path.join(path_results_pop_A_metric_folder,"1_A_1_B")
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)
        mkdir_if_nexist(path_results_pop_A_metric_folder_1_A_1_B)
        text_proba = "_proba_" if proba else ""
        path_fig = os.path.join(path_results_pop_A_metric_folder_1_A_1_B,dict_param_metric_visualisation[metric_name]["filename"]+"_A_"+pop_A+"_B_"+pop_B+text_proba+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None 
    return fig

def display_1_A_1_B_iou(df_temporal_analysis,pop_A,pop_B,metric_name, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ Permet de visualiser metric_name dans chaque regions (1 subplots/region) pour chaque temps (4 subplots) pour les populations pop_A et pop_B
    2 y_axis acceptées
    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 

    metric_code = metric_name

    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"] + pop_A+" vs "+pop_B
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"]
    legend_name_y = dict_param_metric_visualisation[metric_name]["legend_name_y"]
    a_bellow = dict_param_metric_visualisation[metric_name]["a_bellow"]
    #If double y axis
    if "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [  find_feature_name_in_physiological_df(associated_metric)]
        legend_name_y2 = dict_param_metric_visualisation[metric_name]["legend_name_y2"]
        range_y = dict_param_metric_visualisation[metric_name]["range_y"]
        double = True
        if  "associated_metric_2" in list(dict_param_metric_visualisation[metric_name].keys()) :
            associated_metric_2 = dict_param_metric_visualisation[metric_name]["associated_metric_2"]
            y_name_2 = y_name_2 + [  find_feature_name_in_physiological_df(associated_metric_2)]
            legend_name_y3 = dict_param_metric_visualisation[metric_name]["legend_name_y3"]
            triple = True
        else :
            triple = False
    else :
        double = False

    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name =  find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en micro meter")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter
    df_temporal_analysis_pop_A_pop_B = df_temporal_analysis[(df_temporal_analysis["pop_A"] == pop_A) & (df_temporal_analysis["pop_B"] == pop_B)]
    
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=0.5)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    # fig.update_layout(plot_bgcolor = 'grey',paper_bgcolor = "white",)


    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        df_temporal_analysis_pop_A_pop_B_physiological_part = df_temporal_analysis_pop_A_pop_B[df_temporal_analysis_pop_A_pop_B["physiological_part"] == physiological_part]

        if a_bellow : 
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name,color=df_temporal_analysis_pop_A_pop_B_physiological_part["pop_B"].apply(lambda x:"Both_bc_symetrie"),color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_A",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) 
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"),y=y_name_2[1],color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) 
        else : 
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name,color=df_temporal_analysis_pop_A_pop_B_physiological_part["pop_B"].apply(lambda x:"Both_bc_symetrie"),color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_A",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) if double else None
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"),y=y_name_2[1],color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) if triple else None 


        # fig.add_trace(px.scatter(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2,color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None
        # fig.add_trace(px.scatter(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None

        i+=1
    
    #Custom y axis 
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True,rangemode="tozero")
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    #Custom x axis 
    # fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)[pop_A]) 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))

    #Custom y2 axis if exists 
    # fig.update_yaxes(side="right",title_text = "<b>"+legend_name_y2,title_standoff=0,tickfont=dict(size=15),showticklabels=True, secondary_y=True)  if double else None
    fig.update_layout(plot_bgcolor = 'white',paper_bgcolor = "white",)

    # if double: 
    #     if range_y == "0_1":
    #         fig.update_yaxes(secondary_y=True,range=[0,1]) 
    #     elif range_y == "range_all_by_max" : 
    #         max = 0
    #         for t in fig.data:
    #             if t.y.max() > max : 
    #                 max = t.y.max()
    #         print("Max value = ",max)
    #         fig.update_yaxes(range=[0,max]) 
    #     else:
    #         idx_y = 2
    #         for idx_row in range(2):
    #             for idx_col in range(2):
    #                 fig.update_yaxes(matches = "y"+str(idx_y),row=idx_row+1, col=idx_col+1) if double else None
    #                 idx_y +=2

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))

    #Custom lines
    # fig.update_traces(textposition='top center',textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_B],)
    #Remove marker of px.line 
    #Legend
    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'markers+lines+text':
            trace.mode = 'text+lines'
        if legend_showed==0:
            trace.name = legend_name_y.replace("<A>",pop_A).replace("<B>",pop_B)
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                trace.name = legend_name_y2.replace("<A>",pop_A).replace("<B>",pop_B)
            else :
                trace.showlegend = False
            legend_showed+=1
        elif legend_showed==2:
            if triple:
                trace.name = legend_name_y3.replace("<A>",pop_A).replace("<B>",pop_B)
            else : 
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(legend = dict(font = dict(size = 20)),legend_title = dict(font = dict(family = "Courier", size = 30, color = "blue")))

    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    fig.update_traces(line=dict(width=SIZE_LINE))
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    
    #Displaying and saving 
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    
    if dict_param_visualisation_saving["save_fig"]:
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)

        path_fig = os.path.join(path_results_pop_A_metric_folder,dict_param_metric_visualisation[metric_name]["filename"]+"_A_"+pop_A+"_B_"+pop_B+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None 
    return fig

def display_1_A_1_B_fract_B_in_A_clusters(df_temporal_analysis,pop_A,pop_B,metric_name,proba, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ Permet de visualiser metric_name dans chaque regions (1 subplots/region) pour chaque temps (4 subplots) pour les populations pop_A et pop_B
    2 y_axis acceptées

    dbscan_fraction_B_in_A_clusters
    dbscan_fraction_clustered_B_in_intersect
    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 
    triple = False
    metric_code = dict_param_metric_visualisation[metric_name]["proba_metric_name"] if proba else metric_name
    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"].replace("<A>",pop_A).replace("<B>",pop_B)
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"]
    legend_name_y = dict_param_metric_visualisation[metric_name]["legend_name_y"]
    a_bellow = dict_param_metric_visualisation[metric_name]["a_bellow"]
    #If double y axis
    if "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [  find_feature_name_in_physiological_df(associated_metric)]
        legend_name_y2 = dict_param_metric_visualisation[metric_name]["legend_name_y2"]
        range_y = dict_param_metric_visualisation[metric_name]["range_y"]
        double = True
        if  "associated_metric_2" in list(dict_param_metric_visualisation[metric_name].keys()) :
            associated_metric_2 = dict_param_metric_visualisation[metric_name]["associated_metric_2"]
            y_name_2 = y_name_2 + [  find_feature_name_in_physiological_df(associated_metric_2)]
            legend_name_y3 = dict_param_metric_visualisation[metric_name]["legend_name_y3"]
            triple = True
        else :
            triple = False
    else :
        double = False

    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name =  find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en micro meter")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter
    df_temporal_analysis_pop_A_pop_B = df_temporal_analysis[(df_temporal_analysis["pop_A"] == pop_A) & (df_temporal_analysis["pop_B"] == pop_B)]
    
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=0.5)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    # fig.update_layout(plot_bgcolor = 'grey',paper_bgcolor = "white",)


    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        df_temporal_analysis_pop_A_pop_B_physiological_part = df_temporal_analysis_pop_A_pop_B[df_temporal_analysis_pop_A_pop_B["physiological_part"] == physiological_part]
        if a_bellow:
                
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
            # fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)

            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_A",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) if double else None


            # fig.add_trace(px.scatter(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2,color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None
            # fig.add_trace(px.scatter(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_B",symbol = "physiological_part",symbol_map = symbol_map,color_discrete_map = adaptative_color_classification(coef_alpha=1)).data[0],row=(i-1)//2+1, col=(i-1)%2+1,secondary_y=True) if double else None
        else :
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name,color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+str(int(x))+" pcw <b>"), y=y_name_2[0],color="pop_A",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) if double else None

        i+=1
    
    #Custom y axis 
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True,rangemode="tozero")
    #Custom x axis 
    fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)[pop_A]) if a_bellow else None 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))

    #Custom y2 axis if exists 
    # fig.update_yaxes(side="right",title_text = "<b>"+legend_name_y2,title_standoff=0,tickfont=dict(size=15),showticklabels=True, secondary_y=True)  if double else None

    if double: 
        if range_y == "0_1":
            fig.update_yaxes(secondary_y=True,range=[0,1]) 
        elif range_y == "range_all_by_max" : 
            max = 0
            for t in fig.data:
                if t.y.max() > max : 
                    max = t.y.max()
            fig.update_yaxes(range=[0,max]) 
        else:
            idx_y = 2
            for idx_row in range(2):
                for idx_col in range(2):
                    fig.update_yaxes(matches = "y"+str(idx_y),row=idx_row+1, col=idx_col+1) if double else None
                    idx_y +=2

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))

    #Custom lines
    # fig.update_traces(textposition='top center',textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_B],)
    #Remove marker of px.line 
    #Legend
    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'markers+lines+text':
            trace.mode = 'text+lines'
        if legend_showed==0:
            # trace.name = pop_B + "  &#8712;<br>" + pop_A + "<br>Clusters"
            trace.name = legend_name_y.replace("<A>",pop_A).replace("<B>",pop_B)
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                # trace.name = pop_A + " &#8712;<br>" + pop_B + "<br>Clusters"
                trace.name = legend_name_y2.replace("<A>",pop_A).replace("<B>",pop_B)
            else :
                trace.showlegend = False
            legend_showed+=1
        elif legend_showed==2:
            if triple:
                trace.name = legend_name_y3+pop_B+" area"
            else : 
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(legend = dict(font = dict(size = 20)),legend_title = dict(font = dict(family = "Courier", size = 30, color = "blue")))

    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    fig.update_traces(line=dict(width=SIZE_LINE))
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    
    #Displaying and saving 
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    
    if dict_param_visualisation_saving["save_fig"]:
        text_proba = "_proba_" if proba else ""
        path_results_pop_A = os.path.join(dataset_config.dir_base_stat_analysis_PER_CELL_STATE,"A_"+pop_A)
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        path_results_pop_A_metric_folder_1_A_1_B = os.path.join(path_results_pop_A_metric_folder,"1_A_1_B")
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)
        mkdir_if_nexist(path_results_pop_A_metric_folder_1_A_1_B)
        text_proba = "_proba_" if proba else ""
        path_fig = os.path.join(path_results_pop_A_metric_folder_1_A_1_B,dict_param_metric_visualisation[metric_name]["filename"]+"_A_"+pop_A+"_B_"+pop_B+text_proba+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None 
    return fig

def display_1_A_1_B_3_first_B_around_A(df_temporal_analysis,pop_A,pop_B,metric_name,proba, dict_param_metric_visualisation, dict_param_visualisation_saving):
    """ Permet de visualiser metric_name dans chaque regions (1 subplots/region) pour chaque temps (4 subplots) pour les populations pop_A et pop_B
    2 y_axis acceptées
    """
    # print("Pop A : "+pop_A,"VS Pop B : "+pop_B)
    #Recuperation parameters 
    metric_code = dict_param_metric_visualisation[metric_name]["proba_metric_name"] if proba else metric_name
    proba_title_txt = " (proba)" if proba else ""
    title_figure = dict_param_metric_visualisation[metric_name]["title_figure"] +proba_title_txt +  " A: "+pop_A+" vs B: "+pop_B
    y_axis_name = dict_param_metric_visualisation[metric_name]["y_axis_name"]
    legend_name_y = dict_param_metric_visualisation[metric_name]["legend_name_y"]
    a_bellow = dict_param_metric_visualisation[metric_name]["a_bellow"]
    #If double y axis
    if "associated_metric" in list(dict_param_metric_visualisation[metric_name].keys()) :
        associated_metric = dict_param_metric_visualisation[metric_name]["associated_metric"]
        y_name_2 = [  find_feature_name_in_physiological_df(associated_metric)]
        legend_name_y2 = dict_param_metric_visualisation[metric_name]["legend_name_y2"]
        range_y = dict_param_metric_visualisation[metric_name]["range_y"]
        double = True
        if  "associated_metric_2" in list(dict_param_metric_visualisation[metric_name].keys()) :
            associated_metric_2 = dict_param_metric_visualisation[metric_name]["associated_metric_2"]
            y_name_2 = y_name_2 + [  find_feature_name_in_physiological_df(associated_metric_2)]
            legend_name_y3 = dict_param_metric_visualisation[metric_name]["legend_name_y3"]
            triple = True
    else :
        double = False

    #Recuperation of the dataframe
    liste_physiological_part  =["striatum","ganglionic_eminence","cortical_boundary","neocortex"] 
    y_name =  find_feature_name_in_physiological_df(metric_code) #Find the name of the feature in the dataframe
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en micro meter")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]*dataset_config.conversion_px_micro_meter
    df_temporal_analysis_pop_A_pop_B = df_temporal_analysis[(df_temporal_analysis["pop_A"] == pop_A) & (df_temporal_analysis["pop_B"] == pop_B)]
    fig = make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[c for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    # fig = make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  "  for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.05) if not double else make_subplots(2, 2,subplot_titles=[TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[c]+" <br>  " for c in liste_physiological_part],vertical_spacing = 0.15,horizontal_spacing = 0.09,specs=[[{"secondary_y": True} for i in range(2)] for i in range(2)])
    fig.update_layout(title_text=title_figure,title_x=0.5, title_font=dict(size=TITLE_SIZE),
                    showlegend=True,
                    width=WIDTH_FIG,
                    height=HEIGHT_FIG,
                    margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                    )
    for i in fig['layout']['annotations']: #Change subtitle 
        physiological_part = i["text"]
        i["text"] = TABLE_PHYSIOLOGICAL_PART_FILENAMES_NAME[physiological_part]
        # i["bgcolor"] = adaptative_color_tissue_segmentation(coef_alpha=0.5)[physiological_part]
        i['font']["size"] = 20
    # fig.update_layout(plot_bgcolor = adaptative_color_tissue_segmentation(coef_alpha=0.5)["striatum"],paper_bgcolor = "white",)
    # fig.update_layout(plot_bgcolor = 'grey',paper_bgcolor = "white",)


    #Courbes dans les subplots
    i = 1
    for physiological_part in liste_physiological_part:
        df_temporal_analysis_pop_A_pop_B_physiological_part = df_temporal_analysis_pop_A_pop_B[df_temporal_analysis_pop_A_pop_B["physiological_part"] == physiological_part]


        fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name, text = ["","<b>1st "+pop_B,""],color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=1),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1)
        fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"), y=y_name_2[0], text = ["","<b>2nd "+pop_B,""],color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=0.7),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) 
        fig.add_trace(px.line(df_temporal_analysis_pop_A_pop_B_physiological_part, x=df_temporal_analysis_pop_A_pop_B_physiological_part["pcw"].apply(lambda x:"<b>"+pop_A+" <br>"+str(int(x))+" pcw <b>"),y=y_name_2[1], text = ["","<b>3rd "+pop_B,""],color="pop_B",color_discrete_map = adaptative_color_classification(coef_alpha=0.5),markers=False).data[0],row=(i-1)//2+1, col=(i-1)%2+1) 

        i+=1
    
    #Custom lines
    fig.update_traces(textposition=['top right', 'top center', 'top left'],textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_B])
    #Custom y axis 
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True,rangemode="tozero")
    fig.update_yaxes(side="left",title_text = "<b>"+y_axis_name,title_standoff=0,tickfont=dict(size=15),showticklabels=True)
    #Custom x axis 
    fig.update_xaxes(color=adaptative_color_classification(coef_alpha=1)[pop_A]) 
    # fig.for_each_xaxis(lambda x: x.update(showgrid=False))
    fig.update_layout(plot_bgcolor = 'white',paper_bgcolor = "white",)

    #Custom y2 axis if exists 
    # fig.update_yaxes(side="right",title_text = "<b>"+legend_name_y2,title_standoff=0,tickfont=dict(size=15),showticklabels=True, secondary_y=True)  if double else None
    # if double: 
    #     if range_y == "0_1":
    #         fig.update_yaxes(secondary_y=True,range=[0,1]) 
    #     elif range_y == "range_all_by_max" : 
    #         max = 0
    #         for t in fig.data:
    #             if t.y.max() > max : 
    #                 max = t.y.max()
    #         print("Max value = ",max)
    #         fig.update_yaxes(range=[0,max]) 
    #     else:
    #         idx_y = 2
    #         for idx_row in range(2):
    #             for idx_col in range(2):
    #                 fig.update_yaxes(matches = "y"+str(idx_y),row=idx_row+1, col=idx_col+1) if double else None
    #                 idx_y +=2

    fig.for_each_yaxis(lambda x: x.update(showgrid=False))
    #Custom lines
    # fig.update_traces(textposition='top center',textfont_size=15, textfont_color=adaptative_color_classification(coef_alpha=1)[pop_B],)
    #Remove marker of px.line 
    #Legend
    legend_showed = 0 
    for trace in fig.data:
        if trace.mode == 'text+lines+markers':
            trace.mode = 'text+lines'
        if legend_showed==0:
            trace.name = legend_name_y+"<br>"+pop_B 
            legend_showed+=1
        elif legend_showed==1:
            if double : 
                trace.name = legend_name_y2 +"<br>"+pop_B 
            else :
                trace.showlegend = False
            legend_showed+=1
        elif legend_showed==2:
            if triple:
                trace.name = legend_name_y3+"<br>"+pop_B
            else : 
                trace.showlegend = False
            legend_showed+=1
        else:
            trace.showlegend = False
    fig.update_layout(legend = dict(font = dict(size = 20)),legend_title = dict(font = dict(family = "Courier", size = 30, color = "blue")))
    fig.update_layout(legend_traceorder="reversed")
    # fig.update_layout(legend = dict(font = dict(size = 20)),legend_title="Legend",legend_title_font_size=20)
    fig.update_layout(showlegend=False) if not double else None
    fig.update_traces(marker={'size': 15}) #Change size of marker
    fig.update_traces(line=dict(width=SIZE_LINE))
    if metric_code in LIST_METRIC_TO_CONVERT_IN_MICROMETER:
        # print("Conversion en pixels")
        df_temporal_analysis[y_name] = df_temporal_analysis[y_name]/dataset_config.conversion_px_micro_meter
    #Displaying and saving 
    fig.show() if dict_param_visualisation_saving["display_fig"] else None
    if dict_param_visualisation_saving["save_fig"]:
        # path_results_pop_A = os.path.join(dataset_config.dir_base_stat_analysis_PER_CELL_STATE,"A_"+pop_A)
        path_results_pop_A_metric_subgroup = os.path.join(dataset_config.dir_base_stat_analysis,dict_param_metric_visualisation[metric_name]["metric_subgroup_folder_name"])
        path_results_pop_A_metric_folder = os.path.join(path_results_pop_A_metric_subgroup,dict_param_metric_visualisation[metric_name]["metric_folder_name"])
        path_results_pop_A_metric_folder_1_A_1_B = os.path.join(path_results_pop_A_metric_folder,"1_A_1_B")

        # mkdir_if_nexist(path_results_pop_A)
        mkdir_if_nexist(path_results_pop_A_metric_subgroup)
        mkdir_if_nexist(path_results_pop_A_metric_folder)
        text_proba = "_proba_" if proba else ""
        mkdir_if_nexist(path_results_pop_A_metric_folder_1_A_1_B)
        path_fig = os.path.join(path_results_pop_A_metric_folder_1_A_1_B,dict_param_metric_visualisation[metric_name]["filename"]+"A_"+pop_A+"_B_"+pop_B+text_proba+".png")
        fig.write_image(path_fig)
        print(green("Figure saved : "+path_fig,"bold")) if dict_param_visualisation_saving["verbose"] else None 
    return fig




################################## Developped with Covid Dataset 



def plot_cell_proportion(dataset_config, results_several_images, metric_config, display = False):
    """ 
    Used in context of : 
    - Covid data 
    - Stats on entire images
    Possible args for 
    dict : one of [dict_n_cells_per_image, ] 
    """
    #Df pre processing 
    mc=metric_config
    # y = ["n_cells_{}_roi".format(name) for name in dataset_config.cell_class_names]

    for group in list(results_several_images["group"].unique()):

        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        
        sub_df = df_filtered_by_group[mc["y"]+["slide_num"]]
        unique_values_df = sub_df.drop_duplicates(subset=['slide_num'])

        #Fig creation 
        fig = px.bar(unique_values_df, x = list(unique_values_df[mc["col_x_label"]].apply(lambda x: "Img "+str(x))), y=mc["y"],text_auto=mc["text_auto"])

        #Fig layout
        for name in dataset_config.cell_class_names:
            fig.data[0].marker.color = adaptative_color_lines_transparency(coef_alpha=
                                                                        mc["coef_alpha"])[name]


        newnames = dict(zip(mc["y"], dataset_config.cell_class_names))
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))

        fig.update_layout(title_text=mc["title"],
                        title_x=mc["title_x"], 
                        title_font=mc["title_font"],
                        margin=mc["margin"],
                        legend_title=mc["legend_title"],)
        fig.update_yaxes(side=mc["side"],title_text = mc["y_title"],title_standoff=mc["title_standoff"],tickfont=mc["tickfont"],showticklabels=mc["showticklabels"])
        fig.update_xaxes(title_text = "")

        fig.show() if display else None 

        path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,mc["metric_folder_name"])
        mkdir_if_nexist(path_folder_fig)
        fig.write_image(os.path.join(path_folder_fig,group+"_"+mc["figname"]))
        # fig.write_image("figures/{}_n_cells_per_image.png".format(dataset_config.dataset_name),width=1000, height=500, scale=1)



def n_cells_normalised_by_tissue_area(dataset_config, results_several_images, metric_config, display = False):
    """ 
    Used in context of : 
    - Covid data 
    - Stats on entire images
    Possible args for 
    dict : one of [dict_n_cells_per_image, ] 
    """
    #Df pre processing 
    mc=metric_config
    for group in list(results_several_images["group"].unique()):
        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        # y = ["n_cells_{}_roi".format(name) for name in dataset_config.cell_class_names]
        sub_df = df_filtered_by_group[mc["y"]+["slide_num","area_tissue_slide"]]
        unique_values_df = sub_df.drop_duplicates(subset=['slide_num'])

        list_new_y = []
        for celltype_idx, celltype_n_cells_colname in enumerate(mc["y"]):
            unique_values_df["n_cells_{}_roi_normalised_by_tissue_area".format(dataset_config.cell_class_names[celltype_idx])] = unique_values_df[celltype_n_cells_colname]/(unique_values_df["area_tissue_slide"]*dataset_config.conversion_px_micro_meter**2)
            list_new_y.append("n_cells_{}_roi_normalised_by_tissue_area".format(dataset_config.cell_class_names[celltype_idx]))
        #Fig creation 
        fig = px.bar(unique_values_df, x = list(unique_values_df[mc["col_x_label"]].apply(lambda x: "Img "+str(x))), y=list_new_y,text_auto=mc["text_auto"])

        #Fig layout
        for name in dataset_config.cell_class_names:
            fig.data[0].marker.color = adaptative_color_lines_transparency(coef_alpha=
                                                                        mc["coef_alpha"])[name]


        newnames = dict(zip(list_new_y, dataset_config.cell_class_names))
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))

        fig.update_layout(title_text=mc["title"]+"<br> "+group,
                        title_x=mc["title_x"], 
                        title_font=mc["title_font"],
                        margin=mc["margin"],
                        legend_title=mc["legend_title"],)
        fig.update_yaxes(side=mc["side"],title_text = mc["y_title"],title_standoff=mc["title_standoff"],tickfont=mc["tickfont"],showticklabels=mc["showticklabels"])
        fig.update_xaxes(title_text = "")
        #change y range 
        # fig.update_yaxes(range = [0,25*10E-9])


        fig.show() if display else None 

        path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,mc["metric_folder_name"])
        mkdir_if_nexist(path_folder_fig)
        fig.write_image(os.path.join(path_folder_fig,group+"_"+mc["figname"]))
        # fig.write_image("figures/{}_n_cells_per_image.png".format(dataset_config.dataset_name),width=1000, height=500, scale=1)

def n_cells_normalised_by_tissue_area_mean(dataset_config, results_several_images, metric_config, display = False):
    """ 
    Used in context of : 
    - Covid data 
    - Stats on entire images
    Possible args for 
    dict : one of [dict_n_cells_per_image, ] 
    """
    #Df pre processing 
    mc=metric_config
    for group in list(results_several_images["group"].unique()):
        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        # y = ["n_cells_{}_roi".format(name) for name in dataset_config.cell_class_names]
        sub_df = df_filtered_by_group[mc["y"]+["slide_num","area_tissue_slide"]]
        unique_values_df = sub_df.drop_duplicates(subset=['slide_num'])

        list_new_y = []
        for celltype_idx, celltype_n_cells_colname in enumerate(mc["y"]):
            unique_values_df["n_cells_{}_roi_normalised_by_tissue_area".format(dataset_config.cell_class_names[celltype_idx])] = unique_values_df[celltype_n_cells_colname]/(unique_values_df["area_tissue_slide"]*dataset_config.conversion_px_micro_meter**2)
            list_new_y.append("n_cells_{}_roi_normalised_by_tissue_area".format(dataset_config.cell_class_names[celltype_idx]))
            average = unique_values_df.mean()
            if celltype_idx == 0 : 
                fig = px.bar(average[["n_cells_{}_roi_normalised_by_tissue_area".format(dataset_config.cell_class_names[celltype_idx])]],text_auto=True)
            else : 
                fig.add_trace(px.bar(average[["n_cells_{}_roi_normalised_by_tissue_area".format(dataset_config.cell_class_names[celltype_idx])]],text_auto=True).data[0])

        fig.update_xaxes(tickvals=np.arange(4), ticktext=dataset_config.cell_class_names)

        #Fig layout
        for idx, name in enumerate(dataset_config.cell_class_names):
            fig.data[idx].marker.color = adaptative_color_lines_transparency(coef_alpha=
                                                                        mc["coef_alpha"])[name]
        newnames = dict(zip(mc["y"], dataset_config.cell_class_names))
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name] if t.name in list(newnames.keys()) else "jgregre"))

        fig.update_layout(title_text=mc["title"]+"<br> Mean on all images - "+group,
                        title_x=mc["title_x"], 
                        title_font=mc["title_font"],
                        margin=mc["margin"],
                        legend_title=mc["legend_title"],showlegend=False)
        fig.update_yaxes(side=mc["side"],title_text = mc["y_title"],title_standoff=mc["title_standoff"],tickfont=mc["tickfont"],showticklabels=mc["showticklabels"])
        fig.update_xaxes(title_text = "Object type")
        #change y range 
        # fig.update_yaxes(range = [0,25*10E-9])


        fig.show() if display else None 

        path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,mc["metric_folder_name"])
        mkdir_if_nexist(path_folder_fig)
        fig.write_image(os.path.join(path_folder_fig,group+"_mean_"+mc["figname"]))
        # fig.write_image("figures/{}_n_cells_per_image.png".format(dataset_config.dataset_name),width=1000, height=500, scale=1)






def _create_matrix_results(dataset_config,results_several_images,metric_config,images_list, colname_to_play_z, round_z=-1, convert_micro_m = False):
    import math
    dict_matrix = dict()
    cells = dataset_config.cell_class_names
    colname_to_check_multiple_parameters = metric_config["colname_to_check_multiple_parameters"]

    for image in images_list:
        sub_df = results_several_images.query("slide_num == @image")
        for id_parameter_set, parameter_set in enumerate(sub_df[colname_to_check_multiple_parameters].unique()):
            # print("parameter_set",parameter_set)
            # print(type(parameter_set))
            # print(pd.isna(parameter_set))
            if pd.isna(parameter_set):
                continue 
            sub_df_parameter_set = sub_df[sub_df[metric_config["colname_to_check_multiple_parameters"]] == parameter_set]
            matrix = np.zeros((len(cells),(len(cells))))
            for idx_A, type_A in enumerate(cells):
                for idx_B, type_B in enumerate(cells):
                    try:
                        z = sub_df_parameter_set[(sub_df_parameter_set["type_A"] == type_A) & (sub_df_parameter_set["type_B"] == type_B)][colname_to_play_z].values[0]
                    except IndexError:
                        # IndexError occurs when there is no value that satisfies the condition
                        z = np.nan
                    z = z*dataset_config.conversion_px_micro_meter if convert_micro_m else z
                    if round_z != -1:
                        z = round(z,round_z) if z != math.inf else np.nan
                    matrix[idx_A,idx_B] = z
                    # print(parameter_set)
                    # print("len(str(parameter_set))",len(str(parameter_set)))
                    if type(parameter_set)==str : #colocalisation 
                        dict_matrix["Img "+str(image)+" - "+metric_config["txt_param"]+str(parameter_set[:50])+"<br>"+str(parameter_set[50:])] = matrix
                    else : #Dbscan 
                        dict_matrix["Img "+str(image)+" - "+metric_config["txt_param"]+str(parameter_set)] = matrix

    return dict_matrix

def display_A_B_results(dataset_config,images_list, results_several_images, metric_config, display = False):
    """ Display A-B results
    
    dict : one of 
    #Coloc 
        [dict_coloc_association,
        dict_coloc_p_value,
        dict_coloc_distance ] 
    #DBSCAN 
        [dict_dbscan_iou,
        dict_dbscan_ioa,
        dict_dbscan_fraction_clustered_A_in_B,
        dict_dbscan_fraction_A_in_B_clusters
        ]
    #Neighbors analysis 
        [dict_neighbors_analysis_disk, 
        ]
    #
    """
    mc = metric_config
    for group in list(results_several_images["group"].unique()):

        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        images_list = list(df_filtered_by_group["slide_num"].unique())
        dict_matrix = _create_matrix_results(dataset_config,df_filtered_by_group,mc,images_list, mc["colname_to_play_z"],round_z=mc["round_z"], convert_micro_m = mc["convert_micro_m"])
        n_img = len(dict_matrix)
        # print("dict_matrix.keys()",dict_matrix.keys())
        # print("n_img",n_img)
        # print("n_img//2",n_img//2)
        # print("n_img%2",n_img%2)
        fig = make_subplots(n_img//2+(n_img%2), 2,subplot_titles=list(dict_matrix.keys()),vertical_spacing = 0.09,horizontal_spacing = 0.15) 

        for idx_plot, indice_img in enumerate(list(dict_matrix.keys())):
            # print("idx_plot",idx_plot)
            matrix = dict_matrix[indice_img]
            # print("indice_img",indice_img)
            # print("idx_plot//3+1, idx_plot%2+1",idx_plot//3+1, idx_plot%2+1)
            if mc["range_0_1"] : 

                fig.add_trace(px.imshow(np.flipud(matrix), labels=dict(x="Type B", y="Type A"),
                                x=dataset_config.cell_class_names,
                                y=dataset_config.cell_class_names[::-1], text_auto=True, aspect="auto",origin = 0,range_color = [0,1]).data[0], row=idx_plot//2+1, col=idx_plot%2+1)
            else : 
                fig.add_trace(px.imshow(np.flipud(matrix), labels=dict(x="Type B", y="Type A"),
                                x=dataset_config.cell_class_names,
                                y=dataset_config.cell_class_names[::-1], text_auto=True, aspect="auto",origin = 0).data[0], row=idx_plot//2+1, col=idx_plot%2+1)             
            # fig.add_trace(px.imshow(matrix, labels=dict(x="Type B", y="Type A"),
            #                 x=dataset_config.cell_class_names,
            #                 y=dataset_config.cell_class_names, text_auto=True, aspect="auto",origin = 0).data[0], row=idx_plot//2+1, col=idx_plot%2+1)

        fig.update_layout(title_text=mc["title"]+" " + group,
                        title_x=mc["title_x"], 
                        title_font=mc["title_font"],
                        margin=mc["margin"],
                        # legend_title=mc["legend_title"],
                        width=1100,
                        height=600*n_img//2+(n_img%2),
                        coloraxis_colorbar=dict(title=mc["legend_title"]))
        # fig.update_yaxes(side=mc["side"],title_text = mc["y_title"],title_standoff=mc["title_standoff"],tickfont=mc["tickfont"],showticklabels=mc["showticklabels"])
        fig.update_xaxes(title_text = "Type B")
        fig.update_yaxes(title_text = "Type A")
        fig.update_annotations(font=dict(size=10))
        # fig.update_yaxes(range = [0,1]) if mc["range_0_1"] else None 

        fig.show() if display else None

        path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,mc["metric_folder_name"])
        mkdir_if_nexist(path_folder_fig)
        fig.write_image(os.path.join(path_folder_fig,group+"_all_"+mc["figname"]+".png"))


def _create_matrix_results_mean_over_images(dataset_config,results_several_images,metric_config,images_list, colname_to_play_z,round_z=-1, convert_micro_m = False):
    """ Create the matrix containing mean values of "col_z" over the different image of image_list
      
    If several parameters have been used (ex Min_sample = [4,5], create a dictionnary with key:value = parameter:matrix)  
    In that case -> all images should have been processed with the same parameters
    """
    import math
    dict_matrix = dict()
    cells = dataset_config.cell_class_names
    colname_to_check_multiple_parameters = metric_config["colname_to_check_multiple_parameters"]
    # print("results_several_images.shape",results_several_images.shape)
    different_parameter_set = [k for k in results_several_images[colname_to_check_multiple_parameters].unique() if not pd.isna(k)]
    for id_parameter_set, parameter_set in enumerate(different_parameter_set):
        # print("parameter_set",parameter_set)
        sub_df = results_several_images[results_several_images[metric_config["colname_to_check_multiple_parameters"]] == parameter_set]
        # print("sub_df.shape",sub_df.shape)
        matrix = np.zeros((len(cells),(len(cells))))
        for image in images_list:
            # print("image",image)
            sub_df_image = sub_df.query("slide_num == @image")
            # print("sub_df.shape",sub_df.shape)
            for idx_A, type_A in enumerate(cells):
                for idx_B, type_B in enumerate(cells):
                    try:
                        z = sub_df_image[(sub_df_image["type_A"] == type_A) & (sub_df_image["type_B"] == type_B)][colname_to_play_z].values[0]
                    except IndexError:
                        # IndexError occurs when there is no value that satisfies the condition
                        z = np.nan
                    z = z*dataset_config.conversion_px_micro_meter if convert_micro_m else z
                    # z = z/len(images_list)

                    matrix[idx_A,idx_B] += z
        matrix = matrix/len(images_list)
        if round_z != -1:
            matrix = np.round(matrix,round_z) 
        # print("matrix parameter_set",parameter_set,matrix)
        dict_matrix[metric_config["txt_param"]+str(parameter_set)] = matrix
    return dict_matrix

def display_A_B_results_mean(dataset_config,images_list, results_several_images, metric_config, display = False):
    mc = metric_config
    for group in list(results_several_images["group"].unique()):
        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        images_list = list(df_filtered_by_group["slide_num"].unique())
        print("images_list",images_list)
        dict_matrix = _create_matrix_results_mean_over_images(dataset_config,df_filtered_by_group,mc,images_list, mc["colname_to_play_z"],round_z=mc["round_z"], convert_micro_m = mc["convert_micro_m"])
        # print("dict_matrix.keys()",dict_matrix.keys())
        # print("n_img",n_img)
        # print("n_img//2",n_img//2)
        # print("n_img%2",n_img%2)
        for k in dict_matrix.keys():
            matrix = dict_matrix[k]
                # print("indice_img",indice_img)
                # print("idx_plot//3+1, idx_plot%2+1",idx_plot//3+1, idx_plot%2+1)
            if mc["range_0_1"] : 
                fig = px.imshow(matrix, labels=dict(x="Type B", y="Type A"),
                                    x=dataset_config.cell_class_names,
                                    y=dataset_config.cell_class_names, text_auto=True, aspect="auto",origin = 0,range_color = [0,1])
            else : 
                fig = px.imshow(matrix, labels=dict(x="Type B", y="Type A"),
                                    x=dataset_config.cell_class_names,
                                    y=dataset_config.cell_class_names, text_auto=True, aspect="auto",origin = 0)    
            if "4_Neighbors_analysis" in mc["metric_folder_name"]:
                title_text = mc["title"]+" "+group
                txt_imgs  = "_mean_imgs_"+"_".join(map(str,images_list))
                filename = group+"_"+mc["figname"]+"_"+txt_imgs+".png"
            else : 
                title_text = mc["title"]+" - "+group+" -"+"<br> "+mc["txt_param"]+str(k[:50])+" "+"<br>"+str(k[50:])
                txt_imgs  = "_mean_imgs_"+"_".join(map(str,images_list))
                filename = group+"_"+mc["figname"]+"_"+str(k)+txt_imgs+".png"
            fig.update_layout(title_text=title_text,
                            title_x=mc["title_x"], 
                            title_font=mc["title_font"],
                            margin=mc["margin"],
                            legend_title=mc["legend_title"],
                            width=800,
                            height=700,
                            coloraxis_colorbar=dict(title=mc["legend_title"]))
            # fig.update_yaxes(side=mc["side"],title_text = mc["y_title"],title_standoff=mc["title_standoff"],tickfont=mc["tickfont"],showticklabels=mc["showticklabels"])
            # fig.update_xaxes(title_text = "")
            fig.update_annotations(font=dict(size=10))
            fig.show() if display else None

            path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,mc["metric_folder_name"])
            mkdir_if_nexist(path_folder_fig)
            

            fig.write_image(os.path.join(path_folder_fig,filename))


def plot_dbscan_clusters_analysis_results_mean(dataset_config, results_several_images, display = False):
    """ Plot the results of the clusters analysis. """
    for group in list(results_several_images["group"].unique()):

        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        cols_of_interest = ["slide_num","type_A","min_sample_A","n_robust_cluster_A", "n_robust_clustered_cells_A", "n_robust_isolated_cells_A", "fraction_robust_clustered_A","mean_n_cells_robust_clusters_A"]
        df_filtered = df_filtered_by_group[cols_of_interest]
        unique_values_df = df_filtered.drop_duplicates()
        for min_sample in df_filtered_by_group["min_sample_A"].unique():
            sub_df = unique_values_df[unique_values_df["min_sample_A"] == min_sample]
            if len(sub_df)==0 : 
                continue 
            fig = make_subplots(2, 2,subplot_titles=["<b>Number of clusters", "<b>Mean size clusters","<b>Number of clustered cells", "<b>Fraction of clustered cells"],vertical_spacing = 0.15,horizontal_spacing = 0.05) 
            for celltype in dataset_config.cell_class_names:
                dubb_df = sub_df[sub_df["type_A"]==celltype]
                summed_df = dubb_df.mean()
                summed_df["type_A"] = celltype
                df = pd.DataFrame([summed_df])
                fig.add_trace(px.bar(df,x="type_A", y="n_robust_cluster_A").data[0],row=1, col=1)
                fig.add_trace(px.bar(df,x="type_A", y="mean_n_cells_robust_clusters_A").data[0],row=1, col=2)
                fig.add_trace(px.bar(df,x="type_A", y="n_robust_clustered_cells_A").data[0],row=2, col=1)
                fig.add_trace(px.bar(df,x="type_A", y="fraction_robust_clustered_A").data[0],row=2, col=2)
            fig.update_layout(title_text="<b>DBSCAN statistics - Min sample : "+str(int(min_sample))+"<br> "+group,title_x=0.5, title_font=dict(size=20),
                            showlegend=True,
                            width=1000,
                            height=1000,
                            margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                            )
            #Fig layout
            
            names = set()
            fig.for_each_trace(
                lambda trace: trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name))
            
            for bar_ref in range(len(fig.data)):
                cell_type = fig.data[bar_ref]["x"][0]
                fig.data[bar_ref].marker.color = adaptative_color_lines_transparency(coef_alpha=0.5)[cell_type]
                # fig.data[bar_ref].showlegend = False if bar_ref > len(dataset_config.cell_class_names) else None 
            fig.show() if display else None 

            figname = group+"_mean_clusters_cells_proportion_ms_"+str(int(min_sample))+".png"
            path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,"3_DBSCAN")
            mkdir_if_nexist(path_folder_fig)
            fig.write_image(os.path.join(path_folder_fig,figname))



def plot_dbscan_clusters_analysis_results_details(dataset_config, results_several_images, display = False):
    """ Plot the results of the clusters analysis. """
    for group in list(results_several_images["group"].unique()):

        df_filtered_by_group = results_several_images[results_several_images["group"] == group]
        images_list = list(df_filtered_by_group["slide_num"].unique())

        cols_of_interest = ["slide_num","type_A","min_sample_A","n_robust_cluster_A", "n_robust_clustered_cells_A", "n_robust_isolated_cells_A", "fraction_robust_clustered_A","mean_n_cells_robust_clusters_A"]
        df_filtered = df_filtered_by_group[cols_of_interest]
        unique_values_df = df_filtered.drop_duplicates() 
        for min_sample in unique_values_df["min_sample_A"].unique():
            sub_df = unique_values_df[unique_values_df["min_sample_A"] == min_sample]
            if len(sub_df)==0 : 
                continue 
            fig = make_subplots(2, 2,subplot_titles=["<b>Number of clusters", "<b>Mean size clusters","<b>Number of clustered cells", "<b>Fraction of clustered cells"],vertical_spacing = 0.15,horizontal_spacing = 0.05) 
            
            for celltype in dataset_config.cell_class_names:
                dub_df_celltype = sub_df[sub_df["type_A"]==celltype]

                fig.add_trace(px.bar(dub_df_celltype, x=dub_df_celltype["slide_num"].apply(lambda x:"<b> Img "+str(int(x))+"<b>"), y="n_robust_cluster_A",color = "type_A").data[0],row=1, col=1)
                fig.add_trace(px.bar(dub_df_celltype, x=dub_df_celltype["slide_num"].apply(lambda x:"<b> Img "+str(int(x))+"<b>"), y="mean_n_cells_robust_clusters_A",color = "type_A").data[0],row=1, col=2)
                fig.add_trace(px.bar(dub_df_celltype, x=dub_df_celltype["slide_num"].apply(lambda x:"<b> Img "+str(int(x))+"<b>"), y="n_robust_clustered_cells_A",color = "type_A").data[0],row=2, col=1)
                fig.add_trace(px.bar(dub_df_celltype, x=dub_df_celltype["slide_num"].apply(lambda x:"<b> Img "+str(int(x))+"<b>"), y="fraction_robust_clustered_A",color = "type_A").data[0],row=2, col=2)

            fig.update_layout(title_text="<b>DBSCAN statistics - Min sample : "+str(int(min_sample))+"<br> "+group,title_x=0.5, title_font=dict(size=20),
                            showlegend=True,
                            width=1000,
                            height=1000,
                            margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                            )
            #Fig layout-> Remove some legent that are duplicated otherwise 
            names = set()
            fig.for_each_trace(
                lambda trace:
                    trace.update(showlegend=False)
                    if (trace.name in names) else names.add(trace.name))
            #Change color of the bars 
            for bar_ref in range(len(fig.data)):
                fig.data[bar_ref].marker.color = adaptative_color_lines_transparency(coef_alpha=0.5)[fig.data[bar_ref]["legendgroup"]]

            fig.show() if display else None 
            figname =group+ "_clusters_cells_proportion_ms_"+str(int(min_sample))+".png"
            path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,"3_DBSCAN")
            mkdir_if_nexist(path_folder_fig)
            fig.write_image(os.path.join(path_folder_fig,figname))


# def plot_dbscan_clusters_analysis_results(dataset_config, results_several_images, display = False):
#     """ Plot the results of the clusters analysis. """
#     n_closest_neighbors_of_interest = results_several_images["n_closest_neighbors_of_interest"].iloc[0]
#     # cols_of_interest = ["slide_num","type_A","type_B"]+["mean_dist_"+str(k)+"_first_cell_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
#     cols_of_interest = ["slide_num","type_A","type_B"]+["mean_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
    
#     y = ["mean_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
#     df_filtered = results_several_images[cols_of_interest]
    
#     n_cells = len(dataset_config.cell_class_names)
#     fig = make_subplots(n_cells//2+(n_cells%2), 2,subplot_titles=["<b>Neighbors of "+ k for k in dataset_config.cell_class_names],vertical_spacing = 0.07,horizontal_spacing = 0.15) 
        
#     for idx_plot, celltype in enumerate(dataset_config.cell_class_names):
#         dub_df_celltype = df_filtered[df_filtered["type_A"]==celltype]
#         summed_df = dub_df_celltype.mean()
#         unique_values_df = dub_df_celltype.drop_duplicates() 
#         df = pd.DataFrame(columns = unique_values_df.columns)
#         for celltype_B in dataset_config.cell_class_names:
#             df_subbb = unique_values_df[unique_values_df["type_B"] == celltype_B]
#             # print("df after all filtering : ")
#             # print(df_subbb)
#             df_subbb = df_subbb.mean()
#             # print("celltype_B",celltype_B)
#             # print(df_subbb)
#             df_subbb["type_B"] = celltype_B
#             df_subbb["type_A"] = celltype
#             df_sub = pd.DataFrame([df_subbb])
#             df = pd.concat([df,df_sub], axis = 0)
#         for k in range(n_closest_neighbors_of_interest):
#             # fig.add_trace(px.bar(unique_values_df, x=unique_values_df["slide_num"].apply(lambda x:"<b> Img "+str(int(x))+"<b>"), y=y[k]).data[0],row=idx_plot//2+1, col=idx_plot%2+1)
#             # fig.add_trace(px.bar(df, x="type_B", y=y[k]).data[0],row=idx_plot//2+1, col=idx_plot%2+1)
#             fig.add_trace(px.bar(df, x="type_B", y=df[y[k]].apply(lambda x:x*dataset_config.conversion_px_micro_meter)).data[0],row=idx_plot//2+1, col=idx_plot%2+1)

#         fig.update_layout(title_text="<b>Neighbors - First "+str(n_closest_neighbors_of_interest)+" B neighbors of A",title_x=0.5, title_font=dict(size=20),
#                         showlegend=True,
#                         width=1000,
#                         height=1000,
#                         margin=dict(l=50,r=50,b=50,t=100, pad=4))
#         i = 0 
#         j= 0 
#         for bar_ref in range(len(fig.data)):
#             fig.data[bar_ref].marker.color=adaptative_color_lines_transparency(coef_alpha=1)[dataset_config.cell_class_names[j]]
#             i+=1 
#             if i == 3 : 
#                 i=0 
#                 j+=1 

#     fig.show() if display else None 
#     figname = "first_"+str(n_closest_neighbors_of_interest)+"_B_neighbors_arround_A.png"
#     path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,"4_Neighbors_analysis")
#     mkdir_if_nexist(path_folder_fig)
#     fig.write_image(os.path.join(path_folder_fig,figname))


def plot_first_A_neighbors_results(dataset_config, results_several_images, display = False, per_image = False ):
    """ Plot the results of the clusters analysis. """
    for group in list(results_several_images["group"].unique()):

        df_filtered_by_group = results_several_images[results_several_images["group"] == group]   
        n_closest_neighbors_of_interest = df_filtered_by_group["n_closest_neighbors_of_interest"].iloc[0]
        cols_of_interest = ["slide_num","type_A"]+["mean_dist_"+str(k)+"_first_cell_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        # cols_of_interest = ["slide_num","type_A","type_B"]+["mean_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        
        y = ["mean_dist_"+str(k)+"_first_cell_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        df_filtered = df_filtered_by_group[cols_of_interest]
        df_filtered = df_filtered.drop_duplicates() 
        df_filtered[y] *= dataset_config.conversion_px_micro_meter
        if per_image:

            result = df_filtered.groupby(['slide_num', 'type_A']).mean().reset_index()
        else : 
            result = df_filtered.groupby(['type_A']).mean().reset_index()

        fig = px.bar(result,x = "type_A", y = y,barmode='group',color_discrete_map = degrade_colors_neighbors_analysis())
        fig.update_layout(title_text="<b>Neighbors - Distance of the " + str(n_closest_neighbors_of_interest)+" first A neighbors <br>"+group,title_x=0.5, title_font=dict(size=20),
                        showlegend=True,
                        width=1000,
                        height=1000,
                        margin=dict(l=50,r=50,b=50,t=220, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                        legend_title="Distance")
        newnames = {"mean_dist_1_first_cell_around_A":'First neighbor', "mean_dist_2_first_cell_around_A":'Second neighbor',"mean_dist_3_first_cell_around_A":'Third neighbor'}
        fig.update_yaxes(range = [0,100])
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        fig.show() if display else None 
        figname =group+"_mean_first_"+str(n_closest_neighbors_of_interest)+"_A_neighbors.png" if not per_image else "first_"+str(n_closest_neighbors_of_interest)+"_A_neighbors.png"
        path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,"4_Neighbors_analysis")
        mkdir_if_nexist(path_folder_fig)
        fig.write_image(os.path.join(path_folder_fig,figname))

def plot_B_neighbors_arround_A_results(dataset_config, results_several_images, display = False):
    """ Plot the results of the clusters analysis. """
    for group in list(results_several_images["group"].unique()):

        df_filtered_by_group = results_several_images[results_several_images["group"] == group]        
        n_closest_neighbors_of_interest = df_filtered_by_group["n_closest_neighbors_of_interest"].iloc[0]
        # cols_of_interest = ["slide_num","type_A","type_B"]+["mean_dist_"+str(k)+"_first_cell_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        cols_of_interest = ["slide_num","type_A","type_B"]+["mean_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        
        y = ["mean_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        df_filtered = df_filtered_by_group[cols_of_interest]
        
        n_cells = len(dataset_config.cell_class_names)
        fig = make_subplots(n_cells//2+(n_cells%2), 2,subplot_titles=["<b>Neighbors of "+ k for k in dataset_config.cell_class_names],vertical_spacing = 0.07,horizontal_spacing = 0.15) 
            
        for idx_plot, celltype in enumerate(dataset_config.cell_class_names):
            dub_df_celltype = df_filtered[df_filtered["type_A"]==celltype]
            summed_df = dub_df_celltype.mean()
            unique_values_df = dub_df_celltype.drop_duplicates() 
            df = pd.DataFrame(columns = unique_values_df.columns)
            for celltype_B in dataset_config.cell_class_names:
                df_subbb = unique_values_df[unique_values_df["type_B"] == celltype_B]
                # print("df after all filtering : ")
                # print(df_subbb)
                df_subbb = df_subbb.mean()
                # print("celltype_B",celltype_B)
                # print(df_subbb)
                df_subbb["type_B"] = celltype_B
                df_subbb["type_A"] = celltype
                df_sub = pd.DataFrame([df_subbb])
                df = pd.concat([df,df_sub], axis = 0)
            for k in range(n_closest_neighbors_of_interest):
                # fig.add_trace(px.bar(unique_values_df, x=unique_values_df["slide_num"].apply(lambda x:"<b> Img "+str(int(x))+"<b>"), y=y[k]).data[0],row=idx_plot//2+1, col=idx_plot%2+1)
                # fig.add_trace(px.bar(df, x="type_B", y=y[k]).data[0],row=idx_plot//2+1, col=idx_plot%2+1)
                fig.add_trace(px.bar(df, x="type_B", y=df[y[k]].apply(lambda x:x*dataset_config.conversion_px_micro_meter)).data[0],row=idx_plot//2+1, col=idx_plot%2+1)
            fig.update_yaxes(range = [0,800])
            fig.update_layout(title_text="<b> Neighbors - First "+str(n_closest_neighbors_of_interest)+" B neighbors of A <br>"+group,title_x=0.5, title_font=dict(size=20),
                            showlegend=True,
                            width=1000,
                            height=1000,
                            margin=dict(l=50,r=50,b=50,t=100, pad=4))

            i = 0 
            j= 0 
            for bar_ref in range(len(fig.data)):
                fig.data[bar_ref].marker.color=adaptative_color_lines_transparency(coef_alpha=1)[dataset_config.cell_class_names[j]]
                i+=1 
                if i == 3 : 
                    i=0 
                    j+=1 

        fig.show() if display else None 
        figname = group+"_first_"+str(n_closest_neighbors_of_interest)+"_B_neighbors_arround_A.png"
        path_folder_fig = os.path.join(dataset_config.dir_base_stat_analysis,"4_Neighbors_analysis")
        mkdir_if_nexist(path_folder_fig)
        fig.write_image(os.path.join(path_folder_fig,figname))
