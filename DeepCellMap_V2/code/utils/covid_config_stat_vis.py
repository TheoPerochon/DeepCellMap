# Once statistics are calculated on images. This file contain config
# to visualize them. 

from config.dataset_management import take_config_from_dataset
dataset_name = "covid_data_immunofluorescence" #ihc_microglia_fetal_human_brain_database, cancer_data_immunofluorescence, covid_data_immunofluorescence
dataset_config = take_config_from_dataset(dataset_name)
#### Cell number 

#dict_n_cells_per_image

### Colocalisation 

#dict_coloc_association
#dict_coloc_distance
#dict_coloc_distance


### DBSCAN 

# dict_dbscan_iou
# dict_dbscan_ioa
# dict_dbscan_fraction_clustered_A_in_B
# dict_dbscan_fraction_A_in_B_clusters

### Neighbors analysis 

# dict_neighbors_analysis_disk



dict_n_cells_per_image = dict({
    "metric_folder_name": "1_Cells_quantities_and_sizes",
    "figname" : "n_cells_per_image.png",
    "y" : ["n_cells_{}_roi".format(name) for name in dataset_config.cell_class_names],
    "col_x_label" : "slide_num",
    "y_title" : "Cell number",
    "title" : "<b>Object quantities in images",
    "legend_title" : "Objects names",
    "title_x" : 0.5,
    "side" : "left",
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=15),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
})

dict_n_cells_per_image_per_tissue_area = dict({
    "metric_folder_name": "1_Cells_quantities_and_sizes",
    "figname" : "n_cells_per_image_normalised_by_tissue_area.png",
    "y" : ["n_cells_{}_roi".format(name) for name in dataset_config.cell_class_names],
    "col_x_label" : "slide_num",
    "y_title" : "Cell number per tissue size",
    "title" : "<b>Object quantities in images divided by tissue area",
    "legend_title" : "Objects names",
    "title_x" : 0.5,
    "side" : "left",
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=15),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
})
### Colocalisation 


dict_coloc_association = dict({
    "metric_folder_name": "2_Cells_Cells_association",
    "figname" : "association",
    "colname_to_play_z" : "association_score",
    "round_z": 2,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "levelsets",
    "col_x_label" : "slide_num",
    "y_title" : "Association score",
    "title" : "<b>Colocalisation : Association score ",
    "legend_title" : "association<br>score",
    "title_x" : 0.5,
    "side" : "right",
    "text_auto" : True,
    "range_0_1" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "",
})

dict_coloc_p_value = dict({
    "metric_folder_name": "2_Cells_Cells_association",
    "figname" : "p_value",
    "colname_to_play_z" : "p_value",
    "round_z": 6,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "levelsets",
    "col_x_label" : "slide_num",
    "y_title" : "p value",
    "title" : "<b>Colocalisation : p value ",
    "legend_title" : "p_value",
    "title_x" : 0.5,
    "side" : "right",
    "text_auto" : True,
    "range_0_1" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "",
})

dict_coloc_distance = dict({
    "metric_folder_name": "2_Cells_Cells_association",
    "figname" : "distance_association",
    "colname_to_play_z" : "distance_association",
    "round_z": 0,
    "convert_micro_m": True,
    "colname_to_check_multiple_parameters" : "levelsets",
    "col_x_label" : "slide_num",
    "y_title" : "Distance [µm]",
    "title" : "<b>Cell vs Cell colocalisation : Mean distance of association [µm]",
    "legend_title" : "distance [µm]",
    "title_x" : 0.5,
    "range_0_1" : False,
    "side" : "right",
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "",
})


### DBSCAN 
dict_dbscan_iou = dict({
    "metric_folder_name": "3_DBSCAN",
    "figname" : "DBSCAN_IoU",
    "colname_to_play_z" : "iou",
    "round_z": 2,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "min_sample_A",
    "col_x_label" : "slide_num",
    "y_title" : "iou",
    "title" : "<b>DBSCAN - Intersection over Union of the convex hulls<br>",
    "legend_title" : "iou",
    "title_x" : 0.5,
    "side" : "right",
    "range_0_1" : True,
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "MS_",
})

dict_dbscan_ioa = dict({
    "metric_folder_name": "3_DBSCAN",
    "figname" : "DBSCAN_I_A",
    "colname_to_play_z" : "i_A",
    "round_z": 2,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "min_sample_A",
    "col_x_label" : "slide_num",
    "y_title" : "ioA",
    "title" : "<b>DBSCAN - Intersection over Union of the convex hulls of A <br>",
    "legend_title" : "ioa",
    "title_x" : 0.5,
    "range_0_1" : True,
    "side" : "right",
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "MS_",
})

# dict_dbscan_iob = dict({
#     "metric_folder_name": "3_DBSCAN",
#     "figname" : "DBSCAN_I_B.png",
#     "colname_to_play_z" : "i_B",
#     "round_z": 2,
#     "convert_micro_m": False,
#     "colname_to_check_multiple_parameters" : "min_sample_A",
#     "col_x_label" : "slide_num",
#     "y_title" : "ioA",
#     "title" : "<b>DBSCAN - Intersection over Union of the convex hulls of B <br>",
#     "legend_title" : "ioa",
#     "title_x" : 0.5,
#     "side" : "right",
#     "text_auto" : True,
#     "margin" : dict(t=100),
#     "title_font" : dict(size=15),
#     "tickfont" : dict(size=10),
#     "showticklabels" : True,
#     "title_standoff" : 0,
#     "coef_alpha" : 1,
#     "txt_param" : "MS_",
# })

dict_dbscan_fraction_clustered_A_in_B = dict({
    "metric_folder_name": "3_DBSCAN",
    "figname" : "fraction_clustered_A_in_B",
    "colname_to_play_z" : "fraction_clustered_A_in_clustered_B",
    "round_z": 2,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "min_sample_A",
    "col_x_label" : "slide_num",
    "y_title" : "fraction_clustered_A_in_B",
    "title" : "<b>DBSCAN - fraction_clustered_A_in_B <br>",
    "legend_title" : "fraction_clustered_A_in_B",
    "title_x" : 0.5,
    "side" : "right",
    "range_0_1" : True,
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "MS_",
})

dict_dbscan_fraction_A_in_B_clusters = dict({
    "metric_folder_name": "3_DBSCAN",
    "figname" : "fraction_A_in_B_clusters",
    "colname_to_play_z" : "fraction_A_in_B_clusters",
    "round_z": 2,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "min_sample_A",
    "col_x_label" : "slide_num",
    "y_title" : "fraction_A_in_B_clusters",
    "title" : "<b>DBSCAN - fraction_A_in_B_clusters <br>",
    "legend_title" : "fraction_A_in_B_clusters",
    "title_x" : 0.5,
    "side" : "right",
    "text_auto" : True,
    "range_0_1" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "MS_",
})



### Neighbors analysis

dict_neighbors_analysis_disk = dict({
    "metric_folder_name": "4_Neighbors_analysis",
    "figname" : "fraction_disk_containing_B",
    "colname_to_play_z" : "fraction_disk_dc_containing_at_least_one_B",
    "round_z": 2,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "n_closest_neighbors_of_interest",
    "col_x_label" : "slide_num",
    "y_title" : "Fraction A balls",
    "title" : "<b>Fraction of A balls(r) having at least 1 B inside<br> R = max(Mean colocalisation distance A vs all B)",
    "legend_title" : "Fraction A balls",
    "title_x" : 0.5,
    "side" : "right",
    "range_0_1" : True,
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "",
})

dict_neighbors_analysis_fraction_B_first_A_neighbor = dict({
    "metric_folder_name": "4_Neighbors_analysis",
    "figname" : "fraction_B_first_A_neighbor",
    "colname_to_play_z" : "fraction_B_first_A_neighbor",
    "round_z": 3,
    "convert_micro_m": False,
    "colname_to_check_multiple_parameters" : "n_closest_neighbors_of_interest",
    "col_x_label" : "slide_num",
    "y_title" : "Fraction A cells",
    "title" : "<b>Fraction of A cells having B as first neighbor<br> ",
    "legend_title" : "Fraction A cells",
    "title_x" : 0.5,
    "range_0_1" : True,
    "side" : "right",
    "text_auto" : True,
    "margin" : dict(t=100),
    "title_font" : dict(size=15),
    "tickfont" : dict(size=10),
    "showticklabels" : True,
    "title_standoff" : 0,
    "coef_alpha" : 1,
    "txt_param" : "",
})