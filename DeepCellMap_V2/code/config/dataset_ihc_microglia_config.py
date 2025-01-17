import os
import json
from config.base_config import BaseConfig
from config.datasets_config import DatasetBaseConfig, PreprocessingConfig


class IhcMicrogliaFetalHumanBrain(DatasetBaseConfig):
    """
    Link data : https://unioxfordnexus-my.sharepoint.com/personal/wolf2242_ox_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwolf2242%5Fox%5Fac%5Fuk%2FDocuments%2FHolcman%2DMenassa%2DScans&fromShare=true&ga=1
    """
    def __init__(self):


        dataset_name = "ihc_microglia_fetal_human_brain_database"
        super().__init__(dataset_name)  # Call the constructor of the parent class
        self.mapping_img_name = [
            "B5349-H-17pcw",
            "19pcw-NP-155-10-7",
            "NP-65-08-13-20pcw",
            "N54J02-33pcw",
            "RI-1248-94B-14pcw",
            "CS12-880-4pcw",
            "CS14-737-5pcw",
            "CS16-921-6pcw",
            "CS18-327-7pcw",
            "CS21-777-51-8pcw",#10
            "12044-51-10pcw",
            "13246 10pcw D FRONT 17 IBA1 CASP3",
            "13246 10pcw D FRONT 69 IBA1 SOX2",
            "CS23-12780-9pcw",
            "13239 11pcw D FRONT 3 SOX2 IBA1",#15
            "13247 11pcw D3 117 IBA1 CASP3",
            "NP-146-11-12pcw",
            "14221 13W 19 IBA1 CASP3",
            "15322 12W L POST 7 IBA1 CASP3",
            "14133 14W D HOR INF 18 IBA1 CASP3",
            "13470 15pcw D2 POST 39 IBA1 CASP3",
            "13950 16W L POST 19 IBA1 CASP3",
            "13928 20pcw D POST 19 IBA1 CASP3",
            "15307 CS23 90 IBA1 SOX2", #24
            "25pcw-CD 309 D2 (74) IBA1 KI67",
            "N48E02-23pcw",
            "17pcw-NP-153-12-6",
            "N32K03-38pcw", #28
            "N108M02-IBA1-Ki67-1monthPostnatal",
            "Temporal-N20-65-08-14",
            "12pcw-N1798-74",
            "Theo-10pcw-12044-51-IBA1-Ki67.svs",#32
            "Theo-12pcw-146-2011-IBA1-Ki67-Haem-Frontal.svs",#33
            "Theo-14pcw-RI-1248-94B-IBA1-Ki67.svs",#34
            "Theo-17pcw-B3549-IBA1-Ki67-Striatum.svs",#35 #Deja dans le premier set d'analyse  
            "Theo-19pcw-85-2006-IBA1-Ki67-Striatum.svs",#36
            "Theo-19pcw-155-2010-IBA1-Ki67-Striatum.svs",#37#Deja dans le premier set d'analyse  
            "Theo-20pcw-65-2008-IBA1-Ki67-Striatum.svs",]#38
        self.mapping_img_gender = ["Unknow", "Unknow", "Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow", "Unknow", "Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow", "Unknow", "Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow"]
        self.mapping_img_disease = None 
        self.mapping_img_age = ["17", "19", "20","22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22']
        self.cell_class_names = [
            "proliferative",
            "amoeboid",
            "aggregated",
            "phagocytic",
            "ramified",
        ]
        self.mapping_cells_colors = {
            "proliferative": [173, 40, 125],
            "amoeboid": [79, 157, 214],
            "aggregated": [58, 180, 15],
            "phagocytic": [240, 186, 64],
            "ramified": [35, 32, 179],
            "other" : [63, 4, 126]
        }

        self.debug_mode = False
        # Define the dataset-specific directories
        self.data_type = "wsi"
        self.consider_image_with_channels = False 
        self.data_format = "tif"
        self.conversion_px_micro_meter = 0.45

        self.channel_names = None #Fluorescence specific attribute 
        self.dim_position = None #Fluorescence specific attribute 
        self.has_Z = False #Fluorescence specific attribute 
        # Tiling parameters
        self.tile_width = 1024
        self.tile_height = 1024
        self.roi_border_size = int(1024/4)
        self.border_size_during_segmentation = int(1024/4)
        self.crop_size = 256
        self.scale_factor = 32
        self.tissue_extraction_accept_holes = False
        # Segmentation parameters
        self.preprocessing_config = PreprocessingConfig(
            dataset_name=dataset_name, scale_factor=self.scale_factor,tissue_extraction_accept_holes = self.tissue_extraction_accept_holes
        )

        self.tissue_segmentation_param = dict({
            "default" : dict({
                "manual_threshold": 11,
                "dilation": 4,
                "fill_holes": None,
                "remove_small_objects": 300})
        })
        self.threshold_tissue = 0.95
        self.save_tissue_segmentation_steps = True
        self.channel_used_to_segment_tissue = None 
        # Classification parameters
        self.use_imgs_as_channels = False
        self.channels_cells_to_segment = None 
        self.channels_of_interest = None
        self.cells_from_multiple_channels = None
        self.association_cell_name_channel_number = None 
        self.param_best_cellpose = dict({"model_type" : "cyto2","diameter" : 20,"channels" : [[0,0]], "normalisation" : True, "net_avg" : False})

        # Define the dataset-specific physiological_regions
        self.cellpose_parameters = dict(
            {
                "downscale_factor_cellpose_tissue_segmentation": 4,  # better : scale_factor_tile_crop_cellpose
            }
        )
        # Objects specific parameters
        self.model_segmentation_name = "image_processing_steps"
        self.cell_segmentation_param = dict({
            "default" : dict(
                {#segmentation_microglia_otsu integrer ca  ca dans 
                #apply_process_from_name et voir si dans segment_from_cell_segmentation_param
                "rgb_to_eosin" : "mean_channels",
                # "multi_otsu_thresholding" : [None, 0], #Same as segmentation_microglia_otsu????? 
                "otsu_thresholding" : [None,-1],
                "opening" : 1,
                "dilation" : 2,
                "fill_holes" : None,

                # "model_segmentation_type": "otsu_based_binarization_microglia_IHC",
                # "dilation_radius": 2,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
            }),
            "image_16" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
            "image_18" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
            "image_19" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
            "image_20" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
            "image_21" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
            "image_22" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
            "image_23" : dict(
                    {
                "rgb_to_d" : "mean_channels",
                "multi_otsu_thresholding" : [None,-1],
                "opening" : 2,
                "dilation" : 2,
                "fill_holes" : None,
                "remove_small_objects": 400,
                "remove_large_objects" : 100000,
                "filter_center_cells" : self.tile_width,
                }
            ),
        })
        self.cell_segmentation_param_by_cannnel = None 
        self.classification_param = dict(
            {   
                "model_name": "best_model_microglia_IHC",
                "training_set_config_name": "prol_amoe_clust_phag_rami",
                "dir_base_classif": self.dir_base_classif,
                "mode_prediction" : "summed_proba",
                "n_class":7
            }
        )
        self.cell_class_names_for_classification = [
            "background",
            "proliferative",
            "amoeboid",
            "aggregated",
            "phagocytic",
            "ramified",
            "detected",
        ]
        self.tile_test_segmentation = dict({
            "001": [(18,41),(18,35),(18,36),(19,35),(19,36)],
            "002": [(18,41),(18,35),(18,36),(19,35),(19,36)],
            "003": [(18,41),(18,35),(18,36),(19,35),(19,36)],
            "004": [(14,58),(22,60),(56,63),(52,78),(4,82),(18,81),(8,95),(70,21),(80,9)],
            "005": [(20,51),(35,36),(32,39),(45,40),(9,60),(12,67),(12,49),(17,37)],
            "019": [(13,3),(6,11),(3,22),(8,22)],
            "020": [(16,14),(22,31)]
        })
        #Usage : 
        #slide_num, origin_row, origin_col,end_row, end_col  = dataset_config.roi_test_roi_4_tiles[0]
        self.roi_test_tissue_border = [(1,28,18,29,19)]
        self.roi_cool = [(1,17,35,19,37)]


        self.statistics_with_proba = True
        
        self.path_cells = os.path.join(
            self.dir_base_classif, "cells_per_images", "cells"
        )

        self.physiological_regions_max_square_size = {
            "striatum": 100,
            "ganglionic_eminence": 100,
            "cortical_boundary": 50,
            "neocortex": 50,
        }
        self.physiological_regions_group_for_comparaison = {
            "striatum": 1,
            "ganglionic_eminence": 2,
            "cortical_boundary": 3,
            "neocortex": 4,
        }

        # Image specific parameters

        # methods
        self.cell_cell_colocalisation_config = dict(
            {   "compute_with_proba" : True,
                "cell_types_A": self.cell_class_names,
                "cell_types_B": self.cell_class_names,
                "levelsets": [
                        0,
                        100,
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                        1000,
                        1100,
                        1200,
                        1300,
                        1400,
                        1500,
                        1600,
                        1700,
                        1800,
                        1900,
                        2000,
                        2100
                    ],
                "save_images": True,
            }
        )
        self.dbscan_based_analysis_config = dict(
            {
            "min_sample": 4,
            "range_epsilon_to_test" : [100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6300,6600,7000,7300,7600,8000,8500,9000] ,
            "cell_types_A": self.cell_class_names,
            "cell_types_B": self.cell_class_names,
            "config_cluster_robustess_experiment": {
                "n_experiment_of_removing": 3,
                "ratio_removed_cells_robustess_test": 0.1,
                "threshold_conserved_area": 0.6
            },
            "display_convex_hull_clusters": True,
            "save_figure_dbscan": True
            }
        )
        self.neighbors_analysis_config = dict(
            {
            "n_closest_neighbors_of_interest": 3,
            "cell_types_A": self.cell_class_names,
            "cell_types_B": self.cell_class_names,
            }
        )

        self.colnames_table_cells_base = (
            ["id_cell",
            "tile_row",
            "tile_col",
            "tile_cat",
             "x_tile", 
             "y_tile", 
             "x_img", 
             "y_img",
             "x_tile_border",
             "y_tile_border",
            "size",
            "length_max",
            "check_out_of_borders",
            "check_in_centered_tile",
            "bonification_cluster_size_sup",
            "penalite_cluster_size_inf",
            "cell_type"]
            + ["proba_" + f for f in self.cell_class_names]
        )

        self.colnames_df_image = (
            [
            "dataset_name",
            "slide_num",
            "slide_name",
            "slide_path",
            "slide_type",
            "slide_shape",
            "slide_height",
            "slide_width",
            "area_slide",
            "row_tile_size",
            "col_tile_size",
            "n_tiles_row_slide",
            "n_tiles_col_slide",
            "slide_shape_in_tile",
            "n_tiles_slide",
            "slide_size",
            "slide_dtype",
            "slide_min",
            "slide_max",

            "age",
            "pixel_resolution",
            "gender",
            
            "area_tissue_slide",
            "fraction_tissue_slide",

            #parameters
            "model_segmentation_slide", 
            "model_classification_slide",

            #results cells 
            "n_nuclei_in_slide",
            "mean_nuclei_density_slide",
            "std_nuclei_density_slide",
        
            #all cells
            "n_cells_slide",
            "mean_n_cells_per_tile_slide",
            "std_n_cells_per_tile_slide",
            "mean_cell_size_slide",
            "std_cell_size_slide"]
            #cell type A
            + ["n_cells_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["n_cells_{}_proba_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_proba_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]
            
            + ["mean_size_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names] 
            + ["std_size_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]  
        )
            
        self.colnames_df_roi = (
            [
            #parameters 
            "roi_loc",
            "origin_row",
            "origin_col",
            "end_row",
            "end_col",
            "n_tiles_row_roi",
            "n_tiles_col_roi",
            "roi_shape",
            "roi_shape_in_tiles",
            "roi_height",
            "roi_width",
            "roi_border_size",
            "area_roi",
            "area_tissue_roi",
            "area_physiological_part_roi",
            "fraction_tissue_roi",
            "fraction_physiological_part_roi",
            "fraction_tot_tissue_in_roi",
            "fraction_tot_physiological_part_in_roi",

            #results cells 
            "n_nuclei_in_roi",
            "mean_nuclei_density_roi",
            "std_nuclei_density_roi",
            "ratio_nuclei_density_roi_vs_slide",

            #all cells 
            "n_cells_roi",
            "n_cells_roi_w_border",
            "fraction_tot_cells_in_roi",
            "mean_n_cells_per_tile_roi",
            "std_n_cells_per_tile_roi",
            "mean_cell_size_roi",
            "std_cell_size_roi"]

            #cell type A
            + ["n_cells_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["n_cells_{}_roi_w_border".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["n_all_cells_except_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["n_cells_{}_proba_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["n_cells_{}_proba_roi_w_border".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_proba_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["mean_size_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names] 
            + ["std_size_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
            #Comparaison with entire slide 
            + ["fraction_total_{}_in_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
            + ["fraction_total_{}_proba_in_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
        )

        #Usage : 
        #slide_num, origin_row, origin_col,end_row, end_col  = dataset_config.roi_test_tissue_border[0]

        self.limit_row_col_for_slide = dict({
            "006" : dict({"min_row" : 0, "max_row" : 22, "min_col" : 0, "max_col" : 23}),
        })


    def save(self):
        """
        Save the configuration to a JSON file.
        """
        path_to_save = os.path.join(self.dir_config, self.dataset_name, "config.json")
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        super().save(path_to_save)

    # def create_path(self):
    #     """ Create paths related to the class"""

    #     for

