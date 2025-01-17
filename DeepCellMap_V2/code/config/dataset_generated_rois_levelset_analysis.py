import os
import json
from config.base_config import BaseConfig
from config.datasets_config import DatasetBaseConfig, PreprocessingConfig


class GeneratedRoisLevelsetAnalysis(DatasetBaseConfig):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

        self.mapping_img_name = [
            "B5349-H-17pcw",]#38
        self.mapping_img_gender = ["Unknow", "Unknow", "Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow", "Unknow", "Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow", "Unknow", "Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow","Unknow"]
        self.mapping_img_disease = None 
        self.mapping_img_age = ["22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22',"22","22","22",'22','22','22','22','22']
        self.cell_class_names = [
            "A",
            "B",
            "C",
        ]
        self.mapping_cells_colors = {
            "A": [173, 40, 125],
            "B": [79, 157, 214],
            "C": [58, 180, 15],
            "other" : [63, 4, 126]
        }

        self.debug_mode = False
        # Define the dataset-specific directories
        self.data_type = "wsi"
        self.consider_image_with_channels = False 
        self.data_format = "png"
        self.conversion_px_micro_meter = 1

        self.channel_names = None #Fluorescence specific attribute 
        self.dim_position = None #Fluorescence specific attribute 
        self.has_Z = False #Fluorescence specific attribute 
        # Tiling parameters
        self.tile_width = 1024
        self.tile_height = 1024
        self.roi_border_size = 1
        self.border_size_during_segmentation = 0
        self.crop_size = 256
        self.scale_factor = 1
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
                "dilation" : 4,
                "fill_holes" : None,

                # "model_segmentation_type": "otsu_based_binarization_microglia_IHC",
                # "dilation_radius": 2,
                "remove_small_objects": 1200,
                "remove_large_objects" : 100000,
            })
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
            "A",
            "B",
            "C",
            "detected",
        ]
        self.tile_test_segmentation = dict({
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
        # self.cell_cell_colocalisation_config = dict(
        #     {   "compute_with_proba" : True,
        #         "cell_types_A": ["A"],
        #         "cell_types_B": ["B","C"],
        #         "levelsets": [
        #                 0,
        #                 100,
        #                 200,
        #                 300,
        #                 400,
        #                 500,
        #                 600,
        #                 700,
        #                 800,
        #                 900,
        #                 1000,
        #                 1100,
        #                 1200,
        #                 1300,
        #                 1400,
        #                 1500,
        #                 1600,
        #                 1700,
        #                 1800,
        #                 1900,
        #                 2000,
        #                 2100
        #             ],
        #         "save_images": True,
        #     }
        # )
        self.cell_cell_colocalisation_config = dict(
            {   "compute_with_proba" : True,
                "cell_types_A": ["A"],
                "cell_types_B": ["B","C"],
                "levelsets": [
                        0,
                        50,
                        100,
                        150,
                        200,
                        250,
                        300,
                        350,
                        400,
                        450,
                        500,
                        550,
                        600,
                        650,
                        700,
                        750,
                        800,
                        850,
                        900,
                        950,
                        1000],
                "save_images": True,
            }
        )
        self.dbscan_based_analysis_config = dict(
            {
            "min_sample": 4,
            "range_epsilon_to_test" : [100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000] ,
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

    def set_dataset_name(self, dataset_name):
        """
        Dynamically update the dataset name and refresh paths.
        """
        self.dataset_name = dataset_name
        self.update_paths()
        
    # def create_path(self):
    #     """ Create paths related to the class"""

    