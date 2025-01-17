import os
import json
from config.base_config import BaseConfig
from config.datasets_config import DatasetBaseConfig,PreprocessingConfig

class FluorescenceCovidConfig(DatasetBaseConfig):
    """
    """
    def __init__(self):
        dataset_name = "covid_data_immunofluorescence"
        super().__init__(dataset_name)  # Call the constructor of the parent class
        self.mapping_img_name = ["16204_cd681st_s100b_AcquisitionBlock1_pt1.czi",#not_relevant_for_now 
                                 "16225_cd681st_s100b_AcquisitionBlock1_pt1.czi",# not_relevant_for_now
                                 "HDBR_15548_O4_185_PanLaminin488_CD68568_Iba1647_20x.czi",#control #DONE 
                                 "HDBR_15555_Q4_Panlaminin488_CD68568_Iba1647_20x.czi",#covid #DONE 
                                 "HDBR_15544_U4_186_PanLaminin488_CD68568_Iba1647_20x.czi",#control #DONE 
                                 "HDBR_15532_D20_PanLaminin488_CD68568_edinburgh_20x_section2.czi",#not_relevant_for_now
                                 "HDBR_15538_IHC_186_T5_PanLaminin488_CD68568_Iba1647_20X.czi",#covid
                                 "HDBR_15561_P5_Panlaminin488_CD68568_Iba1647_20x.czi",#covid #DONE 
                                 "HDBR_15659_R6b_Panlaminin488_CD68568_Iba1647_20x",#control #DONE 
                                 "HDBR_15714_R4_186_PanLaminin488_CD68568_Iba1647_20x"#covid
                                 ] #Todo 
        self.mapping_img_gender = None
        self.mapping_img_disease = ["no","not_relevant_for_now",
                                "not_relevant_for_now",
                                "control", #DONE 
                                "covid", #DONE 
                                "control", #DONE 
                                "not_relevant_for_now",
                                "covid",  #DONE
                                "covid", #DONE 
                                "control", #DONE 
                                "covid" # DONE 
                                ]
        self.mapping_img_age = [0,
                                0,
                                13,
                                12,
                                15,
                                0,
                                14,
                                21,
                                14,
                                19,
                                ]
        self.cell_class_names = [
            # "fron_channel_0", #Nuclei -> BLUE
            "cd68", #CD68 phagocytic -> RED 
            "iba1", #Iba1 microglia -> White 
            "blood_vessels",#BLOOD vessels -> Green 
            "iba1_cd68",
        ]
        self.debug_mode = False

        self.data_type = "fluorescence"
        self.consider_image_with_channels = True 
        self.data_format = "czi"
        self.conversion_px_micro_meter = (0.312/1000) # N milimeter per mixeo 


        # C1:DAPI-T1(blue) , C2: AF568-T2(red) , C3:AF647-T3(white), C4(AF488-T4(Green) )
        self.channel_names = ["DAPI-T1","AF568-T2","AF488-T4", "AF647-T3"]
        self.dim_position = dict({"C":0,"X":2,"Y":3,"Z":1})
        self.has_Z = True
        #Tiling parameters
        self.tile_width = 1024
        self.tile_height = 1024
        self.roi_border_size = int(1024/4)
        self.border_size_during_segmentation = int(1024/4)
        self.crop_size = 256
        self.scale_factor = 1
        self.tissue_extraction_accept_holes = True
        # Segmentation parameters
        self.preprocessing_config = PreprocessingConfig(dataset_name = dataset_name,
                                                        scale_factor=self.scale_factor,tissue_extraction_accept_holes = self.tissue_extraction_accept_holes)

        self.tissue_segmentation_param = dict({
            "default" : dict({
                "manual_threshold" : 5,
                "dilation" : 25,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "dilation2" : 20,
                "take_largest_component" : 1}),
            "image_3" : dict({
                "manual_threshold" : 10,
                "dilation2" : 25,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "take_largest_component" : 1}),
            "image_5" : dict({
                "manual_threshold" : 10,
                "dilation2" : 25,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "take_largest_component" : 1}),
            "image_6" : dict({
                "manual_threshold" : 10,
                "dilation2" : 25,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "take_largest_component" : 1,
            }),
            "image_8": dict({
                "manual_threshold" : 10,
                "dilation2" : 30,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "take_largest_component" : 1}),
            "image_9": dict({
                "manual_threshold" : 10,
                "dilation2" : 30,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "take_largest_component" : 1}),
            "image_10" : dict({
                "manual_threshold" : 10,
                "dilation2" : 30,
                "fill_holes" : None,
                "remove_small_objects" : 20000,
                "take_largest_component" : 1})
        })
        self.threshold_tissue = 0.95
        self.save_tissue_segmentation_steps = False #all tissue because small  
        self.channel_used_to_segment_tissue = 0
        # Classification parameters
        self.use_imgs_as_channels = False 
        self.channels_cells_to_segment = [1,2,3,4]
        self.channels_of_interest = [1,2,3,4]
        self.cells_from_multiple_channels = dict({
            "iba1_cd68" : [1,2]
        })
        self.association_cell_name_channel_number = dict({
            "cd68" : 1,
            "iba1" : 2,
            "blood_vessels" : 3,
            "iba1_cd68" : 4
        })

        self.param_best_cellpose = dict({"model_type" : "cyto2","diameter" : 20,"channels" : [[0,0]], "normalisation" : True, "net_avg" : False})
        self.cellpose_parameters = dict(
            {
                "downscale_factor_cellpose_tissue_segmentation": None,  # better : scale_factor_tile_crop_cellpose
                "tile_subdivision_factor" : 16,
                "channel_nuclei" : 0
            }
        )

        #Note : quand 2 cellules de types 1 et 2 constituent une cellule d'un autre type, elles ont "used_to_build_cell_type" = True etune autre ligne du df cell permet d'avoir la cellule mergée 
        self.model_segmentation_name = "image_processing_steps"
        self.cell_segmentation_param = None
        self.cell_segmentation_param_by_cannnel = dict({
            "default": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                },
                1: {
                    "multi_otsu_thresholding" :  [None, 40],
                    "dilation" : 8,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {#Ok pour slide 002, NOT FOR 1 
                    "multi_otsu_thresholding" :  [None, 40],
                    # "otsu_thresholding" :  None,
                    "erosion": 2,
                    "dilation" : 4,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 200,
                    "remove_large_objects" : 10000
                    },
                3: {#Ok pour slide 002, 
                    "multi_otsu_thresholding" :  [None, 40],
                    # "otsu_thresholding" :  None,
                    # "erosion": 1,
                    "dilation" : 2,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 250,
                    "remove_large_objects" : 15000
                    }
                }),
            "image_1": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                },
                1: {
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.15,"multi_otsu":True, "param_erosion_cas_1":2, "param_dil_cas_2":3,"dont_consider_background":True,"min_thresh_second_binarization":10},
                    "dilation2" : 3,
                    "fill_holes" : None,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {
                    # "multi_otsu_thresholding" :  [None, 40],
                    "otsu_thresholding" :  [None,-1],
                    "dilation" : 2,
                    "fill_holes" : None,
                    "remove_small_objects" : 500,
                    "filter_center_cells" : self.tile_width,
                    "remove_large_objects" : 10000
                    },
                3: {#Blood vessel 
                    "multi_otsu_thresholding" :  [None, 100],
                    # "otsu_thresholding" :  None,
                    # "erosion": 1,
                    "dilation" : 2,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 250,
                    "remove_large_objects" : 15000
                    }
                }),
            "image_3": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                },
                1: {
                    "multi_otsu_thresholding" :  [None, 100],
                    "erosion": 2,
                    "dilation" : 4,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.15,"multi_otsu":True, "param_erosion_cas_1":2, "param_dil_cas_2":3,"dont_consider_background":True,"min_thresh_second_binarization":10},
                    "fill_holes" : None,
                    "remove_small_objects" : 200,
                    "filter_center_cells" : self.tile_width,
                    "remove_large_objects" : 10000
                    },
                3: {
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.15,"multi_otsu":False, "param_erosion_cas_1":2, "param_dil_cas_2":4,"dont_consider_background":True,"min_thresh_second_binarization":10},
                    "opening" : 9,
                    "dilation":2,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 400,
                    "remove_large_objects" : 15000
                    }
                }),
            "image_4": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                },
                1: {
                    # "multi_otsu_thresholding" :  [None, 40],
                    # "dilation" : 8,
                    # "filter_center_cells" : self.tile_width,
                    # "remove_small_objects" : 250
                    "otsu_thresholding" :  [None, 40],
                    "erosion":3,
                    "dilation" : 8,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {#Ok pour slide 002, NOT FOR 1 
                    "multi_otsu_thresholding" :  [None, 40],
                    # "otsu_thresholding" :  None,
                    "erosion": 2,
                    "dilation" : 4,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 200,
                    "remove_large_objects" : 10000
                    },
                3: {#Ok pour slide 002, 
                    # "multi_otsu_thresholding" :  [None, 40],
                    # # "otsu_thresholding" :  None,
                    # # "erosion": 1,
                    # "dilation" : 2,
                    # "filter_center_cells" : self.tile_width,
                    # "fill_holes" : None,
                    # "remove_small_objects" : 250,
                    # "remove_large_objects" : 15000
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.15,"multi_otsu":False, "param_erosion_cas_1":3, "param_dil_cas_2":0,"min_thresh_second_binarization":35,"dont_consider_background":True},
                    "opening" : 3,
                    # "erosion":3,
                    "dilation":5,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 400,
                    "remove_large_objects" : 20000
                    },
                }),
            "image_5": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                    },
                1: {
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.15, "multi_otsu":True,"param_erosion_cas_1":2, "param_dil_cas_2":3,"min_thresh_second_binarization":10,"dont_consider_background":True},
                    "dilation" : 4,
                    "fill_holes" : None,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {
                    "multi_otsu_thresholding" :  ["dont_consider_background", 40],
                    "dilation" : 4,
                    "fill_holes" : None,
                    "remove_small_objects" : 110,
                    "filter_center_cells" : self.tile_width,
                    "remove_large_objects" : 10000
                    },
                3: {
                    "otsu_thresholding" :  [None,-1],
                    "dilation" : 2,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 250,
                    "remove_large_objects" : 15000
                    },
                }),
            "image_7": dict({
                0: {
                "cellpose" : self.param_best_cellpose
                    },
                1: {
                    "otsu_thresholding" :  [None, 25],
                    "opening":4,
                    "dilation" : 3,
                    "filter_center_cells" : self.tile_width,
                    "remove_small_objects" : 250
                    },
                2: {
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.10,"multi_otsu":False, "param_erosion_cas_1":2, "param_dil_cas_2":5,"dont_consider_background":True,"min_thresh_second_binarization":10},
                    "dilation" : 1,
                    "fill_holes" : None,
                    "remove_small_objects" : 300,
                    "filter_center_cells" : self.tile_width,
                    "remove_large_objects" : 15000
                    },
                3: {
                    "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.15,"multi_otsu":False, "param_erosion_cas_1":2, "param_dil_cas_2":3,"dont_consider_background":True,"min_thresh_second_binarization":20},
                    "remove_small_objects" : 200,
                    # "opening" : 9,
                    "dilation":2,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 400,
                    "remove_large_objects" : 50000
                    },
                }),
            "image_8": dict({
                    0: {
                    "cellpose" : self.param_best_cellpose
                        },
                    1: {#OKKKKK 
                        "multi_otsu_thresholding" :  [None, 25],
                        "opening":1,
                        "dilation" : 5,
                        "filter_center_cells" : self.tile_width,
                        "remove_small_objects" : 300
                        },
                    2: {#OKKKKK 
                        "test_cas1__cas_2" :  {"dilation_test":4, "thresh_fraction_cells_discovered":0.10,"multi_otsu":False, "param_erosion_cas_1":2, "param_dil_cas_2":5,"dont_consider_background":True,"min_thresh_second_binarization":10},
                        "fill_holes" : None,
                        "remove_small_objects" : 300,
                        "filter_center_cells" : self.tile_width,
                        "remove_large_objects" : 50000
                        },
                    3: {#OKKKKK
                        "otsu_thresholding" :  [None,-1],
                        "dilation" : 3,
                        "filter_center_cells" : self.tile_width,
                        "fill_holes" : None,
                        "remove_small_objects" : 250,
                        "remove_large_objects" : 50000
                    }
                }),
            "image_9": dict({
                    0: {
                    "cellpose" : self.param_best_cellpose
                        },
                    1: {#OKKKKK 
                        "otsu_thresholding" :  [None, 25],
                        "opening":2,
                        "dilation" : 3,
                        "filter_center_cells" : self.tile_width,
                        "remove_small_objects" : 300
                        },
                    2: {#OKKKKK 
                        "otsu_thresholding" :  [None,30],
                        "dilation" : 3,
                        "fill_holes" : None,
                        "remove_small_objects" : 250,
                        "filter_center_cells" : self.tile_width,
                        "remove_large_objects" : 50000
                        },
                    3: {#OKKKKK
                    "otsu_thresholding" :  [None,-1],
                    "dilation" : 3,
                    "filter_center_cells" : self.tile_width,
                    "fill_holes" : None,
                    "remove_small_objects" : 250,
                    "remove_large_objects" : 50000
                    }
                }),
            "image_10": dict({
                    0: {
                    "cellpose" : self.param_best_cellpose
                        },
                    1: { #OK pas ouf (perfectible) mais peut faire le taf 
                        "otsu_thresholding" :  [None, 20],
                        "dilation":4,
                        "opening" : 7,
                        "fill_holes" : None,
                        "filter_center_cells" : self.tile_width,
                        "remove_small_objects" : 400
                        },
                    2: { #Cooool pas trop mal du tout du premier ocup 
                        "otsu_thresholding" :  [None,-1],
                        "dilation" : 2,
                        "fill_holes" : None,
                        "remove_small_objects" : 500,
                        "filter_center_cells" : self.tile_width,
                        "remove_large_objects" : 10000
                        },
                    3: {#OKKKKK
                        "otsu_thresholding" :  [None,-1],
                        "dilation" : 4,
                        "filter_center_cells" : self.tile_width,
                        "fill_holes" : None,
                        "remove_small_objects" : 250,
                        "remove_large_objects" : 50000
                    }
                }),

                })

        self.classification_param = None
        self.cell_class_names_for_classification = None 
        self.tile_test_segmentation = dict({
            "001": [(5,5),(3,4),(3,5),(2,5),(5,3),(4,2),(5,2),(4,7),(5,4)],
            "003" : [(3,10),(4,2),(4,9),(2,2),(4,3), (4,5),(5,5),(5,4),(5,3),(2,5),(4,6),(5,5),(4,4)],
            "004" : [(1,6),(4,1),(2,1),(2,3),(2,7),(1,4),(1,5),(3,2), (4,2), (4,1), (2,6)],
            "005" : [(1,2),(1,3),(2,2),(2,4),(2,7),(3,4),(3,10),(4,11),(3,6)],
            "006" : [(4,1),(1,8),(2,7),(2,1),(2,3),(2,7),(1,4),(1,5),(3,2), (4,2), (4,1), (2,6)],

            "007" : [(1,7),(2,3),(3,1),(3,3),(3,5),(4,1),(4,4),(4,6),(5,5),(6,2),(6,5),(7,4),(7,6)],

            "008" : [(1,4),(7,3),(7,4),(3,2), (5,5),(2,4),(3,4),(6,4),(9,5), (8,3),(9,2)],
            "009" : [(6,2),(8,2),(10,3),(8,3),(6,2),(12,1), (14,3),(15,2), (4,1), (7,1), (10,1)],
            "010" : [(8,5),(11,2),(6,2),(3,4),(8,2),(10,3),(2,2),(8,3),(6,2),(11,1), (3,3),(5,2), (4,1), (7,1), (10,1)],

        })
        self.roi_test_tissue_border = [(2,2,2,3,3)]
        self.roi_cool = [(1,17,35,19,37)]
        #Usage : 
        #slide_num, origin_row, origin_col,end_row, end_col  = dataset_config.roi_test_roi_4_tiles[0]
        

        self.statistics_with_proba = False


        self.path_cells = os.path.join(self.dir_base_classif,"cells_per_images","cells")
        self.physiological_regions_max_square_size = None
        self.physiological_regions_group_for_comparaison = None


        #Colocalisation 
        self.cell_cell_colocalisation_config = dict(
            {   "compute_with_proba" : False,
                "cell_types_A": self.cell_class_names,
                "cell_types_B": self.cell_class_names,
                "levelsets": [0,50,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200],
                "save_images": True,
            }
        )

        self.dbscan_based_analysis_config = dict(
            {
            "min_sample": 3,
            "range_epsilon_to_test" : [100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6300,6600,7000,7300,7600,8000,8500,9000] ,
            "cell_types_A": self.cell_class_names,
            "cell_types_B": self.cell_class_names,
            "config_cluster_robustess_experiment": {
                "n_experiment_of_removing": 100,
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
        #methods 
        # self.create_path()
        self.colnames_table_cells_base = [
            "id_cell",
            "cell_type",
            "channel_number",
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
            ]
        self.colnames_df_image = (
            ["slide_num",
            "slide_shape",
            "area_slide",
            "slide_shape_in_tile",
            "n_tiles_slide",
            "pixel_resolution",
            # "gender",
            
            "area_tissue_slide",
            "fraction_tissue_slide",

            ####parameters
            # "model_segmentation_slide", 
            # "model_classification_slide",

            #####results cells 
            "n_nuclei_in_slide",
            "mean_nuclei_density_slide",
            "std_nuclei_density_slide",
        
            #####all cells
            "n_cells_slide",
            "mean_n_cells_per_tile_slide",
            "std_n_cells_per_tile_slide",
            "mean_cell_size_slide",
            "std_cell_size_slide"]

            #### cell type A
            #cell type A
            + ["n_cells_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_slide".format(cell_type_name) for cell_type_name in self.cell_class_names]

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
            "area_roi",
            "area_tissue_roi",
            # "area_physiological_part_roi",
            "fraction_tissue_roi",
            # "fraction_physiological_part_roi",
            "fraction_tot_tissue_in_roi",
            # "fraction_tot_physiological_part_in_roi",

            #results cells 
            "n_nuclei_in_roi",
            "mean_nuclei_density_roi",
            "std_nuclei_density_roi",
            "ratio_nuclei_density_roi_vs_slide",

            #all cells 
            "n_cells_roi",
            "fraction_tot_cells_in_roi",
            "mean_n_cells_per_tile_roi",
            "std_n_cells_per_tile_roi",
            "mean_cell_size_roi",
            "std_cell_size_roi"]

            #cell type A
            + ["n_cells_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["fraction_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            # + ["n_cells_{}_proba_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            # + ["fraction_{}_proba_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]
            + ["mean_size_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names] 
            + ["std_size_{}_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
            #Comparaison with entire slide 
            + ["fraction_total_{}_in_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
            # + ["fraction_total_{}_proba_in_roi".format(cell_type_name) for cell_type_name in self.cell_class_names]  
        )


    def save(self):
        """
        Save the configuration to a JSON file.
        """
        path_to_save = os.path.join(self.dir_config, self.dataset_name,"config.json")
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        super().save(path_to_save)


