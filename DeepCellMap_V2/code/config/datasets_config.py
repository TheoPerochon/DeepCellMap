import os
import json
from config.base_config import BaseConfig
# drive :  https://unioxfordnexus-my.sharepoint.com/personal/wolf2242_ox_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwolf2242%5Fox%5Fac%5Fuk%2FDocuments%2FMarco%2DKatie%2DScans%2DNatureNeuro&fromShare=true&ga=1



class DatasetBaseConfig(BaseConfig):
    FLUOROPHORE_MAPPING_RGB = dict({'DAPI':1, 'AF647':2 , 'AF488' : 0, 'AF568' : 3})

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.preprocessing_tasks = ["tissue_extraction", "tiling"]
        self.setup_directories()

    def setup_directories(self):
        self.dir_base_classif = os.path.join(self.dir_output_dataset, "2_classification_cells")
        self.dir_classified_img = os.path.join(self.dir_base_classif, "classified_slides")
        self.dir_training_set_cells = os.path.join(self.dir_base_classif, "training_set_cells")
        self.dir_models = os.path.join(self.dir_base_classif, "models")
        self.dir_cells_to_be_classified = os.path.join(self.dir_training_set_cells, "cells_to_be_classified")
        self.dir_base_anatomical_region_segmentation = os.path.join(self.dir_output_dataset, "3_anatomical_region_segmentation")
        self.dir_base_roi = os.path.join(self.dir_output_dataset, "4_region_of_interest")
        self.dir_base_stat_analysis = os.path.join(self.dir_output_dataset, "5_statistical_analysis")

    def create_path(self):
        for key, value in self.__dict__.items():
            if "dir" in key:
                os.makedirs(value, exist_ok=True)


# class DatasetBaseConfig(BaseConfig):

#     def __init__(self, dataset_name):
#         super().__init__(dataset_name)  # Call the constructor of the parent class
#         self.debug_mode = False
#         self.preprocessing_tasks = ["tissue_extraction","tiling"]
#         self.dir_base_classif = os.path.join(self.dir_output_dataset,"2_classification_cells")
#         self.dir_classified_img = os.path.join(self.dir_base_classif,"classified_slides")
#         # self.dir_cells_per_imgs = os.path.join(self.dir_base_classif,"cells_per_images")
#         self.dir_training_set_cells = os.path.join(self.dir_base_classif,"training_set_cells")
#         self.dir_models = os.path.join(self.dir_base_classif,"models")
#         self.dir_cells_to_be_classified = os.path.join(self.dir_training_set_cells,"cells_to_be_classified")
#         self.dir_base_anatomical_region_segmentation = os.path.join(self.dir_output_dataset,"3_anatomical_region_segmentation")
#         self.dir_base_roi = os.path.join(self.dir_output_dataset,"4_region_of_interest")
#         self.dir_base_stat_analysis = os.path.join(self.dir_output_dataset,"5_statistical_analysis")

#     def create_path(self):
#         for key, value in self.__dict__.items():
#             if "dir" in key:
#                 os.makedirs(value, exist_ok=True)

class PreprocessingConfig(BaseConfig): 
    """
    Preprocessing configuration."""
    def __init__(self,dataset_name, scale_factor=32,tissue_extraction_accept_holes = True):
        super().__init__(dataset_name)  # Call the constructor of the parent class
        self.scale_factor = scale_factor
        # self.split_channel = split_channel
        self.tissue_extraction_accept_holes = tissue_extraction_accept_holes
        self.preprocessing_path = self.create_path_preprocessing()
        
    #Those are the default values for the preprocessing parameters
    # I can put them in the constructor if they change 
    dest_train_ext = "png"
    thumbnail_ext = "png"
    filter_result_text = "filtered"
    filter_suffix = ""
    tile_suffix = "tile"
    tile_summary_suffix = "tile_summary"

    tissue_high_thresh = 80
    tissue_low_thresh = 10

    def create_path_preprocessing(self):
        dir_base = os.path.join(self.dir_output_dataset,"1_image_pre_processing")
        preprocessing_path = dict({
            "dir_base": dir_base,
            "dir_downscaled_img" : os.path.join(dir_base,"downscaled_images","original"),
            "dir_thumbnail_original" : os.path.join(dir_base,"downscaled_images","thumbnail_original"),
            "dir_downscaled_filtered_img" : os.path.join(dir_base,"downscaled_images","filtered_original"),
            "dir_filtered_thumbnail_img" : os.path.join(dir_base,"downscaled_images","thumbnail_filtered"),
            # "dir_downscaled_tiles" : os.path.join(dir_base,"downscaled_images","tiles"),
            # "dir_thumbnail_tiles" : os.path.join(dir_base,"downscaled_images","thumbnail_tiles"),
            "dir_tiles_original" : os.path.join(dir_base,"downscaled_images","tiles_original"),
            "dir_tiles_thumbnail_original" : os.path.join(dir_base,"downscaled_images","tiles_thumbnail_original"),
            # "dir_tiles_filtered" : os.path.join(dir_base,"downscaled_images","tiles_filtered"),
            # "dir_tiles_thumbnail_filtered" : os.path.join(dir_base,"downscaled_images","tiles_thumbnail_filtered"),
            # "dir_random_tiles_original" : os.path.join(dir_base,"downscaled_images","random_tiles_original"),
            # "dir_random_tiles_thumbnail_original" : os.path.join(dir_base,"downscaled_images","random_tiles_original"),
            "dir_random_tiles_filtered" : os.path.join(dir_base,"downscaled_images","random_tiles_filtered"),
            "dir_random_tiles_thumbnail_filtered" : os.path.join(dir_base,"downscaled_images","thumbnail_random_tiles_filtered"),
            "dir_stats" : os.path.join(dir_base,"stats"),
            "dir_tiles" : os.path.join(dir_base,"tiles"),
            "dir_tiles_summary" : os.path.join(dir_base,"tiles_summary"),
        })
        for key in preprocessing_path.keys():
            os.makedirs(preprocessing_path[key], exist_ok=True)
        return preprocessing_path

#Neutral example 
class NewIhc(DatasetBaseConfig):
    def __init__(self):
        super().__init__()  # Call the constructor of the parent class
        self.debug_mode = False

        # Define the dataset-specific directories
        self.dataset_name = "new_ihc"

        self.data_type = "ihc"
        self.data_format = "czi"
        self.conversion_px_micro_meter = None

        self.channel_names = None
        self.channel_order = None

        #Tiling parameters
        self.tile_width = 1024
        self.tile_height = 1024
        self.roi_border_size = 1024
        self.crop_size = 256

        # Segmentation parameters
        self.preprocessing_config = PreprocessingConfig(split_channel = False, 
                                                        scale_factor=32)
        
        self.tissue_segmentation_param = dict({
            "manual_threshold" : 11,
            "dilation" : 4,
            "fill_holes" : None,
            "remove_small_objects" : 300
        })
        self.save_tissue_segmentation_steps = True

        # Classification parameters 
        self.tile_test_segmentation = dict({
            "1": {},
        })

        #Objects specific parameters 
        self.cells_label_names = None
        self.cell_segmentation_param = None
        self.classification_param = None

        # Define the dataset-specific physiological_regions
        self.cellpose_parameters = None
        self.physiological_regions_max_square_size = None
        self.physiological_regions_group_for_comparaison = None
        #Image specific parameters 
        self.mapping_img_number = None
        self.mapping_img_name = []

        #methods 
        self.create_path()

    
    def save(self):
        """
        Save the configuration to a JSON file.
        """
        path_to_save = os.path.join(self.dir_config, self.dataset_name,"config.json")
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        super().save(path_to_save)




