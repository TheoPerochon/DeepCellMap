
from utils.util import *
from preprocessing import filter
from preprocessing import slide
from preprocessing import tiles
from stat_analysis import deep_cell_map
from segmentation_classification import segmentation, classification
from segmentation_classification.classification import ModelClassification, segment_classify_cells_wsi
from segmentation_classification.region_of_interest import Roi
from code.stat_analysis.colocalisation_analysis import ColocAnalysis
from code.stat_analysis.dbscan_analysis import DbscanAnalysis
from code.stat_analysis.neighbours_analysis import NeighborsAnalysis

from config.dataset_management import take_config_from_dataset

def main(dataset_name,include_preprocessing,slide_num,with_displays):
    print("main")
    print("dataset_name", dataset_name)
    print("include_preprocessing", include_preprocessing)
    # print("from_image", from_image)
    # print("to_image", to_image)
    # print("image_list", image_list)
    print("slide_num",slide_num)
    print("with_displays", with_displays)

    # Config selection
    dataset_config = take_config_from_dataset(dataset_name)
    print(blue("We are playing with the dataset: {}".format(dataset_config.dataset_name)))
    image_list = [slide_num] #np
    if include_preprocessing:

        #Pre-processing 
        #Info dataset
        df = slide.get_info_data(dataset_config, image_list=image_list)
        # df = slide.get_info_data(from_image, to_image, dataset_config)

        # Dataset-splitting (why?)
        # slide.split_img_into_channels_img_range(from_image, to_image,dataset_config)

        #Downscalling 
        slide.training_slide_range_to_images(dataset_config,image_list=image_list)    
        # slide.training_slide_range_to_images(from_image, to_image,dataset_config)

        #Filtering
        segmentation.multiprocess_apply_filters_to_images(save=True, display=False, html=False, image_num_list=image_list,dataset_config=dataset_config)
        # filter.filtering_slide_range_to_images(from_image, to_image,dataset_config)

        #Tiling
        tiles.multiprocess_filtered_images_to_tiles(image_num_list = image_list,dataset_config=dataset_config)
        # tiles.multiprocess_filtered_images_to_tiles(image_num_list = image_list,dataset_config=dataset_config)
        # tiles.tiles_slide_range_to_images(from_image, to_image,dataset_config)

        #Cell classification 
        particular_roi = None #(2,2,3,3)
        channels_cells_to_segment = None
        for num_slide in image_list:
            print(blue("Slide {}".format(num_slide)))
        table_cells = classification.segment_classify_cells(slide_num, dataset_config,particular_roi=particular_roi, channels_cells_to_segment = channels_cells_to_segment)
        table_cells = classification.segment_cells_coming_from_several_channels(slide_num, dataset_config, particular_roi=None, cells_from_multiple_channels = None)
        df = slide.get_statistics_images(dataset_config=dataset_config,image_list = image_list)
    
    
    # Roi definition
    slide_num =slide_num
    origin_row = 2
    origin_col = 2
    end_row = 5
    end_col = 6
    channels_of_interest=None
    entire_image = True 

    roi = Roi(dataset_config,slide_num, origin_row, origin_col, end_row, end_col,entire_image=entire_image)
    print(blue("Roi shape"),roi.roi_shape,blue("Roi with borders shape :"), roi.roi_w_borders_shape)
    #Get tissue mask 
    roi.get_mask_tissue()

    #Visualisation RGB et channels 
    roi.get_img_original(channels_of_interest = "all_as_rgb")
    roi.display_save_img_original(save = True)

    # #Get table cells 
    roi.get_table_cells(filter_cells_from_several_channels=True)

    # #Get mask regions 
    roi.get_cell_masks()

    # #get predictions and masks 
    roi.display_segmented_cells(save_fig = True)

    # #get roi statistics 
    roi.get_statistics_roi(save = True)

    #Colocalisation analysis
    coloc_analysis = ColocAnalysis(roi)
    coloc_analysis.visualise_z_score()
    coloc_analysis.heatmap_colocalisation(feature_name= "association_score", proba = False)
    coloc_analysis.heatmap_colocalisation(feature_name= "distance_association", proba = False)
    coloc_analysis.display_B_in_A_levelsets() if with_displays else None 

    #Apply DBSCAN 
    dbscan_analysis = DbscanAnalysis(roi,min_sample=5)

    #Apply Neighbors analysis 
    neighbors_analysis = NeighborsAnalysis(roi)
    if with_displays:
        coloc_analysis.display_B_in_A_levelsets()

def none_or_int(value):
    if value.lower() == 'none':
        return None
    else:
        return int(value)
    
def parse_list_or_none(value):
    if value.lower() == 'none':
        return None
    else:
        return value.split(',')
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="dataset_name")
    parser.add_argument("--preprocess", action="store_true", help="include_preprocessing",default = "false")
    parser.add_argument("--image_num", type=int, help="slide_num")
    # parser.add_argument("--image_list", action=parse_list_or_none)

    # parser.add_argument("--from_image", type=none_or_int, help="from_image")
    # parser.add_argument("--to_image", type=none_or_int, help="to_image")
    parser.add_argument("--display", action="store_true", help="with_displays")

    args = parser.parse_args()
    print("info sur args", args)
    main(args.dataset_name, args.preprocess, args.image_num, args.display)
    #python main.py --dataset_name covid_data_immunofluorescence  --image_num 3 --display
    #python main.py --dataset_name covid_data_immunofluorescence --preprocess --from_image 1 --to_image 1 --display