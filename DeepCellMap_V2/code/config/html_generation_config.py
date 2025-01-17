import os
import json
from config.base_config import BaseConfig

class HtmlGenerationConfig(BaseConfig):

    num_random_tiles_to_display = 10
    summary_tile_text_color = (255, 255, 255)
    label_all_tiles_in_top_tile_summary = False
    border_all_tiles_in_top_tile_summary = True
    tile_border_size = 2
    font_filename = "Arial.ttf"
    font_size = 20
    # Construct the full path to the font file
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), font_filename)

    # font_path = "arial.ttf"
    summary_title_font_path = "Arial.ttf"
    tile_label_text_size = 10

    high_color = (0, 255, 0)
    medium_color = (255, 255, 0)
    low_color = (255, 165, 0)
    none_color = (255, 0, 0)

    faded_thresh_color = (128, 255, 128)
    faded_medium_color = (255, 255, 128)
    faded_low_color = (255, 210, 128)
    faded_none_color = (255, 128, 128)
    
    thumbnail_ext = "png" #pe a mettre dans preprocessing 
    thumbnail_size = 300
    filter_pagination_size = 50
    filter_paginate = True
    tile_summary_paginate = True 

    tile_summary_pagination_size = 50
    summary_title_text_size = 24
    summary_tile_text_color = (255, 255, 255)
    tile_text_color = (0, 0, 0)

    tile_text_size = 36
    tile_text_background_color = (255, 255, 255)

    tile_text_w_border = 5
    tile_text_h_border = 4
    
    display_tile_summary_labels = True
    def print_config(self):
        super().print_config()  # Call the parent class's print_config method
        print(f"Debug Mode: {self.debug_mode}")

