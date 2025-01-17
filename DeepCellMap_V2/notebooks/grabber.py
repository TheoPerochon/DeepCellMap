
# Standard library imports
import sys
sys.path.append('../code/') # Add code directory to path for importing
sys.path.append('../../code/') # Add code directory to path for importing
sys.path.append('../../') # Add code directory to path for importing
import os

# Third-party library imports
#Classic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
#Specific


# Project-specific imports
# from utils.util import * # Je dois peut etre spliter le fichier util en plusieurs fichiers selon les cas d'usage 

#Config 
# plt.style.use('seaborn')  # Set a common plotting style
warnings.filterwarnings("ignore")  # Suppress all warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

from simple_colors import red as _red
from simple_colors import blue as _blue
from simple_colors import green as _green
from simple_colors import yellow as _yellow


def red(s):
    print(_red(s))
def blue(s):
    print(_blue(s))
def green(s):
    print(_green(s))
def yellow(s):
    print(_yellow(s))

import ipywidgets as widgets
from IPython.display import display

def on_dropdown_change(change):
    selected_df = change['new']
    # You can use the selected_df variable as the dataframe you want to work with
    # For example, you can assign it to a variable for further processing
    global selected_dataframe
    dataset_name = selected_df

dataframe_dropdown = widgets.Dropdown(
    options={'generated_rois':'generated_rois','ihc_microglia_fetal_human_brain_database': "ihc_microglia_fetal_human_brain_database", 'covid_data_immunofluorescence': "covid_data_immunofluorescence"},
    description='Select dataset:',
    value = "generated_rois",
)

dataframe_dropdown.observe(on_dropdown_change, names='value')
# Customize the style of the description to adjust width
dataframe_dropdown.style.description_width = '150px'  # Change the width as needed
dataframe_dropdown.style.font_size = '300px'  # Change the font size as needed

dataframe_dropdown.style.max_width = '1000px'  # Change the max width of each option