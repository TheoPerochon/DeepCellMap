import numpy as np
import os
from config.html_generation_config import HtmlGenerationConfig


from config.base_config import BaseConfig
from config.datasets_config import *


# base_config = BaseConfig()
# # Get the configuration
# if base_config.dataset_name == "ihc_microglia_fetal_human_brain_database":
#     dataset_config = IhcMicrogliaFetalHumanBrain()
# elif base_config.dataset_name == "cancer_data_immunofluorescence":
#     dataset_config = FluorescenceCancerConfig()
# elif base_config.dataset_name == "covid_data_immunofluorescence":
#     dataset_config = FluorescenceCovidConfig()
# elif base_config.dataset_name == "ihc_pig":
#     dataset_config = PigIhc()
# else : 
#     raise Exception("Dataset config not found")



def html_header(page_title):
  """
  Generate an HTML header for previewing images.

  Returns:
    HTML header for viewing images.
  """
  html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" " + \
         "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n" + \
         "<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">\n" + \
         "  <head>\n" + \
         "    <title>%s</title>\n" % page_title + \
         "    <style type=\"text/css\">\n" + \
         "     img { border: 2px solid black; }\n" + \
         "     td { border: 2px solid black; }\n" + \
         "    </style>\n" + \
         "  </head>\n" + \
         "  <body>\n"
  return html


def html_footer():
  """
  Generate an HTML footer for previewing images.

  Returns:
    HTML footer for viewing images.
  """
  html = "</body>\n" + \
         "</html>\n"
  return html
