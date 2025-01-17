

import os
import json
from config.base_config import BaseConfig
from config.datasets_config import DatasetBaseConfig,PreprocessingConfig

from config.dataset_ihc_microglia_config import IhcMicrogliaFetalHumanBrain
from config.dataset_covid_config import FluorescenceCovidConfig
from config.dataset_generated_rois_levelset_analysis import GeneratedRoisLevelsetAnalysis

def take_config_from_dataset(dataset_name):
    if dataset_name == "ihc_microglia_fetal_human_brain_database":
        dataset_config = IhcMicrogliaFetalHumanBrain()
    elif dataset_name == "covid_data_immunofluorescence":
        dataset_config = FluorescenceCovidConfig()
    elif "Gen_ROI" in dataset_name:
        dataset_config = GeneratedRoisLevelsetAnalysis(dataset_name)
    elif "DBSCAN_" in dataset_name:
        dataset_config = GeneratedRoisLevelsetAnalysis(dataset_name)
    else:
        raise ValueError("Dataset name not recognized")
    return dataset_config

