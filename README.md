# DeepCellMap

Algorithms associated with the submission of the paper titled
\"Unraveling Microglial Spatial Organization in the Developing Human Brain with DeepCellMap\" to Nature
Communications. 

Section `Reproductibility IHC guide and validation DeepCellMap` can be used to reproduce paper results on microglia during brain fetal development and to reproduce validation experiments with generated data. Section `How to use DeepCellMap (general case)` can be use in the general case and to reproduce results on fluorescence data. In the coming version of DeepCellMap package, there will be only one section `How to use DeepCellMap` associated with a section `demo`. 

![Alternative text](docs/images/cover_readme.png)


`DeepCellMap` is a Python package containing methods for analyzing the spatial distribution of cells in tissues (histological images and fluorescence images). DeepCellMap consists of various independent modules enabling cell detection and classification, segmentation of anatomical regions, analysis of cell-cell and cell-region couplings, as well as analysis of the distribution of cell populations in clusters and the interactions between clusters of different populations. 

**Summary** 

- [1.Repository content](#Repository-content)
- [2.System Requirements](#system-requirements)
- [3.Installation Guide (with poetry or conda)](#installation-guide)
- [4.Reproductibility guide - IHC microglia developping brain](#Reproductibility-guide)
- [5.How to use DeepCellMap (general case)](#HowtouseDeepCellMap(generalcase))
- [6.Validation DeepCellMap](#Validation-DeepCellMap)
- [7.License](#license)


# 1. Repository content 

This directory contains all the source code needed to reproduce the
results of the paper using different notebooks (doc/src/notebooks). Some
intermediate results have been provided (doc/data) to speed up the
calculation of spatiotemporal statistics on several pre-defined regions
of interest.

It also contains the documentation to use DeepCellMap for other datasets (see demo). 
Note: the package continues to be improved to offer the best possible user experience. 

- [data](./data): 1 folder per datasets (e.g `data/dataset_1/file_1.tiff` )
- [DeepCellMap_V1](./DeepCellMap_V1): contains all the code needed to reproduce DeepCellMap results on IHC-microglia data 
- [DeepCellMap_V2_general_use](./DeepCellMap_V2_general_use): last version of DeepCellMap for easy use with any dataset of fluorescence or ihc images (can be used to reproduce paper results with fluorescence data) and contains the code needed to validate DeepCellMap results on simulated data.
- [demo](./demo): notebook demo to use DeepCellMap on a ROI 
- [docs](./docs): package documentation to use `DeepCellMap_V2_general_use` with a new dataset
- [validation_outputs](./validation_outputs): files related to the validation of the pipeline 
- [outputs](./ouputs): contains DeepCellMap outputs (V1 and V2). The results of `data/dataset_1` are contained in `outputs/dataset_1`.




# 2. System Requirements

## Hardware requirements
`DeepCellMap` package requires only a standard computer with enough RAM to support the in-memory operations. In particular, the calculation of levelsets in the cell-cell and cell-region coupling analysis module can be memory-intensive. An alternative way of reducing computations is to divide the image into smaller regions. 

## Software requirements
### OS Requirements
The package has been developed and tested on the following operating systems:
+ macOS: Big Sur (11), Monterey(12), Sonoma (14.7)
+ Linux: Ubuntu 16.04

For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB || CPU: 4+ cores || 3.3+ GHz/core

### Python Dependencies
`DeepCellMap` mainly depends on the Python scientific stack.

# 3. Installation Guide:

### 1. Clone or copy/past the repository

To get started, you need to clone the repository from GitHub (when available) to your local machine of download the repository from the provided Drive link.

Ensure that Python 3.8 or later is installed on your machine.

You can verify the version by running:

````
python --version
````

If Python is not installed, download it from the official Python website.

## Poetry-based installation
<details>
<summary><strong> Details </strong></summary>

Poetry is a modern dependency management tool for Python, which uses the pyproject.toml file to handle dependencies.

### Step 1: Install Poetry

If you don’t have Poetry installed, you can install it by following the steps described in the official documentation https://python-poetry.org/docs/. 

You may need to add Poetry to your PATH if it doesn’t automatically do so. You can check if Poetry is installed by running:

````
poetry --version
````


### Step 2: Create the Environment and Install Dependencies

Once you have Poetry installed, create the virtual environment and install the project’s dependencies:

```
cd /path/to/your/repository
poetry install
```

This will automatically create a virtual environment, resolve, and install all dependencies listed in the pyproject.toml file.

### Step 3: Activate the Virtual Environment and add packages

You can activate the virtual environment using:


``` 
poetry shell
```

To install missing dependencies, you can use 

``` 
poetry add <package-name>
```


### 4. Run DeepCellMap step-by-step with the notebooks 

To run the Jupyter notebooks with this environment, you will need to install Jupyter and configure the environment as a kernel in your integrated development environment (IDE) for example VSCode. 

</details>


## Conda-based installation

<details>

<summary><strong> Details </strong></summary>
For users who prefer using Conda, follow these steps to create a Conda environment and install dependencies.

### Step 1: Install Conda

If you don't have Conda installed, you can install it by downloading Miniconda or Anaconda from the official website https://docs.anaconda.com/anaconda/install/ 

To verify Conda installation:

```
conda --version
```

### Step 2: Create Conda Environment

Once Conda is installed, create a new environment with the appropriate Python version (e.g., Python 3.10 or higher):

```
conda create --name your-env-name python=3.12
```
Activate the environment:

```
conda activate your-env-name
```

### Step 3: Install Dependencies from requirements.txt

With the environment activated, install the dependencies listed in the requirements.txt file:


```
conda install --file requirements.txt
```

### Step 4: (Optional) Use Conda-forge Channel

If certain dependencies are missing or need to be fetched from the conda-forge channel, you can run:

``` 
conda install --file requirements.txt -c conda-forge
```

Once the dependencies are installed, your Conda environment is set up, and you can now run the project.

### 5. Run DeepCellMap step-by-step with the notebooks 

To run the Jupyter notebooks with this environment, you will need to install Jupyter and configure the environment as a kernel in your integrated development environment (IDE) for example VSCode. 

</details>

# 4.Reproductibility guide - IHC microglia developping brain


## Reproducibility of results on IHC images (microglia in the developing brain). 

The notebooks in this section enable DeepCellMap results to be reproduced on IHC images at 17, 19, and 20 post conceptional weeks (pcw). Intermediate pre-processing results have been provided (pre-processing, cell classification, anatomical region segmentation). The notebooks are all located in the folder  `DeepCellMap_V1/notebooks/IHC_microglia_notebooks` :

- `DeepCellMap_1_DeepCellMap_on_anatomical_regions.ipynb` can be used to visualize anatomical regions, apply DeepCellMap to the various regions and aggregate the results to visualize spatiotemporal results. 

- `DeepCellMap_2_Spatiotemporal_results_visualization.ipynb` uses the results table obtained with the previous notebook to visualize deepcellmap results on cell number and densities per region, cell-cell and cell-region couplings, cell cluster distribution, and cell neighbor analysis. 


# 5. How to use DeepCellMap (general case)

The notebooks described below can be found in `./DeepCellMap_V2_general_use/`. They provide step-by-step guidance for users (with a minimum of Python experience) in using DeepCellMap with any dataset. In addition to the explanations provided in the notebooks, the user can consult the documentation in `./docs`, where the following documents can be found: 

-  `Application of DeepCellMap to a new Dataset.pdf`
- `ConfigFile_explanation_(1_dataset__1_Config_file).pdf` and 
- `DeepCellMap_files_overview_.pdf`

Brief explanation of each notebook: 

1. **`DeepCellMap_1_image_preprocessing.ipynb`** performs image downscaling, tissue mask extraction and tiling 
2. **`DeepCellMap_2_laboratory_Tissue_segmentation.ipynb`** (optional) : helps to find good method to segment tissue of the entire image 
3. **`DeepCellMap_3_laboratory_Cell_segmentation.ipynb`** (optional) :  helps to find good method to segment cells in tissue
4. **`DeepCellMap_4_cell_classification.ipynb`** (optional) : is used to classify cells using a deep learning based model after cell detection and segmentation 
5. **`DeepCellMap_5_application_DeepCellMap.ipynb`** enables the application of DeepCellMap on entire images or ROI in images. 


# 6. Validation DeepCellMap 

This section contains the notebooks used to reproduce DeepCellMap validation experiments. They are all located in the folder  `DeepCellMap_V2_general_use/notebooks_validation`. The validation results can be found in the folder `validation_outputs`.

### Validation of Cell Detection 


- `Validation_cell_detection.ipynb`. The experiment involves randomly selecting regions within the tissue that contain cells and comparing the number of cells detected by DeepCellMap with the number identified by an expert.

The outputs of this part can be found in `./validation_outputs/validation_segmentation_cells/validation_detected_cells`. The files are as follows:
- `image_gallery.png` contains the ROIs;
- `deepcellmap_masks` contains the images of the ROIs and the cell masks obtained with DeepCellMap;
- `validation_celldetection.csv` contains the number of cells detected by the expert vs. DeepCellMap in each ROI, along with the detailed error calculations.

### Validation of Cell segmentation 


- `Validation_cell_segmentation.ipynb`. This experiment involves extracting masks of 130 cells and asking an expert to rate the quality of each mask as “good,” “medium,” or “bad.”

The outputs of this part can be found in `./validation_outputs/validation_segmentation_cells/validation_masks/`. The files are as follows:
- `cells` contains the cells and the overlaid masks;
- `mask_quality_expert.csv` contains the expert's assessment for each mask.

### Validation of levelset-based analysis 

- `Validation_levelset_analysis.ipynb` enables step-by-step generation of spatially coupled cell distributions and application of DeepCellMap to compare simulation parameters with the results provided by the algorithm. 

### Validation of DBSCAN-based analysis 

- `Validation_DBSCAN_analysis.ipynb` allows to go step-by-step in the generation of clustered cell distributions and apply DeepCellMap to compare the results of the algorithm with the parameters of the laws that generated the simulated data. 
