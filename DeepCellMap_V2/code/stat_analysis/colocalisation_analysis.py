# This python file is supposed to be a generalisation of the file colocalisation_analysis that used to work with the region_of_interest.py file. Now that I have write region_of_interest.py that generalise region_of_interest.py so that it works with wsi (everything is in dataset_config files),  can you write a generalised file that would work for a Roi object ?
#
# I think that the best way to do it is to create a new class that would be a generalisation of the class Roi. This class would be called RoiBase. It would have the same attributes as Roi but it would be more general. It would have a method called get_mask that would return the mask of the roi. This method would be implemented in the class Roi.

# Importation des librairies
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pandas as pd
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
import plotly.express as px

from utils.util_colors_drawing import *
from utils.util import *
from simple_colors import *
from math import isnan

VERBOSE = True

def create_path_colocalisation(Roi,levelset):
    """ """
    filename = "levelsets_"+"_".join(map(str,levelset))
    
    path_folder_colocalisation = os.path.join(
        Roi.path_roi, "2_cell_cell_colocalisation"
    )
    path_levelsets = os.path.join(
        path_folder_colocalisation, filename
    )
    if not os.path.exists(path_folder_colocalisation):
        os.mkdir(path_folder_colocalisation)
    if not os.path.exists(path_levelsets):
        os.mkdir(path_levelsets)
    return path_folder_colocalisation, path_levelsets


class ColocAnalysis:
    """
    A class for computing colocalisation between two cell types in a Roi object.

    Attributes:
    - roi: Roi object containing the cells to analyze
    - dataset_config: configuration object for the dataset
    - cell_types_A: list of cell types to analyze for colocalisation
    - cell_types_B: list of cell types to analyze for colocalisation
    - levelsets: list of levelsets to use for colocalisation analysis (start at 0) 
    - path_folder_colocalisation: path to the folder where colocalisation results will be saved
    - path_levelsets: path to the folder where levelsets will be saved
    - verbose: verbosity level (0 for no output, 1 for some output, 2 for detailed output)

    Methods:
    - compute_colocalisation_VG(): computes the colocalisation between two cell types in a Roi object
    - no_colocalisation_with_A(colocalisation_levelsets_scale, type_A, cell_types_B): handles the case where there are no cells of type A in the roi
    - no_colocalisation_with_B(colocalisation_levelsets_scale, type_A, type_B): handles the case where there are no cells of type B in the roi
    - no_colocalisation_when_same_type(colocalisation_levelsets_scale, type_A): handles the case where type A and type B are the same
    """
    colnames_colocalisation_levelsets_scale = [
        'type_A', 
        'type_B',
        "min_levelset",
        "max_levelset",
        "w_number",
        "m_a",
        "eta_t",
        "k",
        "mt_a",
        "sigma_a",
        "threshold",
        "eta_s",
        "eta_a",
        "z_score",
        "p_value",
        "distance_association",
        "proba"
        ]
    colnames_colocalisation = [
        'type_A', 
        'type_B',
        'accumulation',
        'significant_accumulation',
        'association_score',
        "distance_association",
        "p_value",
        "proba"
        ]
    def __init__(self, Roi, levelsets=None, cell_types_A=None, cell_types_B=None, compute_with_proba=None,P = None, verbose=0):
        
        levelsets = Roi.dataset_config.cell_cell_colocalisation_config["levelsets"] if levelsets is None else levelsets
        cell_types_A = Roi.dataset_config.cell_cell_colocalisation_config["cell_types_A"] if cell_types_A is None else cell_types_A
        cell_types_B = Roi.dataset_config.cell_cell_colocalisation_config["cell_types_B"] if cell_types_B is None else cell_types_B
        compute_with_proba = Roi.dataset_config.cell_cell_colocalisation_config["compute_with_proba"] if compute_with_proba is None else compute_with_proba

        self.roi = Roi
        self.dataset_config = Roi.dataset_config
        self.cell_types_A = cell_types_A
        self.cell_types_B = cell_types_B
        self.levelsets = levelsets
        self.statistical_threshold = np.sqrt(2*np.log(len(levelsets)))
        self.compute_with_proba = compute_with_proba
        self.path_folder_colocalisation = create_path_colocalisation(Roi,levelsets)[0]
        self.path_levelsets = create_path_colocalisation(Roi,levelsets)[1]
        self.verbose = verbose
        self.colocalisation_levelsets_scale, self.colocalisation_results = self.compute_colocalisation_VG()
        # print(green("Colocalisation computed"))
        if compute_with_proba : 
            self.colocalisation_levelsets_scale, self.colocalisation_results = self.compute_corrected_association_score(P)

    def compute_colocalisation_VG(self):
        """
        Compute the colocalisation between two cell types in a Roi object
        """
        verbose = False
        # print(blue("Statistical analysis : Colocalisation image {} and levelsets {}...".format(self.roi.slide_num,self.levelsets)))

        colocalisation_levelsets_scale = pd.DataFrame(columns=self.colnames_colocalisation_levelsets_scale)
        colocalisation_results = pd.DataFrame(columns=self.colnames_colocalisation)

        for type_A in self.cell_types_A:
            if self.roi.get_cell_number_from_type(type_A) == 0:
                colocalisation_levelsets_scale = self.no_colocalisation_with_A(colocalisation_levelsets_scale, type_A=type_A,cell_types_B=self.cell_types_B)
                continue
            if not self.roi.at_least_one_B_cell(self.cell_types_B):
                colocalisation_levelsets_scale = self.no_colocalisation_because_no_B_at_all(colocalisation_levelsets_scale, type_A=type_A,cell_types_B=self.cell_types_B)
                continue
            mask_A_w_border = self.roi.get_mask_cell_type(type_A)

            distance_map_from_A_w_border = distance_transform_edt(1 - mask_A_w_border,sampling=1) 

            m_a, sigma_a_numerator = self._compute_M_Sigma(distance_map_from_A_w_border) # Calcul les sommes des aires 
            for type_B in self.cell_types_B:
                if self.roi.get_cell_number_from_type(type_B) == 0:
                    colocalisation_levelsets_scale = self.no_colocalisation_with_B(colocalisation_levelsets_scale, type_A=type_A,type_B=type_B)
                    continue
                if type_A == type_B:
                    colocalisation_levelsets_scale = self.no_colocalisation_when_same_type(colocalisation_levelsets_scale, type_A=type_A)
                    continue
                print("Compute colocalisation between", type_A, "and", type_B) if verbose else None
                #There is A and there is B, let's play

                df_coloc, df_coloc_proba, df_coloc_results, df_coloc_results_proba = self.get_colocalisation_A_B(type_A,type_B,distance_map_from_A_w_border, m_a, sigma_a_numerator)
                
                colocalisation_levelsets_scale = pd.concat([colocalisation_levelsets_scale, df_coloc], ignore_index=True)
                colocalisation_results = pd.concat([colocalisation_results, df_coloc_results], ignore_index=True)

                if self.compute_with_proba :
                    colocalisation_levelsets_scale = pd.concat([colocalisation_levelsets_scale, df_coloc_proba], ignore_index=True)
                    colocalisation_results = pd.concat([colocalisation_results, df_coloc_results_proba], ignore_index=True)
        colocalisation_levelsets_scale.to_csv(os.path.join(self.path_levelsets, "colocalisation_levelsets_scale.csv"),sep = ";")
        colocalisation_results.to_csv(os.path.join(self.path_levelsets, "colocalisation.csv"),sep = ";",index = False)

        # print("Colocalisation computed and saved in ", self.path_levelsets)
        return colocalisation_levelsets_scale, colocalisation_results

    def _compute_corrected_dist_association(self,type_B,table_cells_B ,eta_t_corrected, eta_a_corrected):
        # Get distance map A to compute distance association corrected 
        mask_A_w_border = self.roi.get_mask_cell_type("A")
        distance_map_from_A_w_border = distance_transform_edt(1 - mask_A_w_border,sampling=1) 
        table_cells_B['min_dist_from_A'] = table_cells_B.apply(lambda x: distance_map_from_A_w_border[x['x_roi_w_borders'], x['y_roi_w_borders']], axis=1)
        table_cells_B["is_B"] = (table_cells_B["cell_type"] == self.dataset_config.cell_class_names.index(type_B)+1)
        list_theta_u_proba = []
        for idx, cell in table_cells_B.iterrows():
            w_ij = self._find_wi_cell_B(cell["min_dist_from_A"], self.levelsets)
            if w_ij == len(self.levelsets):
                list_theta_u_proba.append(0)
                continue
            eta_t_ij_proba = eta_t_corrected.iloc[w_ij] 
            eta_a_ij_proba = eta_a_corrected.iloc[w_ij]
            
            theta_u_proba = eta_a_ij_proba/eta_t_ij_proba if eta_t_ij_proba>0 else 0
            list_theta_u_proba.append(theta_u_proba)

        theta_u_proba_array = np.asarray(list_theta_u_proba)
        distance_association_proba_corrected = np.sum(theta_u_proba_array*table_cells_B['min_dist_from_A'])/np.sum(theta_u_proba_array) if np.sum(theta_u_proba_array)>0 else 0
        return distance_association_proba_corrected

    def compute_corrected_association_score_att(self,P): 


        P_inv = np.linalg.inv(P)
        # print("P_inv :" , P_inv)
        colocalisation_levelsets_scale_corrected = self.colocalisation_levelsets_scale.copy()
        colocalisation_results_corrected = self.colocalisation_results.copy()

        colocalisation_levelsets_scale_corrected["eta_t_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["k_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["eta_s_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["eta_a_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["mt_a_corrected"] = np.nan
        colocalisation_levelsets_scale_corrected["sigma_a_corrected"] = np.nan
        colocalisation_levelsets_scale_corrected["z_score_corrected"] = np.nan


        n_B_proba = self.roi.get_cell_number_from_type("B",with_proba=True)
        n_C_proba = self.roi.get_cell_number_from_type("C",with_proba=True)

        n_B_proba_corrected = n_B_proba*P_inv[0,0]+n_C_proba*P_inv[0,1]
        n_C_proba_corrected = n_B_proba*P_inv[1,0]+n_C_proba*P_inv[1,1]
        
        colocalisation_levelsets_scale_corrected_proba_type_B = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"]) ]
        # print("colocalisation_levelsets_scale_corrected_proba_type_B.shape = ",colocalisation_levelsets_scale_corrected_proba_type_B.shape)
        n_cells_type_B = np.sum(colocalisation_levelsets_scale_corrected_proba_type_B["eta_t"])
        colocalisation_levelsets_scale_corrected_proba_type_C = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"]) ]
        # print("colocalisation_levelsets_scale_corrected_proba_type_B.shape = ",colocalisation_levelsets_scale_corrected_proba_type_C.shape)
        n_cells_type_C = np.sum(colocalisation_levelsets_scale_corrected_proba_type_C["eta_t"])


        omega = self.roi.get_feature_from_statistic("area_tissue_roi")

        #!Compute n cells B and C corrected 
        table_cells_B_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index("B")+1]
        table_cells_B = table_cells_B_w_borders[table_cells_B_w_borders["within_roi"]==True]
        n_B = self.roi.get_cell_number_from_type("B")
        n_B_proba = self.roi.get_cell_number_from_type("B",with_proba=True) 
        n_B_proba_square = np.sum(table_cells_B["proba_{}".format("B")]**2) 
        
        table_cells_C_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index("C")+1]
        table_cells_C = table_cells_C_w_borders[table_cells_C_w_borders["within_roi"]==True]
        n_C = self.roi.get_cell_number_from_type("C")
        n_C_proba = self.roi.get_cell_number_from_type("C",with_proba=True) 
        n_C_proba_square = np.sum(table_cells_C["proba_{}".format("C")]**2) 

        for levelset_number in colocalisation_levelsets_scale_corrected.w_number.unique():
            sub_df_proba_wi = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)&(colocalisation_levelsets_scale_corrected["proba"])]
            idx_B = sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)]["eta_a"].index[0]
            idx_C = sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)]["eta_a"].index[0]
            
            eta_t_B = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["eta_t"].values[0]
            eta_t_C = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["eta_t"].values[0]

            k_B = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["k"].values[0]
            k_C = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["k"].values[0]


            m_a_B = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["m_a"].values[0]
            m_a_C = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["m_a"].values[0]

            threshold = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["threshold"].values[0]
            
            eta_s_B = (k_B > threshold)*eta_t_B
            eta_s_C = (k_C > threshold)*eta_t_C
            
            eta_a_B = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["eta_a"].values[0]
            eta_a_C = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["eta_a"].values[0]

            # print("index b size = ",len(sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)]["eta_a"].index))
            # print("index B: ",sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)].index)
            # print("index C: ",sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)].index)

            eta_t_corrected_B = eta_t_B*P_inv[0,0]+eta_t_C*P_inv[0,1]
            eta_t_corrected_C = eta_t_B*P_inv[1,0]+eta_t_C*P_inv[1,1]

            colocalisation_levelsets_scale_corrected.loc[idx_B,"eta_t_corrected"] = eta_t_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"eta_t_corrected"] = eta_t_corrected_C

            k_corrected_B = eta_t_corrected_B*omega/n_cells_type_B
            k_corrected_C = eta_t_corrected_C*omega/n_cells_type_C
            colocalisation_levelsets_scale_corrected.loc[idx_B,"k_corrected"] = k_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"k_corrected"] = k_corrected_C

            mt_a_B = m_a_B*n_cells_type_B/omega
            mt_a_C = m_a_C*n_cells_type_C/omega


            colocalisation_levelsets_scale_corrected.loc[idx_B,"mt_a_corrected"] = m_a_B*n_cells_type_B/omega
            colocalisation_levelsets_scale_corrected.loc[idx_C,"mt_a_corrected"] = m_a_C*n_cells_type_C/omega

            #! Computational way of recovering the sigma numerator since it has already been computed but not saved 
            sigma_a_B_numerator = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["sigma_a"].values[0]*np.sqrt(n_B_proba)/np.sqrt(n_B_proba_square)
            sigma_a_C_numerator = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["sigma_a"].values[0]*np.sqrt(n_C_proba)/np.sqrt(n_C_proba_square)

            sigma_a_corrected_B = sigma_a_B_numerator/np.sqrt(n_cells_type_B)
            sigma_a_corrected_C = sigma_a_C_numerator/np.sqrt(n_cells_type_C)
            colocalisation_levelsets_scale_corrected.loc[idx_B,"sigma_a_corrected"] = sigma_a_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"sigma_a_corrected"] = sigma_a_corrected_C

            threshold_B = m_a_B + self.statistical_threshold*sigma_a_corrected_B
            threshold_C = m_a_C + self.statistical_threshold*sigma_a_corrected_C
            indicatrice_significant_accumulation_B = (k_corrected_B > threshold_B).astype(bool)
            indicatrice_significant_accumulation_C = (k_corrected_C > threshold_C).astype(bool)

            colocalisation_levelsets_scale_corrected.loc[idx_B,"eta_s_corrected"] = indicatrice_significant_accumulation_B*eta_t_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"eta_s_corrected"] = indicatrice_significant_accumulation_C*eta_t_corrected_C

            colocalisation_levelsets_scale_corrected.loc[idx_B,"eta_a_corrected"] = indicatrice_significant_accumulation_B * (eta_t_corrected_B - mt_a_B)
            colocalisation_levelsets_scale_corrected.loc[idx_C,"eta_a_corrected"] = indicatrice_significant_accumulation_C * (eta_t_corrected_C - mt_a_C)

            z_score_B = (k_corrected_B - m_a_B)/sigma_a_corrected_B
            z_score_C = (k_corrected_C - m_a_C)/sigma_a_corrected_C
            colocalisation_levelsets_scale_corrected.loc[idx_B,"z_score_corrected"] = z_score_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"z_score_corrected"] = z_score_C

        colocalisation_results_corrected["accumulation_score_corrected"] = np.nan 
        colocalisation_results_corrected["significant_accumulation_score_corrected"] = np.nan 
        colocalisation_results_corrected["association_score_corrected"] = np.nan 
        colocalisation_results_corrected["distance_association_corrected"] = np.nan 
        colocalisation_results_corrected["p_value_corrected"] = np.nan 

        eta_t_corrected_B = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_t_corrected"]
        eta_t_corrected_C = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_t_corrected"]
        eta_a_corrected_B = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_a_corrected"]
        eta_a_corrected_C = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_a_corrected"]
        dist_coloc_corrected_B = self._compute_corrected_dist_association("B",table_cells_B ,eta_t_corrected_B, eta_a_corrected_B)
        dist_coloc_corrected_C = self._compute_corrected_dist_association("C",table_cells_C ,eta_t_corrected_C, eta_a_corrected_C)

        for type_2 in ["B", "C"]:
            if type_2 == "B":
                n_cells_proba = n_cells_type_B
                dist_coloc_corrected = dist_coloc_corrected_B
            else :
                n_cells_proba = n_cells_type_C
                dist_coloc_corrected = dist_coloc_corrected_C

            n_cells_proba = self.roi.get_cell_number_from_type(type_2,with_proba=True) 
            colocalisation_levelsets_scale_corrected_proba_type_2 = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]==type_2)&(colocalisation_levelsets_scale_corrected["proba"]) ]
            # print(f"Type = {type_2} and sape is {colocalisation_levelsets_scale_corrected_proba_type_2.shape}")
            accumulation_score_corrected = np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_t_corrected"][:-1])/n_cells_proba
            significant_accumulation_score_corrected = np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_s_corrected"][:-1])/n_cells_proba
            association_score_corrected = np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_a_corrected"][:-1])/n_cells_proba
            idx_row_results = colocalisation_results_corrected[colocalisation_results_corrected["proba"]&(colocalisation_results_corrected["type_B"]==type_2)].index[0]
            # print("This index len sshould be 1 : ",len(colocalisation_results_corrected[colocalisation_results_corrected["proba"]&(colocalisation_results_corrected["type_B"]==type_2)].index))
            colocalisation_results_corrected.loc[idx_row_results,"accumulation_score_corrected"]=accumulation_score_corrected

            colocalisation_results_corrected.loc[idx_row_results,"significant_accumulation_score_corrected"]=significant_accumulation_score_corrected

            colocalisation_results_corrected.loc[idx_row_results,"association_score_corrected"]=association_score_corrected
            colocalisation_results_corrected.loc[idx_row_results,"distance_association_corrected"]=dist_coloc_corrected
            colocalisation_results_corrected.loc[idx_row_results,"p_value_corrected"]=1 - (norm.cdf(max(colocalisation_levelsets_scale_corrected_proba_type_2["z_score_corrected"][:-1]), loc=0, scale=1))**len(self.levelsets)


            # print("Type 2 : ",type_2, " n_cells_proba : ", n_cells_proba, "coloc score ",np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_a_corrected"][:-1])/n_cells_proba)

        colocalisation_levelsets_scale_corrected.to_csv(os.path.join(self.path_levelsets, "colocalisation_levelsets_scale.csv"),sep = ";")
        colocalisation_results_corrected.to_csv(os.path.join(self.path_levelsets, "colocalisation.csv"),sep = ";",index = False)
        return colocalisation_levelsets_scale_corrected, colocalisation_results_corrected


    def compute_corrected_association_score(self,P): 

        P_inv = np.linalg.inv(P)
        # print("P_inv :" , P_inv)
        colocalisation_levelsets_scale_corrected = self.colocalisation_levelsets_scale.copy()
        colocalisation_results_corrected = self.colocalisation_results.copy()

        colocalisation_levelsets_scale_corrected["eta_t_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["k_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["eta_s_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["eta_a_corrected"] = np.nan 
        colocalisation_levelsets_scale_corrected["mt_a_corrected"] = np.nan
        colocalisation_levelsets_scale_corrected["sigma_a_corrected"] = np.nan
        colocalisation_levelsets_scale_corrected["z_score_corrected"] = np.nan

        # n_B_proba = self.roi.get_cell_number_from_type("B",with_proba=True)
        # n_C_proba = self.roi.get_cell_number_from_type("C",with_proba=True)

        # n_B_proba_corrected = n_B_proba*P_inv[0,0]+n_C_proba*P_inv[0,1]
        # n_C_proba_corrected = n_B_proba*P_inv[1,0]+n_C_proba*P_inv[1,1]
        
        colocalisation_levelsets_scale_corrected_proba_type_B = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"]) ]
        # print("colocalisation_levelsets_scale_corrected_proba_type_B.shape = ",colocalisation_levelsets_scale_corrected_proba_type_B.shape)
        n_cells_type_B = np.sum(colocalisation_levelsets_scale_corrected_proba_type_B["eta_t"])
        # print("n_cells_type_B",n_cells_type_B)
        colocalisation_levelsets_scale_corrected_proba_type_C = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"]) ]
        # print("colocalisation_levelsets_scale_corrected_proba_type_B.shape = ",colocalisation_levelsets_scale_corrected_proba_type_C.shape)
        n_cells_type_C = np.sum(colocalisation_levelsets_scale_corrected_proba_type_C["eta_t"])
        # print("n_cells_type_B",n_cells_type_C)    


        omega = self.roi.get_feature_from_statistic("area_tissue_roi")

        #!Compute n cells B and C corrected 
        table_cells_B_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index("B")+1]
        table_cells_B = table_cells_B_w_borders[table_cells_B_w_borders["within_roi"]==True]

        n_B_proba = self.roi.get_cell_number_from_type("B",with_proba=True) 
        n_B_proba_square = np.sum(table_cells_B["proba_{}".format("B")]**2) 
        
        table_cells_C_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index("C")+1]
        table_cells_C = table_cells_C_w_borders[table_cells_C_w_borders["within_roi"]==True]

        n_C_proba = self.roi.get_cell_number_from_type("C",with_proba=True) 
        n_C_proba_square = np.sum(table_cells_C["proba_{}".format("C")]**2) 

        n_cells_type_B_corrected = n_cells_type_B*P_inv[0,0]+n_cells_type_C*P_inv[0,1]
        n_cells_type_C_corrected = n_cells_type_B*P_inv[1,0]+n_cells_type_C*P_inv[1,1]

        for levelset_number in colocalisation_levelsets_scale_corrected.w_number.unique():
            sub_df_proba_wi = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)&(colocalisation_levelsets_scale_corrected["proba"])]
            idx_B = sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)]["eta_a"].index[0]
            idx_C = sub_df_proba_wi[(sub_df_proba_wi["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])&(colocalisation_levelsets_scale_corrected["w_number"]==levelset_number)]["eta_a"].index[0]
            
            eta_t_B = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["eta_t"].values[0]
            eta_t_C = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["eta_t"].values[0]

            m_a_B = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["m_a"].values[0]
            m_a_C = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["m_a"].values[0]

            eta_t_corrected_B = eta_t_B*P_inv[0,0]+eta_t_C*P_inv[0,1]
            eta_t_corrected_C = eta_t_B*P_inv[1,0]+eta_t_C*P_inv[1,1]

            colocalisation_levelsets_scale_corrected.loc[idx_B,"eta_t_corrected"] = eta_t_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"eta_t_corrected"] = eta_t_corrected_C

            k_corrected_B = eta_t_corrected_B*omega/n_cells_type_B
            k_corrected_C = eta_t_corrected_C*omega/n_cells_type_C
            colocalisation_levelsets_scale_corrected.loc[idx_B,"k_corrected"] = k_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"k_corrected"] = k_corrected_C

            mt_a_B = m_a_B*n_cells_type_B_corrected/omega
            mt_a_C = m_a_C*n_cells_type_C_corrected/omega


            colocalisation_levelsets_scale_corrected.loc[idx_B,"mt_a_corrected"] = mt_a_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"mt_a_corrected"] = mt_a_C

            #! Computational way of recovering the sigma numerator since it has already been computed but not saved 
            sigma_a_B_numerator = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="B"]["sigma_a"].values[0]*np.sqrt(n_B_proba)/np.sqrt(n_B_proba_square)
            sigma_a_C_numerator = sub_df_proba_wi[sub_df_proba_wi["type_B"]=="C"]["sigma_a"].values[0]*np.sqrt(n_C_proba)/np.sqrt(n_C_proba_square)

            sigma_a_corrected_B = sigma_a_B_numerator/np.sqrt(n_cells_type_B_corrected)
            sigma_a_corrected_C = sigma_a_C_numerator/np.sqrt(n_cells_type_C_corrected)
            colocalisation_levelsets_scale_corrected.loc[idx_B,"sigma_a_corrected"] = sigma_a_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"sigma_a_corrected"] = sigma_a_corrected_C

            threshold_B = m_a_B + self.statistical_threshold*sigma_a_corrected_B
            threshold_C = m_a_C + self.statistical_threshold*sigma_a_corrected_C
            indicatrice_significant_accumulation_B = (k_corrected_B > threshold_B).astype(bool)
            indicatrice_significant_accumulation_C = (k_corrected_C > threshold_C).astype(bool)

            colocalisation_levelsets_scale_corrected.loc[idx_B,"eta_s_corrected"] = indicatrice_significant_accumulation_B*eta_t_corrected_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"eta_s_corrected"] = indicatrice_significant_accumulation_C*eta_t_corrected_C

            colocalisation_levelsets_scale_corrected.loc[idx_B,"eta_a_corrected"] = indicatrice_significant_accumulation_B * (eta_t_corrected_B - mt_a_B)
            colocalisation_levelsets_scale_corrected.loc[idx_C,"eta_a_corrected"] = indicatrice_significant_accumulation_C * (eta_t_corrected_C - mt_a_C)

            z_score_B = (k_corrected_B - m_a_B)/sigma_a_corrected_B
            z_score_C = (k_corrected_C - m_a_C)/sigma_a_corrected_C
            colocalisation_levelsets_scale_corrected.loc[idx_B,"z_score_corrected"] = z_score_B
            colocalisation_levelsets_scale_corrected.loc[idx_C,"z_score_corrected"] = z_score_C

        colocalisation_results_corrected["accumulation_score_corrected"] = np.nan 
        colocalisation_results_corrected["significant_accumulation_score_corrected"] = np.nan 
        colocalisation_results_corrected["association_score_corrected"] = np.nan 
        colocalisation_results_corrected["distance_association_corrected"] = np.nan 
        colocalisation_results_corrected["p_value_corrected"] = np.nan 

        eta_t_corrected_B = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_t_corrected"]
        eta_t_corrected_C = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_t_corrected"]
        eta_a_corrected_B = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="B")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_a_corrected"]
        eta_a_corrected_C = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]=="C")&(colocalisation_levelsets_scale_corrected["proba"])]["eta_a_corrected"]
        dist_coloc_corrected_B = self._compute_corrected_dist_association("B",table_cells_B ,eta_t_corrected_B, eta_a_corrected_B)
        dist_coloc_corrected_C = self._compute_corrected_dist_association("C",table_cells_C ,eta_t_corrected_C, eta_a_corrected_C)

        
        for type_2 in ["B", "C"]:
            if type_2 == "B":
                n_cells_proba = n_cells_type_B_corrected
                dist_coloc_corrected = dist_coloc_corrected_B
            else :
                n_cells_proba = n_cells_type_C_corrected
                dist_coloc_corrected = dist_coloc_corrected_C

            n_cells_proba = self.roi.get_cell_number_from_type(type_2,with_proba=True) 
            colocalisation_levelsets_scale_corrected_proba_type_2 = colocalisation_levelsets_scale_corrected[(colocalisation_levelsets_scale_corrected["type_B"]==type_2)&(colocalisation_levelsets_scale_corrected["proba"]) ]
            # print(f"Type = {type_2} and sape is {colocalisation_levelsets_scale_corrected_proba_type_2.shape}")
            accumulation_score_corrected = np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_t_corrected"][:-1])/n_cells_proba
            significant_accumulation_score_corrected = np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_s_corrected"][:-1])/n_cells_proba
            association_score_corrected = np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_a_corrected"][:-1])/n_cells_proba
            idx_row_results = colocalisation_results_corrected[colocalisation_results_corrected["proba"]&(colocalisation_results_corrected["type_B"]==type_2)].index[0]
            # print("This index len sshould be 1 : ",len(colocalisation_results_corrected[colocalisation_results_corrected["proba"]&(colocalisation_results_corrected["type_B"]==type_2)].index))
            colocalisation_results_corrected.loc[idx_row_results,"accumulation_score_corrected"]=accumulation_score_corrected

            colocalisation_results_corrected.loc[idx_row_results,"significant_accumulation_score_corrected"]=significant_accumulation_score_corrected

            colocalisation_results_corrected.loc[idx_row_results,"association_score_corrected"]=association_score_corrected
            colocalisation_results_corrected.loc[idx_row_results,"distance_association_corrected"]=dist_coloc_corrected
            colocalisation_results_corrected.loc[idx_row_results,"p_value_corrected"]=1 - (norm.cdf(max(colocalisation_levelsets_scale_corrected_proba_type_2["z_score_corrected"][:-1]), loc=0, scale=1))**len(self.levelsets)


            # print("Type 2 : ",type_2, " n_cells_proba : ", n_cells_proba, "coloc score ",np.sum(colocalisation_levelsets_scale_corrected_proba_type_2["eta_a_corrected"][:-1])/n_cells_proba)

        colocalisation_levelsets_scale_corrected.to_csv(os.path.join(self.path_levelsets, "colocalisation_levelsets_scale.csv"),sep = ";")
        colocalisation_results_corrected.to_csv(os.path.join(self.path_levelsets, "colocalisation.csv"),sep = ";",index = False)
        return colocalisation_levelsets_scale_corrected, colocalisation_results_corrected

    def get_colocalisation_A_B(self, type_A, type_B, distance_map_from_A_w_border, m_a, sigma_a_numerator):
        """
        There is A and there is B, let's play 
        """
        coloc = dict()
        coloc["type_A"] = type_A
        coloc["type_B"] = type_B
        coloc["min_levelset"] = self.levelsets
        coloc["max_levelset"] = self.levelsets[1:]+[np.inf]
        coloc["w_number"] = np.arange(len(self.levelsets))
        coloc["m_a"] = m_a
        coloc_proba = coloc.copy()
        coloc["proba"] = False
        coloc_proba["proba"] = True

        coloc_results = dict()
        coloc_results["type_A"] = type_A
        coloc_results["type_B"] = type_B
        coloc_results_proba = coloc_results.copy()
        coloc_results["proba"] = False
        coloc_results_proba["proba"] = True
    
        if self.compute_with_proba : 

            table_cells_B_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] != self.roi.dataset_config.cell_class_names.index(type_A)+1]
        else : 
            table_cells_B_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index(type_B)+1]
        
        blue(f"N cells of type {type_B} is  {table_cells_B_w_borders.shape[0]}")
        table_cells_B = table_cells_B_w_borders[table_cells_B_w_borders["within_roi"]==True]
        table_cells_B['min_dist_from_A'] = table_cells_B.apply(lambda x: distance_map_from_A_w_border[x['x_roi_w_borders'], x['y_roi_w_borders']], axis=1)
        table_cells_B["is_B"] = (table_cells_B["cell_type"] == self.dataset_config.cell_class_names.index(type_B)+1)

        omega = self.roi.get_feature_from_statistic("area_tissue_roi")
        n_B = self.roi.get_cell_number_from_type(type_B)
        n_B_proba = self.roi.get_cell_number_from_type(type_B,with_proba=True) if self.compute_with_proba else None
        n_B_proba_square = np.sum(table_cells_B["proba_{}".format(type_B)]**2) if self.compute_with_proba else None

        coloc["eta_t"], coloc_proba["eta_t"] = self._compute_eta_t(type_B,table_cells_B)
        coloc["k"] = coloc["eta_t"]*omega/n_B
        coloc["mt_a"] = coloc["m_a"]*n_B/omega
        coloc["sigma_a"] = sigma_a_numerator/np.sqrt(n_B)
        coloc["threshold"] = coloc["m_a"] + self.statistical_threshold*coloc["sigma_a"]
        indicatrice_significant_accumulation = (coloc["k"] > coloc["threshold"]).astype(bool)
        coloc["eta_s"] = indicatrice_significant_accumulation*coloc["eta_t"]
        coloc["eta_a"] = indicatrice_significant_accumulation * (coloc["eta_t"] - coloc["mt_a"])
        coloc["z_score"] = (coloc["k"] - coloc["m_a"])/coloc["sigma_a"]
        coloc["p_value"] = 1 - (norm.cdf(max(coloc["z_score"][:-1]), loc=0, scale=1))**len(self.levelsets)

        if self.compute_with_proba : 
            coloc_proba["k"] = coloc_proba["eta_t"]*omega/n_B_proba
            coloc_proba["mt_a"] = coloc_proba["m_a"]*n_B_proba/omega
            coloc_proba["sigma_a"] = sigma_a_numerator*np.sqrt(n_B_proba_square)/np.sqrt(n_B_proba)

            coloc_proba["threshold"] = coloc_proba["m_a"] + self.statistical_threshold*coloc_proba["sigma_a"]
            indicatrice_significant_accumulation_proba = (coloc_proba["k"] > coloc_proba["threshold"]).astype(bool)
            coloc_proba["eta_s"] = indicatrice_significant_accumulation_proba*coloc_proba["eta_t"]
            coloc_proba["eta_a"] = indicatrice_significant_accumulation_proba * (coloc_proba["eta_t"] - coloc_proba["mt_a"])
            
            # coloc_proba["eta_a"] = indicatrice_significant_accumulation_proba * (coloc_proba["eta_t"] - coloc_proba["mt_a"])

            
            coloc_proba["z_score"] = (coloc_proba["k"] - coloc_proba["m_a"])/coloc_proba["sigma_a"]
            coloc_proba["p_value"] = 1 - (norm.cdf(max(coloc_proba["z_score"][:-1]), loc=0, scale=1))**len(self.levelsets)
        
            coloc["distance_association"], coloc_proba["distance_association"] = self._compute_distance_association(table_cells_B,coloc["eta_t"],coloc["eta_a"],coloc_proba["eta_t"],coloc_proba["eta_a"])
        else : 
            coloc["distance_association"], coloc_proba["distance_association"] = self._compute_distance_association(table_cells_B,coloc["eta_t"],coloc["eta_a"])
        df_coloc = pd.DataFrame(coloc)
        df_coloc_proba = pd.DataFrame(coloc_proba) if self.compute_with_proba else None

        coloc_results["accumulation"] = np.sum(coloc["eta_t"][:-1])/n_B
        coloc_results["significant_accumulation"] = np.sum(coloc["eta_s"][:-1])/n_B
        coloc_results["association_score"] = np.sum(coloc["eta_a"][:-1])/n_B
        coloc_results["distance_association"] = coloc["distance_association"]
        coloc_results["p_value"] = coloc["p_value"]
        df_coloc_results = pd.DataFrame([coloc_results])

        if self.compute_with_proba : 
            coloc_results_proba["accumulation"] = np.sum(coloc_proba["eta_t"][:-1])/n_B_proba
            coloc_results_proba["significant_accumulation"] = np.sum(coloc_proba["eta_s"][:-1])/n_B_proba
            coloc_results_proba["association_score"] = np.sum(coloc_proba["eta_a"][:-1])/n_B_proba
            coloc_results_proba["distance_association"] = coloc_proba["distance_association"]
            coloc_results_proba["p_value"] = coloc_proba["p_value"]
            df_coloc_results_proba = pd.DataFrame([coloc_results_proba])
        else :
            df_coloc_results_proba = None

        return df_coloc, df_coloc_proba, df_coloc_results, df_coloc_results_proba



    def _compute_eta_t(self,type_B,table_cells_B):
        """
        Take as inputs :
        - La carte des distances de chaque pixel de la ROI aux cellules de A (fonction phi dans [1])
        - La liste des levelsets 
        Return: 
        Un vecteur [0,4,5,12,354,1,0,...,0] de taille len(levelset_values) ou le +1 a été ajouté au levelset d'appartenance de B dans un des levelsets 

        Ainsi w_0 contient 0 cells, w_1 contient 4 cells, w_2 contient 5 cells 
        """
        table_cells_B_sorted_by_dist = table_cells_B.sort_values("min_dist_from_A", ignore_index = True)
        idx_cell = 0
        cell_dist_from_A = table_cells_B_sorted_by_dist["min_dist_from_A"][idx_cell] #shortest dist

        cell_distribution_in_wi =np.zeros(len(self.levelsets))
        cell_distribution_in_wi_proba=np.zeros(len(self.levelsets))
        df_end = False
        for idx_levelset, levelset in enumerate(self.levelsets[1:]) : 
            if df_end:
                break
            while cell_dist_from_A < levelset : 
                cell_distribution_in_wi[idx_levelset] += int(table_cells_B_sorted_by_dist["is_B"][idx_cell])
                weight_proba = table_cells_B_sorted_by_dist["proba_{}".format(type_B)][idx_cell] if self.compute_with_proba else 1
                cell_distribution_in_wi_proba[idx_levelset]+= weight_proba
                idx_cell+=1
                if idx_cell < len(table_cells_B_sorted_by_dist):
                    cell_dist_from_A = table_cells_B_sorted_by_dist["min_dist_from_A"][idx_cell] #Ok par la cond du while 
                else : 
                    df_end = True
                    break 
        
        if idx_cell < len(table_cells_B_sorted_by_dist): #Il y a des cellules qui ne sont pas dans les levelsets mais en dehors 
            cell_distribution_in_wi[-1] = np.sum(table_cells_B_sorted_by_dist["is_B"][idx_cell:])
            weight_proba = np.sum(table_cells_B_sorted_by_dist["proba_{}".format(type_B)][idx_cell:]) if self.compute_with_proba else 1
            cell_distribution_in_wi_proba[-1] = weight_proba
            
        return cell_distribution_in_wi, cell_distribution_in_wi_proba 

    def _compute_eta_t_faster(self,type_B,table_cells_B):
        """ Idea : create columns "in_w0", "in_w1" with 0/1 or proba and then sum them """
        cell_distribution_in_wi = np.zeros(len(self.levelsets))
        cell_distribution_in_wi_proba = np.zeros(len(self.levelsets))

        for idx_levelset in range(self.levelsets[1:]):
            table_cells_B["in_w{}".format(idx_levelset)] = (self.levelsets[idx_levelset] <= table_cells_B["min_dist_from_A"]) & (table_cells_B["min_dist_from_A"] < self.levelsets[idx_levelset+1])
            cell_distribution_in_wi[idx_levelset] = np.sum(table_cells_B["in_w{}".format(idx_levelset)])
            cell_distribution_in_wi_proba[idx_levelset] = np.sum(table_cells_B["in_w{}".format(idx_levelset)]*table_cells_B["proba_{}".format(type_B)]) if self.compute_with_proba else 1

        table_cells_B["in_w{}".format(len(self.levelsets))] = (self.levelsets[-1] <= table_cells_B["min_dist_from_A"])
        cell_distribution_in_wi[-1] = np.sum(table_cells_B["in_w{}".format(len(self.levelsets))])
        cell_distribution_in_wi_proba[-1] = np.sum(table_cells_B["in_w{}".format(len(self.levelsets))]*table_cells_B["proba_{}".format(type_B)]) if self.compute_with_proba else 1

        return cell_distribution_in_wi, cell_distribution_in_wi_proba

    def _compute_M_Sigma(self,distance_map_from_A_w_border):
        """
        Calcul les mu associé a chaque levelsets dans une carte de distance 
        je opeux utiliser le display pour montrer des choses 
        """
        omega = self.roi.get_feature_from_statistic("area_tissue_roi")
        m_a=np.zeros(len(self.levelsets))
        sigma_a_numerator = np.ones(len(self.levelsets))*np.inf

        levelset_until_max = self.levelsets+[np.inf]

        roi_border_size = self.roi.dataset_config.roi_border_size
        distance_map_from_A = distance_map_from_A_w_border[roi_border_size:-roi_border_size,roi_border_size:-roi_border_size]
        mask_tissue = self.roi.mask_tissue_w_borders[roi_border_size:-roi_border_size,roi_border_size:-roi_border_size]
        distance_map_from_A = distance_map_from_A*mask_tissue

        for id_ls, ls in enumerate(levelset_until_max[:-1]) : 
            area_inter_ls = np.logical_and(distance_map_from_A>=ls,distance_map_from_A<levelset_until_max[id_ls+1])

            wij = np.sum(area_inter_ls)
            m_a[id_ls] = wij
            sigma_a_numerator[id_ls] = np.sqrt(wij*(omega-wij))

        return m_a, sigma_a_numerator

    def _compute_distance_association(self,table_cells_B,eta_t,eta_a, eta_t_proba=None,eta_a_proba=None):
        list_theta_u = []
        list_theta_u_proba = []
        eta_t_proba = eta_t if eta_t_proba is None else eta_t_proba
        eta_a_proba = eta_a if eta_a_proba is None else eta_a_proba
        for idx, cell in table_cells_B.iterrows():

            w_ij = self._find_wi_cell_B(cell["min_dist_from_A"], self.levelsets)
            if w_ij == len(self.levelsets):
                list_theta_u.append(0)
                list_theta_u_proba.append(0)
                continue
            eta_a_ij = eta_a[w_ij]
            eta_t_ij = eta_t[w_ij]
            theta_u = eta_a_ij/eta_t_ij if eta_t_ij>0 else 0
            list_theta_u.append(theta_u)

            eta_a_ij_proba = eta_a_proba[w_ij]
            eta_t_ij_proba = eta_t_proba[w_ij] 
            theta_u_proba = eta_a_ij_proba/eta_t_ij_proba if eta_t_ij_proba>0 else 0
            list_theta_u_proba.append(theta_u_proba)

        theta_u_array = np.asarray(list_theta_u)
        distance_association = np.sum(theta_u_array*table_cells_B['min_dist_from_A'])/np.sum(theta_u_array) if np.sum(theta_u_array)>0 else 0
        theta_u_proba_array = np.asarray(list_theta_u_proba)
        distance_association_proba = np.sum(theta_u_proba_array*table_cells_B['min_dist_from_A'])/np.sum(theta_u_proba_array) if np.sum(theta_u_proba_array)>0 else 0

        return distance_association, distance_association_proba

    def _find_wi_cell_B(self,distance_cell_B_to_A_levelsets, levelsets):
        """
        donne le numéro du ring(R_i,R_(i+1) auquel appartient la cellule B dans les levelsets de A
        """
        
        for idx_levelset, levelset in enumerate(levelsets[1:]):
            if distance_cell_B_to_A_levelsets < levelset : 
                return idx_levelset
        return len(levelsets)

    def no_colocalisation_because_no_B_at_all(self, colocalisation_levelsets_scale, type_A,cell_types_B):
        """ A present, no B at all"""
        return colocalisation_levelsets_scale
    
    def no_colocalisation_with_A(self,colocalisation_levelsets_scale, type_A, cell_types_B):
        """ A not present"""
        dict_coloc_A_B_all_wi = dict()
        return colocalisation_levelsets_scale
    
    def no_colocalisation_with_B(self,colocalisation_levelsets_scale, type_A, type_B):
        """ A present, B not"""
        # print("No cell of type", type_B, "in the roi")
        return colocalisation_levelsets_scale

    def no_colocalisation_when_same_type(self, colocalisation_levelsets_scale, type_A):
        """ No colocalisation if A = B """
        return colocalisation_levelsets_scale
    
############################# Visualisation results 

    def visualise_z_score(self,proba=False): 
        """
        Il faut que j'arrive a mettre les delta a 
        Creer la le plot du Z score en fonction des levelsets 
        """
        normalise_y = True

        # colocalisation_levelsets_scale = colocalisation_levelsets_scale[colocalisation_levelsets_scale["w_num"]!=colocalisation_levelsets_scale["w_num"].max()] #Pour enlever le dernier rang
        title = "(Considering probabilities)" if proba else "(Considering decision)"
        name_file = "_proba" if proba else "_decision"
        colocalisation_levelsets_scale = self.colocalisation_levelsets_scale.query("proba == @proba")
        
        # fig_stat = px.bar(colocalisation_levelsets_scale, x="w_number", y="z_score", facet_row="type_B",facet_col="type_A",facet_row_spacing=0.06, facet_col_spacing=0.03, category_orders={"type_A": list(colocalisation_levelsets_scale["type_A"].unique()),
        #                         "type_B": list(colocalisation_levelsets_scale["type_B"].unique())},title="Analysis Colocalisation<br>"+title, height=2000, width= 1800)
        fig_stat = px.bar(colocalisation_levelsets_scale, x="w_number", y="z_score", facet_row="type_B",facet_col="type_A",facet_row_spacing=0.06, facet_col_spacing=0.03, category_orders={"type_A": colocalisation_levelsets_scale["type_A"],
                                "type_B": colocalisation_levelsets_scale["type_A"]},title="Analysis Colocalisation<br>"+title, height=2000, width= 1800)
        if normalise_y : 
            fig_stat.update_yaxes(matches='y',showticklabels=True)
            name_file+="_y_normalised"
        else : 
            fig_stat.update_yaxes(matches=None,showticklabels=True)

        fig_stat.update_xaxes(matches=None,showticklabels=True)
        thresh_statistic = np.sqrt(2*np.log(len(self.levelsets)))
        fig_stat.add_hline(y=thresh_statistic, line_dash="dot")
        fig_stat.add_hline(y=-thresh_statistic, line_dash="dot")
        for idx_type_B,type_B_name in enumerate(self.cell_types_A[::-1],1): #idx col 
            for idx_type_A, type_A_name in enumerate(self.cell_types_A,1): #idx row
                if type_B_name == type_A_name:
                    continue
                # print("visualise_Z_Score")
                distance_association = self.colocalisation_results[(self.colocalisation_results["type_A"]==type_A_name) & (self.colocalisation_results["type_B"]==type_B_name) & (self.colocalisation_results["proba"]==0)]["distance_association"].iloc[0]
                if isnan(distance_association):
                    continue
                x_pos = (distance_association*len(self.levelsets))/self.levelsets[-1]-1
                fig_stat.add_vline(x=x_pos, line_width=3, line_dash='dash', line_color='red', row=idx_type_B, col=idx_type_A) if distance_association != 0 else None  #annotation_text=str(delta)+"micro m"

        fig_stat.for_each_annotation(lambda a: a.update(text="State "+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))


        pathfile = os.path.join(self.path_levelsets, "Z_score_analysis"+name_file+".png")
        fig_stat.write_image(pathfile)

    def visualise_Z_Score_both_proba_and_not(self): 
        """
        Il faut que jarrive a mettre les delta a 
        Creer la le plot du Z score en fonction des levelsets 

        self.colocalisation_levelsets_scale, 
        self.colocalisation_results
        """
        if not self.compute_with_proba :
            print("No proba computed")
            raise ValueError
        
        fig_stat = px.bar(self.colocalisation_levelsets_scale, x="w_number", y="z_score", facet_row="type_B",facet_col="type_A",color = "proba" ,barmode="group",color_discrete_map={'1':'green',
                                    '0':'cyan'}, facet_row_spacing=0.06, facet_col_spacing=0.03, category_orders={"type_A": list(self.colocalisation_levelsets_scale["type_A"].unique()),
                                "type_B": list(self.colocalisation_levelsets_scale["type_A"].unique())},title="Analysis Colocalisation", height=2000, width= 1800)
                                
        fig_stat.update_yaxes(matches=None,showticklabels=True)
        fig_stat.update_xaxes(matches=None,showticklabels=True)

        thresh_statistic = np.sqrt(2*np.log(len(self.levelsets)))

        fig_stat.add_hline(y=thresh_statistic, line_dash="dot")
        fig_stat.add_hline(y=-thresh_statistic, line_dash="dot")
        for idx_type_B,type_B_name in enumerate(self.cell_types_A[::-1],1): #idx col 
            for idx_type_A, type_A_name in enumerate(self.cell_types_A,1): #idx row
                if type_B_name == type_A_name:
                    continue
                # print("visualise_Z_Score")
                print("type_A_name",type_A_name)
                print("type_B_name",type_B_name)

                distance_association = self.colocalisation_results[(self.colocalisation_results["type_A"]==type_A_name) & (self.colocalisation_results["type_B"]==type_B_name) & (self.colocalisation_results["proba"]==0)]["distance_association"].iloc[0]
                if isnan(distance_association):
                    continue
                distance_association_proba = self.colocalisation_results[(self.colocalisation_results["type_A"]==type_A_name) & (self.colocalisation_results["type_B"]==type_B_name) & (self.colocalisation_results["proba"]==1)]["distance_association"].iloc[0]
                x_pos = (distance_association*len(self.levelsets))/self.levelsets[-1]-1
                x_pos_proba = (distance_association_proba*len(self.levelsets))/self.levelsets[-1]-1
                fig_stat.add_vline(x=x_pos, line_width=3, line_dash='dash', line_color='red', row=idx_type_B, col=idx_type_A) if distance_association != 0 else None  #annotation_text=str(delta)+"micro m"
                fig_stat.add_vline(x=x_pos_proba, line_width=3, line_dash='dot', line_color='blue', row=idx_type_B, col=idx_type_A) if distance_association_proba != 0 else None 

        fig_stat.for_each_annotation(lambda a: a.update(text="State "+a.text.split("=")[0][-1]+" : "+a.text.split("=")[-1]))

        pathfile = os.path.join(self.path_levelsets, "Z_score_analysis.png")
        fig_stat.write_image(pathfile)

    def heatmap_colocalisation(self,proba = False, feature_name = "association"):

        title_metric = " (considering proba) " if proba else ""
        suffix_filename = "_proba" if proba else ""
        
        matrix = self._preprocess_before_heatmap(feature_name=feature_name,proba=proba)
        fig = px.imshow(np.flipud(matrix),labels=dict(x="type B", y="type A"), #ici np.flipud est le seul moyen d'inversé le y-axis selon https://github.com/plotly/plotly.py/issues/413
                            x=self.cell_types_B,
                            y=self.cell_types_A[::-1], text_auto=True, aspect="auto",origin = "lower")

        fig.update_xaxes(side="top",title_text = "type B") #side top permet de placer les labels en haut 
        fig.update_yaxes(title_text="Type A")
        
        for annotation in fig['layout']['annotations']: #Le style des titres du plot est est modifié ici 
            annotation['font'] = dict(size=20)  
            annotation['yshift'] = 55 #Les titres sont décalés vers le haut pour ne pas overlappe les labels de l'x-axis 
        fig.update_layout(title_text="<b>Cell-cell colocalisation : association"+title_metric+" <br> type B in type A levelsets  </b>",title_x=0.5,title_font=dict(size=30),
                            coloraxis=dict(colorscale='balance', colorbar_thickness=25,colorbar_x=-0.16),
                            coloraxis2=dict(colorscale='curl',colorbar_thickness=25,),
                            showlegend=True,
                            width=1650, #Taille de la figure 
                            height=1000,
                            margin=dict(l=50,r=50,b=50,t=270, pad=4),#Permet d'eloigner les plots du titre   #Gere les marges du plot (left, right, bottom, top)
                            coloraxis_colorbar=dict(title="Mean <br>distance<br>colocalisation<br>(<span>&#181;</span>m)"))

        pathfile = os.path.join(self.path_levelsets,feature_name+suffix_filename+".png")
        fig.write_image(pathfile)

    def _preprocess_before_heatmap(self,feature_name="association_score",proba=True):
        import math
        """Return a matrix 5x5 to plot any metric of colocalisation between 5 states """
        round_z = 2
        labels_type_A = self.cell_types_A
        labels_type_B = self.cell_types_B

        df = self.colocalisation_results[self.colocalisation_results["proba"]==proba]
        matrix = np.zeros((len(labels_type_A), len(labels_type_B)))

        for idx_A, type_A in enumerate(labels_type_A):
            for idx_B, type_B in enumerate(labels_type_B):
                if idx_B != idx_A : 
                    z = df[(df["type_A"] == type_A) & (df["type_B"] == type_B)][feature_name].values[0]
                    if feature_name == "distance_association":
                        z = z*self.dataset_config.conversion_px_micro_meter
                    if round_z != -1:
                        z = round(z,round_z) if z != math.inf else np.nan
                    matrix[idx_A,idx_B] = z
        return matrix

    def display_B_in_A_levelsets(self, display = True,output_path_name="colocalisation",roi_category="all",figsize = (20,20),display_only_levelsets = False ,with_background = True,with_roi_delimiter = False,with_center_of_mass = False):
        """Construit les figures des cellules B dans les levelsets de A 
        - "with_proba" color les cellules B de la même couleur 
        """
        to_expo = True #Bc this github branch of the project 
        with_anatomical_part_mask = False 
        with_center_of_mass = False
        with_tiles_delimitation = False

        comment_figure = "_only_levelsets_" if display_only_levelsets else ""

        if not hasattr(self.roi,"image_w_borders"): 
            with_background = False

        if with_background : 
            if self.dataset_config.consider_image_with_channels :
                    background = self.roi.image_w_borders["RGB"]
            else :
                # background = np.dstack([self.image_w_borders[:,:,1], self.image_w_borders[:,:,1], self.image_w_borders[:,:,1]])
                background = self.roi.image_w_borders
        else : 
            background = np.zeros((self.roi.roi_w_borders_shape[0],self.roi.roi_w_borders_shape[1],3))

        
        for id_type_A, type_A in enumerate(self.cell_types_A,1):  #Boucle sur pop A
            background_pil = np_to_pil(background)
            drawing = ImageDraw.Draw(background_pil, "RGBA")
            if self.roi.get_cell_number_from_type(type_A) == 0:
                continue

            mask_A_w_border = self.roi.get_mask_cell_type(type_A)
            distance_map_from_A_w_border = distance_transform_edt(1 - mask_A_w_border,sampling=1) 

            wi_polynomes = self._get_wi_as_polygones(distance_map_from_A_w_border)
            if display_only_levelsets :
                drawing = self._draw_levelsets(drawing,wi_polynomes)
            else : 
                drawing = self._draw_wi(drawing,wi_polynomes) 
            

            drawing = draw_cells_on_img(self.roi, drawing,cell_type_filter=type_A)

            for id_type_B, type_B in enumerate(self.cell_types_B,1):
                if type_A != type_B:
                    drawing = draw_cells_on_img(self.roi, drawing,cell_type_filter=type_B)

            # im = add_mask_tissu_on_img(RegionOfInterest, img)
            if with_roi_delimiter : 
                drawing = draw_roi_delimiter(self.roi, drawing)
            if with_tiles_delimitation:
                drawing = with_tiles_delimitations(self, drawing)

            fig = plt.figure(figsize=figsize)#, tight_layout = True)
            plt.imshow(background_pil)# if display else None 
            plt.axis('off')

            plt.show() if display else None 
            plt.close("All")
            if to_expo : 
                directory = os.path.join(self.dataset_config.dir_output,OUTPUT_EXPO_NAME,output_path_name,roi_category)
                os.makedirs(directory,exist_ok=True)
                figname = "s"+str(self.roi.slide_num).zfill(3)+"_ro" +str(self.roi.origin_row).zfill(3) + "_co" +str(self.roi.origin_col).zfill(3) + "_re" +str(self.roi.end_row).zfill(3) + "_ce" +str(self.roi.end_col).zfill(3)

                name_fig = figname+"_A_"+type_A+"_levelsets"

                path_save = find_path_last(directory,name_fig)
                fig.savefig(path_save,
                            facecolor='white',
                            dpi="figure",
                            bbox_inches='tight',
                            pad_inches=0.1)
            else : 

                path_folder_fig = os.path.join(self.path_levelsets, "B_in_A_levelsets")
                mkdir_if_nexist(path_folder_fig)
                name_fig = "A_"+type_A+"_levelsets"+comment_figure+".png"

                fig.savefig(os.path.join(path_folder_fig,name_fig ),
                            facecolor='white',
                            dpi="figure",
                            bbox_inches='tight',
                            pad_inches=0.1)

    def _get_wi_as_polygones(self,distance_map):
        """
        Les levelsets sont les concours des wi donc pas besoin de les conservers 
        Input : 
        - Distance map 
        - Levelsets 
        Output : 
        - dictionnaire dict["w_i"] = multipolynomes des regions de wi 
        """
        levelsets = self.levelsets
        wi_polynomes = dict()
        for id_ls, ls in enumerate(levelsets[:-1]):
            next_ls = levelsets[id_ls+1]
            wi = np.logical_and(distance_map>=ls,distance_map<=next_ls) 
            wi_poly = mask_to_polygons_layer(wi)
            wi_polynomes["w_"+str(id_ls)] = wi_poly
        return wi_polynomes

    def _draw_wi(self,drawing, wi_polynomes):
        """Ajoute les wi dans l'image des levelsets """
        cmap = plt.cm.coolwarm  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        liste_colors = [tuple([int(cmaplist[i][0]*255),int(cmaplist[i][1]*255),int(cmaplist[i][2]*255),int(cmaplist[i][3]*100)])  for i in np.linspace(1,255,len(self.levelsets)).astype(int)]
        liste_colors = liste_colors[::-1]
        for idx_color, wi_name in enumerate(list(wi_polynomes.keys())[::-1]):
            wi = wi_polynomes[wi_name]
            Multipol_ = [list(poly.exterior.coords) for poly in list(wi.geoms)]
            for points in Multipol_:
                drawing.polygon(points,fill=liste_colors[idx_color], outline =(0,0,0,255))
            for poly in wi.geoms:
                for hole in poly.interiors:
                    drawing.polygon(hole.coords, fill=(255, 255, 255,0), outline=(0, 0, 0, 255))
        return drawing

    def _draw_levelsets(self,drawing,wi_polynomes):
        """Ajoute les levelsets dans l'image des levelsets (PAS LES W)"""
        
        alpha = 255
        cmap = plt.cm.coolwarm  # define the colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        liste_colors = [tuple([int(cmaplist[i][0]*255),int(cmaplist[i][1]*255),int(cmaplist[i][2]*255),int(cmaplist[i][3]*alpha)])  for i in np.linspace(1,255,len(self.levelsets)).astype(int)]
        liste_colors = liste_colors[::-1]
        for idx_color, wi_name in enumerate(list(wi_polynomes.keys())[::-1]):
            wi = wi_polynomes[wi_name]
            Multipol_ = [list(poly.exterior.coords) for poly in list(wi.geoms)]
            for points in Multipol_:
                drawing.line(points,fill=liste_colors[idx_color], width = 20)
        return drawing