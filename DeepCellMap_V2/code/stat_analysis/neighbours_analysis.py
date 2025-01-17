# Importation des librairies
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pandas as pd
from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
import plotly.express as px

from utils.util_colors_drawing import *
from utils.util import *
from simple_colors import *
from math import isnan


from stat_analysis.colocalisation_analysis import create_path_colocalisation
VERBOSE = False


class NeighborsAnalysis:
    """
    #!!!! -> Cells in roi border can be considered as neighbours
    """
    def __init__(self, Roi,neighbors_analysis_config=None,cell_types_A=None,cell_types_B=None):
        neighbors_analysis_config = Roi.dataset_config.neighbors_analysis_config if neighbors_analysis_config is None else neighbors_analysis_config
        cell_types_A = Roi.dataset_config.neighbors_analysis_config["cell_types_A"] if cell_types_A is None else cell_types_A
        cell_types_B = Roi.dataset_config.neighbors_analysis_config["cell_types_B"] if cell_types_B is None else cell_types_B
        n_closest_neighbors_of_interest = Roi.dataset_config.neighbors_analysis_config["n_closest_neighbors_of_interest"]

        self.roi = Roi
        self.dataset_config = Roi.dataset_config
        self.neighbors_analysis_config = neighbors_analysis_config
        self.cell_types_A = cell_types_A
        self.cell_types_B = cell_types_B
        self.path_folder_neighbors_analysis = self._create_path_neighbors_analysis()
        self.colnames_neighbors_analysis = (
            [
            'type_A', #
            'type_B',#
            "n_closest_neighbors_of_interest"]

            +["distance_caracteristique_dc_from_colocalisation_B_around_A",#
            "fraction_disk_dc_containing_at_least_one_B"]
            + ["fraction_B_first_A_neighbor"]
            + ["mean_dist_"+str(k)+"_first_cell_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
            + ["std_dist_"+str(k)+"_first_cell_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
             + ["mean_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
             + ["std_dist_"+str(k)+"_first_B_around_A" for k in range(1,n_closest_neighbors_of_interest+1)]
        )
        self.neighbors_analysis = self.compute_neighbors_analysis()


    def _create_path_neighbors_analysis(self):
        """ """
        path_folder_neighbors_analysis = os.path.join(
            self.roi.path_roi, "4_neighbors_analysis"
        )
        if not os.path.exists(path_folder_neighbors_analysis):
            os.mkdir(path_folder_neighbors_analysis)
        return path_folder_neighbors_analysis


    def get_dist_colocalisation_max_A(self, type_A):
        list_levelsets = self.roi.dataset_config.cell_cell_colocalisation_config["levelsets"]
        path_folder_colocalisation, path_levelsets = create_path_colocalisation(self.roi,list_levelsets)
        path_df = os.path.join(path_levelsets,"colocalisation.csv")
        df_colocalisation = pd.read_csv(path_df,sep = ";")
        # for col in df_colocalisation.columns:
        #     print("col",col)
        # print("df_colocalisation.shape",df_colocalisation["type_A"].shape)
        dist_coloc_max_A = int(df_colocalisation[df_colocalisation["type_A"]==type_A]["distance_association"].max())
        return dist_coloc_max_A

    def compute_neighbors_analysis(self):
        """

        """
        verbose = False 
        print(blue("Statistical analysis : Neighbors analysis image {}...".format(self.roi.slide_num)))

        df_neighbours_analysis = pd.DataFrame(columns=self.colnames_neighbors_analysis)
        n_closest_neighbors_of_interest = self.neighbors_analysis_config["n_closest_neighbors_of_interest"]

        for type_A in self.cell_types_A:
            print(blue("Analysis with A cells :"),type_A) if verbose else None 
            na = self.roi.get_cell_number_from_type(type_A)
            if na == 0:
                continue
            if not self.roi.at_least_one_B_cell(self.cell_types_B):
                continue
            results_A = dict()
            results_A["type_A"] = type_A
            results_A["n_closest_neighbors_of_interest"] = n_closest_neighbors_of_interest


            A_cells = self.roi.table_cells_w_borders[(self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index(type_A)+1) & (self.roi.table_cells_w_borders["within_roi"]==True)]
            # other_cells = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] != self.roi.dataset_config.cell_class_names.index(type_A)+1]
            other_cells = self.roi.table_cells_w_borders
            #!!!! -> Cells in roi border can be considered as neighbours
            matrix_distance_A_other = distance.cdist(other_cells[['x_roi_w_borders','y_roi_w_borders']],A_cells[['x_roi_w_borders','y_roi_w_borders']]) #(i,j) = dist(cell_A_i,cell_B_j)
            print("n_zero dist",matrix_distance_A_other[matrix_distance_A_other==0].sum())
            print("matrix_distance_A_other",matrix_distance_A_other.shape)
            matrix_distance_A_other[matrix_distance_A_other==0]= np.inf 
            matrix_distance_A_other = pd.DataFrame(matrix_distance_A_other)
            results_A = self._get_dist_first_neighbors(results_A, matrix_distance_A_other,A_vs_all=True)
            
            #Labels first cell around A
            label_first_cell_around_A = self._get_fraction_B_first_A_neighbor(A_cells,other_cells,matrix_distance_A_other)
            
            dist_coloc_max_A = [self.get_dist_colocalisation_max_A(type_A)]
            results_A["distance_caracteristique_dc_from_colocalisation_B_around_A"] = dist_coloc_max_A[0]
            for type_B in self.cell_types_B:
                results_AB = results_A.copy()
                results_AB["type_B"] = type_B
                results_AB["fraction_B_first_A_neighbor"] = np.sum(label_first_cell_around_A == self.dataset_config.association_cell_name_channel_number[type_B])/A_cells.shape[0]
                if type_B == type_A : 
                    print("NA - idx_pop_B ",type_A," pop_B_name ",type_B)
                    print("results_AB[fraction_B_first_A_neighbor]",results_AB["fraction_B_first_A_neighbor"])
                #B cells inside ROI
                B_cells_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index(type_B)+1]

                #Distance matrix between A and B. Shape = (nb_cells_B,nb_cells_A)
                matrix_distance_A_B = distance.cdist(B_cells_w_borders[['x_roi_w_borders','y_roi_w_borders']],A_cells[['x_roi_w_borders','y_roi_w_borders']]) #(i,j) = dist(cell_A_i,cell_B_j)
                matrix_distance_A_B[matrix_distance_A_B==0]= np.inf 
                df_distance_A_B = pd.DataFrame(matrix_distance_A_B)
                #Mean and std dist 3 first B neighbours of A
                results_AB = self._get_dist_first_neighbors(results_AB, df_distance_A_B,A_vs_all=False)

                #Number of balls of radius r containing at least 1 cell B (different radius)
                results_AB["fraction_disk_dc_containing_at_least_one_B"] = self.compute_fraction_A_balls_containing_at_least_1_B( matrix_distance_A_B, dist_coloc_max_A)
                df_A_B = pd.DataFrame([results_AB])
                df_neighbours_analysis = pd.concat([df_neighbours_analysis,df_A_B],  ignore_index=True)

        # df_neighbours_analysis = pd.DataFrame([results],columns = self.colnames_neighbors_analysis)
        df_neighbours_analysis.to_csv(os.path.join(self.path_folder_neighbors_analysis,"df_neighbours_analysis.csv"), sep = ";", index = False)
        return df_neighbours_analysis


    def _get_dist_first_neighbors(self,results_A, matrix_distance_A_B, A_vs_all):
        """
        Columns : A cells 
        Rows : B cells 

        Return the average and std distance of the k closest neighbours of A cells

        Si le nombre de cellules B (nb de lignes du df) est inférieur au nombre de voisins de référence, on complète avec des np.inf
        Si par exemple on a 3 voisins de référence et que le nombre de cellules B est de 2, on complète avec 1 ligne de np.inf
        Il en résulte que l'output sera egale a [mean distance first B,mean distance second B,np.inf]
        """
        n_closest_neighbors_of_interest = results_A["n_closest_neighbors_of_interest"]
        array_average_dist_first_B_neighbors = np.zeros((n_closest_neighbors_of_interest,matrix_distance_A_B.shape[1]))
        
        #Preprocess if not enough B cells
        if matrix_distance_A_B.shape[0] < n_closest_neighbors_of_interest:
            to_complete = np.zeros((n_closest_neighbors_of_interest-matrix_distance_A_B.shape[0],matrix_distance_A_B.shape[1]))
            to_complete[:] = np.inf
            to_complete = pd.DataFrame(to_complete)
            df_enough_B_cells = pd.concat([matrix_distance_A_B,to_complete],axis = 0)
        else : 
            df_enough_B_cells = matrix_distance_A_B
        
        #Get the mean distance of the k closest neighbours of A cells
        for idx_cells in range(matrix_distance_A_B.shape[1]):
            array_average_dist_first_B_neighbors[:,idx_cells] = df_enough_B_cells.nsmallest(n_closest_neighbors_of_interest, columns=idx_cells)[idx_cells] #Couteux en temps de calcul : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nsmallest.html#pandas.DataFrame.nsmallest
        
        mean_dist_first_B_neighbours = np.mean(array_average_dist_first_B_neighbors,axis = 1)
        std_dist_first_B_neighbours = np.std(array_average_dist_first_B_neighbors,axis = 1)

        for k in range(1,n_closest_neighbors_of_interest+1):
            if A_vs_all : 
                results_A["mean_dist_"+str(k)+"_first_cell_around_A"] =mean_dist_first_B_neighbours[k-1]
                results_A["std_dist_"+str(k)+"_first_cell_around_A"] = std_dist_first_B_neighbours[k-1]
            else : 
                results_A["mean_dist_"+str(k)+"_first_B_around_A"] =mean_dist_first_B_neighbours[k-1]
                results_A["std_dist_"+str(k)+"_first_B_around_A"] = std_dist_first_B_neighbours[k-1]
        return results_A

    def _get_fraction_B_first_A_neighbor(self,A_cells,df_other,distances_A_other):
        """
        Return the labels of the closest B cells to each A cells
        """

        A_cells.reset_index(drop = True, inplace = True)
        df_other.reset_index(drop = True, inplace = True)
        na = A_cells.shape[0]
        # print("na-> ",na)
        array_first_neighbours = np.zeros((1,na))
        if distances_A_other.shape[0] < 1 or distances_A_other.shape[1] < 1:
            return array_first_neighbours
        for idx_cells in range(na): #Iter sur les cellules A
            idx_row_first =  distances_A_other.nsmallest(1, columns=idx_cells).index[0] #Couteux en temps de calcul : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nsmallest.html#pandas.DataFrame.nsmallest
            # print("df_other[cell_type]",df_other["cell_type"])
            # print("idx_cells",idx_cells)
            # print("idx_row_first",idx_row_first)
            # print("df_other[cell_type][idx_row_first]",df_other["cell_type"][idx_row_first])
            array_first_neighbours[0,idx_cells] = df_other["cell_type"][idx_row_first]
        return array_first_neighbours

    def compute_fraction_A_balls_containing_at_least_1_B(self, matrix_distance_A_B, dist_coloc_max_A): 
        # print("matrix_distance_A_B<=dist_coloc_max_A.shape",(matrix_distance_A_B<=dist_coloc_max_A).shape)
        # print("matrix_distance_A_B : ",matrix_distance_A_B)
        matrix_dist_inf=(matrix_distance_A_B<=dist_coloc_max_A)
        # print("matrix_dist_inf.shape",matrix_dist_inf.shape)
        dist_inf = np.sum(matrix_dist_inf, axis = 0)
        # print("shape(dist_inf) = ",dist_inf.shape)
        at_least_1_B = (dist_inf>=1)
        # print("at_least_1_B.shape",at_least_1_B.shape)
        # print("matrix_distance_A_B.shape[1] = ",matrix_distance_A_B.shape[1])
        # print("at_least_1_B",at_least_1_B)
        fraction = at_least_1_B.sum()/matrix_distance_A_B.shape[1]
        return fraction 

    # def _get_fraction_disk_containing_1_B(self,results_A,df_A,df_B,dist_coloc_max_A):
    #     """
    #     Return the rate of balls of radius radius_ball containing at least 1 cell B
    #     """
    #     def _util_get_fraction_disk_containing_1_B(df_A,df_B,radius_ball):
    #         """
    #         Return the rate of balls of radius radius_ball containing at least 1 cell B
    #         """
    #         nb_balls = 0
    #         nb_balls_containing_1_B = 0
    #         for idx_A,cell_A in df_A.iterrows():
    #             nb_balls += 1
    #             df_B_in_ball = df_B[(df_B['x_roi_extended']-cell_A['x_roi_extended'])**2 + (df_B['y_roi_extended']-cell_A['y_roi_extended'])**2 <= radius_ball**2]
    #             if df_B_in_ball.shape[0] >= 1: #Je peux changer 1 pour q si je veux qu'il y ait au pmoins q cellules B dans la balle 
    #                 nb_balls_containing_1_B += 1
    #         rate_balls_containing_1_B = nb_balls_containing_1_B/nb_balls if nb_balls>0 else np.nan 
    #         return rate_balls_containing_1_B

    #     rate_balls_containing_1_B = np.zeros(len(dist_coloc_max_A))
    #     for idx_ball,radius_ball in enumerate(dist_coloc_max_A):
    #         rate_balls_containing_1_B[idx_ball] = _util_get_fraction_disk_containing_1_B(df_A,df_B,radius_ball)
    #     return rate_balls_containing_1_B