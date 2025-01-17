

import shutil 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from utils.util import *
from utils.util_colors_drawing import *
from utils.util import *
# from segmentation_classification.region_of_interest import RegionOfInterest
from sklearn.cluster import DBSCAN

from shapely.geometry import MultiPoint,MultiPolygon,polygon, Point, LineString
from scipy.spatial import distance
from simple_colors import *

import random
from scipy.ndimage.filters import gaussian_filter
from collections import Counter
from shapely.geometry.polygon import Polygon

from shapely import affinity
from plotly.subplots import make_subplots
import pickle


class DbscanAnalysis:

    """
    display_convex_hull(roi,dbscan_param_analysis,clustered_cells,min_sample,liste_convex_hull_to_display=liste_convex_hull_to_display,liste_cells_state_to_display=liste_cells_state_to_display,background = background,coef_alpha = 255, display_fig = display_fig,save_fig = save_fig,figsize=figsize,save_to_commun_path=None )
    fig_epsilon_opt_vs_other_epsilon(roi,dbscan_param_analysis,min_sample, save_fig = save_fig, display_fig = display_fig)
    create_gif_experiment(roi,dbscan_param_analysis,clustered_cells,min_sample,param_experiences_pts_removing,figsize=figsize,background=background, save_to_commun_path = None)
    BIG_GIF_create_gif_experiment(roi,dbscan_param_analysis,clustered_cells,min_sample,param_experiences_pts_removing,figsize=figsize,background=background, save_to_commun_path = None)
    
    fig_clustered_vs_isolated_cells(roi,clusters, min_sample = min_sample,save_fig = save_fig,display_fig = display_fig,save_to_commun_path=None)
    fig_area_clusters(roi,clusters, min_sample = min_sample,save_fig = save_fig,display_fig = display_fig,save_to_commun_path = None)
    fig_inside_vs_edge_cells_per_clusters(roi,clusters, min_sample = min_sample,save_fig = save_fig,display_fig = display_fig,save_to_commun_path = None)
    fig_cluster_n_divided_by_area(roi,clusters, min_sample = min_sample,save_fig = save_fig,display_fig = display_fig,save_to_commun_path = None)
    fig_conserved_area_after_rmv(roi,clusters, min_sample = min_sample, param_experiences_pts_removing=param_experiences_pts_removing, save_fig = False,display_fig = display_fig,save_to_commun_path = None)
    fig_histogram_conserved_area_after_rmv(clusters, min_sample = min_sample,param_experiences_pts_removing=param_experiences_pts_removing,roi = roi,save_fig = save_fig,display_fig = display_fig,save_to_commun_path=None)

    #results related metrics 
    stat_heatmap_IoU(roi,dbscan_report_on_roi,min_sample, param_experiences_pts_removing["threshold_conserved_area"], save_fig = save_fig,display=display_fig, save_to_commun_path = None )

    stat_heatmap_IoA(roi,dbscan_report_on_roi,min_sample, param_experiences_pts_removing["threshold_conserved_area"], save_fig = save_fig,display=display_fig, save_to_commun_path = None )
    stat_heatmap_fraction_A_in_B_clusters(roi,dbscan_report_on_roi,min_sample, param_experiences_pts_removing["threshold_conserved_area"], save_fig = save_fig,display=display_fig, save_to_commun_path = None )

    """
    colnames_epsilon_analysis = [
        "cell_type",
        "epsilon",
        "min_sample",
        "n_cluster",
        "n_clustered_cells",
        "n_isolated_cells",
        "is_optimal",
    ]
    colnames_cells_profil = [
        "cell_type",
        "min_sample",
        "epsilon",
        "x_roi_w_borders",
        "y_roi_w_borders",
        "id_cluster",
        "in_cluster_edge",
        "area_cluster"
    ]
    colnames_clusters = [
        "cell_type",
        "min_sample",
        "epsilon",
        "id_cluster",
        "area",
        "n_cells",
        "n_cells_edge",
        "n_cells_inside",
        "x_centroid", #of all cells (inside+edge) not only edge 
        "y_centroid" ,#of all cells (inside+edge) not only edge 
        "n_cells_divided_by_area",
        "ratio_removed_cells_robustess_test",
        "n_experiment_of_removing",
        "mean_area_after_rmv",
        "std_area_after_rmv",
        "mean_conserved_area_after_rmv", #average on experiments 
        "std_conserved_area_after_rmv"
        #Room for improvement here -> new metrics 

        "mean_x_centroid_after_rmv",
        "mean_y_centroid_after_rmv",
        "std_x_centroid_after_rmv",
        "std_y_centroid_after_rmv",
        "mean_distance_old_new_centroid",
        "std_distance_old_new_centroid",
    ]
    colnames_dbscan_statistics = [
        "type_A",
        "type_B",
        "epsilon_A",
        "min_sample_A",
        "epsilon_B",
        "min_sample_B",
        #Test robustess parameters 

        "threshold_conserved_area",
        "ratio_removed_cells_robustess_test",
        "n_experiment_of_removing",

        #Results 
        "n_robust_cluster_A",
        "n_robust_clustered_cells_A",
        "n_robust_isolated_cells_A",
        "fraction_robust_clustered_A",

        "n_robust_cluster_B",
        "n_robust_clustered_cells_B",
        "n_robust_isolated_cells_B",
        "fraction_robust_clustered_B",

        "n_all_cluster_A",
        "n_all_clustered_cells_A",
        "n_all_isolated_cells_A",
        "fraction_all_clustered_A",

        "n_all_cluster_B",
        "n_all_clustered_cells_B",
        "n_all_isolated_cells_B",
        "fraction_all_clustered_B",

        "fraction_robust_clusters_A",

        #Statistics on robust clusters 
        #number of cells 
        "mean_n_cells_robust_clusters_A",
        "std_n_cells_robust_clusters_A",
        "mean_n_edge_cells_robust_clusters_A",
        "std_n_edge_cells_robust_clusters_A",
        "mean_n_inside_cells_robust_clusters_A",
        #area
        "mean_area_robust_clusters_A",
        "std_area_robust_clusters_A",
        "mean_conserved_area_robust_clusters_A",  #average on clusters 
        "std_conserved_area_robust_clusters_A",
        #distances
        "mean_distance_old_new_centroid_robust_clusters_A", #average on clusters 
        "std_distance_old_new_centroid_robust_clusters_A",
        #Statistics on all clusters 
        #number of cells 
        "mean_n_cells_all_clusters_A",
        "std_n_cells_all_clusters_A",
        "mean_n_edge_cells_all_clusters_A",
        "std_n_edge_cells_all_clusters_A",
        "mean_n_inside_cells_all_clusters_A",
        #area
        "mean_area_all_clusters_A",
        "std_area_all_clusters_A",
        "mean_conserved_area_all_clusters_A",
        "std_conserved_area_all_clusters_A",
        #distances
        "mean_distance_old_new_centroid_all_clusters_A",
        "std_distance_old_new_centroid_all_clusters_A",

        #Statistics on robust clusters 
        #number of cells 
        "mean_n_cells_robust_clusters_B",
        "std_n_cells_robust_clusters_B",
        "mean_n_edge_cells_robust_clusters_B",
        "std_n_edge_cells_robust_clusters_B",
        "mean_n_inside_cells_robust_clusters_B",
        #area
        "mean_area_robust_clusters_B",
        "std_area_robust_clusters_B",
        "mean_conserved_area_robust_clusters_B",  #average on clusters 
        "std_conserved_area_robust_clusters_B",
        #distances
        "mean_distance_old_new_centroid_robust_clusters_B", #average on clusters 
        "std_distance_old_new_centroid_robust_clusters_B",
        #Statistics on all clusters 
        #number of cells 
        "mean_n_cells_all_clusters_B",
        "std_n_cells_all_clusters_B",
        "mean_n_edge_cells_all_clusters_B",
        "std_n_edge_cells_all_clusters_B",
        "mean_n_inside_cells_all_clusters_B",
        #area
        "mean_area_all_clusters_B",
        "std_area_all_clusters_B",
        "mean_conserved_area_all_clusters_B",
        "std_conserved_area_all_clusters_B",
        #distances
        "mean_distance_old_new_centroid_all_clusters_B",
        "std_distance_old_new_centroid_all_clusters_B",

        #A-B Statistics ONLY on robusts clusters 
        "iou", #Intersection convex hulls A et B / Union convex hulls 
        "i_A",  #Intersection convex hull / A convex hulls 
        "fraction_A_in_B_clusters", # all A cells in B convex hulls / All A cells
        "fraction_clustered_A_in_clustered_B", # clustered A cells in B convex hulls / All A cells

        "i_B", #Intersection convex hull / B convex hulls 
        "fraction_B_in_A_clusters", # all B cells in B convex hulls / All A cells
        "fraction_clustered_B_in_clustered_A" #
    ]
    def __init__(self, Roi, min_sample=None,cell_types_A=None,cell_types_B=None,range_epsilon_to_test=None, config_cluster_robustess_experiment=None):
        display_save_epsilon_analysis = True 
        min_sample = Roi.dataset_config.dbscan_based_analysis_config["min_sample"] if min_sample is None else min_sample
        cell_types_A = Roi.dataset_config.dbscan_based_analysis_config["cell_types_A"] if cell_types_A is None else cell_types_A
        cell_types_B = Roi.dataset_config.dbscan_based_analysis_config["cell_types_B"] if cell_types_B is None else cell_types_B
        range_epsilon_to_test = Roi.dataset_config.dbscan_based_analysis_config["range_epsilon_to_test"] if range_epsilon_to_test is None else range_epsilon_to_test
        config_cluster_robustess_experiment = Roi.dataset_config.dbscan_based_analysis_config["config_cluster_robustess_experiment"] if config_cluster_robustess_experiment is None else config_cluster_robustess_experiment

        self.roi = Roi
        self.dataset_config = self.roi.dataset_config
        self.min_sample = min_sample
        self.range_epsilon_to_test = range_epsilon_to_test
        self.config_cluster_robustess_experiment = config_cluster_robustess_experiment
        self.cell_types_A = cell_types_A
        self.cell_types_B = cell_types_B
        self.path_folder_dbscan = self._create_path_dbscan_analysis()[0]
        self.path_ms = self._create_path_dbscan_analysis()[1]

        self.epsilon_analysis_df, self.cells_profil, self.clusters = self.compute_dbscan_on_cell_type()
        self.dbscan_statistics = self.DBSCAN_analysis_cell_types_A_B()
        if display_save_epsilon_analysis : 
 
            self.fig_epsilon_opt_vs_other_epsilon(roi = self.roi,dbscan_param_analysis=self.epsilon_analysis_df,min_sample=self.min_sample, save_fig = True, display_fig = False)
    

    def _create_path_dbscan_analysis(self):
        """ """
        path_folder_dbscan = os.path.join(
            self.roi.path_roi, "3_DBSCAN"
        )
        path_ms = os.path.join(
            path_folder_dbscan, "min_sample_"+str(self.min_sample).zfill(2)
        )
        if not os.path.exists(path_folder_dbscan):
            os.mkdir(path_folder_dbscan)
        if not os.path.exists(path_ms):
            os.mkdir(path_ms)
        return path_folder_dbscan, path_ms

### DBSCAN method applied to cell type A 

    def compute_dbscan_on_cell_type(self):
        verbose = False

        epsilon_analysis_df = pd.DataFrame(columns = self.colnames_epsilon_analysis)
        cells_profil = pd.DataFrame(columns = self.colnames_cells_profil)
        clusters = pd.DataFrame(columns = self.colnames_clusters)

        for type_A in self.cell_types_A:
            print(green("DBSCAN analysis on cell type : "+str(type_A))) if verbose else None 
            if self.roi.get_cell_number_from_type(type_A) == 0:
                # print("There is smt to do because cell A is empty of cells")
                continue
            #Calcul de ce qui est utile partout ici 
            table_cells_A_w_borders = self.roi.table_cells_w_borders[self.roi.table_cells_w_borders["cell_type"] == self.roi.dataset_config.cell_class_names.index(type_A)+1]
            table_cells_A_w_borders.reset_index(drop = True, inplace = True)
            distance_matrix_A = pd.DataFrame(distance.cdist(table_cells_A_w_borders[['x_roi_w_borders','y_roi_w_borders']],table_cells_A_w_borders[['x_roi_w_borders','y_roi_w_borders']] ))
            epsilon_analysis_A , cells_profil_A = self._find_apply_optimal_dbscan(type_A, table_cells_A_w_borders, distance_matrix_A)
            cells_profil_A_with_experiment, clusters_A = self.experiment_on_clusters(cells_profil_A)
            # clusters_A = self.compute_cluster_stats(cells_profil_A)

            ###################
            clusters = pd.concat([clusters,clusters_A],axis = 0)
            epsilon_analysis_df = pd.concat([epsilon_analysis_df,epsilon_analysis_A],axis = 0)
            cells_profil = pd.concat([cells_profil,cells_profil_A_with_experiment],axis = 0)

        # dbscan_report_on_roi = self.compute_metrics_dbscan(clusters,epsilon_analysis_df,clustered_cells)  
        epsilon_analysis_df.to_csv(os.path.join(self.path_ms,"epsilon_analysis.csv"),sep = ';',index = False)
        cells_profil.to_csv(os.path.join(self.path_ms,"cells_profil.csv"),index = False,sep = ';')
        clusters.to_csv(os.path.join(self.path_ms,"clusters.csv"),index = False,sep = ';')

        return epsilon_analysis_df, cells_profil, clusters

    def _find_apply_optimal_dbscan(self,type_A, table_cells_A_w_borders, distance_matrix_A):
        """
        Find optimal epsilon for DBSCAN and apply it to the ROI
        return epsilon_analysis_A and cells_profil_A

        Input :     
        RegionOfInterest : classe dentrale dans la librairie : contient les positions (x,y) des cellules et leur state d'appartenance (Amoeboid, Proliferative, ...)
        epsilon : paramètre DBSCAN (rayon dans lequel il doit y avoir au moins min_samples points pour que le point d'intérêt soit considéré comme un core point
        min_sample : nombre de points qu'il doit y avoir dans epsilon 

        Output : 
        dbscan_param_analysis : DataFrame contenant (pop, eps, min_samples, n_cluster, n_isolated, opt)
        n_isolated : nombre de points qui n'est pas clusterisé 
        n_clusters : nombre de clusters de points 
        opt : si le epsilon est l'epsilon optimal ou non 

        Explication méthode : 
        1. https://fr.wikipedia.org/wiki/DBSCAN explication 
        2. https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80 Performances 

        """
        list_epsilon_analysis_A = []
        #Test all epsilon and save the results in a table
        for epsilon in self.range_epsilon_to_test:
            clustering_ = DBSCAN(eps=epsilon, min_samples=self.min_sample).fit(distance_matrix_A)
            label_clusters_ = clustering_.labels_
            epsilon_clustering_profil = dict()
            epsilon_clustering_profil["cell_type"] = type_A
            epsilon_clustering_profil["epsilon"] = epsilon
            epsilon_clustering_profil["min_sample"] = self.min_sample
            epsilon_clustering_profil["n_cluster"] = len(np.unique(label_clusters_))-1 if -1 in label_clusters_ else len(np.unique(label_clusters_))
            epsilon_clustering_profil["n_isolated_cells"] = np.sum(np.array(label_clusters_) == -1, axis=0)
            epsilon_clustering_profil["n_clustered_cells"] = len(label_clusters_) - epsilon_clustering_profil["n_isolated_cells"]
            list_epsilon_analysis_A.append(epsilon_clustering_profil)
        
        #Select optimal epsilon that maximize the number of clusters 
        epsilon_analysis_A = pd.DataFrame(list_epsilon_analysis_A,columns = self.colnames_epsilon_analysis)
        max_n_clusters = epsilon_analysis_A["n_cluster"].max()
        optimal_epsilon_A = np.mean(epsilon_analysis_A[epsilon_analysis_A["n_cluster"] == max_n_clusters]["epsilon"])
    
        #Add optimal epsilon to the table
        clustering_ = DBSCAN(eps=optimal_epsilon_A, min_samples=self.min_sample).fit(distance_matrix_A)
        label_clusters_ = clustering_.labels_
        epsilon_clustering_profil = dict()
        epsilon_clustering_profil["cell_type"] = type_A
        epsilon_clustering_profil["epsilon"] = epsilon
        epsilon_clustering_profil["min_sample"] = self.min_sample
        epsilon_clustering_profil["n_cluster"] = len(np.unique(label_clusters_))-1 if -1 in label_clusters_ else len(np.unique(label_clusters_))
        epsilon_clustering_profil["n_isolated_cells"] = np.sum(np.array(label_clusters_) == -1, axis=0)
        epsilon_clustering_profil["n_clustered_cells"] = len(label_clusters_) - epsilon_clustering_profil["n_isolated_cells"]
        
        if not optimal_epsilon_A in epsilon_analysis_A["epsilon"].values:
            optimal_epsilon_results = pd.DataFrame([epsilon_clustering_profil])
            epsilon_analysis_A = pd.concat([epsilon_analysis_A,optimal_epsilon_results],axis = 0)
        epsilon_analysis_A["is_optimal"] = (epsilon_analysis_A["epsilon"] == optimal_epsilon_A)
        
        cells_profil_A = pd.DataFrame({
            "cell_type":type_A,
            "min_sample":self.min_sample,
            "epsilon":optimal_epsilon_A,
            "x_roi_w_borders" : table_cells_A_w_borders['x_roi_w_borders'],
            "y_roi_w_borders" : table_cells_A_w_borders['y_roi_w_borders'],
            "id_cluster" : label_clusters_,
            "in_cluster_edge" : False,
            "area_cluster" : "to_do"
            })
        cells_profil_A = self._enrich_cells_profil(cells_profil_A)
        return epsilon_analysis_A , cells_profil_A

    def _enrich_cells_profil(self, cells_profil_A):
        """
        Add "in_cluster_edge" and "area_cluster" to cells_profil_A
        """
        cells_profil_A_ = pd.DataFrame(columns =self.colnames_cells_profil)
        for cluster_i in cells_profil_A["id_cluster"].unique(): #Partition de clustered_cells_
            cells_cluster_i = cells_profil_A[cells_profil_A["id_cluster"] == cluster_i]
            if cluster_i == -1: # No clustered cells
                cells_cluster_i["in_cluster_edge"] = False
                cells_cluster_i["area_cluster"] = np.nan
                cells_profil_A_ = pd.concat([cells_profil_A_,cells_cluster_i], axis = 0)
            else: #Cluster numeroted i 
                cells_cluster_i["in_cluster_edge"]
                points = np.asarray(cells_cluster_i[["x_roi_w_borders","y_roi_w_borders"]])
                pt_flipped = MultiPoint(np.flip(points, axis = -1))
                pt_flipped_reversed = affinity.scale(pt_flipped, xfact=1, yfact=-1, origin=(0,0))
                x_centroid = - pt_flipped_reversed.centroid.y 
                y_centroid = pt_flipped_reversed.centroid.x
                hull = pt_flipped_reversed.convex_hull
                if not isinstance(hull,Polygon):
                    continue 
                coords_edge_pts = list(hull.exterior.coords)
                for x_shapely, y_shapely in coords_edge_pts:
                    x , y = -y_shapely, x_shapely
                    cells_cluster_i["in_cluster_edge"][(cells_cluster_i["x_roi_w_borders"] == int(x)) & (cells_cluster_i["y_roi_w_borders"]== int(y))] = True
                cells_cluster_i["area_cluster"] = hull.area
                cells_cluster_i["x_centroid"] = int(x_centroid)
                cells_cluster_i["y_centroid"] = int(y_centroid)

                cells_profil_A_ = pd.concat([cells_profil_A_,cells_cluster_i], axis = 0)
        # cells_profil_A_.drop_duplicates(keep = 'first', inplace=True)
        return cells_profil_A_

    def experiment_on_clusters(self,cells_profil_A):
        """
        Remove percent of points in border of each cluster

        Create clusters_df with columns :
        ["cell_type","min_sample","epsilon","id_cluster","area","n_cells","n_cells_edge","n_cells_inside","n_cells_divided_by_area","ratio_removed_cells_robustess_test","n_experiment_of_removing","mean_conserved_area_after_rmv","mean_conserved_area_after_rmv"]
        """
        n_experiment_of_removing = self.config_cluster_robustess_experiment["n_experiment_of_removing"]
        ratio_removed_cells_robustess_test = self.config_cluster_robustess_experiment["ratio_removed_cells_robustess_test"]
        
        clusters = pd.DataFrame(columns = self.colnames_clusters)
        cells_profil_A_with_experiment = pd.DataFrame()
        for cluster_i in cells_profil_A["id_cluster"].unique(): #Partition de clustered_cells_
            if cluster_i == -1: #Isolated cells 
                cells_cluster_i = cells_profil_A[cells_profil_A["id_cluster"] == cluster_i]
                dict_cluster_i = dict()
                dict_cluster_i["id_cluster"] = cluster_i
                dict_cluster_i["n_cells"] = len(cells_cluster_i)
                dict_cluster_i["area"] = np.nan
                dict_cluster_i["n_cells_edge"] = np.nan
                dict_cluster_i["n_cells_inside"] = np.nan
                dict_cluster_i["x_centroid"] = np.nan
                dict_cluster_i["y_centroid"] = np.nan
                dict_cluster_i["n_cells_divided_by_area"] = np.nan
                for experiment in range(1,n_experiment_of_removing):    
                    cells_cluster_i["edge_cell_in_e"+str(experiment)] = False

                    dict_cluster_i["area_after_e"+str(experiment)] = np.nan
                    dict_cluster_i["area_conserved_e"+str(experiment)  ] = np.nan
                    dict_cluster_i["x_centroid_after_e"+str(experiment)] = np.nan
                    dict_cluster_i["y_centroid_after_e"+str(experiment)] = np.nan
                    dict_cluster_i["distance_old_new_centroid_e"+str(experiment)] = np.nan
            else : #Cluster 
                dict_cluster_i = dict()
                cells_cluster_i = cells_profil_A[cells_profil_A["id_cluster"] == cluster_i]
                dict_cluster_i["id_cluster"] = cluster_i
                dict_cluster_i["area"] = cells_cluster_i["area_cluster"].values[0]
                dict_cluster_i["n_cells"] = len(cells_cluster_i)
                dict_cluster_i["n_cells_edge"] = len(cells_cluster_i[cells_cluster_i["in_cluster_edge"] == 1])
                dict_cluster_i["n_cells_inside"] = len(cells_cluster_i[cells_cluster_i["in_cluster_edge"] == 0])
                dict_cluster_i["n_cells_divided_by_area"] = len(cells_cluster_i)/cells_cluster_i["area_cluster"].values[0]
                dict_cluster_i["x_centroid"] = cells_cluster_i["x_centroid"].values[0]
                dict_cluster_i["y_centroid"] = cells_cluster_i["y_centroid"].values[0]

                edge_cells_idx = list(cells_cluster_i[cells_cluster_i["in_cluster_edge"] == True].index)
                inside_cells_idx = list(cells_cluster_i[cells_cluster_i["in_cluster_edge"] == False].index)
                n_cells_to_keep = int(len(edge_cells_idx)*(1-ratio_removed_cells_robustess_test))
                for id_experiment in range(1,n_experiment_of_removing):   
                    list_cells_to_keep_as_edge = random.sample(edge_cells_idx, k=n_cells_to_keep)
                    idx_cells_cluster_i_after_rmv = list(list_cells_to_keep_as_edge) + list(inside_cells_idx)
                    cells_cluster_i, dict_cluster_i = self._compute_statistics_new_cluster_i(dict_cluster_i, cells_cluster_i, idx_cells_cluster_i_after_rmv, id_experiment) 
                dict_cluster_i = self.get_average_statistics_on_experiment(dict_cluster_i, n_experiment_of_removing)
            
            
            cells_profil_A_with_experiment = pd.concat([cells_profil_A_with_experiment,cells_cluster_i], axis = 0) #J'ajoute les clusters lignes apres lignes
            clusters = pd.concat([clusters,pd.DataFrame([dict_cluster_i])], axis = 0) #J'ajoute les clusters lignes apres lignes
        
        clusters["cell_type"] = cells_profil_A["cell_type"].values[0]
        clusters["min_sample"] = self.min_sample
        clusters["epsilon"] = cells_profil_A["epsilon"].values[0]
        clusters["ratio_removed_cells_robustess_test"] = ratio_removed_cells_robustess_test
        clusters["n_experiment_of_removing"] = n_experiment_of_removing
        return cells_profil_A_with_experiment, clusters
    
    def _compute_statistics_new_cluster_i(self, dict_cluster_i, cells_cluster_i, idx_cells_cluster_i_after_rmv, id_experiment):
        """
        Compute the area of the cluster and attribute 1 to a cell belonging to the border of the cluster and 0 otherwise 

        cells_cluster_i is returned only to create GIF of new clusters 
        """
        coord_cells_new_cluster_i = np.asarray(cells_cluster_i.loc[idx_cells_cluster_i_after_rmv][["x_roi_w_borders","y_roi_w_borders"]])
        coord_flipped = MultiPoint(np.flip(coord_cells_new_cluster_i, axis = -1))
        pt_flipped_reversed = affinity.scale(coord_flipped, xfact=1, yfact=-1, origin=(0,0))
        hull = pt_flipped_reversed.convex_hull
        if not isinstance(hull,Polygon):
            dict_cluster_i["x_centroid_after_e"+str(id_experiment)] = np.nan
            dict_cluster_i["y_centroid_after_e"+str(id_experiment)] = np.nan
            dict_cluster_i["area_after_e"+str(id_experiment)] = np.nan
            dict_cluster_i["area_conserved_e"+str(id_experiment)] = np.nan
            dict_cluster_i["distance_old_new_centroid_e"+str(id_experiment)] = np.nan
            cells_cluster_i["edge_cell_in_e"+str(id_experiment)] = False
            return cells_cluster_i, dict_cluster_i
        coords_edge_cells = list(hull.exterior.coords)
        cells_cluster_i["edge_cell_in_e"+str(id_experiment)] = False
        for x_shapely,y_shapely in coords_edge_cells:
            x,y = -y_shapely, x_shapely
            cells_cluster_i["edge_cell_in_e"+str(id_experiment)][(cells_cluster_i["x_roi_w_borders"]== int(x))&(cells_cluster_i["y_roi_w_borders"]== int(y))] = True
        
        #Statistics on this new cluster 

        x_new_centroid = - pt_flipped_reversed.centroid.y 
        y_new_centroid = pt_flipped_reversed.centroid.x

        dict_cluster_i["x_centroid_after_e"+str(id_experiment)] = int(x_new_centroid)
        dict_cluster_i["y_centroid_after_e"+str(id_experiment)] = int(y_new_centroid)
        dict_cluster_i["area_after_e"+str(id_experiment)] = hull.area
        dict_cluster_i["area_conserved_e"+str(id_experiment)] = hull.area/dict_cluster_i["area"]
        dict_cluster_i["distance_old_new_centroid_e"+str(id_experiment)] = np.sqrt((dict_cluster_i["x_centroid"]-x_new_centroid)**2+(dict_cluster_i["y_centroid"]-y_new_centroid)**2)
        return cells_cluster_i, dict_cluster_i

    def get_average_statistics_on_experiment(self, dict_cluster_i, n_experiment_of_removing): 
        list_area_after_rmv = []
        list_conserved_area_after_rmv = []
        list_distance_old_new_centroid = []
        list_x_centroid_after_rmv = []
        list_y_centroid_after_rmv = []
        for experiment in range(1,n_experiment_of_removing):
            list_area_after_rmv.append(dict_cluster_i["area_after_e"+str(experiment)])
            list_conserved_area_after_rmv.append(dict_cluster_i["area_conserved_e"+str(experiment)])
            list_distance_old_new_centroid.append(dict_cluster_i["distance_old_new_centroid_e"+str(experiment)])
            list_x_centroid_after_rmv.append(dict_cluster_i["x_centroid_after_e"+str(experiment)])
            list_y_centroid_after_rmv.append(dict_cluster_i["y_centroid_after_e"+str(experiment)])

        dict_cluster_i["mean_area_after_rmv"] = np.mean(list_area_after_rmv)
        dict_cluster_i["std_area_after_rmv"] = np.std(list_area_after_rmv)
        dict_cluster_i["mean_conserved_area_after_rmv"] = np.mean(list_conserved_area_after_rmv)
        dict_cluster_i["std_conserved_area_after_rmv"] = np.std(list_conserved_area_after_rmv)
        dict_cluster_i["mean_distance_old_new_centroid"] = np.mean(list_distance_old_new_centroid)
        dict_cluster_i["std_distance_old_new_centroid"] = np.std(list_distance_old_new_centroid)
        dict_cluster_i["mean_x_centroid_after_rmv"] = np.mean(list_x_centroid_after_rmv)
        dict_cluster_i["std_x_centroid_after_rmv"] = np.std(list_x_centroid_after_rmv)
        dict_cluster_i["mean_y_centroid_after_rmv"] = np.mean(list_y_centroid_after_rmv)
        dict_cluster_i["std_y_centroid_after_rmv"] = np.std(list_y_centroid_after_rmv)
        return dict_cluster_i

### DBSCAN statistics on cell type A and B
    def DBSCAN_analysis_cell_types_A_B(self):
        """  
        NOTE : je ne considère que les min sample qui sont les mêmes pour la population A et la population B (mais je pourrais très bien utiliser des min sampels différent) 
        self.epsilon_analysis_df, self.cells_profil, self.clusters
        """
        # print(blue("Statistical analysis : DBSCAN image {} and min sample {}...".format(self.roi.slide_num, self.min_sample)))

        threshold_conserved_area = self.config_cluster_robustess_experiment["threshold_conserved_area"]
        # clustered_cells_satisfying_criteria = self._filter_robust_clusters()
        list_results = []
        for type_A in self.cell_types_A:
            # print(red("A : "),type_A)
            if self.roi.get_cell_number_from_type(type_A) == 0:
                # print("DBSCAN_analysis_cell_types_A_B No cells A")
                continue
            if not self.roi.at_least_one_B_cell(self.cell_types_B):
                continue
            A_cells = self.cells_profil[self.cells_profil["cell_type"] == type_A]
            A_clusters = self.clusters[(self.clusters["cell_type"] == type_A)&(self.clusters["id_cluster"] != -1)]
            id_robust_A_clusters = list(self.clusters[(self.clusters["cell_type"] == type_A) & (self.clusters["mean_conserved_area_after_rmv"]>threshold_conserved_area)]["id_cluster"].unique())
            A_robust_clusters = A_clusters[A_clusters['id_cluster'].isin(id_robust_A_clusters)]
            A_robust_cells = A_cells[A_cells['id_cluster'].isin(id_robust_A_clusters)]

            multipol_convex_hull_A_clusters = self._create_convex_hull(A_cells,id_robust_A_clusters)
            cells_coord_A = self._create_cells_as_multipoints(A_cells)

            dict_A_results = dict()
            dict_A_results["type_A"] = type_A
            #dbscan parameters 
            dict_A_results["epsilon_A"] = A_cells["epsilon"].values[0]
            dict_A_results["min_sample_A"] = A_cells["min_sample"].values[0]
            #Results number of cells 
            dict_A_results["n_robust_cluster_A"]  = len(id_robust_A_clusters)
            dict_A_results["n_robust_clustered_cells_A"] = len(A_robust_cells)
            dict_A_results["n_robust_isolated_cells_A"] = len(A_cells) - len(A_robust_cells)
            dict_A_results["fraction_robust_clustered_A"] = dict_A_results["n_robust_clustered_cells_A"]/len(A_cells)

            dict_A_results["n_all_cluster_A"]  = len(A_clusters)
            dict_A_results["n_all_clustered_cells_A"] = len(A_cells[A_cells["id_cluster"] != -1])
            dict_A_results["n_all_isolated_cells_A"] = len(A_cells[A_cells["id_cluster"] == -1])
            dict_A_results["fraction_all_clustered_A"] = dict_A_results["n_all_clustered_cells_A"]/len(A_cells)
            #Statistics on robust clusters 
            # if dict_A_results["n_all_cluster_A"] == 0 : 

                # print("dict_A_results[n_all_cluster_A] vaut 0 -",type_A )
            dict_A_results["fraction_robust_clusters_A"] = len(id_robust_A_clusters)/dict_A_results["n_all_cluster_A"] if dict_A_results["n_all_cluster_A"] != 0 else np.nan
            #number of cells
            dict_A_results["mean_n_cells_robust_clusters_A"] = A_robust_clusters["n_cells"].mean()
            dict_A_results["std_n_cells_robust_clusters_A"] = A_robust_clusters["n_cells"].std()
            dict_A_results["mean_n_edge_cells_robust_clusters_A"] = A_robust_clusters["n_cells_edge"].mean()
            dict_A_results["std_n_edge_cells_robust_clusters_A"] = A_robust_clusters["n_cells_edge"].std()
            dict_A_results["mean_n_inside_cells_robust_clusters_A"] = A_robust_clusters["n_cells_inside"].mean()
            dict_A_results["std_n_inside_cells_robust_clusters_A"] = A_robust_clusters["n_cells_inside"].std()
            #area
            dict_A_results["mean_area_robust_clusters_A"] = A_robust_clusters["area"].mean()
            dict_A_results["std_area_robust_clusters_A"] = A_robust_clusters["area"].std()
            dict_A_results["mean_conserved_area_robust_clusters_A"] = A_robust_clusters["mean_conserved_area_after_rmv"].mean()
            dict_A_results["std_conserved_area_robust_clusters_A"] = A_robust_clusters["mean_conserved_area_after_rmv"].std()
            #distances
            dict_A_results["mean_distance_old_new_centroid_robust_clusters_A"] = A_robust_clusters["mean_distance_old_new_centroid"].mean()
            dict_A_results["std_distance_old_new_centroid_robust_clusters_A"] = A_robust_clusters["mean_distance_old_new_centroid"].std()
            #Statistics on all clusters 
            #number of cells
            dict_A_results["mean_n_cells_all_clusters_A"] = A_clusters["n_cells"].mean()
            dict_A_results["std_n_cells_all_clusters_A"] = A_clusters["n_cells"].std()
            dict_A_results["mean_n_edge_cells_all_clusters_A"] = A_clusters["n_cells_edge"].mean()
            dict_A_results["std_n_edge_cells_all_clusters_A"] = A_clusters["n_cells_edge"].std()
            dict_A_results["mean_n_inside_cells_all_clusters_A"] = A_clusters["n_cells_inside"].mean()
            dict_A_results["std_n_inside_cells_all_clusters_A"] = A_robust_clusters["n_cells_inside"].std()
            #area
            dict_A_results["mean_area_all_clusters_A"] = A_clusters["area"].mean()
            dict_A_results["std_area_all_clusters_A"] = A_clusters["area"].std()
            dict_A_results["mean_conserved_area_all_clusters_A"] = A_clusters["mean_conserved_area_after_rmv"].mean()
            dict_A_results["std_conserved_area_all_clusters_A"] = A_clusters["mean_conserved_area_after_rmv"].std()
            #distances
            dict_A_results["mean_distance_old_new_centroid_all_clusters_A"] = A_clusters["mean_distance_old_new_centroid"].mean()
            dict_A_results["std_distance_old_new_centroid_all_clusters_A"] = A_clusters["mean_distance_old_new_centroid"].std()

            for type_B in self.cell_types_B:
                # print(red("B : "),type_B)
                # if type_B == type_A: 
                #     continue 
                if self.roi.get_cell_number_from_type(type_B) == 0:
                    continue

                # print(green("DBSCAN analysis on cell type : "+str(type_A)+" and "+str(type_B)))
                B_cells = self.cells_profil[self.cells_profil["cell_type"] == type_B]
                B_clusters = self.clusters[(self.clusters["cell_type"] == type_B)&(self.clusters["id_cluster"] != -1)]
                id_robust_B_clusters = list(self.clusters[(self.clusters["cell_type"] == type_B) & (self.clusters["mean_conserved_area_after_rmv"]>threshold_conserved_area)]["id_cluster"].unique())
                B_robust_clusters = B_clusters[B_clusters['id_cluster'].isin(id_robust_B_clusters)]
                B_robust_cells = B_cells[B_cells['id_cluster'].isin(id_robust_B_clusters)]

                multipol_convex_hull_B_clusters = self._create_convex_hull(B_cells,id_robust_B_clusters)
                cells_coord_B = self._create_cells_as_multipoints(B_cells)

                dict_A_B_results = dict_A_results.copy()
                dict_A_B_results["type_B"] = type_B
                #dbscan parameters 
                dict_A_B_results["epsilon_B"] = B_cells["epsilon"].values[0]
                dict_A_B_results["min_sample_B"] = B_cells["min_sample"].values[0]
                #Results number of cells 
                dict_A_B_results["n_robust_cluster_B"]  = len(id_robust_B_clusters)
                dict_A_B_results["n_robust_clustered_cells_B"] = len(B_robust_cells)
                dict_A_B_results["n_robust_isolated_cells_B"] = len(B_cells) - len(B_robust_cells)
                dict_A_B_results["fraction_robust_clustered_B"] = dict_A_B_results["n_robust_clustered_cells_B"]/len(B_cells)

                dict_A_B_results["n_all_cluster_B"]  = len(B_clusters)
                dict_A_B_results["n_all_clustered_cells_B"] = len(B_cells[B_cells["id_cluster"] != -1])
                dict_A_B_results["n_all_isolated_cells_B"] = len(B_cells[B_cells["id_cluster"] == -1])
                dict_A_B_results["fraction_all_clustered_B"] = dict_A_B_results["n_all_clustered_cells_B"]/len(B_cells)
                
                #Statistics on robust clusters 
                dict_A_B_results["fraction_robust_clusters_B"] = len(id_robust_B_clusters)/dict_A_B_results["n_all_cluster_B"] if dict_A_B_results["n_all_cluster_B"] != 0 else np.nan
                #number of cells
                dict_A_B_results["mean_n_cells_robust_clusters_B"] = B_robust_clusters["n_cells"].mean()
                dict_A_B_results["std_n_cells_robust_clusters_B"] = B_robust_clusters["n_cells"].std()
                dict_A_B_results["mean_n_edge_cells_robust_clusters_B"] = B_robust_clusters["n_cells_edge"].mean()
                dict_A_B_results["std_n_edge_cells_robust_clusters_B"] = B_robust_clusters["n_cells_edge"].std()
                dict_A_B_results["mean_n_inside_cells_robust_clusters_B"] = B_robust_clusters["n_cells_inside"].mean()
                dict_A_B_results["std_n_inside_cells_robust_clusters_B"] = B_robust_clusters["n_cells_inside"].std()
                #area
                dict_A_B_results["mean_area_robust_clusters_B"] = B_robust_clusters["area"].mean()
                dict_A_B_results["std_area_robust_clusters_B"] = B_robust_clusters["area"].std()
                dict_A_B_results["mean_conserved_area_robust_clusters_B"] = B_robust_clusters["mean_conserved_area_after_rmv"].mean()
                dict_A_B_results["std_conserved_area_robust_clusters_B"] = B_robust_clusters["mean_conserved_area_after_rmv"].std()
                #distances
                dict_A_B_results["mean_distance_old_new_centroid_robust_clusters_B"] = B_robust_clusters["mean_distance_old_new_centroid"].mean()
                dict_A_B_results["std_distance_old_new_centroid_robust_clusters_B"] = B_robust_clusters["mean_distance_old_new_centroid"].std()
                
                #Statistics on all clusters 
                #number of cells
                dict_A_B_results["mean_n_cells_all_clusters_B"] = B_clusters["n_cells"].mean()
                dict_A_B_results["std_n_cells_all_clusters_B"] = B_clusters["n_cells"].std()
                dict_A_B_results["mean_n_edge_cells_all_clusters_B"] = B_clusters["n_cells_edge"].mean()
                dict_A_B_results["std_n_edge_cells_all_clusters_B"] = B_clusters["n_cells_edge"].std()
                dict_A_B_results["mean_n_inside_cells_all_clusters_B"] = B_clusters["n_cells_inside"].mean()
                dict_A_B_results["std_n_inside_cells_all_clusters_B"] = B_robust_clusters["n_cells_inside"].std()
                #area
                dict_A_B_results["mean_area_all_clusters_B"] = B_clusters["area"].mean()
                dict_A_B_results["std_area_all_clusters_B"] = B_clusters["area"].std()
                dict_A_B_results["mean_conserved_area_all_clusters_B"] = B_clusters["mean_conserved_area_after_rmv"].mean()
                dict_A_B_results["std_conserved_area_all_clusters_B"] = B_clusters["mean_conserved_area_after_rmv"].std()
                #distances
                dict_A_B_results["mean_distance_old_new_centroid_all_clusters_B"] = B_clusters["mean_distance_old_new_centroid"].mean()
                dict_A_B_results["std_distance_old_new_centroid_all_clusters_B"] = B_clusters["mean_distance_old_new_centroid"].std()

                #A-B Statistics 
                if multipol_convex_hull_A_clusters == None or multipol_convex_hull_B_clusters == None :
                    dict_A_B_results["iou"], dict_A_B_results["i_A"], dict_A_B_results["i_B"] = 0, 0, 0
                    dict_A_B_results["fraction_A_in_B_clusters"] = 0 if multipol_convex_hull_B_clusters == None else self._compute_A_in_B_clusters(cells_coord_A,multipol_convex_hull_B_clusters)
                    dict_A_B_results["fraction_B_in_A_clusters"] = 0 if multipol_convex_hull_A_clusters == None else self._compute_A_in_B_clusters(cells_coord_B,multipol_convex_hull_A_clusters)  
                    dict_A_B_results["fraction_clustered_A_in_clustered_B"] = 0
                    dict_A_B_results["fraction_clustered_B_in_clustered_A"] = 0
                else : 
                    dict_A_B_results["iou"] = multipol_convex_hull_A_clusters.intersection(multipol_convex_hull_B_clusters).area/multipol_convex_hull_A_clusters.union(multipol_convex_hull_B_clusters).area
                    dict_A_B_results["i_A"] = multipol_convex_hull_A_clusters.intersection(multipol_convex_hull_B_clusters).area/multipol_convex_hull_A_clusters.area
                    dict_A_B_results["i_B"] = multipol_convex_hull_A_clusters.intersection(multipol_convex_hull_B_clusters).area/multipol_convex_hull_B_clusters.area
                    dict_A_B_results["fraction_A_in_B_clusters"] = self._compute_A_in_B_clusters(cells_coord_A,multipol_convex_hull_B_clusters)
                    dict_A_B_results["fraction_B_in_A_clusters"] = self._compute_A_in_B_clusters(cells_coord_B,multipol_convex_hull_A_clusters)
                    dict_A_B_results["fraction_clustered_A_in_clustered_B"] = self._compute_fraction_clustered_A_in_intersection(cells_coord_A,multipol_convex_hull_A_clusters,multipol_convex_hull_B_clusters) 

                    dict_A_B_results["fraction_clustered_B_in_clustered_A"] = self._compute_fraction_clustered_A_in_intersection(cells_coord_B,multipol_convex_hull_B_clusters,multipol_convex_hull_A_clusters)
                list_results.append(dict_A_B_results)
        dbscan_statistics = pd.DataFrame(list_results, columns = self.colnames_dbscan_statistics)
        dbscan_statistics["threshold_conserved_area"] = threshold_conserved_area
        dbscan_statistics["ratio_removed_cells_robustess_test"] = self.config_cluster_robustess_experiment["ratio_removed_cells_robustess_test"]
        dbscan_statistics["n_experiment_of_removing"] = self.config_cluster_robustess_experiment["n_experiment_of_removing"]

        dbscan_statistics.to_csv(os.path.join(self.path_ms,"dbscan_statistics.csv"),sep = ';',index = False)
        return dbscan_statistics

    def _create_convex_hull(self,A_cells,id_robust_clusters): 
        """
        Créer un Multi Polynome des enveloppes convexes de la population A dans la config DBSCAN (epsilon_A,min_sample_A) à partir des polynomes stockés dans convex_hulls_dict
        Empèche ls polynomes invalides grace aux unions (sinon overlap de polynomes )
        """
        list_convex_hull_clusters = []
        for id_cluster in id_robust_clusters:
            cells_robust_clusters_A = A_cells[A_cells["id_cluster"] == id_cluster]
            coord_A = np.asarray(cells_robust_clusters_A[["x_roi_w_borders","y_roi_w_borders"]])
            pt_flipped = MultiPoint(np.flip(coord_A, axis = -1))
            pt_flipped_reversed = affinity.scale(pt_flipped, xfact=1, yfact=-1, origin=(0,0))
            hull = pt_flipped_reversed.convex_hull
            list_convex_hull_clusters.append(hull)
        if len(list_convex_hull_clusters) == 0 :
            return None
        Multipol_convex_hull_clusters = list_convex_hull_clusters[0]
        if len(list_convex_hull_clusters)>1:
            for pol in list_convex_hull_clusters[1:]:
                Multipol_convex_hull_clusters = pol.union(Multipol_convex_hull_clusters)
        return Multipol_convex_hull_clusters

    def _create_cells_as_multipoints(self, A_cells):
        """Creer les multipoints shapely d'un state de cellules à partir du clustered_cells dans la config DBSCAN (epsilon_A,min_sample_A) (qui n'a pas d'effets sur les points de la population !)
        """
        coord_A_np = np.asarray(A_cells[["x_roi_w_borders","y_roi_w_borders"]])
        coord_A_temp = MultiPoint(np.flip(coord_A_np, axis = -1))
        coord_A_image = affinity.scale(coord_A_temp, xfact=1, yfact=-1, origin=(0,0))
        return coord_A_image

    def _compute_A_in_B_clusters(self,coord_A,multipol_convex_hull_B_clusters):
        numerateur = coord_A.intersection(multipol_convex_hull_B_clusters)
        if numerateur.is_empty:
            return 0
        elif isinstance(numerateur, Point):
            return 1/len(coord_A.geoms)
            # return 1
        else : 

            tx = len(numerateur.geoms)/len(coord_A.geoms)
            return tx 
        
    # def _compute_fraction_clustered_A_in_intersection(self,coord_A,multipol_convex_hull_A_clusters,multipol_convex_hull_B_clusters):
    #     """ Calcul le ratio de la population A dans l'intersection des enveloppes convexes de A et B"""
    #     clustered_A = coord_A.intersection(multipol_convex_hull_A_clusters)
    #     clustered_A_in_clustered_B_area = clustered_A.intersection(multipol_convex_hull_B_clusters)  
    #     if clustered_A_in_clustered_B_area.is_empty:
    #         return 0
    #     elif isinstance(clustered_A_in_clustered_B_area, Point):
    #         return 1
    #     else :
    #         tx = len(clustered_A_in_clustered_B_area.geoms)/len(coord_A.geoms)
    #         return tx
    def _compute_fraction_clustered_A_in_intersection(self,coord_A,multipol_convex_hull_A_clusters,multipol_convex_hull_B_clusters):
        """ Calcul le ratio de la population A dans l'intersection des enveloppes convexes de A et B"""
        clustered_A = coord_A.intersection(multipol_convex_hull_A_clusters)
        
        clustered_A_in_clustered_B_area = clustered_A.intersection(multipol_convex_hull_B_clusters)  
        if clustered_A_in_clustered_B_area.is_empty:
            return 0
        elif isinstance(clustered_A_in_clustered_B_area, Point):
            return 1/len(clustered_A.geoms)
        else :
            tx = len(clustered_A_in_clustered_B_area.geoms)/len(clustered_A.geoms)
            return tx

## Displaying functions (all todo) 

    def display_convex_hull(self, list_cell_types_draw=None, list_clusters_to_draw=None,figsize = (30,30),with_background = True, robust_clusters_only = True,output_path_name="colocalisation",roi_category="all",color_name="dataset_config" ): 
        """   
        """
        display_fig = False
        to_expo = False 
        with_roi_delimiter = False 
        with_anatomical_part_mask = False 
        with_center_of_mass = False
        with_tiles_delimitation = False

        draw_cells = True
        if list_clusters_to_draw == None:
            list_clusters_to_draw =  self.cell_types_A
        if list_cell_types_draw == None:
            list_cell_types_draw = self.cell_types_A

        if not hasattr(self.roi,"image_w_borders"): 
            with_background = False
        if with_background : 
            if self.dataset_config.consider_image_with_channels :
                background = self.roi.image_w_borders["RGB"]
            else :
                # background = np.dstack([self.image_w_borders[:,:,1], self.image_w_borders[:,:,1], self.image_w_borders[:,:,1]])
                background = self.roi.image_w_borders

        # if with_background : 
        #     if self.dataset_config.data_type == "fluorescence":
        #         background = self.roi.image_w_borders["RGB"]
        #     elif self.dataset_config.data_type == "wsi": 
        #         background = self.roi.image_w_borders
        #     else : 
        #         raise Exception("There is no code for this kind of data")
        else : 
            background = np.zeros((self.roi.roi_w_borders_shape[0],self.roi.roi_w_borders_shape[1],3))
        
        background_pil = np_to_pil(background)
        drawing = ImageDraw.Draw(background_pil, "RGBA")
        #Drawing convex hulls 
        for cell_type in list_clusters_to_draw:
            img = self._add_polygons_hull_to_draw(drawing,cell_type,robust_clusters_only,color_name=color_name)
        #Drawing cells 
        if draw_cells : 
            for cell_type in list_cell_types_draw:
                img = draw_cells_on_img(self.roi,img,cell_type_filter = cell_type,color_name=color_name)

        if with_roi_delimiter : 
            drawing = draw_roi_delimiter(self.roi, drawing)
        if with_anatomical_part_mask:
            drawing = draw_anatomical_part_mask(self.roi,drawing)
        if with_center_of_mass : 
            drawing = draw_center_of_mass(self.roi,drawing)
        if with_tiles_delimitation:
            drawing = with_tiles_delimitations(self.roi, drawing)

        fig = plt.figure(figsize=figsize)#, tight_layout = True)
        plt.imshow(background_pil)
        plt.axis('off')

        if robust_clusters_only : 
            fig_name = "dbscan_ms_"+str(self.min_sample)+"_clusters_r"+str(self.config_cluster_robustess_experiment["ratio_removed_cells_robustess_test"])+"_th"+str(self.config_cluster_robustess_experiment["threshold_conserved_area"])+"_robust_clusters_only"
        else : 
            fig_name = "dbscan_ms_"+str(self.min_sample)+"_all_clusters"

        if to_expo : 
            directory = os.path.join(self.dataset_config.dir_output,OUTPUT_EXPO_NAME,output_path_name,roi_category)
            os.makedirs(directory,exist_ok=True)
            figname = "s"+str(self.roi.slide_num).zfill(3)+"_ro" +str(self.roi.origin_row).zfill(3) + "_co" +str(self.roi.origin_col).zfill(3) + "_re" +str(self.roi.end_row).zfill(3) + "_ce" +str(self.roi.end_col).zfill(3)

            path_save = find_path_last(directory,figname+"_"+fig_name)
            fig.savefig(path_save,
                        facecolor='white',
                        dpi="figure",
                        bbox_inches='tight',
                        pad_inches=0.1)
        else : 

            path_save = find_path_last(self.roi.path_roi, fig_name)
            fig.savefig(path_save,
                        facecolor='white',
                        dpi="figure",
                        bbox_inches='tight',
                        pad_inches=0.1)
        plt.imshow(background_pil) if display_fig else plt.close("all") 
        plt.axis('off')

    def _add_polygons_hull_to_draw(self, drawing, cell_type, robust_clusters_only,color_name="dataset_config"):
        """
        Draw convex hull 
        """
        coef_alpha = 170
        if color_name == "dataset_config":
            color_convex_hull = create_colors(coef_alpha, self.dataset_config.mapping_cells_colors)
        else : 
            color_convex_hull = create_colors(coef_alpha, dict_colors[color_name])

        cells = self.cells_profil[self.cells_profil["cell_type"]==cell_type]
        if robust_clusters_only:
            threshold_conserved_area = self.config_cluster_robustess_experiment["threshold_conserved_area"]
            A_clusters = self.clusters[(self.clusters["cell_type"] == cell_type)&(self.clusters["id_cluster"] != -1)]
            id_robust_A_clusters = list(A_clusters[A_clusters["mean_conserved_area_after_rmv"]>threshold_conserved_area]["id_cluster"].unique())
            A_robust_clusters = A_clusters[A_clusters['id_cluster'].isin(id_robust_A_clusters)]
            cells = cells[cells["id_cluster"].isin(id_robust_A_clusters)]

        coord_cells = self._create_list_coord_by_cluster_id(cells) 
        if len(coord_cells) == 0:
            return drawing  
        convex_hull_list = []
        for points in coord_cells:
            Multipoints = MultiPoint(np.flip(points, axis = -1))
            hull = Multipoints.convex_hull
            convex_hull_list.append(hull)
        Multipol_A = convex_hull_list[0]
        for pol in convex_hull_list[1:]:
                if isinstance(pol,LineString):
                        # Linesstring_list.append(pol.coords[:])
                        continue
                Multipol_A = pol.union(Multipol_A)
        if isinstance(Multipol_A,MultiPolygon):
                Multipol_ = [list(poly.exterior.coords) for poly in list(Multipol_A.geoms)]
        else : 
                Multipol_ = [Multipol_A.exterior.coords]
        for points in Multipol_:
                drawing.polygon(points,fill=color_convex_hull[cell_type], outline ="black")
        return drawing
    
    def _create_list_coord_by_cluster_id(self,cells):
        coord_cells = []
        for id_cluster in cells["id_cluster"].unique():
            if id_cluster != -1 : #Noisy points are excluded 
                coord_cells.append(np.asarray(cells[cells["id_cluster"]==id_cluster][["x_roi_w_borders","y_roi_w_borders"]]))
        return coord_cells


    def fig_epsilon_opt_vs_other_epsilon(self,roi, dbscan_param_analysis,min_sample, save_fig = False, display_fig = False,save_to_commun_path = None):
        """
        J'aurais pu le faire avec des make_subplots comme partout c'est plus facile a gérer je crpis 
        Créer la figure de display du nombre de noisy points en fonction du nombre de clusters créer en fonction des paramètres de DBSCAN (MinSample et epsilon)
        """
        symbols = ['circle', 'hexagram']
        symbol_list = [symbols[k] for k in dbscan_param_analysis["is_optimal"]]
        color_size = list(dbscan_param_analysis["epsilon"])
        def custom_legend_name(fig_stat):
            """      
            Je peux manipuler ce que je veux en allant voir le dictionnaire de fig.data et en changeant les trucs 
            """
            num_legend_circle = 0 
            num_legend_hexagram = 0 
            for i in range(len(fig_stat.data)):
                if fig_stat.data[i].marker.symbol == "circle":
                    fig_stat.data[i].name = "Epsilon"
                    num_legend_circle+=1
                if fig_stat.data[i].marker.symbol == "diamond":
                    fig_stat.data[i].marker.symbol = "hexagram"
                    fig_stat.data[i].name = "Epsilon Optimal"
                    if num_legend_hexagram == 0:
                        fig_stat.data[i].showlegend = True
                        num_legend_hexagram+=1

        fig_stat = px.scatter(dbscan_param_analysis, x="n_cluster", y="n_isolated_cells", color=color_size,symbol = symbol_list,size =color_size, facet_row="min_sample",facet_col="cell_type",facet_row_spacing = 0.01, facet_col_spacing = 0.05)
        fig_stat.update_xaxes(matches=None,showticklabels=True)
        fig_stat.update_yaxes(side="left",title_text = "N isolated cells",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig_stat.update_xaxes(side="bottom",title_text = "N cluster",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig_stat.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=-0.1,ticks="inside",ticksuffix="",title="Epsilon"),legend=dict(orientation="h"),legend_title_text='Legend')
        fig_stat.update_layout(title_text="<b>Epsilon optimal vs other epsilon on n cluster and n isolated cells <br> Min sample = "+str(min_sample),title_x=0.5,title_font=dict(size=30),
                showlegend=True,
                width=1700,
                height=600,margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=140,
                    pad=4))
        custom_legend_name(fig_stat)
        if save_to_commun_path is not None : 
            filepath = os.path.join(save_to_commun_path,"epsilon_opt_vs_other_epsilon_ms_"+str(min_sample)+".png")
            fig_stat.write_image(filepath)


        if save_fig :

            fig_name = "epsilon_opt_vs_other_epsilon_ms_"+str(min_sample).zfill(2)+".png"
            path_save = find_path_last(self.roi.path_roi, fig_name)


            # path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
            # path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
            # mkdir_if_nexist(path_save_clustering)
            # mkdir_if_nexist(path_save_clustering_ms)
            # path_dbscan_param_analysis_file = os.path.join(path_save_clustering_ms, "Statistics")
            # mkdir_if_nexist(path_dbscan_param_analysis_file)
            # filepath = os.path.join(path_dbscan_param_analysis_file,"epsilon_opt_vs_other_epsilon_ms_"+str(min_sample).zfill(2)+".png")
            fig_stat.write_image(path_save)
        if display_fig :
            fig_stat.show()

    def create_gif_experiment(roi,dbscan_param_analysis,clustered_cells_with_experiment,min_sample,param_experiences_pts_removing,figsize=(15,15),background=True, save_to_commun_path = None):
        percent = param_experiences_pts_removing["percent"]
        n_experiment = param_experiences_pts_removing["n_experiment"]

        path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
        path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
        path_save_clustering_experiment = os.path.join(path_save_clustering_ms, "Removing_"+str(percent)+"_percent_edge_points_clusters")
        path_save_clustering_experiment_min_sample = os.path.join(path_save_clustering_experiment, "min_sample_"+str(min_sample).zfill(2))
        mkdir_if_nexist(path_save_clustering)
        mkdir_if_nexist(path_save_clustering_ms)
        mkdir_if_nexist(path_save_clustering_experiment)
        mkdir_if_nexist(path_save_clustering_experiment_min_sample)
        if not hasattr(roi,"img_rgb_borders"):
            background = False
        for exp in range(1,n_experiment):
            colname_exp = "e"+str(exp)+"_rmv_"+str(percent)+"percent"
            path_save_clustering_experiment_min_sample_pop_name_exp = os.path.join(path_save_clustering_experiment_min_sample, "experiment_"+str(exp)+".png")
            display_convex_hull(roi,dbscan_param_analysis,clustered_cells_with_experiment[clustered_cells_with_experiment[colname_exp]==True],min_sample,liste_convex_hull_to_display=["Amoeboid", "Proliferative", "Cluster", "Phagocytic", "Ramified"]  ,liste_cells_state_to_display=["Amoeboid", "Proliferative", "Cluster", "Phagocytic", "Ramified"] ,background = background,coef_alpha = 255, display_fig = False,save_fig = False,figsize=figsize,save_to_commun_path=path_save_clustering_experiment_min_sample_pop_name_exp,exp_number = exp,percent=percent )

        list_frames = []
        for exp in range(1,n_experiment):
            path_save_clustering_experiment_min_sample_pop_name_exp = os.path.join(path_save_clustering_experiment_min_sample, "experiment_"+str(exp)+".png")
            img = Image.open(path_save_clustering_experiment_min_sample_pop_name_exp) 
            list_frames.append(img)

        frame_one = list_frames[0]
        if save_to_commun_path is not None : 
            path_to_save = os.path.join(save_to_commun_path, "Clusters_area_evolution_ms_"+str(min_sample)+".gif")
        else : 
            path_to_save = os.path.join(path_save_clustering_experiment, "Clusters_area_evolution_ms_"+str(min_sample)+".gif")

        frame_one.save(path_to_save, format="GIF", append_images=list_frames,save_all=True, duration=300, loop=0)
        # Supression des images ayant servi a créer le GIF 
        for exp in range(1,n_experiment):
            path_save_clustering_experiment_min_sample_pop_name_exp = os.path.join(path_save_clustering_experiment_min_sample, "experiment_"+str(exp)+".png")
            os.remove(path_save_clustering_experiment_min_sample_pop_name_exp)
        shutil.rmtree(path_save_clustering_experiment_min_sample)

    def BIG_GIF_create_gif_experiment(roi,dbscan_param_analysis,clustered_cells_with_experiment,min_sample,param_experiences_pts_removing,figsize=(15,15),background=True, save_to_commun_path = None):
        percent = param_experiences_pts_removing["percent"]
        n_experiment = param_experiences_pts_removing["n_experiment"]

        path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
        path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
        path_save_clustering_experiment = os.path.join(path_save_clustering_ms, "Removing_"+str(percent)+"_percent_edge_points_clusters")
        path_save_clustering_experiment_min_sample = os.path.join(path_save_clustering_experiment, "min_sample_"+str(min_sample).zfill(2))
        mkdir_if_nexist(path_save_clustering)
        mkdir_if_nexist(path_save_clustering_ms)
        mkdir_if_nexist(path_save_clustering_experiment)
        mkdir_if_nexist(path_save_clustering_experiment_min_sample)
        if not hasattr(roi,"img_rgb_borders"):
            background = False
        for exp in range(1,n_experiment):
            colname_exp = "e"+str(exp)+"_rmv_"+str(percent)+"percent"
            path_save_clustering_experiment_min_sample_pop_name_exp = os.path.join(path_save_clustering_experiment_min_sample, "experiment_"+str(exp)+".png")
            path_figure = os.path.join(roi.path_classification,"0_Images_ROI")
            last_fig_number = get_last_fig_number(path_figure)
            name_fig = str(last_fig_number+1).zfill(3)+"_experiment_"+str(exp)+"_ms_"+str(min_sample)+".png"
            path = os.path.join(path_figure,name_fig)
            id_img = 1
            while os.path.exists(path):
                path = os.path.join(path_figure,str(id_img).zfill(2)+"_"+name_fig)
                id_img+=1
            display_convex_hull(roi,dbscan_param_analysis,clustered_cells_with_experiment[clustered_cells_with_experiment[colname_exp]==True],min_sample,liste_convex_hull_to_display=["Amoeboid", "Proliferative", "Cluster", "Phagocytic", "Ramified"]  ,liste_cells_state_to_display=["Amoeboid", "Proliferative", "Cluster", "Phagocytic", "Ramified"] ,background = background,coef_alpha = 255, display_fig = False,save_fig = False,figsize=figsize,save_to_commun_path=path,exp_number = exp,percent=percent )

## Displaying statistics 

    def fig_clustered_vs_isolated_cells(roi,clusters, min_sample = 5,save_fig = False,display_fig = False,save_to_commun_path=None):
        labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
        fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.1,horizontal_spacing = 0.06)
        
        i = 1
        if len(clusters) == 0:
            fig.update_layout(title_text="<b>There is no cells in the ROI - DBSCAN",title_x=0.5,title_font=dict(size=30))
            return fig
        for pop_A in labels:
            df = clusters.query("pop_name == @pop_A & min_sample == @min_sample")
            # print("pop_Avpop_A",pop_A,len(df))
            if len(df) <= 2:
                # print("Pas de cellules")
                continue
            fig.add_trace(go.Bar(x = ["Isolated"],y = [df.query("cluster_number < 0")["n_cells"].sum()],texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker_color = COLORS_STATES_PLOTLY[i]), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.add_trace(go.Bar(x = ["Clustered"],y = [df.query("cluster_number >= 0")["n_in_border"].sum()],texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker_color = COLORS_STATES_PLOTLY[i]), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.add_trace(go.Bar(x = ["Clustered"],y = [df.query("cluster_number >= 0")["n_inside"].sum()],texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i]), row=(i-1)//3+1, col=(i-1)%3+1)

            i+=1
        fig.update_layout(title_text="<b>Clustered vs Isolated cells - DBSCAN<br> Min sample = "+str(min_sample),title_x=0.5,title_font=dict(size=30),
                    showlegend=False,
                    width=1200,
                    height=700,margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=140,
                        pad=4))
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig.update_layout(barmode='stack')
        fig.update_yaxes(side="left",title_text = "Number of cells",title_standoff=0,tickfont=dict(size=15),showticklabels=True)

        if save_to_commun_path is not None:
            filepath = os.path.join(save_to_commun_path,"Isolated_vs_Clustered_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)
        if save_fig :
            path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
            path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
            mkdir_if_nexist(path_save_clustering)
            path_dbscan_statistics = os.path.join(path_save_clustering_ms, "Statistics")
            mkdir_if_nexist(path_dbscan_statistics)
            filepath = os.path.join(path_dbscan_statistics,"Isolated_vs_Clustered_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)

        if display_fig :
            fig.show()

    def fig_area_clusters(roi,clusters, min_sample = 5,save_fig = False,display_fig = True,save_to_commun_path = None):
        """ Todo : ajouter des "no cells quand il n'y a pas de cellules dans la ROI"""
        labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
        fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.15,horizontal_spacing = 0.1)
        if len(clusters) == 0:
            fig.update_layout(title_text="<b>There is no cells in the ROI - DBSCAN",title_x=0.5,title_font=dict(size=30))
            return fig
        i = 1
        for pop_A in labels:
            df = clusters[(clusters["pop_name"]==pop_A) & (clusters["min_sample"]==min_sample)]
            df = df[df["cluster_number"] != -1]
            if len(df) == 0:
                continue

                # fig.add_trace(go.Histogram(x = df[df.in_border ==1]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY_TRANSPARENCY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)
                # fig.add_trace(go.Histogram(x = df[df.in_border ==0]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)

            fig.add_trace(go.Bar(
            name =labels[i-1],
            x = df["cluster_number"],
            y = df["area"],
            text= "",
            marker_color = COLORS_STATES_PLOTLY[i]
            ), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.update_layout(title_text="<b>Clusters area - DBSCAN<br>Min sample = "+str(min_sample),title_x=0.5,title_font=dict(size=30),
                showlegend=False,
                width=1300,
                height=700,margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=170,
                    pad=4
                )
            )
            i+=1
        fig.update_layout(barmode='stack')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig.update_yaxes(side="left",title_text = "Area",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_xaxes(side="bottom",title_text = "N° cluster",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        if save_to_commun_path is not None:
            filepath = os.path.join(save_to_commun_path,"Clusters_area_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)

        if save_fig :
            path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
            path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
            mkdir_if_nexist(path_save_clustering)
            mkdir_if_nexist(path_save_clustering_ms)
            path_dbscan_statistics = os.path.join(path_save_clustering_ms, "Statistics")
            mkdir_if_nexist(path_dbscan_statistics)
            filepath = os.path.join(path_dbscan_statistics,"Clusters_area_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)

        if display_fig :
            fig.show()

    def fig_inside_vs_edge_cells_per_clusters(roi,clusters, min_sample = 5,save_fig = False,display_fig = True,save_to_commun_path = None):
        labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
        fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.15,horizontal_spacing = 0.1)
        if len(clusters) == 0:
            fig.update_layout(title_text="<b>There is no cells in the ROI - DBSCAN",title_x=0.5,title_font=dict(size=30))
            return fig
        i = 1
        for pop_A in labels:
            df = clusters[(clusters["pop_name"]==pop_A) & (clusters["min_sample"]==min_sample)]
            df = df[df["cluster_number"] != -1]
            if len(df) == 0:
                continue

                # fig.add_trace(go.Histogram(x = df[df.in_border ==1]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY_TRANSPARENCY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)
                # fig.add_trace(go.Histogram(x = df[df.in_border ==0]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)

            fig.add_trace(go.Bar(
            name =labels[i-1],
            x = df["cluster_number"],
            y = df["n_inside"],texttemplate="%{y}",
            text= "",
            marker_color = COLORS_STATES_PLOTLY[i]
            ), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.add_trace(go.Bar(
            name =labels[i-1],
            x = df["cluster_number"],
            y = df["n_in_border"],texttemplate="%{y}",
            text= "",
            marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i]
            ), row=(i-1)//3+1, col=(i-1)%3+1)

            fig.update_layout(title_text="<b>Inside (opaque) vs edge cells (transparent) for each cluster<br>Min sample = "+str(min_sample),title_x=0.5,title_font=dict(size=30),
                showlegend=False,
                width=1300,
                height=700,margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=170,
                    pad=4
                )
            )
            i+=1
        fig.update_layout(barmode='stack')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig.update_yaxes(side="left",title_text = "Number of cells",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_xaxes(side="bottom",title_text = "N° cluster",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        if save_to_commun_path is not None:
            filepath = os.path.join(save_to_commun_path,"n_Inside_vs_edge_cells_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)
        if save_fig :
            path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
            path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
            mkdir_if_nexist(path_save_clustering)
            mkdir_if_nexist(path_save_clustering_ms)
            path_dbscan_statistics = os.path.join(path_save_clustering_ms, "Statistics")
            mkdir_if_nexist(path_dbscan_statistics)
            filepath = os.path.join(path_dbscan_statistics,"n_Inside_vs_edge_cells_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)

        if display_fig :
            fig.show()

    def fig_cluster_n_divided_by_area(roi,clusters, min_sample = 5,save_fig = False,display_fig = True,save_to_commun_path = None):
        labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
        fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.15,horizontal_spacing = 0.1)
        if len(clusters) == 0:
            fig.update_layout(title_text="<b>There is no cells in the ROI - DBSCAN",title_x=0.5,title_font=dict(size=30))
            return fig
        i = 1
        for pop_A in labels:
            df = clusters[(clusters["pop_name"]==pop_A) & (clusters["min_sample"]==min_sample)]
            df = df[df["cluster_number"] != -1]
            if len(df) == 0:
                continue

                # fig.add_trace(go.Histogram(x = df[df.in_border ==1]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY_TRANSPARENCY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)
                # fig.add_trace(go.Histogram(x = df[df.in_border ==0]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)

            fig.add_trace(go.Bar(
            name =labels[i-1],
            x = df["cluster_number"],
            y = df["n_in_border"]/df["area"],
            text= "",
            marker_color = COLORS_STATES_PLOTLY_TRANSPARENCY[i]
            ), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.add_trace(go.Bar(
            name =labels[i-1],
            x = df["cluster_number"],
            y = df["n_inside"]/df["area"],
            text= "",
            marker_color = COLORS_STATES_PLOTLY[i]
            ), row=(i-1)//3+1, col=(i-1)%3+1)
            fig.update_layout(title_text="<b>Inside (opaque) vs edge cells (transparent) divided by cluster size<br>Min sample = "+str(min_sample),title_x=0.5,title_font=dict(size=30),
                showlegend=False,
                width=1200,
                height=700,margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=170,
                    pad=4
                )
            )
            i+=1
        fig.update_yaxes(side="left",title_text = "Cell number / area",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_xaxes(side="bottom",title_text = "N° cluster",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_layout(barmode='stack')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        if save_to_commun_path is not None:
            filepath = os.path.join(save_to_commun_path,"Inside_vs_edge_cells_divided_by_area_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)
        if save_fig :
            path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
            path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
            mkdir_if_nexist(path_save_clustering)
            mkdir_if_nexist(path_save_clustering_ms)
            path_dbscan_statistics = os.path.join(path_save_clustering_ms, "Statistics")
            mkdir_if_nexist(path_dbscan_statistics)
            filepath = os.path.join(path_dbscan_statistics,"Inside_vs_edge_cells_divided_by_area_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)

        if display_fig :
            fig.show()

    def fig_conserved_area_after_rmv(roi,clusters, min_sample, param_experiences_pts_removing,save_fig = False,display_fig = True,save_to_commun_path = None):
        labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
        percent =param_experiences_pts_removing["percent"]
        n_experiment = param_experiences_pts_removing["n_experiment"]
        fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.15,horizontal_spacing = 0.1)
        if len(clusters) == 0:
            fig.update_layout(title_text="<b>There is no cells in the ROI - DBSCAN",title_x=0.5,title_font=dict(size=30))
            return fig
        conserved_area_threshold = 0.6
        i = 1
        for pop_A in labels:
            df = clusters[(clusters["pop_name"]==pop_A) & (clusters["min_sample"]==min_sample)]
            df = df[df["cluster_number"] != -1]
            if len(df) == 0:
                continue
            # fig.add_trace(go.Histogram(x = df[df.in_border ==1]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY_TRANSPARENCY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)
            # fig.add_trace(go.Histogram(x = df[df.in_border ==0]["label_clusterisation"].apply(lambda x: str(x) if x != -1 else "Isolated"),histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)

            fig.add_trace(go.Bar(
            name =labels[i-1],
            x = df["cluster_number"],
            y = df["mean_conserved_area_after_rmv_10"],
            text= "",error_y=dict(type='data', array=df["std_conserved_area_after_rmv_10"]),
            marker_color = COLORS_STATES_PLOTLY[i]
            ), row=(i-1)//3+1, col=(i-1)%3+1)
            
            i+=1
        fig.update_layout(title_text="<b>Averaged % of conserved area after removing "+str(percent)+"% of the cells at the edges of the clusters<br>Min sample = "+str(min_sample)+" - "+str(n_experiment) + " experiments",title_x=0.5,title_font=dict(size=25),
                    showlegend=False,
                    width=1400,
                    height=700,margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=170,
                        pad=4
                    )
                )
        fig.update_yaxes(side="left",title_text = "% conserved area",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_xaxes(side="bottom",title_text = "N° cluster",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_layout(barmode='stack')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig.add_hline(y=conserved_area_threshold, line_width=4, line_dash='dash', line_color='black')
        if save_to_commun_path is not None:
            filepath = os.path.join(save_to_commun_path,"Conserved_area_after_rmv_per_clusters_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)
        if save_fig :
            path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
            path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
            mkdir_if_nexist(path_save_clustering)
            mkdir_if_nexist(path_save_clustering_ms)
            path_dbscan_statistics = os.path.join(path_save_clustering_ms, "Statistics")
            mkdir_if_nexist(path_dbscan_statistics)
            filepath = os.path.join(path_dbscan_statistics,"Conserved_area_after_rmv_per_clusters_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)

        if display_fig :
            fig.show()

    def fig_histogram_conserved_area_after_rmv(df_all_clusters, min_sample,param_experiences_pts_removing,roi = None, save_fig = False,display_fig = False,save_to_commun_path=None):
        labels = ["Proliferative", "Amoeboid", "Cluster", "Phagocytic", "Ramified"]
        percent = param_experiences_pts_removing["percent"]
        n_experiment = param_experiences_pts_removing["n_experiment"]
        fig = make_subplots(2, 3,subplot_titles=[c for c in labels],vertical_spacing = 0.2,horizontal_spacing = 0.06)
        if len(df_all_clusters) == 0:
            fig.update_layout(title_text="<b>There is no cells in the ROI - DBSCAN",title_x=0.5,title_font=dict(size=30))
            return fig
        conserved_area_threshold = 0.6
        i = 1
        percent = 10
        for pop_A in labels:
            df = df_all_clusters.query("pop_name == @pop_A & min_sample == @min_sample")
            if len(df) == 0:
                continue
            fig.add_trace(go.Histogram(x = df["mean_conserved_area_after_rmv_10"],histnorm = 'density',texttemplate="%{y}" ,name = '',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)

            # fig.add_trace(go.Histogram(x = df[df.in_border ==1]["label_clusterisation"].apply(lambda x: "Clustered cells" if x != -1 else "Isolated"),histnorm = 'density',name = 'Grass Bar',opacity = 0.9,xaxis = 'x1',yaxis = 'y1',texttemplate="%{y}" ,marker = go.histogram.Marker({"color" :COLORS_STATES_PLOTLY[i]})), row=(i-1)//3+1, col=(i-1)%3+1)
            i+=1
        fig.update_layout(title_text="<b>Histogram % of conserved cluster's area after removing "+str(percent)+"% of the cells at the edges of the clusters<br> Min sample = "+str(min_sample)+ " - "+str(n_experiment)+" experiments",title_x=0.5,title_font=dict(size=20),
                    showlegend=False,
                    width=1200,
                    height=700,margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=140,
                        pad=4))
        fig.add_vline(x=conserved_area_threshold, line_width=4, line_dash='dash', line_color='black')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig.update_layout(barmode='stack')
        fig.update_yaxes(side="left",title_text = "Number of cluster",title_standoff=0,tickfont=dict(size=15),showticklabels=True)
        fig.update_xaxes(side="bottom",title_text = "Conserved area",title_standoff=0,tickfont=dict(size=15),showticklabels=True)

        if save_to_commun_path is not None:
            filepath = os.path.join(save_to_commun_path,"Histogram_conserved_area_ms_"+str(min_sample).zfill(2)+".png")
            fig.write_image(filepath)
        if save_fig :
            if roi is not None : 
                path_save_clustering = os.path.join(roi.path_classification, "3_DBSCAN")
                path_save_clustering_ms = os.path.join(path_save_clustering, "min_sample_"+str(min_sample).zfill(2))
                mkdir_if_nexist(path_save_clustering)
                mkdir_if_nexist(path_save_clustering_ms)
                path_dbscan_statistics = os.path.join(path_save_clustering_ms, "Statistics")
                mkdir_if_nexist(path_dbscan_statistics)
                filepath = os.path.join(path_dbscan_statistics,"Histogram_conserved_area_ms_"+str(min_sample).zfill(2)+".png")
                fig.write_image(filepath)
            else :
                print(red("No roi given, cannot save the figure"))
        if display_fig :
            fig.show()

