import numpy as np
from scipy.spatial import KDTree

def get_clusters(filled_surface_sites, nn_threshold) -> list[int]:
    coordinates_np = np.array([entry[1] for entry in filled_surface_sites])
    kdtree = KDTree(coordinates_np)
    visited_indices = set()
    clusters = []

    def dfs(index, cluster):
        if index in visited_indices:
            return
        visited_indices.add(index)
        cluster.append(index)

        neighbors = kdtree.query_ball_point(coordinates_np[index], r=nn_threshold, p=2)
        for neighbor_index in neighbors:
            if neighbor_index != index and filled_surface_sites[neighbor_index][0] == filled_surface_sites[index][0]:
                dfs(neighbor_index, cluster)

    for index in range(len(filled_surface_sites)):
        if index not in visited_indices:
            new_cluster = []
            dfs(index, new_cluster)
            if new_cluster:
                clusters.append(new_cluster)

    return clusters

def analyze_clusters(filled_surface_sites, nn_threshold, num_ligands_1, num_ligands_2) -> tuple[list[float], list[float]]:
    type1_clusters=[]
    type2_clusters=[]
    ligand_types = set(entry[0] for entry in filled_surface_sites)
    for ligand_type in ligand_types:
        ligand_sites = [entry for entry in filled_surface_sites if entry[0] == ligand_type]
        ligand_clusters = get_clusters(ligand_sites, nn_threshold)

        ligand_clusters = [cluster for cluster in ligand_clusters if len(cluster) > 1]
        ligand_cluster_sizes = [len(cluster) for cluster in ligand_clusters]
        print(ligand_cluster_sizes)
        
        if ligand_type == 'ch3':
            ligand_cluster_fractions = [cluster/num_ligands_1 for cluster in ligand_cluster_sizes]
            type1_clusters.extend(ligand_cluster_fractions)
        elif ligand_type == 'oh':
            ligand_cluster_fractions = [cluster/num_ligands_2 for cluster in ligand_cluster_sizes]
            type2_clusters.extend(ligand_cluster_fractions)

    return type1_clusters, type2_clusters