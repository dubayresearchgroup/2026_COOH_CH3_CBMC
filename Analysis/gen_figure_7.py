import numpy as np
import matplotlib.pyplot as plt
import cluster_functions
from matplotlib.cbook import boxplot_stats

#variables
trials = 3
anchor_type = 5
num_configs = 1
replicates = 1
fragment_size = 4
threshold = 6
nbins = 75

num_ligands_1 = [274,263,256,227] # the number of alkanethiol ligands for DDT/MPA, DDT/MHA, HT/MUA, and DDT/MUA
num_ligands_2 = [274,263,256,227] # the number of carboxylic acid ligands for DDT/MPA, DDT/MHA, HT/MUA, and DDT/MUA
num_ligands_1_len = [13,13,7,13] # length of ligand type 1 (the alkanethiol)
num_ligands_2_len = [7,10,15,15] # length of ligand type 2 (the carboxylic acid)
num_np_atoms = 2456 # number of metal nanoparticle atoms 

monolayers = ['ddt_mpa', 'ddt_mha','ht_mua','ddt_mua']
conf_types = ['janus', 'random', 'striped']

def get_clusters() -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float],
                            list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:

    """
    Function that determines the cluster size fraction of each ligand type in the binary monolayer by using the sulfur anchor positions, a KDTree, and a First Depth Search algo
    
    :return: lists containing the number weighted average cluster size fractions found as well along with lists that contain raw cluster size fractions
    :rtype: tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]
    """

    #initialize empty arrays
    ddtmpa_all_clusters, ddtmha_all_clusters, htmua_all_clusters, ddtmua_all_clusters = [], [], [], []
    ddt_mpa_1_clusters, ddt_mpa_2_clusters = [], []
    ddt_mua_1_clusters, ddt_mua_2_clusters = [], []
    ddt_mha_1_clusters, ddt_mha_2_clusters = [], []
    ht_mua_1_clusters, ht_mua_2_clusters = [], []
    ddt_mpa_1_nw, ddt_mpa_2_nw = [], []
    ddt_mua_1_nw, ddt_mua_2_nw = [], []
    ddt_mha_1_nw, ddt_mha_2_nw = [], []
    ht_mua_1_nw, ht_mua_2_nw = [], []
    
    for i, monolayer in enumerate(monolayers):
        for j, conf_type in enumerate(conf_types):
            for trial in range(trials):
                filepath = f'../Datafiles/{monolayer}/{conf_type}/t{trial+1}/trajectory.xyz'
                data = []
                with open(filepath, 'r') as file:
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1':
                                data.append(columns)
                total_lines = len(data)
                print("Total lines:", total_lines)
                raw_data = np.array(data, dtype=float)
                for config in range(num_configs):
                    start_line = total_lines - (config+1) * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                    end_line = start_line + (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                    lig_sites = raw_data[start_line:end_line, 1:4]
                    type_sites = raw_data[start_line:end_line, 0]
                    ch3_s_sites = []
                    oh_s_sites = [] 
                    count_s = 0
                    for k in range(len(type_sites)):
                        if type_sites[k] == anchor_type:
                            count_s += 1
                            if count_s <= num_ligands_1[i]:
                                ch3_s_sites.append(lig_sites[k])
                            else:
                                oh_s_sites.append(lig_sites[k])
                            
                    oh_s_sites = np.array(oh_s_sites)
                    ch3_s_sites = np.array(ch3_s_sites)
                    print(f'{monolayer}/{conf_type}/{trial+1} : #CH3 S anchors in array: ', len(ch3_s_sites))
                    print(f'{monolayer}/{conf_type}/{trial+1} : #COOH S anchors in array: ', len(oh_s_sites))
                    ch3_s_sites_occupied = {tuple(site) for site in ch3_s_sites}
                    oh_s_sites_occupied = {tuple(site) for site in oh_s_sites}
                    filled_s_sites = [['oh', list(site)] for site in oh_s_sites_occupied] + [['ch3', list(site)] for site in ch3_s_sites_occupied]
                    type1_clusters, type2_clusters = cluster_functions.analyze_clusters(filled_surface_sites=filled_s_sites, nn_threshold=threshold, num_ligands_1=num_ligands_1[i], num_ligands_2=num_ligands_2[i])
                    if monolayer == 'ddt_mpa':
                        ddtmpa_all_clusters.append([type1_clusters, type2_clusters])
                        ddt_mpa_1_clusters.append(type1_clusters)
                        ddt_mpa_2_clusters.append(type1_clusters)
                        ddt_mpa_1_nw.append((sum(n ** 2 for n in type1_clusters) / sum(type1_clusters)))
                        ddt_mpa_2_nw.append((sum(n ** 2 for n in type2_clusters) / sum(type2_clusters)))
                    elif monolayer == 'ddt_mha':
                        ddtmha_all_clusters.append([type1_clusters, type2_clusters])
                        ddt_mha_1_clusters.append(type1_clusters)
                        ddt_mha_2_clusters.append(type2_clusters)
                        ddt_mha_1_nw.append((sum(n ** 2 for n in type1_clusters) / sum(type1_clusters)))
                        ddt_mha_2_nw.append((sum(n ** 2 for n in type2_clusters) / sum(type2_clusters)))
                    elif monolayer == 'ddt_mua':
                        ddtmua_all_clusters.append([type1_clusters, type2_clusters])
                        ddt_mua_1_clusters.append(type1_clusters)
                        ddt_mua_2_clusters.append(type2_clusters)
                        ddt_mua_1_nw.append((sum(n ** 2 for n in type1_clusters) / sum(type1_clusters)))
                        ddt_mua_2_nw.append((sum(n ** 2 for n in type2_clusters) / sum(type2_clusters)))
                    else:
                        htmua_all_clusters.append([type1_clusters, type2_clusters])
                        ht_mua_1_clusters.append(type1_clusters)
                        ht_mua_2_clusters.append(type2_clusters)
                        ht_mua_1_nw.append((sum(n ** 2 for n in type1_clusters) / sum(type1_clusters)))
                        ht_mua_2_nw.append((sum(n ** 2 for n in type2_clusters) / sum(type2_clusters)))
    return ddt_mpa_1_nw, ddt_mpa_2_nw, ddt_mha_1_nw, ddt_mha_2_nw, ht_mua_1_nw, ht_mua_2_nw, ddt_mua_1_nw, ddt_mua_2_nw, ddt_mpa_1_clusters, ddt_mpa_2_clusters, ddt_mha_1_clusters, ddt_mha_2_clusters, ht_mua_1_clusters, ht_mua_2_clusters, ddt_mua_1_clusters, ddt_mua_2_clusters

def plot(ddt_mpa_1_nw, ddt_mpa_2_nw, ddt_mha_1_nw, ddt_mha_2_nw, ht_mua_1_nw, ht_mua_2_nw, ddt_mua_1_nw, ddt_mua_2_nw, ddt_mpa_1_clusters, ddt_mpa_2_clusters, ddt_mha_1_clusters, ddt_mha_2_clusters, ht_mua_1_clusters, ht_mua_2_clusters, ddt_mua_1_clusters, ddt_mua_2_clusters) -> None:
    """
    Plot the number weighted cluster size fraction as a boxplot along with the raw cluster size fractions found during the analysis
    """
    
    plt.figure()
    box_1 = plt.boxplot(ddt_mpa_1_nw, positions=[0.145], vert=True, patch_artist=True, widths=0.04)
    box_2 = plt.boxplot(ddt_mpa_2_nw, positions=[0.245], vert=True, patch_artist=True, widths=0.04)
    box_1['boxes'][0].set_facecolor((0, 0, 0, 0.6))  # black with alpha 0.6
    box_2['boxes'][0].set_facecolor((1, 0, 0, 0.6))
    box_1['medians'][0].set_color('k')
    box_2['medians'][0].set_color('k')
    box_1['medians'][0].set_linewidth(2)
    box_2['medians'][0].set_linewidth(2)
    box_3 = plt.boxplot(ddt_mha_1_nw, positions=[0.345], vert=True, patch_artist=True, widths=0.04)
    box_4 = plt.boxplot(ddt_mha_2_nw, positions=[0.445], vert=True, patch_artist=True, widths=0.04)
    box_3['boxes'][0].set_facecolor((0, 0, 0, 0.6))
    box_4['boxes'][0].set_facecolor((1, 0, 0, 0.6))
    box_3['medians'][0].set_color('k')
    box_4['medians'][0].set_color('k')
    box_3['medians'][0].set_linewidth(2)
    box_4['medians'][0].set_linewidth(2)
    box_5 = plt.boxplot(ht_mua_1_nw, positions=[0.545], vert=True, patch_artist=True, widths=0.04)
    box_6 = plt.boxplot(ht_mua_2_nw, positions=[0.645], vert=True, patch_artist=True, widths=0.04)
    box_5['boxes'][0].set_facecolor((0, 0, 0, 0.6))
    box_6['boxes'][0].set_facecolor((1, 0, 0, 0.6))
    box_5['medians'][0].set_color('k')
    box_6['medians'][0].set_color('k')
    box_5['medians'][0].set_linewidth(2)
    box_6['medians'][0].set_linewidth(2)
    box_7 = plt.boxplot(ddt_mua_1_nw, positions=[0.745], vert=True, patch_artist=True,  widths=0.04)
    box_8 = plt.boxplot(ddt_mua_2_nw, positions=[0.845], vert=True, patch_artist=True,  widths=0.04)
    box_7['boxes'][0].set_facecolor((0, 0, 0, 0.6))
    box_8['boxes'][0].set_facecolor((1, 0, 0, 0.6))
    box_7['medians'][0].set_color('k')
    box_8['medians'][0].set_color('k')
    box_7['medians'][0].set_linewidth(2)
    box_8['medians'][0].set_linewidth(2)
    x_ch3_0, x_ch3_1, x_ch3_2, x_ch3_3 = 0.11, 0.31, 0.51, 0.71 
    x_cooh_0, x_cooh_1, x_cooh_2, x_cooh_3 = 0.21, 0.41, 0.61, 0.81
    for i in range(9):
        if i == 0:
            plt.scatter(x=[x_ch3_0]*len(ddt_mpa_1_clusters[i]), y=ddt_mpa_1_clusters[i], color='k', marker='o', label='-CH$_3$', s=10) #type: ignore
            plt.scatter(x=[x_cooh_0]*len(ddt_mpa_2_clusters[i]), y=ddt_mpa_2_clusters[i], color='r', marker='o', label='-COOH', s=10) #type: ignore
        else:
            plt.scatter(x=[x_ch3_0]*len(ddt_mpa_1_clusters[i]), y=ddt_mpa_1_clusters[i], color='k', marker='o', s=10) #type: ignore
            plt.scatter(x=[x_cooh_0]*len(ddt_mpa_2_clusters[i]), y=ddt_mpa_2_clusters[i], color='r', marker='o', s=10) #type: ignore

        plt.scatter(x=[x_ch3_1]*len(ddt_mha_1_clusters[i]), y=ddt_mha_1_clusters[i], color='k', marker='o', s=10) #type: ignore
        plt.scatter(x=[x_cooh_1]*len(ddt_mha_2_clusters[i]), y=ddt_mha_2_clusters[i], color='r', marker='o', s=10) #type: ignore

        plt.scatter(x=[x_ch3_2]*len(ht_mua_1_clusters[i]), y=ht_mua_1_clusters[i], color='k', marker='o', s=10) #type: ignore
        plt.scatter(x=[x_cooh_2]*len(ht_mua_2_clusters[i]), y=ht_mua_2_clusters[i], color='r', marker='o', s=10) #type: ignore

        plt.scatter(x=[x_ch3_3]*len(ddt_mua_1_clusters[i]), y=ddt_mua_1_clusters[i], color='k', marker='o', s=10) #type: ignore
        plt.scatter(x=[x_cooh_3]*len(ddt_mua_2_clusters[i]), y=ddt_mua_2_clusters[i], color='r', marker='o', s=10) #type: ignore
        x_ch3_0 += 0.01
        x_cooh_0 += 0.01
        x_ch3_1 += 0.01
        x_cooh_1 += 0.01
        x_ch3_2 += 0.01
        x_cooh_2 += 0.01
        x_ch3_3 += 0.01
        x_cooh_3 += 0.01
    x_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    [plt.axvline(x, color='k', linestyle='--', alpha=0.5) for x in x_vals]
    plt.xticks([0.2,0.4,0.6,0.8],['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA'])
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.ylabel('Cluster Size Fraction', fontsize=14)
    plt.xlabel('Monolayers', fontsize=14)
    plt.legend(frameon=False, fontsize='large', bbox_to_anchor=(0.5,1.1), ncols=2, loc='upper center')
    plt.xlim(0.05,0.95)
    plt.tight_layout()
    plt.savefig(f'../Figures/NW_boxplots_thresh_{threshold}A.png', format='png', dpi=600)

    return print(f"Number weighted cluster plot saved to ../Figures/NW_boxplots_thresh_{threshold}A.png")

def write_stats(ddt_mpa_1_nw, ddt_mpa_2_nw, ddt_mha_1_nw, ddt_mha_2_nw, ht_mua_1_nw, ht_mua_2_nw, ddt_mua_1_nw, ddt_mua_2_nw, ddt_mpa_1_clusters, ddt_mpa_2_clusters, ddt_mha_1_clusters, ddt_mha_2_clusters, ht_mua_1_clusters, ht_mua_2_clusters, ddt_mua_1_clusters, ddt_mua_2_clusters) -> None:
    """
    Save results to text file ../Datafiles/cluster_data.txt
    """

    ddt_mpa_1_boxplot_stats, ddt_mpa_2_boxplot_stats = boxplot_stats(ddt_mpa_1_nw)[0], boxplot_stats(ddt_mpa_2_nw)[0]
    ddt_mha_1_boxplot_stats, ddt_mha_2_boxplot_stats = boxplot_stats(ddt_mha_1_nw)[0], boxplot_stats(ddt_mha_2_nw)[0]
    ht_mua_1_boxplot_stats, ht_mua_2_boxplot_stats = boxplot_stats(ht_mua_1_nw)[0], boxplot_stats(ht_mua_2_nw)[0]
    ddt_mua_1_boxplot_stats, ddt_mua_2_boxplot_stats = boxplot_stats(ddt_mua_1_nw)[0], boxplot_stats(ddt_mua_2_nw)[0]

    with open("../Datafiles/cluster_data.txt", "w") as file: 
        file.write("Cluster information\n")
        file.write("Cluster\tClusters_found\tLow_Whisker\tNW_Q1\tNW_Median\tNW_Q3\tHigh_Whisker\n")
        file.write(f'DDT/MPA Alkane\t{sum(len(c) for c in ddt_mpa_1_clusters)}\t'
                f'{ddt_mpa_1_boxplot_stats["whislo"]}\t{ddt_mpa_1_boxplot_stats["q1"]}\t{ddt_mpa_1_boxplot_stats["med"]}\t{ddt_mpa_1_boxplot_stats["q3"]}\t{ddt_mpa_1_boxplot_stats["whishi"]}\n')
        file.write(f'DDT/MPA Carboxylic Acid\t{sum(len(c) for c in ddt_mpa_2_clusters)}\t'
                f'{ddt_mpa_2_boxplot_stats["whislo"]}\t{ddt_mpa_2_boxplot_stats["q1"]}\t{ddt_mpa_2_boxplot_stats["med"]}\t{ddt_mpa_2_boxplot_stats["q3"]}\t{ddt_mpa_2_boxplot_stats["whishi"]}\n\n')
        file.write(f'DDT/MHA Alkane\t{sum(len(c) for c in ddt_mha_1_clusters)}\t'
                f'{ddt_mha_1_boxplot_stats["whislo"]}\t{ddt_mha_1_boxplot_stats["q1"]}\t{ddt_mha_1_boxplot_stats["med"]}\t{ddt_mha_1_boxplot_stats["q3"]}\t{ddt_mha_1_boxplot_stats["whishi"]}\n')
        file.write(f'DDT/MHA Carboxylic Acid\t{sum(len(c) for c in ddt_mha_2_clusters)}\t'
                f'{ddt_mha_2_boxplot_stats["whislo"]}\t{ddt_mha_2_boxplot_stats["q1"]}\t{ddt_mha_2_boxplot_stats["med"]}\t{ddt_mha_2_boxplot_stats["q3"]}\t{ddt_mha_2_boxplot_stats["whishi"]}\n\n')
        file.write(f'HT/MUA Alkane\t{sum(len(c) for c in ht_mua_1_clusters)}\t'
                f'{ht_mua_1_boxplot_stats["whislo"]}\t{ht_mua_1_boxplot_stats["q1"]}\t{ht_mua_1_boxplot_stats["med"]}\t{ht_mua_1_boxplot_stats["q3"]}\t{ht_mua_1_boxplot_stats["whishi"]}\n')
        file.write(f'HT/MUA Carboxylic Acid\t{sum(len(c) for c in ht_mua_2_clusters)}\t'
                f'{ht_mua_2_boxplot_stats["whislo"]}\t{ht_mua_2_boxplot_stats["q1"]}\t{ht_mua_2_boxplot_stats["med"]}\t{ht_mua_2_boxplot_stats["q3"]}\t{ht_mua_2_boxplot_stats["whishi"]}\n\n')
        file.write(f'DDT/MUA Alkane\t{sum(len(c) for c in ddt_mua_1_clusters)}\t'
                f'{ddt_mua_1_boxplot_stats["whislo"]}\t{ddt_mua_1_boxplot_stats["q1"]}\t{ddt_mua_1_boxplot_stats["med"]}\t{ddt_mua_1_boxplot_stats["q3"]}\t{ddt_mua_1_boxplot_stats["whishi"]}\n')
        file.write(f'DDT/MUA Carboxylic Acid\t{sum(len(c) for c in ddt_mua_2_clusters)}\t'
                f'{ddt_mua_2_boxplot_stats["whislo"]}\t{ddt_mua_2_boxplot_stats["q1"]}\t{ddt_mua_2_boxplot_stats["med"]}\t{ddt_mua_2_boxplot_stats["q3"]}\t{ddt_mua_2_boxplot_stats["whishi"]}\n')

    return 

def main() -> None:
    """
    This python script generates figure 7 of the manuscript.
    
    Computes and plots the Cluster Statistics of the four monolayers studied
    """
    ddt_mpa_1_nw, ddt_mpa_2_nw, ddt_mha_1_nw, ddt_mha_2_nw, ht_mua_1_nw, ht_mua_2_nw, ddt_mua_1_nw, ddt_mua_2_nw, ddt_mpa_1_clusters, ddt_mpa_2_clusters, ddt_mha_1_clusters, ddt_mha_2_clusters, ht_mua_1_clusters, ht_mua_2_clusters, ddt_mua_1_clusters, ddt_mua_2_clusters= get_clusters()
    plot(ddt_mpa_1_nw, ddt_mpa_2_nw, ddt_mha_1_nw, ddt_mha_2_nw, ht_mua_1_nw, ht_mua_2_nw, ddt_mua_1_nw, ddt_mua_2_nw, ddt_mpa_1_clusters, ddt_mpa_2_clusters, ddt_mha_1_clusters, ddt_mha_2_clusters, ht_mua_1_clusters, ht_mua_2_clusters, ddt_mua_1_clusters, ddt_mua_2_clusters)
    write_stats(ddt_mpa_1_nw, ddt_mpa_2_nw, ddt_mha_1_nw, ddt_mha_2_nw, ht_mua_1_nw, ht_mua_2_nw, ddt_mua_1_nw, ddt_mua_2_nw, ddt_mpa_1_clusters, ddt_mpa_2_clusters, ddt_mha_1_clusters, ddt_mha_2_clusters, ht_mua_1_clusters, ht_mua_2_clusters, ddt_mua_1_clusters, ddt_mua_2_clusters)
    
    return

if __name__ == '__main__':
    main()