import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

###variables
monolayers = ['ddt_mpa', 'ddt_mha', 'ht_mua', 'ddt_mua']
conf_types = ['janus', 'random', 'striped']
trials = 3
anchor_type = 5
num_configs = 5
num_configs_baseline=50

num_ligands_1_TGA = [274, 263, 256, 227]
num_ligands_2_TGA = [274, 263, 256, 227]
num_ligands_1_len = [13,13,7,13]
num_ligands_2_len = [7,10,15,15]

###HT only baseline filepaths
ht_baseline_filepaths = ['../Datafiles/ht_only/7p4lnm/trajectory.xyz', '../Datafiles/ht_only/7p3lnm/trajectory.xyz', '../Datafiles/ht_only/7p1lnm/trajectory.xyz', '../Datafiles/ht_only/6p7lnm/trajectory.xyz']

###plot details
monolayer_titles = ['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA'] 
color_x_values = [2, 3.5, 4.6, 6.1, 6.8, 8.4]
colors_for_colorbar = ['#c7cad1', '#a3a8b5','#818799', '#61687f', '#424a65', '#242e4c']


def get_distance_from_edge(edge_atoms, anchor_points) -> list[float]:
    """
    Iterates through all of the anchor points (coordinates corresponding to the sulfur anchors) and computes the euclidean distances between the current anchor
    and all edge sites to find the nearest edge site (i.e. min(distances)). The nearest edge atom can be assumed to be on the same plane as the anchor (though the anchor
    coordinate exists just above the coordinate plane of the edge atoms associated with the facet that the anchor is located on)
    
    returns a list of the minimum distances for each anchor as min_distances
    """
    
    min_distances = []
    for a in anchor_points:
        distances = []
        for e in edge_atoms:
            distances.append(np.linalg.norm(a - e))
        min_distances.append(min(distances))
        #print(min(min_distances))
    
    return min_distances

def compute_edge_distances_and_plot_normalized(np_edge_atoms) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    """
    Docstring for compute_edge_distances_and_plot_normalized
    
    :param np_edge_atoms: an np.ndarray list of coordinates corresponding to all of the nanoparticle edge atoms on a 4.7 nm icosahedral nanoparticle
    :return: xbins: x-coordinate used to plot the histograms. This is invariable for each subplot but is regenerated every loop in compute_edge_distances_and_plot_normalized() hence the #type: ignore
    :return: monolayer_distances_cooh: list of lists. Where each list contains COOH ligand sulfur anchor-edge atom minimum distances for one of the four monolayers
    :return: monolayer_distances_ch3: list of lists. Where each list contains CH3 ligand sulfur anchor-edge atom minimum distances for one of the four monolayers
    :return: monolayer_baselines: list of lists. Where each list contains HT ligand sulfur anchor-edge atom minimum distances for one of the four densities. The order of the densities corresponds
                                to the order of the monolayers in the lists above so that the densities correspond and the baseline is meaningful
    :rtype: tuple[ndarray[Any, Any], list[float], list[float], list[float]]
    """
        
    monolayer_distances_ch3 = []
    monolayer_distances_cooh=  []
    monolayer_baselines = []
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    plot_index = 0
    for m, monolayer in enumerate(monolayers):
        data = []
        min_distances_baseline = []
        with open(ht_baseline_filepaths[m], 'r') as file:
            #print(f'{ht_baseline_filepaths[m]} Found')
            for line in file:
                if line.strip():
                    columns = line.split()
                    if len(columns) == 4 and columns[0] != '1':
                        data.append(columns)
        total_lines = len(data)
        data = np.array(data[(total_lines - num_configs_baseline * (num_ligands_1_TGA[m] * 7 + num_ligands_2_TGA[m] * 7)):], dtype=float)
        #print(len(data)/(num_ligands_1_TGA[m] * 7 + num_ligands_2_TGA[m] * 7))
        for config in range(num_configs_baseline):
            #print(f'{monolayer.capitalize()}\t{conf_type.capitalize()}\tTRIAL{trial+1}\tCONFIG{config+1}')
            num_ligand = num_ligands_1_TGA[m]
            if config != 0:
                start_line = (config*(num_ligands_1_TGA[m] * 7 + num_ligands_2_TGA[m] * 7))-1
            else:
                start_line = (config*(num_ligands_1_TGA[m] * 7 + num_ligands_2_TGA[m] * 7))
            lig_sites = data[start_line:,1:4]
            type_sites = data[start_line:,0]
            #print(type_sites)
            lig_1_sites = []
            count = 0
            for k in range(len(type_sites)):
                if type_sites[k] == anchor_type:
                    count += 1
                    if count <= num_ligands_1_TGA[m] + num_ligands_2_TGA[m]:
                        lig_1_sites.append(lig_sites[k])
            print(f'HT only CONFIG {config+1} Baseline sites', len(lig_1_sites))
            lig_1_sites = np.array(lig_1_sites)
            min_distances_baseline.extend(get_distance_from_edge(np_edge_atoms, lig_1_sites))
            #print(len(min_distances_baseline))
        nx_baseline, xbins = np.histogram(min_distances_baseline, bins=10, density=False, range=(2,8.5))
        sum_counts_baseline = np.sum(nx_baseline)
        nx_baseline_prob = nx_baseline / sum_counts_baseline
        monolayer_baselines.append(nx_baseline_prob)
        min_distances_ch3 = []
        min_distances_oh = []
        for conf_type in conf_types:
            for trial in range(trials):
                filepath = f'../Datafiles/{monolayer}/{conf_type}/t{trial+1}/trajectory.xyz'
                data = []
                with open(filepath, 'r') as file:
                    #print(f'{filepath} Found')
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1':
                                data.append(columns)
                total_lines = len(data)
                data = np.array(data[(total_lines - num_configs * (num_ligands_1_TGA[m] * num_ligands_1_len[m] + num_ligands_2_TGA[m] * num_ligands_2_len[m])):], dtype=float)
                #print(len(data)/((num_ligands_1_TGA[m] * num_ligands_1_len[m] + num_ligands_2_TGA[m] * num_ligands_2_len[m])))
                for config in range(num_configs):
                    #print(f'{monolayer.capitalize()}\t{conf_type.capitalize()}\tTRIAL{trial+1}\tCONFIG{config+1}')
                    num_ligand = num_ligands_1_TGA[m]
                    if config != 0:
                        start_line = (config*(num_ligands_1_TGA[m] * num_ligands_1_len[m] + num_ligands_2_TGA[m] * num_ligands_2_len[m]))-1
                    else:
                        start_line = (config*(num_ligands_1_TGA[m] * num_ligands_1_len[m] + num_ligands_2_TGA[m] * num_ligands_2_len[m]))
                    lig_sites = data[start_line:,1:4]
                    type_sites = data[start_line:,0]
                    #print(type_sites)
                    lig_1_sites = []
                    lig_2_sites = [] 
                    count = 0
                    for k in range(len(type_sites)):
                        if type_sites[k] == anchor_type:
                            count += 1
                            if count <= num_ligand:
                                lig_1_sites.append(lig_sites[k])
                            elif count > num_ligand and count <= (num_ligands_1_TGA[m] + num_ligands_2_TGA[m]):
                                lig_2_sites.append(lig_sites[k])
                    print(f'{monolayer} {conf_type} {trial+1} CONFIG {config+1} Ligand Lists', len(lig_1_sites), len(lig_2_sites))
                    lig_1_sites = np.array(lig_1_sites)
                    lig_2_sites = np.array(lig_2_sites)
                    min_distances_ch3.extend(get_distance_from_edge(np_edge_atoms, lig_1_sites))
                    min_distances_oh.extend(get_distance_from_edge(np_edge_atoms, lig_2_sites))
                    #print(len(min_distances_ch3), len(min_distances_oh))
                            
        with open(f'../Datafiles/{monolayer}_edge_distances.txt', 'w') as edge_file:
            edge_file.write('CH3\tCOOH\n')
            for x_i, x in enumerate(min_distances_ch3):
                edge_file.write(f'{x}\t{min_distances_oh[x_i]}\n')

        axes[m].set_xlabel('$d$ (\u212b)', fontsize=15)
        if m == 0:
            axes[m].set_ylabel('$P(d)/P_0(d)$', fontsize=15)

        nx_oh, xbins = np.histogram(min_distances_oh, bins=10, density=False, range=(2,8.5))
        nx_ch3, _ = np.histogram(min_distances_ch3, bins=10, density=False, range=(2,8.5))
        
        nx_oh_sum, nx_ch3_sum = np.sum(nx_oh), np.sum(nx_ch3)
        nx_oh_prob = np.array(nx_oh) / nx_oh_sum
        nx_ch3_prob = np.array(nx_ch3) / nx_ch3_sum
        
        monolayer_distances_ch3.append(nx_ch3_prob)
        monolayer_distances_cooh.append(nx_oh_prob)
        
        nx_oh_prob = nx_oh_prob / nx_baseline_prob
        nx_ch3_prob = nx_ch3_prob / nx_baseline_prob
        
        axes[m].set_xlim(0, 10)
        axes[m].set_ylim(0,4.0)
        axes[m].set_xticks([0,2,4,6,8,10])
        axes[m].tick_params(axis='both', which='major', labelsize=14)        
        axes[m].set_title(f'{monolayer_titles[m]}', fontweight='bold', fontsize=14)
        axes[m].stairs(nx_oh_prob, xbins, orientation='vertical', fill=False, label='COOH', edgecolor='r', linewidth=2.5)
        axes[m].stairs(nx_ch3_prob, xbins, orientation='vertical', fill=False, label='CH3', edgecolor='k', linewidth=2.5) 
        axes[m].axhline(y=1.0, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
        axes[m].legend(fontsize='large', frameon=False, loc='upper center', ncols=2)      
        for i in range(len(color_x_values)):
            rect = patches.Rectangle(
                (color_x_values[i]-0.05, 0.1), 
                0.1, 0.15,    
                facecolor=colors_for_colorbar[i],
                edgecolor='none'
            )
            axes[m].add_patch(rect)
        plot_index += 1
        
    plt.tight_layout()
    plt.savefig(f'../Figures/edge_dependency_normalized.png', format='png', dpi=400)
    plt.close()
    
    return xbins, monolayer_distances_cooh, monolayer_distances_ch3, monolayer_baselines #type: ignore

def plot_unnormalized(xbins, monolayer_distances_cooh, monolayer_distances_ch3, monolayer_baselines) -> None:
    """
    Takes the below params generated from compute_edge_distances_and_plot_normalized() to plot the all three raw, unnormalized, datasets (the COOH ligand distances, the CH3 ligand distances
    and the baseline HT ligand distances) against each other for each of the four monolayers studied.
    
    :param xbins: x-coordinate used to plot the histograms. This is invariable for each subplot but is regenerated every loop in compute_edge_distances_and_plot_normalized() hence the #type: ignore
    :param monolayer_distances_cooh: list of lists. Where each list contains COOH ligand sulfur anchor-edge atom minimum distances for one of the four monolayers
    :param monolayer_distances_ch3: list of lists. Where each list contains CH3 ligand sulfur anchor-edge atom minimum distances for one of the four monolayers
    :param monolayer_baselines: list of lists. Where each list contains HT ligand sulfur anchor-edge atom minimum distances for one of the four densities. The order of the densities corresponds
                                to the order of the monolayers in the lists above so that the densities correspond and the baseline is meaningful
    """
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for m,monolayer in enumerate(monolayers):
        axes[m].set_xlim(0, 10)
        #axes[m].set_ylim(0,1.0)
        axes[m].set_xticks([0,2,4,6,8,10])
        axes[m].tick_params(axis='both', which='major', labelsize=14)
        axes[m].set_title(f'{monolayer_titles[m]}', fontweight='bold', fontsize=14)
        axes[m].stairs(monolayer_distances_cooh[m], xbins, orientation='vertical', fill=False, label='COOH', edgecolor='r', linewidth=1.5)
        axes[m].stairs(monolayer_distances_ch3[m], xbins, orientation='vertical', fill=False, label='CH3', edgecolor='k', linewidth=1.5)
        axes[m].stairs(monolayer_baselines[m], xbins, orientation='vertical', fill=False, label='$P_0(d)$', edgecolor='b', linewidth=1.5)
        axes[m].legend(fontsize='xx-small', frameon=False, loc='upper center', ncols=3)
        axes[m].set_xlabel('$d$')
        for i in range(len(color_x_values)):
            rect = patches.Rectangle(
                (color_x_values[i]-0.15, 0.01), 
                0.3, 0.015,    
                facecolor=colors_for_colorbar[i],
                edgecolor='none'
            )
            axes[m].add_patch(rect)
    axes[0].set_ylabel('$P(d)$')
    plt.tight_layout()
    plt.savefig(f'../Figures/edge_dependency_unnormalized.png', format='png', dpi=400)
    plt.close()
    
    return

def read_np_edge_file(filepath) -> np.ndarray:
    """"
    parses the nanoparticle edge site coordinate file and stores its contents in the NDArray np_edge_atoms
    
    np_edge_atoms will then be used in get_distances() to compute the minimum euclidean distance between a sulfur anchor atom coordinate and the nanoparticle edge atoms.
    """
    
    np_edge_atoms = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():
                columns = line.split()
                if len(columns) == 4:
                    np_edge_atoms.append(columns[1:])
    #print(f'NUMBER OF EDGE ATOMS FOR SIZE: {size}', len(np_edge_atoms))
    
    return np.array(np_edge_atoms, dtype=float)

def main() -> None:
    """
    This python script generates figure S6 of the Supplementary Information for the manuscript.
    
    Computes and plots the edge distance distributions both normalized, as presented in the main text, and unnormalized presented
    alongside the HT-only distribution (shown as blue in figure S6 - P_0) for each of the four monolayers studied 
    """    
    
    np_edge_atoms = read_np_edge_file('../Datafiles/np_10h_edge_atoms.xyz')

    xbins, monolayer_distances_cooh, monolayer_distances_ch3, monolayer_baselines = compute_edge_distances_and_plot_normalized(np_edge_atoms)
    plot_unnormalized(xbins, monolayer_distances_cooh, monolayer_distances_ch3, monolayer_baselines)

    return


if __name__ == '__main__':
    main()