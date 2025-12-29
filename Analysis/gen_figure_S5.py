
import numpy as np
import matplotlib.pyplot as plt
import rdf_functions
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

def get_rdf() -> None:

    """
    Function that determines the cluster size fraction of each ligand type in the binary monolayer by using the sulfur anchor positions, a KDTree, and a First Depth Search algo
    """

    plot_index = 0
    fig, axs = plt.subplots(4,2, sharex=True,sharey=True)
    for i, monolayer in enumerate(monolayers):
        #print('ok')
        rdf_data_ohs_sum = np.zeros(nbins)
        rdf_data_ch3s_sum = np.zeros(nbins)
        rdf_data_both_sum = np.zeros(nbins)
        rdf_anchor_data_ohs_sum = np.zeros(nbins)
        rdf_anchor_data_ch3s_sum = np.zeros(nbins)
        rdf_anchor_data_both_sum = np.zeros(nbins)
        
        total_trials = len(conf_types) * trials
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
                    oh_sites = []
                    ch3_sites = []
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
                        elif type_sites[k] == 4: #CH3
                            ch3_sites.append(lig_sites[k])
                        elif type_sites[k] == 7: #H-bearing oxygen
                            oh_sites.append(lig_sites[k])

                    oh_sites = np.array(oh_sites)
                    ch3_sites = np.array(ch3_sites)
                    oh_s_sites = np.array(oh_s_sites)
                    ch3_s_sites = np.array(ch3_s_sites)
                    print(f'{monolayer}/{conf_type}/{trial+1} : #CH3 S anchors in array: ', len(ch3_s_sites))
                    print(f'{monolayer}/{conf_type}/{trial+1} : #CH3 head sites in array: ', len(ch3_sites))
                    print(f'{monolayer}/{conf_type}/{trial+1} : #COOH S anchors in array: ', len(oh_s_sites))
                    print(f'{monolayer}/{conf_type}/{trial+1} : #COOH head sites in array: ', len(oh_sites))
                    oh_sites_occupied = {tuple(site) for site in oh_sites}
                    ch3_sites_occupied = {tuple(site) for site in ch3_sites}
                    ch3_s_sites_occupied = {tuple(site) for site in ch3_s_sites}
                    oh_s_sites_occupied = {tuple(site) for site in oh_s_sites}
                    filled_s_sites = [['oh', list(site)] for site in oh_s_sites_occupied] + [['ch3', list(site)] for site in ch3_s_sites_occupied]
                    head_group_sites = [['oh', list(site)] for site in oh_sites_occupied] + [['ch3', list(site)] for site in ch3_sites_occupied]
                    rdf_data_ohs, rdf_data_ch3s, rdf_data_both, rdf_anchor_data_ohs, rdf_anchor_data_ch3s, rdf_anchor_data_both, bin_edges = rdf_functions.get_rdf(
                        data_anchor=filled_s_sites, 
                        data_head=head_group_sites, 
                        nbins=nbins, 
                        num_mols=(num_ligands_1[i] + num_ligands_2[i]), 
                        r_max=20
                    )
                rdf_data_ohs_sum += rdf_data_ohs #type: ignore
                rdf_data_ch3s_sum += rdf_data_ch3s #type: ignore
                rdf_data_both_sum += rdf_data_both #type: ignore
                rdf_anchor_data_ohs_sum += rdf_anchor_data_ohs #type: ignore
                rdf_anchor_data_ch3s_sum += rdf_anchor_data_ch3s #type: ignore
                rdf_anchor_data_both_sum += rdf_anchor_data_both #type: ignore
        rdf_data_ohs_avg = rdf_data_ohs_sum / total_trials
        rdf_data_ch3s_avg = rdf_data_ch3s_sum / total_trials
        rdf_data_both_avg = rdf_data_both_sum / total_trials
        rdf_anchor_data_ohs_avg = rdf_anchor_data_ohs_sum / total_trials
        rdf_anchor_data_ch3s_avg = rdf_anchor_data_ch3s_sum / total_trials
        rdf_anchor_data_both_avg = rdf_anchor_data_both_sum / total_trials

        axs[i,0].plot(bin_edges[:-1], rdf_data_ohs_avg, label='COOH\u2192COOH', color='r') #type: ignore
        axs[i,0].plot(bin_edges[:-1], rdf_data_ch3s_avg, label='CH3\u2192CH3', color='k') #type: ignore
        axs[i,0].plot(bin_edges[:-1], rdf_data_both_avg, label='CH3\u2192COOH', color='#67CC8E') #type: ignore
        if i == 3:
            axs[i,0].set_xlabel(r'$r\ (\mathrm{\AA})$', fontsize=14)
            axs[i,0].set_xticks([0,5,10,15,20])
        axs[i,0].set_ylabel('g(r)', fontsize=14)
        axs[i,0].set_xlim(0.0, 20.0)
        axs[i,0].set_ylim(0.0, 6.5)
        if i == 0:
            axs[i,0].set_title('Tail RDF')
            axs[i,0].legend(loc='upper center', bbox_to_anchor=(1, 1.8),
            ncol=3, frameon=False, fontsize='large')
        axs[i,0].set_yticks([1,6.5])
        
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_ohs_avg, label='COOH\u2192COOH', color='r') #type: ignore
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_ch3s_avg, label='CH3\u2192CH3', color='k') #type: ignore
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_both_avg, label='CH3\u2192COOH', color='#67CC8E') #type: ignore
        if i == 3:
            axs[i,1].set_xlabel(r'$r\ (\mathrm{\AA})$', fontsize=14)
            axs[i,1].set_xticks([0,5,10,15,20])
        axs[i,1].set_xlim(0.0, 20.0)
        axs[i,1].set_ylim(0.0, 6.5)
        if i == 0:
            axs[i,1].set_title('Anchor RDF')
        
        axs[i,0].tick_params(axis='both', which='major', labelsize=13)
        axs[i,1].tick_params(axis='both', which='major', labelsize=13)
        plot_index += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig('../Figures/radial_distribution_functions.png', format='png', dpi=600)

    return 

def main() -> None:
    """
    This python script generates figure S5 of the Supplementary Information for the manuscript.
    
    Computes and plots the Sulfur-Sulfur, CH3-CH3, and CH3-OH (using the H-bearing O as the reference) radial distribution functions for each of the four monolayers studied.
    """
    get_rdf()
    return


if __name__ == '__main__':
    main()