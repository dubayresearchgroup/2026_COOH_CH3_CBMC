import numpy as np
import matplotlib.pyplot as plt

#variables
trials = 3
anchor_type = 5
num_configs = 1
replicates = 1
threshold_distance = 2.7

num_ligands_1 = [274,263,256,227] # the number of alkanethiol ligands for DDT/MPA, DDT/MHA, HT/MUA, and DDT/MUA
num_ligands_2 = [274,263,256,227] # the number of carboxylic acid ligands for DDT/MPA, DDT/MHA, HT/MUA, and DDT/MUA
num_ligands_1_len = [13,13,7,13] # length of ligand type 1 (the alkanethiol)
num_ligands_2_len = [7,10,15,15] # length of ligand type 2 (the carboxylic acid)

monolayers = ['ddt_mpa', 'ddt_mha','ht_mua','ddt_mua']
conf_types = ['janus', 'random', 'striped']

fragment_prob_xticklabels = ['DDT/MPA', 'HT/MUA', 'DDT/MHA', 'DDT/MUA']
fragment_prob_xticks = []
for i in range(len(fragment_prob_xticklabels)):
    fragment_prob_xticks.append(i+1)

def get_conformer(data, num_ch3s, num_coohs, num_ligands_2_len, anchor_type=5) -> tuple[float, list[float]]:
    oxygens = []
    hydrogens = []
    distances = []
    anti = 0
    syn = 0
    count = 0
    for i, atom in enumerate(data):
        if count < num_ch3s:
            if int(atom[0]) == anchor_type:
                count += 1
        elif int(atom[0]) == anchor_type:
            oxygens.append(data[i + (num_ligands_2_len-2)][1:4])
            hydrogens.append(data[i + (num_ligands_2_len-1)][1:4])
    if len(hydrogens) == num_coohs:
        for i in range(len(hydrogens)):
            oh_dist = np.linalg.norm(np.array(oxygens[i]) - np.array(hydrogens[i]))
            distances.append(oh_dist)
            #print(oh_dist)
            if oh_dist >= threshold_distance:
                anti += 1
            else:
                syn += 1
    #print(anti,syn)
    if (anti+syn) != 0:
        return float(anti / (anti + syn)), distances
    else:
        return 0.0, []

def plot_all_distances(distances) -> None:
    
    plt.hist(distances, bins=50, color='k')
    plt.axvline(x=2.7, color='r', label='Set Threshold Distance')
    plt.xlabel('Distances (A)')
    plt.ylabel('Counts')
    #plt.xlim(2,3.2)
    plt.tight_layout()
    plt.savefig('../Figures/H_distances_to_carbonyl_histogram.png', format='png', dpi=400)
    plt.close()
    
    return

def plot_anti_ratios(anti_ratios) -> None:
    
    num_to_consider_per_monolayer =  replicates * num_configs * trials * len(conf_types)
    ddt_mha_anti_ratio_avg, ddt_mha_anti_ratio_std  =  100*np.mean(anti_ratios[num_to_consider_per_monolayer:(2*num_to_consider_per_monolayer-1)]), 100*np.std(anti_ratios[num_to_consider_per_monolayer:(2*num_to_consider_per_monolayer-1)])
    ddt_mpa_anti_ratio_avg, ddt_mpa_anti_ratio_std = 100*np.mean(anti_ratios[0:(num_to_consider_per_monolayer-1)]), 100*np.std(anti_ratios[0:(num_to_consider_per_monolayer-1)])
    ddt_mua_anti_ratio_avg, ddt_mua_anti_ratio_std = 100*np.mean(anti_ratios[(2*num_to_consider_per_monolayer):(3*num_to_consider_per_monolayer-1)]), 100*np.std(anti_ratios[(2*num_to_consider_per_monolayer):(3*num_to_consider_per_monolayer-1)])
    ht_mua_anti_ratio_avg, ht_mua_anti_ratio_std = 100*np.mean(anti_ratios[(3*num_to_consider_per_monolayer):]), 100*np.std(anti_ratios[(3*num_to_consider_per_monolayer):])
    
    plt.figure()
    plt.errorbar(1, ddt_mpa_anti_ratio_avg, yerr=ddt_mpa_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
    plt.errorbar(3, ht_mua_anti_ratio_avg, yerr=ht_mua_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
    plt.errorbar(2, ddt_mha_anti_ratio_avg, yerr=ddt_mha_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
    plt.errorbar(4, ddt_mua_anti_ratio_avg, yerr=ddt_mua_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
    plt.xticks(fragment_prob_xticks, ['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA'])
    plt.xlabel('Monolayers')
    plt.ylabel('Anti COOH Conformer % Occurrence')
    plt.xlim(0,5)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig('../Figures/anti_conformer_ratios.png', format='png', dpi=600)
    
    return

def get_cooh_conformer_prevalence() -> tuple[list[float], list[float]]:
    
    anti_ratios = []
    distances = []
    
    for i, monolayer in enumerate(monolayers):
        for j, conf_type in enumerate(conf_types):
            for trial in range(trials):
                print(f'{monolayer} / {conf_type} / {trial+1}')
                filepath = f'../Datafiles/{monolayer}/{conf_type}/t{trial+1}/trajectory.xyz'
                data = []
                with open(filepath, 'r') as file:
                    #print(True)
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1':
                                data.append(columns)
                total_lines = len(data)
                data = np.array(data[(total_lines - num_configs * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])):], dtype=float)
                for config in range(num_configs):
                    for replicate in range(replicates):
                        start_line = config * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                        end_line = start_line + (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                        anti_conformer_ratio, x = get_conformer(data[start_line:end_line], num_ligands_1[i], num_ligands_2[i], num_ligands_2_len[i])
                        anti_ratios.append(anti_conformer_ratio)
                        distances.extend(x)
    
    return anti_ratios, distances

def main() -> None:
    """
    This python script generates figure S7 of the Supplementary Information for the manuscript.
    
    Computes and plots the distribution of distances found between a carboxylic acid groups acidic H and carbonyl O to determine a threshold distance to consider as anti or syn conformer.
    """
         
    anti_ratios, distances = get_cooh_conformer_prevalence()
    plot_all_distances(distances)
    plot_anti_ratios(anti_ratios)
                    
    return


if __name__ == '__main__':
    main()