
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy.stats as ss

#variables
trials = 3
anchor_type = 5
num_configs = 1
replicates = 1
fragment_size = 4
threshold = 6

num_ligands_1 = [274,263,256,227] # the number of alkanethiol ligands for DDT/MPA, DDT/MHA, HT/MUA, and DDT/MUA
num_ligands_2 = [274,263,256,227] # the number of carboxylic acid ligands for DDT/MPA, DDT/MHA, HT/MUA, and DDT/MUA
num_ligands_1_len = [13,13,7,13] # length of ligand type 1 (the alkanethiol)
num_ligands_2_len = [7,10,15,15] # length of ligand type 2 (the carboxylic acid)
num_np_atoms = 2456 # number of metal nanoparticle atoms 

monolayers = ['ddt_mpa', 'ddt_mha','ht_mua','ddt_mua']
conf_types = ['janus', 'random', 'striped']
lig_1 = ['ddt', 'ddt','ht','ddt']
lig_2 = ['mpa','mha','mua','mua']


fragment_prob_xticklabels = ['DDT/MPA', 'HT/MUA', 'DDT/MHA', 'DDT/MUA']
fragment_prob_xticks = []
for i in range(len(fragment_prob_xticklabels)):
    fragment_prob_xticks.append(i+1)


def get_binomial(fraction, fragment_size):
    return ss.binom.pmf(np.arange(fragment_size + 1), fragment_size, fraction)

def get_maldi_ssr(filled_surface_sites, num_iterations, nn_threshold, target_fractions, fragment_size, mua=False):
    fragments = [0] * (fragment_size + 1)
    filled_surface_sites_np = [[entry[0], np.array(entry[1])] for entry in filled_surface_sites]
    coordinates_np = np.array([entry[1] for entry in filled_surface_sites_np])
    kdtree = KDTree(coordinates_np)
    total_fragments = 0
    for i in range(num_iterations):
        chosen_index = np.random.randint(0, len(filled_surface_sites_np))
        query_point = filled_surface_sites[chosen_index][1]
        distances, neighbor_indices = kdtree.query(query_point, k=fragment_size, distance_upper_bound=nn_threshold, p=2)
        valid_pairs = [(d, i) for d, i in zip(distances, neighbor_indices) if i != chosen_index and d <= nn_threshold]
        #print(valid_pairs[:fragment_size-1])
        valid_pairs.sort()
       # print(valid_pairs[:fragment_size-1])
        valid_indices = [i for d, i in valid_pairs]#neighbor_indices[valid_mask]
        #print(f'Chosen Index: {chosen_index}\t Neighbor Indices: {valid_indices[:fragment_size-1]}')
        if len(valid_indices) > fragment_size-1:
            valid_indices = valid_indices[:fragment_size-1]
        if len(valid_indices) == fragment_size - 1:
            if mua:
                fragment_index = int(filled_surface_sites[chosen_index][0] == 2) + sum(1 for index in valid_indices if filled_surface_sites[index][0] == 2)
            else:
                fragment_index = int(filled_surface_sites[chosen_index][0] == 1) + sum(1 for index in valid_indices if filled_surface_sites[index][0] == 1)
            fragments[fragment_index] += 1
            total_fragments +=1
    if total_fragments == 0:
        raise ValueError("No valid fragments found.")

    fragment_probabilities = [fragment / total_fragments for fragment in fragments]
    ssr = sum((fragment_probabilities[i] - target_fractions[i]) ** 2 for i in range(len(target_fractions)))

    return fragment_probabilities, ssr, total_fragments

def compute_frags_and_ssr() -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], 
                                     list[float], list[float], list[float], list[float], list[float], list[float],
                                     list[float], list[float], list[float], list[float], list[float], list[float],
                                     list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
    Computes the fragment distribution and SSR for each of the 9 trials performed for each monolayer studied and returns the average and standard deviation for the initial and final states
    
    :return: average and standard deviation for each monolayers Configuration subgroups in the initial and final states to plot the change in SSR shown in figure S4
    """        
    for i, monolayer in enumerate(monolayers):
        initial_janus, final_janus = [], []
        initial_random, final_random = [], []
        initial_striped, final_striped = [], []
        
        for j, conf_type in enumerate(conf_types):
            for trial in range(trials):
                filepath = f'../Datafiles/{monolayer}/{conf_type}/t{trial+1}/trajectory.xyz'
                data = []
                with open(filepath, 'r') as file:
                    print(True)
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1':
                                data.append(columns)
                total_lines = len(data)
                data_from_end = np.array(data[(total_lines - num_configs * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])):], dtype=float)
                data_from_start = np.array(data[:(num_configs * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i]))], dtype=float)
                for config in range(num_configs):
                    for replicate in range(replicates):
                        print(f'config {config} / replicate {replicate}')
                        start_line = config * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                        end_line = start_line + (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                        
                        lig_sites_final = data_from_end[start_line:end_line, 1:4]
                        type_sites_final = data_from_end[start_line:end_line, 0]
                        lig_1_sites_final = []
                        lig_2_sites_final = []
                        
                        lig_sites_initial = data_from_start[start_line:end_line, 1:4]
                        type_sites_initial = data_from_start[start_line:end_line, 0]
                        lig_1_sites_initial = []
                        lig_2_sites_initial = []
                        
                        count = 0
                        for k in range(len(type_sites_initial)):
                            if int(type_sites_initial[k]) == anchor_type:
                                if count < num_ligands_1[i]:
                                    lig_1_sites_initial.append(lig_sites_initial[k])
                                else:
                                    lig_2_sites_initial.append(lig_sites_initial[k])
                                count += 1
                                
                        count = 0
                        for k in range(len(type_sites_final)):
                            if int(type_sites_final[k]) == anchor_type:
                                if count < num_ligands_1[i]:
                                    lig_1_sites_final.append(lig_sites_final[k])
                                else:
                                    lig_2_sites_final.append(lig_sites_final[k])
                                count += 1
                                
                        initial_L1_sites_occupied = {tuple(site) for site in lig_1_sites_initial}
                        initial_L2_sites_occupied = {tuple(site) for site in lig_2_sites_initial}
                        initial_L2_sites_occupied = initial_L2_sites_occupied - initial_L1_sites_occupied
                        print('Number of Ligands Identified in Initial State:', len(initial_L1_sites_occupied), len(initial_L2_sites_occupied)) #print len to manually check that length matches the number of atoms 
                        
                        final_L1_sites_occupied = {tuple(site) for site in lig_1_sites_final}
                        final_L2_sites_occupied = {tuple(site) for site in lig_2_sites_final}
                        final_L2_sites_occupied = final_L2_sites_occupied - final_L1_sites_occupied
                        print('Number of Ligands Identified in Final State:', len(final_L1_sites_occupied), len(final_L2_sites_occupied))
                        
                        filled_sites = [[1, list(site)] for site in initial_L1_sites_occupied] + [[2, list(site)] for site in initial_L2_sites_occupied]
                        if monolayer == 'ht_mua' or monolayer == 'ddt_mua': #use MUA as the ligand to count (heaviest ligand in monolayer)
                            initial_target_fractions = get_binomial(float(num_ligands_2[i]) / float(num_ligands_1[i] + num_ligands_2[i]), fragment_size)
                            _, initial_ssr, _ = get_maldi_ssr(filled_sites, 100000, threshold, initial_target_fractions, fragment_size, mua=True)
                        else: #use DDT
                            initial_target_fractions = get_binomial(float(num_ligands_1[i]) / float(num_ligands_1[i] + num_ligands_2[i]), fragment_size)
                            _, initial_ssr, _ = get_maldi_ssr(filled_sites, 100000, threshold, initial_target_fractions, fragment_size, mua=False)
                            
                        filled_sites = [[1, list(site)] for site in final_L1_sites_occupied] + [[2, list(site)] for site in final_L2_sites_occupied]
                        if monolayer == 'ht_mua' or monolayer == 'ddt_mua': #use MUA as the ligand to count (heaviest ligand in monolayer)
                            final_target_fractions = get_binomial(float(num_ligands_2[i]) / float(num_ligands_1[i] + num_ligands_2[i]), fragment_size)
                            _, final_ssr, _ = get_maldi_ssr(filled_sites, 100000, threshold, final_target_fractions, fragment_size, mua=True)
                        else: #use DDT
                            final_target_fractions = get_binomial(float(num_ligands_1[i]) / float(num_ligands_1[i] + num_ligands_2[i]), fragment_size)
                            _, final_ssr, _ = get_maldi_ssr(filled_sites, 100000, threshold, final_target_fractions, fragment_size, mua=False)
                        
                        #partition per conf stats to record
                        if conf_type == 'janus':
                            initial_janus.append(initial_ssr)
                            final_janus.append(final_ssr)
                        elif conf_type == 'random':
                            initial_random.append(initial_ssr)
                            final_random.append(final_ssr)
                        else:
                            initial_striped.append(initial_ssr)
                            final_striped.append(final_ssr)

        if monolayer == 'ddt_mpa':
            ddtmpa_janus_i, ddtmpa_random_i, ddtmpa_striped_i = [float(np.mean(initial_janus)), float(np.std(initial_janus))], [float(np.mean(initial_random)), float(np.std(initial_random))], [float(np.mean(initial_striped)), float(np.std(initial_striped))]
            ddtmpa_janus_f, ddtmpa_random_f, ddtmpa_striped_f = [float(np.mean(final_janus)), float(np.std(final_janus))], [float(np.mean(final_random)), float(np.std(final_random))], [float(np.mean(final_striped)), float(np.std(final_striped))]   
        elif monolayer == 'ddt_mha':
            ddtmha_janus_i, ddtmha_random_i, ddtmha_striped_i = [float(np.mean(initial_janus)), float(np.std(initial_janus))], [float(np.mean(initial_random)), float(np.std(initial_random))], [float(np.mean(initial_striped)), float(np.std(initial_striped))]
            ddtmha_janus_f, ddtmha_random_f, ddtmha_striped_f = [float(np.mean(final_janus)), float(np.std(final_janus))], [float(np.mean(final_random)), float(np.std(final_random))], [float(np.mean(final_striped)), float(np.std(final_striped))] 
        elif monolayer == 'ht_mua':
            htmua_janus_i, htmua_random_i, htmua_striped_i = [float(np.mean(initial_janus)), float(np.std(initial_janus))], [float(np.mean(initial_random)), float(np.std(initial_random))], [float(np.mean(initial_striped)), float(np.std(initial_striped))]
            htmua_janus_f, htmua_random_f, htmua_striped_f = [float(np.mean(final_janus)), float(np.std(final_janus))], [float(np.mean(final_random)), float(np.std(final_random))], [float(np.mean(final_striped)), float(np.std(final_striped))] 
        elif monolayer == 'ddt_mua':
            ddtmua_janus_i, ddtmua_random_i, ddtmua_striped_i = [float(np.mean(initial_janus)), float(np.std(initial_janus))], [float(np.mean(initial_random)), float(np.std(initial_random))], [float(np.mean(initial_striped)), float(np.std(initial_striped))]
            ddtmua_janus_f, ddtmua_random_f, ddtmua_striped_f = [float(np.mean(final_janus)), float(np.std(final_janus))], [float(np.mean(final_random)), float(np.std(final_random))], [float(np.mean(final_striped)), float(np.std(final_striped))] 
        
        
        
        print(f'{monolayer} JANUS avg: initial -> {np.mean(initial_janus):.2f} {np.std(initial_janus):.2f} final ->{np.mean(final_janus):.2f} {np.std(final_janus):.2f}')
        print(f'{monolayer} Random avg: initial -> {np.mean(initial_random):.2f} {np.std(initial_random):.2f} final ->{np.mean(final_random):.2f} {np.std(final_random):.2f}')
        print(f'{monolayer} Striped avg: initial -> {np.mean(initial_striped):.2f} {np.std(initial_striped):.2f} final ->{np.mean(final_striped):.2f} {np.std(final_striped):.2f}')
            
    return ddtmpa_janus_i, ddtmpa_random_i, ddtmpa_striped_i, ddtmpa_janus_f, ddtmpa_random_f, ddtmpa_striped_f, ddtmha_janus_i, ddtmha_random_i, ddtmha_striped_i, ddtmha_janus_f, ddtmha_random_f, ddtmha_striped_f, htmua_janus_i, htmua_random_i, htmua_striped_i, htmua_janus_f, htmua_random_f, htmua_striped_f, ddtmua_janus_i, ddtmua_random_i, ddtmua_striped_i, ddtmua_janus_f, ddtmua_random_f, ddtmua_striped_f #type: ignore

def plot(ddtmpa_janus_i, ddtmpa_random_i, ddtmpa_striped_i, ddtmpa_janus_f, ddtmpa_random_f, ddtmpa_striped_f, ddtmha_janus_i, ddtmha_random_i, ddtmha_striped_i, ddtmha_janus_f, ddtmha_random_f, ddtmha_striped_f, htmua_janus_i, htmua_random_i, htmua_striped_i, htmua_janus_f, htmua_random_f, htmua_striped_f, ddtmua_janus_i, ddtmua_random_i, ddtmua_striped_i, ddtmua_janus_f, ddtmua_random_f, ddtmua_striped_f) -> None:
    x = [0.9,1,1.1,1.9,2,2.1,2.9,3,3.1,3.9,4,4.1]
    plt.figure()
    plt.errorbar(x=[x[0], x[3], x[6], x[9]], 
                y=[ddtmpa_janus_f[0], ddtmha_janus_f[0], htmua_janus_f[0], ddtmua_janus_f[0]],
                yerr=[ddtmpa_janus_f[1], ddtmha_janus_f[1], htmua_janus_f[1], ddtmua_janus_f[1]],
                capsize=5, label='Janus',
                fmt='o',
                color='#232D4B')
    plt.errorbar(x=[x[1], x[4], x[7], x[10]], 
                y=[ddtmpa_random_f[0], ddtmha_random_f[0], htmua_random_f[0], ddtmua_random_f[0]],
                yerr=[ddtmpa_random_f[1], ddtmha_random_f[1], htmua_random_f[1], ddtmua_random_f[1]],
                capsize=5, label='Random',
                fmt='o',
                color='#E57200')
    plt.errorbar(x=[x[2], x[5], x[8], x[11]], 
                y=[ddtmpa_striped_f[0], ddtmha_striped_f[0], htmua_striped_f[0], ddtmua_striped_f[0]],
                yerr=[ddtmpa_striped_f[1], ddtmha_striped_f[1], htmua_striped_f[1], ddtmua_striped_f[1]],
                capsize=5, label='Striped',
                fmt='o',
                color='#495E9D')

    plt.errorbar(x=[x[0], x[3], x[6], x[9]], 
                y=[ddtmpa_janus_i[0], ddtmha_janus_i[0], htmua_janus_i[0], ddtmua_janus_i[0]],
                yerr=[ddtmpa_janus_i[1], ddtmha_janus_i[1], htmua_janus_i[1], ddtmua_janus_i[1]],
                capsize=5,label='Janus Initial Configuration',
                fmt='*',
                color='#232D4B',
                alpha=0.3)
    plt.errorbar(x=[x[1], x[4], x[7], x[10]], 
                y=[ddtmpa_random_i[0], ddtmha_random_i[0], htmua_random_i[0], ddtmua_random_i[0]],
                yerr=[ddtmpa_random_i[1], ddtmha_random_i[1], htmua_random_i[1], ddtmua_random_i[1]],
                capsize=5, label='Random Initial Configuration',
                fmt='*',
                color='#E57200',
                alpha=0.3)
    plt.errorbar(x=[x[2], x[5], x[8], x[11]], 
                y=[ddtmpa_striped_i[0], ddtmha_striped_i[0], htmua_striped_i[0], ddtmua_striped_i[0]],
                yerr=[ddtmpa_striped_i[1], ddtmha_striped_i[1], htmua_striped_i[1], ddtmua_striped_i[1]],
                capsize=5, label='Striped Initial Configuration',
                fmt='*',
                color='#495E9D',
                alpha=0.3)
    plt.xlabel('Monolayers')
    plt.ylabel('SSR')
    plt.xticks(fragment_prob_xticks, fragment_prob_xticklabels)
    plt.ylim(0, 0.8)
    plt.legend(loc='upper center', frameon=False, fontsize='small', ncols=2)

    plt.axvline(x=x[0], ymax=(ddtmpa_janus_i[0]/0.8), ymin=(ddtmpa_janus_f[0]/0.8), color='#232D4B',alpha=0.3,linestyle=':')
    plt.axvline(x=x[1], ymax=(ddtmpa_random_i[0]/0.8), ymin=(ddtmpa_random_f[0]/0.8), color='#E57200',alpha=0.3,linestyle=':')
    plt.axvline(x=x[2], ymax=(ddtmpa_striped_i[0]/0.8), ymin=(ddtmpa_striped_f[0]/0.8), color='#495E9D',alpha=0.3,linestyle=':')

    plt.axvline(x=x[0], ymax=(ddtmha_janus_i[0]/0.8), ymin=(ddtmha_janus_f[0]/0.8), color='#232D4B',alpha=0.3,linestyle=':')
    plt.axvline(x=x[1], ymax=(ddtmha_random_i[0]/0.8), ymin=(ddtmha_random_f[0]/0.8), color='#E57200',alpha=0.3,linestyle=':')
    plt.axvline(x=x[2], ymax=(ddtmha_striped_i[0]/0.8), ymin=(ddtmha_striped_f[0]/0.8), color='#495E9D',alpha=0.3,linestyle=':')

    plt.axvline(x=x[0], ymax=(htmua_janus_i[0]/0.8), ymin=(htmua_janus_f[0]/0.8), color='#232D4B',alpha=0.3,linestyle=':')
    plt.axvline(x=x[1], ymax=(htmua_random_i[0]/0.8), ymin=(htmua_random_f[0]/0.8), color='#E57200',alpha=0.3,linestyle=':')
    plt.axvline(x=x[2], ymax=(htmua_striped_i[0]/0.8), ymin=(htmua_striped_f[0]/0.8), color='#495E9D',alpha=0.3,linestyle=':')

    plt.axvline(x=x[0], ymax=(ddtmua_janus_i[0]/0.8), ymin=(ddtmua_janus_f[0]/0.8), color='#232D4B',alpha=0.3,linestyle=':')
    plt.axvline(x=x[1], ymax=(ddtmua_random_i[0]/0.8), ymin=(ddtmua_random_f[0]/0.8), color='#E57200',alpha=0.3,linestyle=':')
    plt.axvline(x=x[2], ymax=(ddtmua_striped_i[0]/0.8), ymin=(ddtmua_striped_f[0]/0.8), color='#495E9D',alpha=0.3,linestyle=':')

    plt.xlim(0,5)
    plt.tight_layout()
    plt.savefig('../Figures/initial_vs_final_ssrs.png', format='png', dpi=400)
    plt.close()
    
    return 

def main() -> None:
    """
    This python script generates figure S4 of the Supplementary Information for the manuscript.
    
    Computes and plots the average initial and average final SSR values for each subgroup (Janus, Random, and Striped) of the dataset for each of the four monolayers studied.
    """
    
    ddtmpa_janus_i, ddtmpa_random_i, ddtmpa_striped_i, ddtmpa_janus_f, ddtmpa_random_f, ddtmpa_striped_f, ddtmha_janus_i, ddtmha_random_i, ddtmha_striped_i, ddtmha_janus_f, ddtmha_random_f, ddtmha_striped_f, htmua_janus_i, htmua_random_i, htmua_striped_i, htmua_janus_f, htmua_random_f, htmua_striped_f, ddtmua_janus_i, ddtmua_random_i, ddtmua_striped_i, ddtmua_janus_f, ddtmua_random_f, ddtmua_striped_f = compute_frags_and_ssr()
    plot(ddtmpa_janus_i, ddtmpa_random_i, ddtmpa_striped_i, ddtmpa_janus_f, ddtmpa_random_f, ddtmpa_striped_f, ddtmha_janus_i, ddtmha_random_i, ddtmha_striped_i, ddtmha_janus_f, ddtmha_random_f, ddtmha_striped_f, htmua_janus_i, htmua_random_i, htmua_striped_i, htmua_janus_f, htmua_random_f, htmua_striped_f, ddtmua_janus_i, ddtmua_random_i, ddtmua_striped_i, ddtmua_janus_f, ddtmua_random_f, ddtmua_striped_f)
    
    return


if __name__ == '__main__':
    main()