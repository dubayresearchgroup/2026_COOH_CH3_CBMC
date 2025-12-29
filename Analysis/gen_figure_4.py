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


#experimental data obtained 
experimental_fragment_distributions = [[0.357685, 0.109651, 0.103253, 0.116551, 0.312861],
                                       [0.304508, 0.108998, 0.118901, 0.166772, 0.300822],
                                       [0.282248, 0.136605, 0.141126, 0.153033, 0.286989],
                                       [0.172014, 0.238135, 0.281251, 0.184366, 0.124234],
                                       ]

experimental_fragment_distributions_std = [[0.015815, 0.019752, 0.016672, 0.010287, 0.01999],
                                       [0.015577, 0.02012, 0.014, 0.009778, 0.026703],
                                       [0.01712, 0.01143, 0.014134, 0.009764, 0.014903],
                                       [0.014494, 0.011432, 0.027544, 0.008248, 0.013283],
                                       ]

exp_ssr = [0.26075,0.2075,0.175580538804374,0.0225]
exp_ssr_std = [0.024474,0.025626,0.0160727173799454,0.012834]

def get_binomial(fraction, fragment_size) -> np.ndarray:
    return ss.binom.pmf(np.arange(fragment_size + 1), fragment_size, fraction)

def get_maldi_ssr(filled_surface_sites, num_iterations, nn_threshold, target_fractions, fragment_size, mua=False) -> tuple[list[float], float, int]:
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

def compute_frags_and_ssr() -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    """
    Computes the fragment distribution and SSR for each of the 9 trials performed for each monolayer studied and returns the average and standard deviation for all metrics assessed
    
    :return: average and standard deviation of the fragment fractions for each monolayer and the corresponding average and standard deviation SSR values
    :rtype: tuple[list[list[float]], list[list[float]], list[float], list[float]]
    """
    
    with open(f'../Datafiles/fragment_data.txt', 'w') as file:
        file.write('#\nmonolayer\tfragment avg\t fragment std\tssr avg\tssr std\n')
        file.close()
    
    theoretical_fragment_distributions = []
    theoretical_fragment_distributions_std = []
    
    sim_ssr = []
    sim_ssr_std = []
    
    for i, monolayer in enumerate(monolayers):
        fragment_probabilities_all = []
        janus = []
        random = []
        striped = []
        ssrs_general = []
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
                data = np.array(data[(total_lines - num_configs * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])):], dtype=float)
                for config in range(num_configs):
                    for replicate in range(replicates):
                        print(f'config {config} / replicate {replicate}')
                        start_line = config * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                        end_line = start_line + (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])
                        lig_sites = data[start_line:end_line, 1:4]
                        type_sites = data[start_line:end_line, 0]
                        lig_1_sites = []
                        lig_2_sites = []
                        count = 0
                        for k in range(len(type_sites)):
                            if int(type_sites[k]) == anchor_type:
                                if count < num_ligands_1[i]:
                                    lig_1_sites.append(lig_sites[k])
                                else:
                                    lig_2_sites.append(lig_sites[k])
                                count += 1
                        L1_sites_occupied = {tuple(site) for site in lig_1_sites}
                        L2_sites_occupied = {tuple(site) for site in lig_2_sites}
                        L2_sites_occupied = L2_sites_occupied - L1_sites_occupied
                        print(len(L1_sites_occupied), len(L2_sites_occupied)) #print len to manually check that length matches the number of atoms 
                        filled_sites = [[1, list(site)] for site in L1_sites_occupied] + [[2, list(site)] for site in L2_sites_occupied]
                        if monolayer == 'ht_mua' or monolayer == 'ddt_mua': #use MUA as the ligand to count (heaviest ligand in monolayer)
                            target_fractions = get_binomial(float(num_ligands_2[i]) / float(num_ligands_1[i] + num_ligands_2[i]), fragment_size)
                            fragment_probabilities, ssr, total_fragments = get_maldi_ssr(filled_sites, 100000, threshold, target_fractions, fragment_size, mua=True)
                        else: #use DDT
                            target_fractions = get_binomial(float(num_ligands_1[i]) / float(num_ligands_1[i] + num_ligands_2[i]), fragment_size)
                            fragment_probabilities, ssr, total_fragments = get_maldi_ssr(filled_sites, 100000, threshold, target_fractions, fragment_size, mua=False)
                        
                        #partition per conf stats to record
                        if conf_type == 'janus':
                            janus.append(ssr)
                        elif conf_type == 'random':
                            random.append(ssr)
                        else:
                            striped.append(ssr)

                        fragment_probabilities_all.append(fragment_probabilities)
                        print(f'{monolayer}/{conf_type}/Trial {trial+1}',fragment_probabilities, ssr, total_fragments)
        
        ssrs_general.append(janus)
        ssrs_general.append(random)
        ssrs_general.append(striped)
        
        sim_ssr.append(np.mean(ssrs_general))
        sim_ssr_std.append(np.std(ssrs_general))
        
        print(f'{monolayer} JANUS avg: {np.mean(janus)} {np.std(janus)}')
        print(f'{monolayer} Random avg: {np.mean(random)} {np.std(random)}')
        print(f'{monolayer} Striped avg: {np.mean(striped)} {np.std(striped)}')
        print(f'{monolayer} General avg: {np.mean(ssrs_general)} {np.std(ssrs_general)}')

        fragment_probabilities_all = np.array(fragment_probabilities_all)
        fps_avg = np.mean(fragment_probabilities_all, axis=0)
        fps_std = np.std(fragment_probabilities_all, axis=0)
        
        theoretical_fragment_distributions.append(fps_avg)
        theoretical_fragment_distributions_std.append(fps_std)
        
        with open(f'../Datafiles/fragment_data.txt', 'a') as file:
            file.write(f'{monolayer}\t{fps_avg}\t{fps_std}\t{np.mean(ssrs_general)}\t{np.std(ssrs_general)}\n')
            file.close()
    
    return theoretical_fragment_distributions, theoretical_fragment_distributions_std, sim_ssr, sim_ssr_std

def plot(theoretical_fragment_distributions, theoretical_fragment_distributions_std, sim_ssr, sim_ssr_std) -> None:
    """
    Plots the SSRs and fragment distributions to compare between experiment and simulation which are reported in Figure 6 of the manuscript.
    
    :param theoretical_fragment_distributions: list of lists of floats corresponding to the average fragment fractions of each monolayers fragment distribution
    :param theoretical_fragment_distributions_std: list of lists of floats corresponding to the standard deviation fragment fractions of each monolayers fragment distribution
    :param sim_ssr: average SSR value computed of the 9 trials for each monolayer
    :param sim_ssr_std: standard deviation of the SSRs found in the 9 trials for each monolayer
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    x = [0,1,2,3,4]	
    
    plot_index = 0
    plot_titles = ['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA']
    for i in range(2):
        for j in range(2):
            if i==0 and j==0:
                axs[i,j].bar([x-0.25 for x in x], experimental_fragment_distributions[plot_index], yerr = experimental_fragment_distributions_std[plot_index], capsize=5, label ='Experiment', color='#232D4B', ecolor='k', width=0.5, alpha=0.8)
                axs[i,j].bar([x+0.25 for x in x], theoretical_fragment_distributions[plot_index], yerr = theoretical_fragment_distributions_std[plot_index], capsize=5, label ='Simulation', color='#E57200', ecolor='k', width=0.5, alpha=0.8)
                axs[i,j].legend(loc='upper center', bbox_to_anchor=(1.05, 1.25), ncols=2, frameon=False, fontsize='large')
            else:
                axs[i,j].bar([x-0.25 for x in x], experimental_fragment_distributions[plot_index], yerr = experimental_fragment_distributions_std[plot_index], capsize=5, color='#232D4B', ecolor='k', width=0.5, alpha=0.8)
                axs[i,j].bar([x+0.25 for x in x], theoretical_fragment_distributions[plot_index], yerr = theoretical_fragment_distributions_std[plot_index], capsize=5, color='#E57200', ecolor='k', width=0.5, alpha=0.8)
            axs[i,j].set_title(plot_titles[plot_index])
            axs[0,0].set_ylabel('Fragment Probability')
            axs[1,0].set_ylabel('Fragment Probability')
            if plot_index < 2:
                axs[i,j].set_xticks([],[])
            axs[0,1].set_yticks([0.1,0.2,0.3,0.4,0.5],['','','','',''])
            axs[1,1].set_yticks([0.1,0.2,0.3,0.4,0.5],['','','','',''])
            axs[i,j].set_xlabel('# DDT in Fragment')
            axs[1,0].set_xticks([0,1,2,3,4])
            axs[1,1].set_xticks([0,1,2,3,4])
            axs[0,1].set_xticks([0,1,2,3,4],['','','','',''])
            axs[0,0].set_xticks([0,1,2,3,4],['','','','',''])
            axs[1,0].set_xlabel('# MUA in Fragment')
            axs[1,1].set_xlabel('# MUA in Fragment')
            axs[i,j].set_xlim(-1,5)
            axs[i,j].set_ylim(0,0.5)
            plot_index += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.savefig('../Figures/fragment_distributions.png', format='png', dpi=400)
        
    #the below values were obtained and reported in the manuscript
    #sim_ssr = [0.25701611468148156, 0.1046326760366252,0.18068079034717727,0.0019020954646963048]
    #sim_ssr_std = [0.10546676166545496,0.018991263644837914,0.042575937165310954,0.0018083988484822915]
    #exp_ssr = [0.26075,0.2075,0.175580538804374,0.0225]
    #exp_ssr_std = [0.024474,0.025626,0.0160727173799454,0.012834]

    x = [0.45,0.55,0.95,1.05,1.45,1.55,1.95,2.05]
    plt.figure(figsize=(8,8))
    plt.errorbar(x=[x[0], x[2], x[4], x[6]], 
                y=[exp_ssr[0], exp_ssr[1], exp_ssr[2], exp_ssr[3]],
                yerr=[exp_ssr_std[0], exp_ssr_std[1], exp_ssr_std[2], exp_ssr_std[3]],
                capsize=5, label='Experiment',
                fmt='o',
                marker='s',
                color='#232D4B',
                alpha=0.8)
    plt.errorbar(x=[x[1], x[3], x[5], x[7]], 
                y=[sim_ssr[0], sim_ssr[1], sim_ssr[2], sim_ssr[3]],
                yerr=[sim_ssr_std[0], sim_ssr_std[1], sim_ssr_std[2], sim_ssr_std[3]],
                capsize=5, label='Simulation',
                fmt='o',
                marker='o',
                color='#E57200',
                alpha=0.8)
    plt.xlabel('Monolayers')
    plt.ylabel('SSR')
    plt.xticks([0.5,1,1.5,2], fragment_prob_xticklabels)
    plt.legend(frameon=False, fontsize='large')
    plt.xlim(0.35,2.15)
    plt.tight_layout()
    plt.savefig('../Figures/SSRs_simulation_vs_experiment.png', format='png', dpi=400)
    plt.close()
    
    return 

def main() -> None:
    """
    This python script generates both figures shown in figure 4 of the manuscript.
    
    Computes and plots the SSR and Fragment Distributions for the four monolayers studied.
    """
    
    theoretical_fragment_distributions, theoretical_fragment_distributions_std, sim_ssr, sim_ssr_std = compute_frags_and_ssr()
    plot(theoretical_fragment_distributions, theoretical_fragment_distributions_std, sim_ssr, sim_ssr_std)
    
    return


if __name__ == '__main__':
    main()