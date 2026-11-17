import matplotlib.pyplot as plt
import numpy as np
import rdf_functions

anchor_type = 5
num_configs = 5
trials = 3
r_max = 20
nbins = 75
threshold = 5
w = [12, 12, 10, 8, 12, 12, 12, 8]
num_ligands_1 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_2 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_1_len = [13, 13, 7, 13, 13, 13, 7, 13]
num_ligands_2_len = [7, 10, 15, 15, 7, 10, 15, 15]
monolayers = ["ddt_apt", "ddt_aht", "ht_aut", "ddt_aut", "ddt_mpa", "ddt_mha", "ht_mua", "ddt_mua"]
conf_types = ["janus", "random", "striped"]

plot_index = 0
fig, axs = plt.subplots(8, 2, sharex=True, sharey=True, figsize=(7.48,10))
for i, monolayer in enumerate(monolayers):
    rdf_data_ohs_sum = np.zeros(nbins)
    rdf_data_ch3s_sum = np.zeros(nbins)
    rdf_data_both_sum = np.zeros(nbins)
    rdf_anchor_data_ohs_sum = np.zeros(nbins)
    rdf_anchor_data_ch3s_sum = np.zeros(nbins)
    rdf_anchor_data_both_sum = np.zeros(nbins)
    total_trials = len(conf_types) * trials * num_configs
    trial_count = 0
    for j, conf_type in enumerate(conf_types):
        for trial in range(trials):
            filepath = f"../{monolayer}/w{w[i]}/{conf_type}/t{trial+1}/trajectory.xyz"
            data = []
            with open(filepath, "r") as file:
                print(True)
                for line in file:
                    if line.strip():
                        columns = line.split()
                        if len(columns) == 4 and columns[0] != "1":
                            data.append(columns)
                total_lines = len(data)
                data = np.array(data[(total_lines - num_configs * (num_ligands_1[i] * num_ligands_1_len[i] + num_ligands_2[i] * num_ligands_2_len[i])):], dtype=float)
            for config in range(num_configs):
                print(trial_count)
                start_line = config * (
                    num_ligands_1[i] * num_ligands_1_len[i]
                    + num_ligands_2[i] * num_ligands_2_len[i]
                )
                end_line = start_line + (
                    num_ligands_1[i] * num_ligands_1_len[i]
                    + num_ligands_2[i] * num_ligands_2_len[i]
                )
                lig_sites = data[start_line:end_line, 1:4]
                type_sites = data[start_line:end_line, 0]
                oh_sites = []
                ch3_sites = []
                ch3_s_sites = []
                oh_s_sites = []
                count_s = 0
                if i > 3:
                    type_2_site_label = 2
                else:
                    type_2_site_label = 6
                for k in range(len(type_sites)):
                    if type_sites[k] == anchor_type:
                        count_s += 1
                        if count_s <= num_ligands_1[i]:
                            ch3_s_sites.append(lig_sites[k])
                        else:
                            oh_s_sites.append(lig_sites[k])
                    elif type_sites[k] == 4:  # CH3
                        ch3_sites.append(lig_sites[k])
                    elif type_sites[k] == type_2_site_label:  # H-bearing oxygen or N
                        oh_sites.append(lig_sites[k])
                oh_sites = np.array(oh_sites)
                ch3_sites = np.array(ch3_sites)
                oh_s_sites = np.array(oh_s_sites)
                ch3_s_sites = np.array(ch3_s_sites)
                print('NUM CH3 surf sites in array: ', len(ch3_s_sites))
                print('NUM CH3 head sites in array: ', len(ch3_sites))
                print('NUM COOH surf sites in array: ', len(oh_s_sites))
                print('NUM COOH head sites in array: ', len(oh_sites))
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
                rdf_data_ohs_sum += rdf_data_ohs
                rdf_data_ch3s_sum += rdf_data_ch3s
                rdf_data_both_sum += rdf_data_both
                rdf_anchor_data_ohs_sum += rdf_anchor_data_ohs
                rdf_anchor_data_ch3s_sum += rdf_anchor_data_ch3s
                rdf_anchor_data_both_sum += rdf_anchor_data_both
            trial_count += 1
    rdf_data_ohs_avg = rdf_data_ohs_sum / total_trials
    rdf_data_ch3s_avg = rdf_data_ch3s_sum / total_trials
    rdf_data_both_avg = rdf_data_both_sum / total_trials
    rdf_anchor_data_ohs_avg = rdf_anchor_data_ohs_sum / total_trials
    rdf_anchor_data_ch3s_avg = rdf_anchor_data_ch3s_sum / total_trials
    rdf_anchor_data_both_avg = rdf_anchor_data_both_sum / total_trials
    if i > 3:
        axs[i,0].plot(bin_edges[:-1], rdf_data_ohs_avg, label='COOH\u2192COOH', color='r')
        axs[i,0].plot(bin_edges[:-1], rdf_data_both_avg, label='CH3\u2192COOH', color='#67CC8E')
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_both_avg, label='CH3\u2192COOH', color='#67CC8E')
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_ohs_avg, label='COOH\u2192COOH', color='r')
    else:
        axs[i,0].plot(bin_edges[:-1], rdf_data_ohs_avg, label='NH2\u2192NH2', color='b')
        axs[i,0].plot(bin_edges[:-1], rdf_data_both_avg, label='CH3\u2192NH2', color='#67CC8E')
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_both_avg, label='CH3\u2192NH2', color='#67CC8E')
        axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_ohs_avg, label='NH2\u2192NH2', color='b')

    axs[i,0].plot(bin_edges[:-1], rdf_data_ch3s_avg, label='CH3\u2192CH3', color='k')
    axs[i,1].plot(bin_edges[:-1], rdf_anchor_data_ch3s_avg, label='CH3\u2192CH3', color='k')
    if i == len(monolayers)-1:
        axs[i,0].set_xticks([0,5,10,15,20])
        axs[i,1].set_xticks([0,5,10,15,20])
    axs[i,0].set_xlim(0.0, 20.0)
    axs[i,0].set_ylim(0.0, 6.0)
    if i == 0:
        axs[i,0].set_title('Tail RDF')
        axs[i,1].set_title('Anchor RDF')
    axs[i,0].set_yticks([1,6.0])
    axs[i,1].set_xlim(0.0, 20.0)
    axs[i,1].set_ylim(0.0, 6.0)
    axs[i,0].tick_params(axis='both', which='major', labelsize=13)
    axs[i,1].tick_params(axis='both', which='major', labelsize=13)
    plot_index += 1
fig.supylabel("g(r)", fontsize=14)
fig.supxlabel(r'$r\ (\mathrm{\AA})$', fontsize=14)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.savefig('../figures/figure_S8.png', format='png', dpi=400)
print("RDF analysis complete")
