import matplotlib.pyplot as plt
import numpy as np
import cluster_functions
from matplotlib.cbook import boxplot_stats

anchor_type = 5
num_configs = 5
trials = 3
threshold = 5
w = [12, 12, 10, 8, 12, 12, 12, 8]
num_ligands_1 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_2 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_1_len = [13, 13, 7, 13, 13, 13, 7, 13]
num_ligands_2_len = [7, 10, 15, 15, 7, 10, 15, 15]
monolayers = ["ddt_apt", "ddt_aht", "ht_aut", "ddt_aut", "ddt_mpa", "ddt_mha", "ht_mua", "ddt_mua"]
conf_types = ["janus", "random", "striped"]

ddt_mpa_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_mpa_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_mua_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_mua_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_mha_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_mha_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ht_mua_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ht_mua_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_apt_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_apt_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_aut_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_aut_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_aht_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ddt_aht_2_clusters = [[] for _ in range(len(conf_types) * trials)]
ht_aut_1_clusters = [[] for _ in range(len(conf_types) * trials)]
ht_aut_2_clusters = [[] for _ in range(len(conf_types) * trials)]

ddt_mpa_1_nw = []
ddt_mpa_2_nw = []
ddt_mua_1_nw = []
ddt_mua_2_nw = []
ddt_mha_1_nw = []
ddt_mha_2_nw = []
ht_mua_1_nw = []
ht_mua_2_nw = []
ddt_apt_1_nw = []
ddt_apt_2_nw = []
ddt_aut_1_nw = []
ddt_aut_2_nw = []
ddt_aht_1_nw = []
ddt_aht_2_nw = []
ht_aut_1_nw = []
ht_aut_2_nw = []

for i, monolayer in enumerate(monolayers):
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
                oh_s_sites = []
                ch3_s_sites = []
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
                oh_s_sites = np.array(oh_s_sites)
                ch3_s_sites = np.array(ch3_s_sites)
                oh_s_sites_occupied = {tuple(site) for site in oh_s_sites}
                ch3_s_sites_occupied = {tuple(site) for site in ch3_s_sites}
                filled_s_sites = [['oh', list(site)] for site in oh_s_sites_occupied] + [['ch3', list(site)] for site in ch3_s_sites_occupied]
                aa, bb = cluster_functions.analyze_clusters(filled_surface_sites=filled_s_sites, nn_threshold=threshold, num_ligands_1=num_ligands_1[i], num_ligands_2=num_ligands_2[i])
                if monolayer == 'ddt_mpa':
                    ddt_mpa_1_clusters[trial_count].extend(aa)
                    ddt_mpa_2_clusters[trial_count].extend(bb)
                    ddt_mpa_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ddt_mpa_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ddt_mha':
                    ddt_mha_1_clusters[trial_count].extend(aa)
                    ddt_mha_2_clusters[trial_count].extend(bb)
                    ddt_mha_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ddt_mha_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ddt_mua':
                    ddt_mua_1_clusters[trial_count].extend(aa)
                    ddt_mua_2_clusters[trial_count].extend(bb)
                    ddt_mua_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ddt_mua_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ht_mua':
                    ht_mua_1_clusters[trial_count].extend(aa)
                    ht_mua_2_clusters[trial_count].extend(bb)
                    ht_mua_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ht_mua_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ddt_apt':
                    ddt_apt_1_clusters[trial_count].extend(aa)
                    ddt_apt_2_clusters[trial_count].extend(bb)
                    ddt_apt_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ddt_apt_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ddt_aht':
                    ddt_aht_1_clusters[trial_count].extend(aa)
                    ddt_aht_2_clusters[trial_count].extend(bb)
                    ddt_aht_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ddt_aht_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ddt_aut':
                    ddt_aut_1_clusters[trial_count].extend(aa)
                    ddt_aut_2_clusters[trial_count].extend(bb)
                    ddt_aut_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ddt_aut_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
                elif monolayer == 'ht_aut':
                    ht_aut_1_clusters[trial_count].extend(aa)
                    ht_aut_2_clusters[trial_count].extend(bb)
                    ht_aut_1_nw.append((sum(n ** 2 for n in aa) / sum(aa)) if sum(aa) > 0 else 0)
                    ht_aut_2_nw.append((sum(n ** 2 for n in bb) / sum(bb)) if sum(bb) > 0 else 0)
            trial_count += 1

plt.figure(figsize=(8.5,6))
x_ch3_0, x_ch3_1, x_ch3_2, x_ch3_3, x_ch3_4, x_ch3_5, x_ch3_6, x_ch3_7 = 0.11, 0.31, 0.51, 0.71, 1.01, 1.21, 1.41, 1.61
x_cooh_0, x_cooh_1, x_cooh_2, x_cooh_3, x_cooh_4, x_cooh_5, x_cooh_6, x_cooh_7 = 0.21, 0.41, 0.61, 0.81, 1.11, 1.31, 1.51, 1.71
print(len(ddt_mpa_1_clusters), len(ddt_mpa_2_clusters))

for i in range(9):
    if i == 0:
        plt.scatter(x=[x_ch3_0]*len(ddt_mpa_1_clusters[i]), y=ddt_mpa_1_clusters[i], color='k', marker='o', label='-CH$_3$', s=10)
        plt.scatter(x=[x_cooh_0]*len(ddt_mpa_2_clusters[i]), y=ddt_mpa_2_clusters[i], color='r', marker='o', label='-COOH', s=10)
        plt.scatter(x=[x_cooh_4]*len(ddt_apt_2_clusters[i]), y=ddt_apt_2_clusters[i], color='b', marker='o', label='-NH$_2$', s=10)
        plt.scatter(x=[x_ch3_4]*len(ddt_apt_1_clusters[i]), y=ddt_apt_1_clusters[i], color='k', marker='o', s=10)
    else:
        plt.scatter(x=[x_ch3_0]*len(ddt_mpa_1_clusters[i]), y=ddt_mpa_1_clusters[i], color='k', marker='o', s=10)
        plt.scatter(x=[x_cooh_0]*len(ddt_mpa_2_clusters[i]), y=ddt_mpa_2_clusters[i], color='r', marker='o', s=10,)
        plt.scatter(x=[x_ch3_4]*len(ddt_apt_1_clusters[i]), y=ddt_apt_1_clusters[i], color='k', marker='o', s=10)
        plt.scatter(x=[x_cooh_4]*len(ddt_apt_2_clusters[i]), y=ddt_apt_2_clusters[i], color='b', marker='o', s=10)
    plt.scatter(x=[x_ch3_1]*len(ddt_mha_1_clusters[i]), y=ddt_mha_1_clusters[i], color='k', marker='o', s=10)
    plt.scatter(x=[x_cooh_1]*len(ddt_mha_2_clusters[i]), y=ddt_mha_2_clusters[i], color='r', marker='o', s=10)
    plt.scatter(x=[x_ch3_2]*len(ht_mua_1_clusters[i]), y=ht_mua_1_clusters[i], color='k', marker='o', s=10)
    plt.scatter(x=[x_cooh_2]*len(ht_mua_2_clusters[i]), y=ht_mua_2_clusters[i], color='r', marker='o', s=10)
    plt.scatter(x=[x_ch3_3]*len(ddt_mua_1_clusters[i]), y=ddt_mua_1_clusters[i], color='k', marker='o', s=10)
    plt.scatter(x=[x_cooh_3]*len(ddt_mua_2_clusters[i]), y=ddt_mua_2_clusters[i], color='r', marker='o', s=10)
    plt.scatter(x=[x_ch3_5]*len(ddt_aht_1_clusters[i]), y=ddt_aht_1_clusters[i], color='k', marker='o', s=10)
    plt.scatter(x=[x_cooh_5]*len(ddt_aht_2_clusters[i]), y=ddt_aht_2_clusters[i], color='b', marker='o', s=10)
    plt.scatter(x=[x_ch3_7]*len(ddt_aut_1_clusters[i]), y=ddt_aut_1_clusters[i], color='k', marker='o', s=10)
    plt.scatter(x=[x_cooh_7]*len(ddt_aut_2_clusters[i]), y=ddt_aut_2_clusters[i], color='b', marker='o', s=10)
    plt.scatter(x=[x_ch3_6]*len(ht_aut_1_clusters[i]), y=ht_aut_1_clusters[i], color='k', marker='o', s=10)
    plt.scatter(x=[x_cooh_6]*len(ht_aut_2_clusters[i]), y=ht_aut_2_clusters[i], color='b', marker='o', s=10)
    x_ch3_0 += 0.01
    x_cooh_0 += 0.01
    x_ch3_1 += 0.01
    x_cooh_1 += 0.01
    x_ch3_2 += 0.01
    x_cooh_2 += 0.01
    x_ch3_3 += 0.01
    x_cooh_3 += 0.01
    x_ch3_4 += 0.01
    x_cooh_4 += 0.01
    x_ch3_5 += 0.01
    x_cooh_5 += 0.01
    x_ch3_6 += 0.01
    x_cooh_6 += 0.01
    x_ch3_7 += 0.01
    x_cooh_7 += 0.01

box_1 = plt.boxplot(ddt_mpa_1_nw, positions=[0.145], vert=True, patch_artist=True, widths=0.04)
box_2 = plt.boxplot(ddt_mpa_2_nw, positions=[0.245], vert=True, patch_artist=True, widths=0.04)
box_1['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_1['boxes'][0].set_edgecolor('orange')
box_1['medians'][0].set_color('orange')
box_1['medians'][0].set_linewidth(2)
box_1['caps'][0].set_color('orange')
box_1['caps'][1].set_color('orange')
box_1['whiskers'][0].set_color('orange')
box_1['whiskers'][1].set_color('orange')
box_2['boxes'][0].set_facecolor((1, 0, 0, 0.6))
box_2['boxes'][0].set_edgecolor('orange')
box_2['medians'][0].set_color('orange')
box_2['medians'][0].set_linewidth(2)
box_2['caps'][0].set_color('orange')
box_2['caps'][1].set_color('orange')
box_2['whiskers'][0].set_color('orange')
box_2['whiskers'][1].set_color('orange')

box_3 = plt.boxplot(ddt_mha_1_nw, positions=[0.345], vert=True, patch_artist=True, widths=0.04)
box_4 = plt.boxplot(ddt_mha_2_nw, positions=[0.445], vert=True, patch_artist=True, widths=0.04)
box_3['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_3['boxes'][0].set_edgecolor('orange')
box_3['medians'][0].set_color('orange')
box_3['medians'][0].set_linewidth(2)
box_3['caps'][0].set_color('orange')
box_3['caps'][1].set_color('orange')
box_3['whiskers'][0].set_color('orange')
box_3['whiskers'][1].set_color('orange')
box_4['boxes'][0].set_facecolor((1, 0, 0, 0.6))
box_4['boxes'][0].set_edgecolor('orange')
box_4['medians'][0].set_color('orange')
box_4['medians'][0].set_linewidth(2)
box_4['caps'][0].set_color('orange')
box_4['caps'][1].set_color('orange')
box_4['whiskers'][0].set_color('orange')
box_4['whiskers'][1].set_color('orange')

box_5 = plt.boxplot(ht_mua_1_nw, positions=[0.545], vert=True, patch_artist=True, widths=0.04)
box_6 = plt.boxplot(ht_mua_2_nw, positions=[0.645], vert=True, patch_artist=True, widths=0.04)
box_5['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_5['boxes'][0].set_edgecolor('orange')
box_5['medians'][0].set_color('orange')
box_5['medians'][0].set_linewidth(2)
box_5['caps'][0].set_color('orange')
box_5['caps'][1].set_color('orange')
box_5['whiskers'][0].set_color('orange')
box_5['whiskers'][1].set_color('orange')
box_6['boxes'][0].set_facecolor((1, 0, 0, 0.6))
box_6['boxes'][0].set_edgecolor('orange')
box_6['medians'][0].set_color('orange')
box_6['medians'][0].set_linewidth(2)
box_6['caps'][0].set_color('orange')
box_6['caps'][1].set_color('orange')
box_6['whiskers'][0].set_color('orange')
box_6['whiskers'][1].set_color('orange')

box_7 = plt.boxplot(ddt_mua_1_nw, positions=[0.745], vert=True, patch_artist=True, widths=0.04)
box_8 = plt.boxplot(ddt_mua_2_nw, positions=[0.845], vert=True, patch_artist=True, widths=0.04)
box_7['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_7['boxes'][0].set_edgecolor('orange')
box_7['medians'][0].set_color('orange')
box_7['medians'][0].set_linewidth(2)
box_7['caps'][0].set_color('orange')
box_7['caps'][1].set_color('orange')
box_7['whiskers'][0].set_color('orange')
box_7['whiskers'][1].set_color('orange')
box_8['boxes'][0].set_facecolor((1, 0, 0, 0.6))
box_8['boxes'][0].set_edgecolor('orange')
box_8['medians'][0].set_color('orange')
box_8['medians'][0].set_linewidth(2)
box_8['caps'][0].set_color('orange')
box_8['caps'][1].set_color('orange')
box_8['whiskers'][0].set_color('orange')
box_8['whiskers'][1].set_color('orange')

box_9 = plt.boxplot(ddt_apt_1_nw, positions=[1.045], vert=True, patch_artist=True, widths=0.04)
box_10 = plt.boxplot(ddt_apt_2_nw, positions=[1.145], vert=True, patch_artist=True, widths=0.04)
box_9['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_9['boxes'][0].set_edgecolor('orange')
box_9['medians'][0].set_color('orange')
box_9['medians'][0].set_linewidth(2)
box_9['caps'][0].set_color('orange')
box_9['caps'][1].set_color('orange')
box_9['whiskers'][0].set_color('orange')
box_9['whiskers'][1].set_color('orange')
box_10['boxes'][0].set_facecolor((0, 0, 1, 0.6))
box_10['boxes'][0].set_edgecolor('orange')
box_10['medians'][0].set_color('orange')
box_10['medians'][0].set_linewidth(2)
box_10['caps'][0].set_color('orange')
box_10['caps'][1].set_color('orange')
box_10['whiskers'][0].set_color('orange')
box_10['whiskers'][1].set_color('orange')

box_11 = plt.boxplot(ddt_aht_1_nw, positions=[1.245], vert=True, patch_artist=True, widths=0.04)
box_12 = plt.boxplot(ddt_aht_2_nw, positions=[1.345], vert=True, patch_artist=True, widths=0.04)
box_11['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_11['boxes'][0].set_edgecolor('orange')
box_11['medians'][0].set_color('orange')
box_11['medians'][0].set_linewidth(2)
box_11['caps'][0].set_color('orange')
box_11['caps'][1].set_color('orange')
box_11['whiskers'][0].set_color('orange')
box_11['whiskers'][1].set_color('orange')
box_12['boxes'][0].set_facecolor((0, 0, 1, 0.6))
box_12['boxes'][0].set_edgecolor('orange')
box_12['medians'][0].set_color('orange')
box_12['medians'][0].set_linewidth(2)
box_12['caps'][0].set_color('orange')
box_12['caps'][1].set_color('orange')
box_12['whiskers'][0].set_color('orange')
box_12['whiskers'][1].set_color('orange')

box_13 = plt.boxplot(ht_aut_1_nw, positions=[1.445], vert=True, patch_artist=True, widths=0.04)
box_14 = plt.boxplot(ht_aut_2_nw, positions=[1.545], vert=True, patch_artist=True, widths=0.04)
box_13['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_13['boxes'][0].set_edgecolor('orange')
box_13['medians'][0].set_color('orange')
box_13['medians'][0].set_linewidth(2)
box_13['caps'][0].set_color('orange')
box_13['caps'][1].set_color('orange')
box_13['whiskers'][0].set_color('orange')
box_13['whiskers'][1].set_color('orange')
box_14['boxes'][0].set_facecolor((0, 0, 1, 0.6))
box_14['boxes'][0].set_edgecolor('orange')
box_14['medians'][0].set_color('orange')
box_14['medians'][0].set_linewidth(2)
box_14['caps'][0].set_color('orange')
box_14['caps'][1].set_color('orange')
box_14['whiskers'][0].set_color('orange')
box_14['whiskers'][1].set_color('orange')

box_15 = plt.boxplot(ddt_aut_1_nw, positions=[1.645], vert=True, patch_artist=True, widths=0.04)
box_16 = plt.boxplot(ddt_aut_2_nw, positions=[1.745], vert=True, patch_artist=True, widths=0.04)
box_15['boxes'][0].set_facecolor((0, 0, 0, 0.6))
box_15['boxes'][0].set_edgecolor('orange')
box_15['medians'][0].set_color('orange')
box_15['medians'][0].set_linewidth(2)
box_15['caps'][0].set_color('orange')
box_15['caps'][1].set_color('orange')
box_15['whiskers'][0].set_color('orange')
box_15['whiskers'][1].set_color('orange')
box_16['boxes'][0].set_facecolor((0, 0, 1, 0.6))
box_16['boxes'][0].set_edgecolor('orange')
box_16['medians'][0].set_color('orange')
box_16['medians'][0].set_linewidth(2)
box_16['caps'][0].set_color('orange')
box_16['caps'][1].set_color('orange')
box_16['whiskers'][0].set_color('orange')
box_16['whiskers'][1].set_color('orange')

x_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
[plt.axvline(x, color='k', linestyle='--', alpha=0.5) for x in x_vals]
plt.axvline(0.95, color='k', linewidth=3.5)
plt.xticks([0.2,0.4,0.6,0.8,1.1,1.3,1.5,1.7],['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA', 'DDT/APT', 'DDT/AHT', 'HT/AUT', 'DDT/AUT'])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.ylabel('Cluster Size Fraction', fontsize=14)
plt.xlabel('Monolayers', fontsize=14)
plt.legend(frameon=False, fontsize='large', bbox_to_anchor=(0.5,1.1), ncols=3, loc='upper center')
plt.xlim(0.05,1.85)
plt.ylim(bottom=0, top=1.2)
plt.tight_layout()
plt.savefig('../figures/figure_7.png', format='png', dpi=400)

ddt_mpa_1_boxplot_stats = boxplot_stats(ddt_mpa_1_nw)[0]
ddt_mpa_2_boxplot_stats = boxplot_stats(ddt_mpa_2_nw)[0]
ddt_mha_1_boxplot_stats = boxplot_stats(ddt_mha_1_nw)[0]
ddt_mha_2_boxplot_stats = boxplot_stats(ddt_mha_2_nw)[0]
ht_mua_1_boxplot_stats = boxplot_stats(ht_mua_1_nw)[0]
ht_mua_2_boxplot_stats = boxplot_stats(ht_mua_2_nw)[0]
ddt_mua_1_boxplot_stats = boxplot_stats(ddt_mua_1_nw)[0]
ddt_mua_2_boxplot_stats = boxplot_stats(ddt_mua_2_nw)[0]
ddt_apt_1_boxplot_stats = boxplot_stats(ddt_apt_1_nw)[0]
ddt_apt_2_boxplot_stats = boxplot_stats(ddt_apt_2_nw)[0]
ddt_aht_1_boxplot_stats = boxplot_stats(ddt_aht_1_nw)[0]
ddt_aht_2_boxplot_stats = boxplot_stats(ddt_aht_2_nw)[0]
ht_aut_1_boxplot_stats = boxplot_stats(ht_aut_1_nw)[0]
ht_aut_2_boxplot_stats = boxplot_stats(ht_aut_2_nw)[0]
ddt_aut_1_boxplot_stats = boxplot_stats(ddt_aut_1_nw)[0]
ddt_aut_2_boxplot_stats = boxplot_stats(ddt_aut_2_nw)[0]

with open("../datafiles/cluster_data.txt", "w") as file:
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
               f'{ddt_mua_2_boxplot_stats["whislo"]}\t{ddt_mua_2_boxplot_stats["q1"]}\t{ddt_mua_2_boxplot_stats["med"]}\t{ddt_mua_2_boxplot_stats["q3"]}\t{ddt_mua_2_boxplot_stats["whishi"]}\n\n')
    file.write(f'DDT/APT Alkane\t{sum(len(c) for c in ddt_apt_1_clusters)}\t'
               f'{ddt_apt_1_boxplot_stats["whislo"]}\t{ddt_apt_1_boxplot_stats["q1"]}\t{ddt_apt_1_boxplot_stats["med"]}\t{ddt_apt_1_boxplot_stats["q3"]}\t{ddt_apt_1_boxplot_stats["whishi"]}\n')
    file.write(f'DDT/APT Amine\t{sum(len(c) for c in ddt_apt_2_clusters)}\t'
               f'{ddt_apt_2_boxplot_stats["whislo"]}\t{ddt_apt_2_boxplot_stats["q1"]}\t{ddt_apt_2_boxplot_stats["med"]}\t{ddt_apt_2_boxplot_stats["q3"]}\t{ddt_apt_2_boxplot_stats["whishi"]}\n\n')
    file.write(f'DDT/AHT Alkane\t{sum(len(c) for c in ddt_aht_1_clusters)}\t'
               f'{ddt_aht_1_boxplot_stats["whislo"]}\t{ddt_aht_1_boxplot_stats["q1"]}\t{ddt_aht_1_boxplot_stats["med"]}\t{ddt_aht_1_boxplot_stats["q3"]}\t{ddt_aht_1_boxplot_stats["whishi"]}\n')
    file.write(f'DDT/AHT Amine\t{sum(len(c) for c in ddt_aht_2_clusters)}\t'
               f'{ddt_aht_2_boxplot_stats["whislo"]}\t{ddt_aht_2_boxplot_stats["q1"]}\t{ddt_aht_2_boxplot_stats["med"]}\t{ddt_aht_2_boxplot_stats["q3"]}\t{ddt_aht_2_boxplot_stats["whishi"]}\n\n')
    file.write(f'HT/AUT Alkane\t{sum(len(c) for c in ht_aut_1_clusters)}\t'
               f'{ht_aut_1_boxplot_stats["whislo"]}\t{ht_aut_1_boxplot_stats["q1"]}\t{ht_aut_1_boxplot_stats["med"]}\t{ht_aut_1_boxplot_stats["q3"]}\t{ht_aut_1_boxplot_stats["whishi"]}\n')
    file.write(f'HT/AUT Amine\t{sum(len(c) for c in ht_aut_2_clusters)}\t'
               f'{ht_aut_2_boxplot_stats["whislo"]}\t{ht_aut_2_boxplot_stats["q1"]}\t{ht_aut_2_boxplot_stats["med"]}\t{ht_aut_2_boxplot_stats["q3"]}\t{ht_aut_2_boxplot_stats["whishi"]}\n\n')
    file.write(f'DDT/AUT Alkane\t{sum(len(c) for c in ddt_aut_1_clusters)}\t'
               f'{ddt_aut_1_boxplot_stats["whislo"]}\t{ddt_aut_1_boxplot_stats["q1"]}\t{ddt_aut_1_boxplot_stats["med"]}\t{ddt_aut_1_boxplot_stats["q3"]}\t{ddt_aut_1_boxplot_stats["whishi"]}\n')
    file.write(f'DDT/AUT Amine\t{sum(len(c) for c in ddt_aut_2_clusters)}\t'
               f'{ddt_aut_2_boxplot_stats["whislo"]}\t{ddt_aut_2_boxplot_stats["q1"]}\t{ddt_aut_2_boxplot_stats["med"]}\t{ddt_aut_2_boxplot_stats["q3"]}\t{ddt_aut_2_boxplot_stats["whishi"]}\n')

print("Cluster analysis complete")
