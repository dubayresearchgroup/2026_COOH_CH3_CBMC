import numpy as np
import matplotlib.pyplot as plt
import random

anchor_type = 5
num_configs = 5
num_configs_baseline = 50
w = [12, 12, 12, 8, 12, 12, 10, 8]
np_sizes = [9,10,9,10,11,9,11,11]
bin_range = [8, 10, 10, 12, 10, 10, 10, 10]
num_ligands_1 = [175, 214, 174, 208, 265, 173, 260, 250]
num_ligands_2 = [175, 214, 174, 208, 265, 173, 260, 250]
num_ligands_1_len = [13, 13, 7, 13, 13, 13, 7, 13]
num_ligands_2_len = [7, 10, 15, 15, 7, 10, 15, 15]
monolayers = ["ddt_mpa", "ddt_mha", "ht_mua", "ddt_mua", "ddt_apt", "ddt_aht", "ht_aut", "ddt_aut"]
lig_1 = ["ddt", "ddt", "ht", "ddt", "ddt", "ddt", "ht", "ddt"]
lig_2 = ["mpa", "mha", "mua", "mua", "apt", "aht", "aut", "aut"]
conf_types = ["janus", "random", "striped"]
trials = 3
monolayer_titles = ['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA', 'DDT/APT', 'DDT/AHT', 'HT/AUT', 'DDT/AUT']

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
    return min_distances

def compute_edge_distances() -> tuple:
    """
    Computes edge distances and returns statistics for plotting.
    """
    avg_monolayer_distances_ch3 = []
    avg_monolayer_distances_cooh = []
    std_monolayer_distances_ch3 = []
    std_monolayer_distances_cooh = []
    monolayer_baselines = []
    xbins_all = []
    _fig, axes = plt.subplots(2, 4, figsize=(7.48, 4.3), sharex=True)

    for m, monolayer in enumerate(monolayers):
        if np_sizes[m] == 9:
            bin_edges = [2.0, 3.3, 5.5, 8]
        elif np_sizes[m] == 10:
            bin_edges = [2.0, 3.3, 5.5, 9]
        elif np_sizes[m] == 11:
            bin_edges = [2.0, 3.3, 5.5, 10]
        ht_filepath = f"../ht_only/ht_{monolayer}/t1/trajectory.xyz"
        np_edge_atoms = read_np_edge_file(f'../datafiles/np_{np_sizes[m]}h_edge_atoms.xyz')
        print('number of edge atoms read: ', len(np_edge_atoms))
        data = []
        min_distances_baseline = []
        row = m // 4
        col = m % 4
        ax = axes[row, col]
        if m < 4:
            label2 = "COOH"
            edge_color='r'
        else:
            label2 = 'NH$_2$'
            edge_color='b'
        with open(ht_filepath, 'r') as file:
            for line in file:
                if line.strip():
                    columns = line.split()
                    if len(columns) == 4 and columns[0] != '1':
                        data.append(columns)
        total_lines = len(data)
        data = np.array(data[(total_lines - num_configs_baseline * (num_ligands_1[m] * 7 + num_ligands_2[m] * 7)):], dtype=float)

        for config in range(num_configs_baseline):
            num_ligand = num_ligands_1[m]
            if config != 0:
                start_line = (config * (num_ligands_1[m] * 7 + num_ligands_2[m] * 7)) - 1
            else:
                start_line = (config * (num_ligands_1[m] * 7 + num_ligands_2[m] * 7))
            lig_sites = data[start_line:,1:4]
            type_sites = data[start_line:,0]
            lig_1_sites = []
            count = 0
            for k in range(len(type_sites)):
                if type_sites[k] == anchor_type:
                    count += 1
                    if count <= num_ligands_1[m] + num_ligands_2[m]:
                        lig_1_sites.append(lig_sites[k])
            lig_1_sites = np.array(lig_1_sites)
            min_distances_baseline.extend(get_distance_from_edge(np_edge_atoms, lig_1_sites))

        min_distances_baseline = [val for val in min_distances_baseline if val <= bin_edges[-1]]
        print(monolayer, 'baseline range detected: ', min(min_distances_baseline), max(min_distances_baseline))
        nx_baseline, xbins = np.histogram(min_distances_baseline, bins=bin_edges, density=False)
        sum_counts_baseline = np.sum(nx_baseline)
        nx_baseline_prob = nx_baseline / sum_counts_baseline
        monolayer_baselines.append(nx_baseline_prob)

        baseline_y = []
        for i in range(len(min_distances_baseline)):
            baseline_y.append(random.uniform(-0.4, 0.4))
        ax.fill_betweenx(y=[-0.45,2.45], x1=bin_edges[0], x2=bin_edges[1], color="#011959", alpha=0.6)
        ax.fill_betweenx(y=[-0.45,2.45], x1=bin_edges[1], x2=bin_edges[2], color="#6F7D3A", alpha=0.8)
        ax.fill_betweenx(y=[-0.45,2.45], x1=bin_edges[2], x2=bin_edges[3], color="#FACCFA", alpha=0.8)
        ax.scatter(min_distances_baseline, baseline_y, color='k', alpha=0.4, edgecolor='k', s=0.1)
        min_distances_ch3 = []
        min_distances_oh = []
        for conf_type in conf_types:
            for trial in range(trials):
                distances_per_trial_ch3 = []
                distances_per_trial_oh = []
                filepath = f"../{monolayer}/hinterwirth_set/w{w[m]}/{conf_type}/t{trial+1}/trajectory.xyz"
                data = []
                with open(filepath, 'r') as file:
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1':
                                data.append(columns)
                total_lines = len(data)
                data = np.array(data[(total_lines - num_configs * (num_ligands_1[m] * num_ligands_1_len[m] + num_ligands_2[m] * num_ligands_2_len[m])):], dtype=float)
                for config in range(num_configs):
                    num_ligand = num_ligands_1[m]
                    if config != 0:
                        start_line = (config * (num_ligands_1[m] * num_ligands_1_len[m] + num_ligands_2[m] * num_ligands_2_len[m])) - 1
                    else:
                        start_line = (config * (num_ligands_1[m] * num_ligands_1_len[m] + num_ligands_2[m] * num_ligands_2_len[m]))
                    lig_sites = data[start_line:,1:4]
                    type_sites = data[start_line:,0]
                    lig_1_sites = []
                    lig_2_sites = []
                    count = 0
                    for k in range(len(type_sites)):
                        if type_sites[k] == anchor_type:
                            count += 1
                            if count <= num_ligand:
                                lig_1_sites.append(lig_sites[k])
                            elif count > num_ligand and count <= (num_ligands_1[m] + num_ligands_2[m]):
                                lig_2_sites.append(lig_sites[k])
                    lig_1_sites = np.array(lig_1_sites)
                    lig_2_sites = np.array(lig_2_sites)
                    distances_per_trial_ch3.extend(get_distance_from_edge(np_edge_atoms, lig_1_sites))
                    distances_per_trial_oh.extend(get_distance_from_edge(np_edge_atoms, lig_2_sites))
                distances_per_trial_ch3 = [val for val in distances_per_trial_ch3 if val <= bin_edges[-1]]
                distances_per_trial_oh = [val for val in distances_per_trial_oh if val <= bin_edges[-1]]
                ch3_y, oh_y = [], []
                for i in range(len(distances_per_trial_ch3)):
                    ch3_y.append(random.uniform(0.6, 1.4))
                for i in range(len(distances_per_trial_oh)):
                    oh_y.append(random.uniform(1.6, 2.4))
                ax.scatter(distances_per_trial_ch3, ch3_y, color='k', alpha=0.4, edgecolor='k', s=0.1)
                ax.scatter(distances_per_trial_oh, oh_y, color='k', alpha=0.4, edgecolor='k', s=0.1)
                nx_oh, xbins = np.histogram(distances_per_trial_oh, bins=bin_edges, density=False)
                nx_ch3, _ = np.histogram(distances_per_trial_ch3, bins=bin_edges, density=False)
                nx_oh_sum, nx_ch3_sum = np.sum(nx_oh), np.sum(nx_ch3)
                nx_oh_prob = np.array(nx_oh) / nx_oh_sum if nx_oh_sum > 0 else np.zeros_like(nx_oh)
                nx_ch3_prob = np.array(nx_ch3) / nx_ch3_sum if nx_ch3_sum > 0 else np.zeros_like(nx_ch3)
                min_distances_ch3.append(nx_ch3_prob)
                min_distances_oh.append(nx_oh_prob)
        with open(f'../datafiles/{monolayer}_edge_distances.txt', 'w') as edge_file:
            edge_file.write('Type 1\tType 2\n')
            for x_i, x in enumerate(min_distances_ch3):
                edge_file.write(f'{x}\t{min_distances_oh[x_i]}\n')
        avg_monolayer_distances_ch3.append(np.mean(min_distances_ch3, axis=0))
        avg_monolayer_distances_cooh.append(np.mean(min_distances_oh, axis=0))
        std_monolayer_distances_ch3.append(np.std(min_distances_ch3, axis=0))
        std_monolayer_distances_cooh.append(np.std(min_distances_oh, axis=0))
        print(monolayer, 'data collected. N samples:', len(min_distances_ch3))
        ax.set_ylim(-0.45, 2.45)
        ax.set_xlim(-0.2, 10.2)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_yticks([0,1,2], labels=[" ", " ", " "])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.1)
        ax.axhline(y=1, color='k', linestyle=':', alpha=0.1)
        ax.axhline(y=2, color='k', linestyle=':', alpha=0.1)
        for x_val in bin_edges:
            ax.axvline(x=x_val, color='orange', linestyle='-')
        ax.set_title(f'{monolayer_titles[m]}', fontweight='bold', fontsize=14)
        if row == 1:
            ax.set_xlabel('$d$', fontsize=14)
        xbins_all.append(xbins)
    axes[0, 0].set_yticks([0,1,2], labels=["HT", "-CH$_3$", "-COOH"])
    axes[1, 0].set_yticks([0,1,2], labels=["HT", "-CH$_3$", "-NH$_2$"])
    #axes[0, 0].set_yticks([0], labels=["HT"])
    #axes[1, 0].set_yticks([0], labels=["HT"])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('../figures/figure_S9.png', format='png', dpi=400)
    plt.close()
    print("Shapes - ch3:", len(avg_monolayer_distances_ch3), "cooh:", len(avg_monolayer_distances_cooh))

    return xbins_all, avg_monolayer_distances_cooh, avg_monolayer_distances_ch3, std_monolayer_distances_cooh, std_monolayer_distances_ch3, monolayer_baselines


def plot_unnormalized(xbins, avg_monolayer_distances_cooh, avg_monolayer_distances_ch3, std_monolayer_distances_cooh, std_monolayer_distances_ch3, monolayer_baselines) -> None:
    """
    Plots unnormalized histograms with errorbars representing standard deviation.
    """
    _fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
    for m, monolayer in enumerate(monolayers):
        if m < 4:
            label2 = "COOH"
            edge_color='r'
        else:
            label2 = 'NH$_2$'
            edge_color='b'
        row = m // 4
        col = m % 4
        ax = axes[row, col]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f'{monolayer_titles[m]}', fontweight='bold', fontsize=14)
        ax.stairs(avg_monolayer_distances_cooh[m], xbins[m], orientation='vertical',
                  fill=False, label=label2, edgecolor=edge_color, linewidth=1.5)
        ax.stairs(avg_monolayer_distances_ch3[m], xbins[m], orientation='vertical',
                  fill=False, label='CH$_3$', edgecolor='k', linewidth=1.5)
        ax.stairs(monolayer_baselines[m], xbins[m], orientation='vertical',
                  fill=False, label='$P_0(d)$', edgecolor='k', linewidth=1.5, linestyle=':')
        bin_centers = (xbins[m][:-1] + xbins[m][1:]) / 2
        ax.errorbar(bin_centers, avg_monolayer_distances_cooh[m],
                   yerr=std_monolayer_distances_cooh[m], fmt='none',
                   color=edge_color, capsize=3, alpha=0.8)
        ax.errorbar(bin_centers, avg_monolayer_distances_ch3[m],
                   yerr=std_monolayer_distances_ch3[m], fmt='none',
                   color='k', capsize=3, alpha=0.8)
        ax.legend(fontsize='large', frameon=False, loc='upper right', ncols=1)
        if row == 1:
            ax.set_xlabel('$d$', fontsize=14)
    axes[0, 0].set_ylabel('$P(d)$')
    axes[1, 0].set_ylabel('$P(d)$')
    plt.tight_layout()
    plt.savefig('../figures/figure_S10.png', format='png', dpi=400)
    plt.close()

def plot_normalized(xbins, avg_monolayer_distances_cooh, avg_monolayer_distances_ch3, std_monolayer_distances_cooh, std_monolayer_distances_ch3, monolayer_baselines) -> None:
    """
    Plots normalized histograms (P/P0) with errorbars.
    Baseline error is assumed negligible, so normalized error = std / baseline.
    """
    _fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
    for m, monolayer in enumerate(monolayers):
        if m < 4:
            label2 = "COOH"
            edge_color='r'
        else:
            label2 = 'NH$_2$'
            edge_color='b'
        row = m // 4
        col = m % 4
        ax = axes[row, col]
        ax.set_xlim(-0.2, 10.2)
        ax.set_ylim(0, 2.0)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.axhline(y=1, color='k', linestyle=":", alpha=0.1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f'{monolayer_titles[m]}', fontweight='bold', fontsize=14)
        norm_ch3 = avg_monolayer_distances_ch3[m] / monolayer_baselines[m]
        norm_cooh = avg_monolayer_distances_cooh[m] / monolayer_baselines[m]
        baseline_safe = np.where(monolayer_baselines[m] == 0, np.inf, monolayer_baselines[m])
        norm_std_ch3 = std_monolayer_distances_ch3[m] / baseline_safe
        norm_std_cooh = std_monolayer_distances_cooh[m] / baseline_safe
        ax.stairs(norm_ch3, xbins[m], orientation='vertical', fill=False,
                 label='CH$_3$', edgecolor='k', linewidth=3.5)
        ax.stairs(norm_cooh, xbins[m], orientation='vertical', fill=False,
                 label=label2, edgecolor=edge_color, linewidth=3.5)
        bin_centers = (xbins[m][:-1] + xbins[m][1:]) / 2
        ax.errorbar(bin_centers, norm_ch3, yerr=norm_std_ch3,
                   fmt='none', color='k', capsize=3, alpha=0.6)
        ax.errorbar(bin_centers, norm_cooh, yerr=norm_std_cooh,
                   fmt='none', color=edge_color, capsize=3, alpha=0.6)
        ax.legend(fontsize='large', frameon=False, loc='upper right', ncols=1)
        if row == 1:
            ax.set_xlabel('$d$', fontsize=14)
    axes[0, 0].set_ylabel('$P(d)$/$P_0(d)$', fontsize=14)
    axes[1, 0].set_ylabel('$P(d)$/$P_0(d)$', fontsize=14)
    plt.tight_layout()
    plt.savefig('../figures/figure_6b.png', format='png', dpi=400)
    plt.close()

def read_np_edge_file(filepath) -> np.ndarray:
    """
    parses the nanoparticle edge site coordinate file and stores its contents in the NDArray np_edge_atoms
    """
    np_edge_atoms = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():
                columns = line.split()
                if len(columns) == 4:
                    np_edge_atoms.append(columns[1:])
    return np.array(np_edge_atoms, dtype=float)

def main() -> None:
    """
    Generates figure S6 and related plots.
    """
    xbins, monolayer_distances_cooh, monolayer_distances_ch3, std_cooh, std_ch3, monolayer_baselines = compute_edge_distances()
    plot_unnormalized(xbins, monolayer_distances_cooh, monolayer_distances_ch3,
                     std_cooh, std_ch3, monolayer_baselines)
    plot_normalized(xbins, monolayer_distances_cooh, monolayer_distances_ch3,
                   std_cooh, std_ch3, monolayer_baselines)

if __name__ == '__main__':
    main()
