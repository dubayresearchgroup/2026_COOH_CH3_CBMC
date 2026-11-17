import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.spatial import KDTree

def get_binomial(fraction, fragment_size):
    return ss.binom.pmf(np.arange(fragment_size + 1), fragment_size, fraction)

def get_maldi_ssr(filled_surface_sites, num_iterations, nn_threshold, target_fractions, fragment_size, mua=False):
    fragments = [0] * (fragment_size + 1)
    filled_surface_sites_np = [
        [entry[0], np.array(entry[1])] for entry in filled_surface_sites
    ]
    coordinates_np = np.array([entry[1] for entry in filled_surface_sites_np])
    kdtree = KDTree(coordinates_np)
    total_fragments = 0
    for i in range(num_iterations):
        chosen_index = np.random.randint(0, len(filled_surface_sites_np))
        query_point = filled_surface_sites[chosen_index][1]
        distances, neighbor_indices = kdtree.query(
            query_point, k=fragment_size, distance_upper_bound=nn_threshold, p=2
        )
        valid_pairs = [
            (d, i)
            for d, i in zip(distances, neighbor_indices)
            if i != chosen_index and d <= nn_threshold
        ]
        valid_pairs.sort()
        valid_indices = [i for d, i in valid_pairs]
        if len(valid_indices) > fragment_size - 1:
            valid_indices = valid_indices[: fragment_size - 1]
        if len(valid_indices) == fragment_size - 1:
            if mua:
                fragment_index = int(filled_surface_sites[chosen_index][0] == 2) + sum(
                    1 for index in valid_indices if filled_surface_sites[index][0] == 2
                )
            else:
                fragment_index = int(filled_surface_sites[chosen_index][0] == 1) + sum(
                    1 for index in valid_indices if filled_surface_sites[index][0] == 1
                )
            fragments[fragment_index] += 1
            total_fragments += 1
    if total_fragments == 0:
        raise ValueError("No valid fragments found.")
    fragment_probabilities = [fragment / total_fragments for fragment in fragments]
    ssr = sum(
        (fragment_probabilities[i] - target_fractions[i]) ** 2
        for i in range(len(target_fractions))
    )
    return fragment_probabilities, ssr, total_fragments

anchor_type = 5
num_configs = 5
replicates = 3
fragment_size = 4
threshold = 5
w = [12, 12, 10, 8, 12, 12, 12, 8]
num_ligands_1 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_2 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_1_len = [13, 13, 7, 13, 13, 13, 7, 13]
num_ligands_2_len = [7, 10, 15, 15, 7, 10, 15, 15]
monolayers = ["ddt_apt", "ddt_aht", "ht_aut", "ddt_aut", "ddt_mpa", "ddt_mha", "ht_mua", "ddt_mua"]
lig_1 = ["ddt", "ddt", "ht", "ddt", "ddt", "ddt", "ht", "ddt"]
lig_2 = ["apt", "aht", "aut", "aut", "mpa", "mha", "mua", "mua"]
cooh_fragment_prob_xticklabels = ["DDT/MPA", "HT/MUA", "DDT/MHA", "DDT/MUA"]
nh2_fragment_prob_xticklabels = ["DDT/APT", "HT/AUT", "DDT/AHT", "DDT/AUT"]
fragment_prob_xticks = [i + 1 for i in range(len(cooh_fragment_prob_xticklabels))]
conf_types = ["janus", "random", "striped"]
trials = 3

ddt_mpa_ssrs, ddt_mpa_all = [], []
ddt_mha_ssrs, ddt_mha_all = [], []
ddt_mua_ssrs, ddt_mua_all = [], []
ht_mua_ssrs, ht_mua_all = [], []
ddt_apt_ssrs, ddt_apt_all = [], []
ddt_aht_ssrs, ddt_aht_all = [], []
ddt_aut_ssrs, ddt_aut_all = [], []
ht_aut_ssrs, ht_aut_all = [], []

with open(f"../datafiles/fragment_data_{threshold}.txt", "w") as file:
    file.write("#\nmonolayer\tfragment avg\t fragment std\tssr avg\tssr std\n")
    file.close()

for i, monolayer in enumerate(monolayers):
    fragment_probabilities_all = []
    janus = []
    random = []
    striped = []
    ssrs_general = []
    for j, conf_type in enumerate(conf_types):
        for trial in range(trials):
            filepath = f"../{monolayer}/w{w[i]}/{conf_type}/t{trial+1}/trajectory.xyz"
            print(filepath)
            data = []
            with open(filepath, "r") as file:
                print(True)
                for line in file:
                    if line.strip():
                        columns = line.split()
                        if len(columns) == 4 and columns[0] != "1":
                            data.append(columns)
            total_lines = len(data)
            data = np.array(
                data[
                    (
                        total_lines
                        - num_configs
                        * (
                            num_ligands_1[i] * num_ligands_1_len[i]
                            + num_ligands_2[i] * num_ligands_2_len[i]
                        )
                    ) :
                ],
                dtype=float,
            )
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
                print(len(L1_sites_occupied), len(L2_sites_occupied))
                filled_sites = [[1, list(site)] for site in L1_sites_occupied] + [
                    [2, list(site)] for site in L2_sites_occupied
                ]
                for replicate in range(replicates):
                    print(f"config {config} / replicate {replicate}")
                    if lig_2[i] == "mua" or (lig_2[i] == "aut" and lig_1[i] == 'ht'):
                        target_fractions = get_binomial(
                            float(num_ligands_2[i])
                            / float(num_ligands_1[i] + num_ligands_2[i]),
                            fragment_size,
                        )
                        fragment_probabilities, ssr, total_fragments = get_maldi_ssr(
                            filled_sites,
                            100000,
                            threshold,
                            target_fractions,
                            fragment_size,
                            mua=True,
                        )
                    else:
                        target_fractions = get_binomial(
                            float(num_ligands_1[i])
                            / float(num_ligands_1[i] + num_ligands_2[i]),
                            fragment_size,
                        )
                        fragment_probabilities, ssr, total_fragments = get_maldi_ssr(
                            filled_sites,
                            100000,
                            threshold,
                            target_fractions,
                            fragment_size,
                            mua=False,
                        )
                    if conf_type == "janus":
                        janus.append(ssr)
                    elif conf_type == "random":
                        random.append(ssr)
                    else:
                        striped.append(ssr)
                    fragment_probabilities_all.append(fragment_probabilities)
                    if replicate == 0:
                        print(
                            f"{monolayer}/{conf_type}/Trial {trial + 1}",
                            fragment_probabilities,
                            ssr,
                            total_fragments,
                        )
    ssrs_general.append(janus)
    ssrs_general.append(random)
    ssrs_general.append(striped)
    print(f"{monolayer} JANUS avg: {np.mean(janus)} {np.std(janus)}")
    print(f"{monolayer} Random avg: {np.mean(random)} {np.std(random)}")
    print(f"{monolayer} Striped avg: {np.mean(striped)} {np.std(striped)}")

    if monolayer == 'ddt_apt':
        ddt_apt_ssrs.append(np.mean(janus))
        ddt_apt_ssrs.append(np.std(janus))
        ddt_apt_ssrs.append(np.mean(random))
        ddt_apt_ssrs.append(np.std(random))
        ddt_apt_ssrs.append(np.mean(striped))
        ddt_apt_ssrs.append(np.std(striped))
        ddt_apt_all.append(ssrs_general)
    elif monolayer == 'ddt_aht':
        ddt_aht_ssrs.append(np.mean(janus))
        ddt_aht_ssrs.append(np.std(janus))
        ddt_aht_ssrs.append(np.mean(random))
        ddt_aht_ssrs.append(np.std(random))
        ddt_aht_ssrs.append(np.mean(striped))
        ddt_aht_ssrs.append(np.std(striped))
        ddt_aht_all.append(ssrs_general)
    elif monolayer == 'ddt_aut':
        ddt_aut_ssrs.append(np.mean(janus))
        ddt_aut_ssrs.append(np.std(janus))
        ddt_aut_ssrs.append(np.mean(random))
        ddt_aut_ssrs.append(np.std(random))
        ddt_aut_ssrs.append(np.mean(striped))
        ddt_aut_ssrs.append(np.std(striped))
        ddt_aut_all.append(ssrs_general)
    elif monolayer == 'ht_aut':
        ht_aut_ssrs.append(np.mean(janus))
        ht_aut_ssrs.append(np.std(janus))
        ht_aut_ssrs.append(np.mean(random))
        ht_aut_ssrs.append(np.std(random))
        ht_aut_ssrs.append(np.mean(striped))
        ht_aut_ssrs.append(np.std(striped))
        ht_aut_all.append(ssrs_general)
    elif monolayer == 'ddt_mpa':
        ddt_mpa_ssrs.append(np.mean(janus))
        ddt_mpa_ssrs.append(np.std(janus))
        ddt_mpa_ssrs.append(np.mean(random))
        ddt_mpa_ssrs.append(np.std(random))
        ddt_mpa_ssrs.append(np.mean(striped))
        ddt_mpa_ssrs.append(np.std(striped))
        ddt_mpa_all.append(ssrs_general)
    elif monolayer == 'ddt_mha':
        ddt_mha_ssrs.append(np.mean(janus))
        ddt_mha_ssrs.append(np.std(janus))
        ddt_mha_ssrs.append(np.mean(random))
        ddt_mha_ssrs.append(np.std(random))
        ddt_mha_ssrs.append(np.mean(striped))
        ddt_mha_ssrs.append(np.std(striped))
        ddt_mha_all.append(ssrs_general)
    elif monolayer == 'ddt_mua':
        ddt_mua_ssrs.append(np.mean(janus))
        ddt_mua_ssrs.append(np.std(janus))
        ddt_mua_ssrs.append(np.mean(random))
        ddt_mua_ssrs.append(np.std(random))
        ddt_mua_ssrs.append(np.mean(striped))
        ddt_mua_ssrs.append(np.std(striped))
        ddt_mua_all.append(ssrs_general)
    elif monolayer == 'ht_mua':
        ht_mua_ssrs.append(np.mean(janus))
        ht_mua_ssrs.append(np.std(janus))
        ht_mua_ssrs.append(np.mean(random))
        ht_mua_ssrs.append(np.std(random))
        ht_mua_ssrs.append(np.mean(striped))
        ht_mua_ssrs.append(np.std(striped))
        ht_mua_all.append(ssrs_general)

    fragment_probabilities_all = np.array(fragment_probabilities_all)
    fps_avg = np.mean(fragment_probabilities_all, axis=0)
    fps_std = np.std(fragment_probabilities_all, axis=0)
    with open(f"../datafiles/fragment_data_{threshold}.txt", "a") as file:
        file.write(
            f"{monolayer}\t{fps_avg}\t{fps_std}\t{np.mean(ssrs_general)}\t{np.std(ssrs_general)}\n"
        )
        file.close()

x = [0.6, 0.8, 1.0, 1.2, 1.6, 1.8, 2.0, 2.2, 2.6, 2.8, 3.0, 3.2, 3.6, 3.8, 4.0, 4.2]
fig, axs = plt.subplots(2, 1, figsize=(6,6))

axs[0].errorbar(
    x=[x[1], x[5], x[9], x[13]],
    y=[ddt_mpa_ssrs[0], ht_mua_ssrs[0], ddt_mha_ssrs[0], ddt_mua_ssrs[0]],
    yerr=[ddt_mpa_ssrs[1], ht_mua_ssrs[1], ddt_mha_ssrs[1], ddt_mua_ssrs[1]],
    capsize=5,
    label="Janus",
    fmt="o",
    marker="o",
)
axs[0].errorbar(
    x=[x[2], x[6], x[10], x[14]],
    y=[ddt_mpa_ssrs[2], ht_mua_ssrs[2], ddt_mha_ssrs[2], ddt_mua_ssrs[2]],
    yerr=[ddt_mpa_ssrs[3], ht_mua_ssrs[3], ddt_mha_ssrs[3], ddt_mua_ssrs[3]],
    capsize=5,
    label="Random",
    fmt="o",
    marker="o",
)
axs[0].errorbar(
    x=[x[3], x[7], x[11], x[15]],
    y=[ddt_mpa_ssrs[4], ht_mua_ssrs[4], ddt_mha_ssrs[4], ddt_mua_ssrs[4]],
    yerr=[ddt_mpa_ssrs[5], ht_mua_ssrs[5], ddt_mha_ssrs[5], ddt_mua_ssrs[5]],
    capsize=5,
    label="Striped",
    fmt="o",
    marker="o",
)
axs[0].errorbar(
    x=[x[0], x[4], x[8], x[12]],
    y=[np.mean(ddt_mpa_all), np.mean(ht_mua_all), np.mean(ddt_mha_all), np.mean(ddt_mua_all)],
    yerr=[np.std(ddt_mpa_all), np.std(ht_mua_all), np.std(ddt_mha_all), np.std(ddt_mua_all)],
    capsize=5,
    label="Monolayer Average",
    fmt="o",
    marker="o",
    color="k",
)
axs[0].errorbar(
    x=[x[0], x[4], x[8], x[12]],
    y=[0.26075, 0.1756, 0.2075, 0.0225],
    yerr=[0.024474, 0.016073, 0.025626, 0.012834],
    capsize=5,
    label="Experimental Average",
    fmt="o",
    marker="s",
    color="r",
)

axs[1].errorbar(
    x=[x[1], x[5], x[9], x[13]],
    y=[ddt_apt_ssrs[0], ht_aut_ssrs[0], ddt_aht_ssrs[0], ddt_aut_ssrs[0]],
    yerr=[ddt_apt_ssrs[1], ht_aut_ssrs[1], ddt_aht_ssrs[1], ddt_aut_ssrs[1]],
    capsize=5,
    label="Janus",
    fmt="o",
    marker="o",
)
axs[1].errorbar(
    x=[x[2], x[6], x[10], x[14]],
    y=[ddt_apt_ssrs[2], ht_aut_ssrs[2], ddt_aht_ssrs[2], ddt_aut_ssrs[2]],
    yerr=[ddt_apt_ssrs[3], ht_aut_ssrs[3], ddt_aht_ssrs[3], ddt_aut_ssrs[3]],
    capsize=5,
    label="Random",
    fmt="o",
    marker="o",
)
axs[1].errorbar(
    x=[x[3], x[7], x[11], x[15]],
    y=[ddt_apt_ssrs[4], ht_aut_ssrs[4], ddt_aht_ssrs[4], ddt_aut_ssrs[4]],
    yerr=[ddt_apt_ssrs[5], ht_aut_ssrs[5], ddt_aht_ssrs[5], ddt_aut_ssrs[5]],
    capsize=5,
    label="Striped",
    fmt="o",
    marker="o",
)
axs[1].errorbar(
    x=[x[0], x[4], x[8], x[12]],
    y=[np.mean(ddt_apt_all), np.mean(ht_aut_all), np.mean(ddt_aht_all), np.mean(ddt_aut_all)],
    yerr=[np.std(ddt_apt_all), np.std(ht_aut_all), np.std(ddt_aht_all), np.std(ddt_aut_all)],
    capsize=5,
    label="Monolayer Average",
    fmt="o",
    marker="o",
    color="k",
)
axs[1].errorbar(
    x=[x[0], x[4], x[8], x[12]],
    y=[0.2764,0.2224,0.1696,0.0063],
    yerr=[0.0542,0.0588,0.0257,0.0046],
    capsize=5,
    label="Experimental Average",
    fmt="o",
    marker="s",
    color="r",
)
axs[1].set_xlabel("Monolayers")
axs[0].set_ylabel("SSR")
axs[1].set_ylabel("SSR")
axs[0].set_ylim(bottom=0, top=0.4)
axs[1].set_ylim(bottom=0, top=0.4)
axs[0].set_xticks(fragment_prob_xticks, cooh_fragment_prob_xticklabels)
axs[1].set_xticks(fragment_prob_xticks, nh2_fragment_prob_xticklabels)
axs[0].legend(loc='upper right', frameon=False, fontsize='medium')
plt.tight_layout()
plt.savefig("../figures/figure_S7.png", format="png", dpi=600)
plt.close()
print("SSR and fragment analysis complete")
