import matplotlib.pyplot as plt
import numpy as np

def get_conformer(data, num_ch3s, num_coohs, num_ligands_2_len, anchor_type=5):
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
            oxygens.append(data[i + (num_ligands_2_len - 2)][1:4])
            hydrogens.append(data[i + (num_ligands_2_len - 1)][1:4])
    if len(hydrogens) == num_coohs:
        for i in range(len(hydrogens)):
            oh_dist = np.linalg.norm(np.array(oxygens[i]) - np.array(hydrogens[i]))
            distances.append(oh_dist)
            if oh_dist >= 2.7:
                anti += 1
            else:
                syn += 1
    if (anti + syn) != 0:
        return float(anti / (anti + syn)), distances
    else:
        return 0, None

anchor_type = 5
num_configs = 5
w = [12, 12, 10, 8, 12, 12, 12, 8]
num_ligands_1 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_2 = [265, 173, 260, 250, 175, 214, 174, 208]
num_ligands_1_len = [13, 13, 7, 13, 13, 13, 7, 13]
num_ligands_2_len = [7, 10, 15, 15, 7, 10, 15, 15]
monolayers = ["ddt_apt", "ddt_aht", "ht_aut", "ddt_aut", "ddt_mpa", "ddt_mha", "ht_mua", "ddt_mua"]
conf_types = ["janus", "random", "striped"]
trials = 3
anti_ratios = []
distances = []
for i, monolayer in enumerate(monolayers):
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
                if i > 3:  # COOH-containing monolayers only
                    anti_conformer_ratio, x = get_conformer(
                        data[start_line:end_line],
                        num_ligands_1[i],
                        num_ligands_2[i],
                        num_ligands_2_len[i]
                    )
                    print(anti_conformer_ratio)
                    anti_ratios.append(anti_conformer_ratio)
                    distances.extend(x)

plt.hist(distances, bins=50, color='k')
plt.axvline(x=2.7, color='r', label='Set Threshold Distance')
plt.xlabel('Distances (A)')
plt.ylabel('Counts')
plt.xlim(2,3.2)
plt.tight_layout()
plt.savefig('../figures/figure_S11.png', format='png', dpi=600)
plt.close()
print("Length of anti_ratios list: ", len(anti_ratios))
num_to_consider_per_monolayer = num_configs * trials * len(conf_types)
ddt_mpa_anti_ratio_avg = 100*np.mean(anti_ratios[0:(num_to_consider_per_monolayer-1)])
ddt_mpa_anti_ratio_std = 100*np.std(anti_ratios[0:(num_to_consider_per_monolayer-1)])
ddt_mha_anti_ratio_avg = 100*np.mean(anti_ratios[num_to_consider_per_monolayer:(2*num_to_consider_per_monolayer-1)])
ddt_mha_anti_ratio_std = 100*np.std(anti_ratios[num_to_consider_per_monolayer:(2*num_to_consider_per_monolayer-1)])
ht_mua_anti_ratio_avg = 100*np.mean(anti_ratios[(3*num_to_consider_per_monolayer):])
ht_mua_anti_ratio_std = 100*np.std(anti_ratios[(3*num_to_consider_per_monolayer):])
ddt_mua_anti_ratio_avg = 100*np.mean(anti_ratios[(2*num_to_consider_per_monolayer):(3*num_to_consider_per_monolayer-1)])
ddt_mua_anti_ratio_std = 100*np.std(anti_ratios[(2*num_to_consider_per_monolayer):(3*num_to_consider_per_monolayer-1)])
plt.figure()
plt.errorbar(1, ddt_mpa_anti_ratio_avg, yerr=ddt_mpa_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
plt.errorbar(3, ht_mua_anti_ratio_avg, yerr=ht_mua_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
plt.errorbar(2, ddt_mha_anti_ratio_avg, yerr=ddt_mha_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
plt.errorbar(4, ddt_mua_anti_ratio_avg, yerr=ddt_mua_anti_ratio_std, marker='o', capsize=4, color='#232D4B')
plt.fill_between([0,1,2,3,4,5], y1=3, y2=7, color='k', alpha=0.25)
plt.xticks([1,2,3,4], ['DDT/MPA', 'DDT/MHA', 'HT/MUA', 'DDT/MUA'])
plt.xlabel('Monolayers')
plt.ylabel('Anti COOH Conformer % Occurrence')
plt.xlim(0,5)
plt.ylim(0,10)
plt.tight_layout()
plt.savefig('../figures/figure_S12.png', format='png', dpi=400)
print('Conformer plot complete')
