import matplotlib.pyplot as plt

fragment_prob_xticklabels = ['DDT/MPA', 'HT/MUA', 'DDT/MHA', 'DDT/MUA']
fragment_prob_xticks = []
for i in range(len(fragment_prob_xticklabels)):
    fragment_prob_xticks.append(i+1)
print(fragment_prob_xticks)

experimental_fragment_distributions = [
    [0.357684574, 0.109651206, 0.103252844, 0.116550753,0.312860623],
    [0.30450756, 0.108998382, 0.118900598, 0.166771936, 0.300821524],
    [0.282248079, 0.13660489, 0.141126003, 0.153032514, 0.286988515],
    [0.151770595, 0.234702584, 0.290907084, 0.195149734, 0.127470003],
]
experimental_fragment_distributions_std = [
    [0.015814535, 0.019751575, 0.016671764,0.01028718, 0.01999021],
    [0.015577321,0.020119983, 0.013999559, 0.009777529, 0.036702554],
    [0.017119533, 0.011429576, 0.014133992, 0.009764157, 0.014903076],
    [0.035870309, 0.01119446, 0.026286626, 0.022650962, 0.017968601],
]
theoretical_fragment_distributions = [
    [0.27184473, 0.14567885, 0.14135559, 0.16795333, 0.2731675 ],
    [0.19220206, 0.21060975, 0.19178349, 0.19521074, 0.21019396],
    [0.23394166, 0.17981008, 0.16051354, 0.17415042, 0.2515843 ],
    [0.06195761, 0.24840842, 0.36833704, 0.25024191, 0.07105503],
]
theoretical_fragment_distributions_std = [
    [0.04018639, 0.02744366, 0.02866186, 0.03089606, 0.03742432],
    [0.02326736, 0.01829872, 0.03510068, 0.02666496, 0.01779426],
    [0.02777769, 0.03387226, 0.02338232, 0.0160062 , 0.02428756],
    [0.01364467, 0.01673301, 0.02738086, 0.02471669, 0.01535253],
]
sim_ssr = [
    0.16595502527160158,
    0.12514108602000149,
    0.07989379728419234,
    0.002182966701426807,
]
sim_ssr_std = [
    0.056612201095394464,
    0.03138490782880945,
    0.020961892486618974,
    0.0014100595718975885,
]
exp_ssr = [0.2608,0.1756, 0.2060,  0.0225] #DDT/MPA, HT/MUA, DDT/MHA, DDT/MUA
exp_ssr_std = [0.0245,0.0161, 0.0273, 0.0128] #DDT/MPA, HT/MUA, DDT/MHA, DDT/MUA
for i in range(len(theoretical_fragment_distributions)):
    ssr_exp_sim = 0
    for frag in range(len(theoretical_fragment_distributions[i])):
        ssr_exp_sim += (
            theoretical_fragment_distributions[i][frag]
            - experimental_fragment_distributions[i][frag]
        ) ** 2
    print(ssr_exp_sim)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
x = [0, 1, 2, 3, 4]
plot_index = 0
plot_titles = ["DDT/MPA", "DDT/MHA", "HT/MUA", "DDT/MUA"]
for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            axs[i, j].bar(
                [x - 0.25 for x in x],
                experimental_fragment_distributions[plot_index],
                yerr=experimental_fragment_distributions_std[plot_index],
                capsize=5,
                label="Experiment",
                color="k",
                ecolor="orange",
                width=0.5,
                alpha=0.8,
            )
            axs[i, j].bar(
                [x + 0.25 for x in x],
                theoretical_fragment_distributions[plot_index],
                yerr=theoretical_fragment_distributions_std[plot_index],
                capsize=5,
                label="Simulation",
                color="k",
                ecolor="orange",
                width=0.5,
                alpha=0.4,
            )
        else:
            axs[i, j].bar(
                [x - 0.25 for x in x],
                experimental_fragment_distributions[plot_index],
                yerr=experimental_fragment_distributions_std[plot_index],
                capsize=5,
                color="k",
                ecolor="orange",
                width=0.5,
                alpha=0.8,
            )
            axs[i, j].bar(
                [x + 0.25 for x in x],
                theoretical_fragment_distributions[plot_index],
                yerr=theoretical_fragment_distributions_std[plot_index],
                capsize=5,
                color="k",
                ecolor="orange",
                width=0.5,
                alpha=0.4,
            )
        #axs[i, j].set_title(plot_titles[plot_index], fontsize=16)
        axs[i,j].tick_params(axis='both', which='major', labelsize=16)
        if plot_index < 2:
            axs[i, j].set_xticks([], [])
        axs[0, 1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["", "", "", "", ""])
        axs[1, 1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["", "", "", "", ""])
        axs[i, j].set_xlabel("# DDT", fontsize=16)
        axs[1, 0].set_xticks([0, 1, 2, 3, 4])
        axs[1, 1].set_xticks([0, 1, 2, 3, 4])
        axs[0, 1].set_xticks([0, 1, 2, 3, 4], ["", "", "", "", ""])
        axs[0, 0].set_xticks([0, 1, 2, 3, 4], ["", "", "", "", ""])
        axs[1, 0].set_xlabel("# MUA", fontsize=16)
        axs[1, 1].set_xlabel("# MUA", fontsize=16)
        axs[i, j].set_xlim(-1, 5)
        axs[i, j].set_ylim(0, 0.5)
        plot_index += 1
fig.supylabel("Fragment Probability", fontsize=16)
plt.subplots_adjust(wspace=0.05, hspace=0.25)
plt.tight_layout()
plt.savefig("../figures/figure_4a.png", format="png", dpi=400)

x = [0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1]
plt.figure(figsize=(8, 6))
plt.errorbar(
    x=[x[0], x[2], x[4], x[6]],
    y=[exp_ssr[0], exp_ssr[1], exp_ssr[2], exp_ssr[3]],
    yerr=[exp_ssr_std[0], exp_ssr_std[1], exp_ssr_std[2], exp_ssr_std[3]],
    capsize=10,
    label="Experiment",
    fmt="o",
    marker="s",
    color="k",
    alpha=0.8,
    markersize=10,
)
plt.errorbar(
    x=[x[1], x[3], x[5], x[7]],
    y=[sim_ssr[0], sim_ssr[1], sim_ssr[2], sim_ssr[3]],
    yerr=[sim_ssr_std[0], sim_ssr_std[1], sim_ssr_std[2], sim_ssr_std[3]],
    capsize=10,
    label="Simulation",
    fmt="o",
    marker="o",
    color="k",
    alpha=0.4,
    markersize=10,
)
plt.xlabel("Monolayers", fontsize=16)
plt.ylabel("SSR", fontsize=16)
# plt.axhline(y=0.1, color='k', alpha=0.4, linestyle='--')
plt.xticks(fragment_prob_xticks, fragment_prob_xticklabels)
plt.ylim(top=0.3812)
plt.yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
plt.legend(frameon=False, fontsize="x-large")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig("../figures/figure_4c.png", format="png", dpi=400)
