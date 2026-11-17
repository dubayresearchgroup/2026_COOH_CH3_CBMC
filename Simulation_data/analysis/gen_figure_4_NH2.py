import matplotlib.pyplot as plt

fragment_prob_xticklabels = ['DDT/APT', 'HT/AUT', 'DDT/AHT', 'DDT/AUT']
fragment_prob_xticks = []
for i in range(len(fragment_prob_xticklabels)):
    fragment_prob_xticks.append(i+1)
print(fragment_prob_xticks)

experimental_fragment_distributions = [
    [0.332717851, 0.152744931, 0.054112875, 0.134297979, 0.326126364],
    [0.267504783, 0.151784113, 0.146510299, 0.141263203, 0.292937603],
    [0.319642334, 0.092958992, 0.14131806, 0.135768248, 0.310312366],
    [0.071949187, 0.209451227,0.413777417, 0.267930939, 0.03689123],
]
experimental_fragment_distributions_std = [
    [0.058700564, 0.070424496, 0.012818082, 0.03681686, 0.026279643],
    [0.03271045, 0.017992058, 0.014070178, 0.026176981, 0.009936074],
    [0.03523017, 0.020912714, 0.030293172, 0.029884758, 0.036133552],
    [0.026724222, 0.018465257, 0.03654443, 0.028636915, 0.013821839],
]
theoretical_fragment_distributions = [
    [0.32288997, 0.11520943, 0.10276548, 0.13385496, 0.32528016],
    [0.22848688, 0.17632872, 0.17236938, 0.1811512 , 0.24166384],
    [0.25317989, 0.16482171, 0.15064748, 0.16442821, 0.26692271],
    [0.06569634, 0.2430945 , 0.37053486, 0.26042691, 0.06024739],
]
theoretical_fragment_distributions_std = [
    [0.03490031, 0.02658378, 0.02795088, 0.02405628, 0.0354346 ],
    [0.02612598, 0.02022127, 0.02441045, 0.01717121, 0.01957108],
    [0.01819993, 0.01845212, 0.01655233, 0.01019449, 0.01170587],
    [0.01613198, 0.01568098, 0.03615816, 0.01859545, 0.01355308],
]
sim_ssr = [
   0.2471664461409512,
   0.1442460427542736,
   0.11324332387517884,
   0.0025346582334198017,
]
sim_ssr_std = [
    0.06075042814181896,
    0.017244613388992733,
    0.02651359994556536,
    0.0018469307690306385,
]

exp_ssr = [0.2764,0.2224,0.1696,0.0063] #DDT/APT, HT/AUT, DDT/AHT, DDT/AUT
exp_ssr_std = [0.0542,0.0588,0.0257,0.0046] #DDT/APT, HT/AUT, DDT/AHT, DDT/AUT

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
plot_titles = ["DDT/APT", "DDT/AHT", "HT/AUT", "DDT/AUT"]
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
        axs[1, 0].set_xlabel("# AUT", fontsize=16)
        axs[1, 1].set_xlabel("# AUT", fontsize=16)
        axs[i, j].set_xlim(-1, 5)
        axs[i, j].set_ylim(0, 0.5)
        plot_index += 1
fig.supylabel("Fragment Probability", fontsize=16)
plt.subplots_adjust(wspace=0.05, hspace=0.25)
plt.tight_layout()
plt.savefig("../figures/figure_4b.png", format="png", dpi=400)

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
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig("../figures/figure_4d.png", format="png", dpi=400)
