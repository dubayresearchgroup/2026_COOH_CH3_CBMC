import numpy as np
import matplotlib.pyplot as plt


def get_atoms_by_type(mols, len_typ, rdf_typ):
    if rdf_typ == 'head':
        mol_index = len_typ - 1
    elif rdf_typ == 'sulfur':
        mol_index = 0
    atoms = [mols[mol].getAtomByMolIndex(mol_index) for mol in mols if len(mols[mol]) == len_typ]
    return atoms

def get_distance(atom1, atom2):
    x1, y1, z1 = atom1
    x2, y2, z2 = atom2

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    return np.sqrt(dx**2 + dy**2 + dz**2)

def get_rdf_oxygens(atoms, numbins, num_mols, cutoff):
    # Extract the coordinates of oxygen atoms
    oxygens = [atom[1] for atom in atoms if atom[0] == 'oh']
    oxygens = np.array(oxygens)
    num_oxygens = len(oxygens)
    dists = []

    #print('NUM ALCOHOL LIGANDS: ', num_oxygens)
    # Calculate distances between all pairs of oxygen atoms using Euclidean distance
    for i in range(num_oxygens):
        for j in range(i + 1, num_oxygens):
            distance = get_distance(oxygens[i], oxygens[j])
            if distance < cutoff:
                dists.append(distance)
    [counts, bins] = np.histogram(dists, bins=numbins, range=(0.0, cutoff))
    # Calculate histogram and normalize for RDF
    dr = bins[1] - bins[0]
    area = 4*np.pi*(np.mean(dists)/2)**2
    dareas = 2 * np.pi * bins[1:] * dr
    norm_factor =  (len(dists)/2)/area  # Update normalization factor for spherical polar coordinates
    counts[0] = 0
    return ((counts/dareas)/norm_factor,bins)



def get_rdf_methyls(atoms, numbins, num_mols, cutoff):
    oxygens = [atom[1] for atom in atoms if atom[0] == 'ch3']
    oxygens = np.array(oxygens)
    num_oxygens = int(num_mols / 2)
    dists = np.zeros(num_oxygens ** 2)
    d = []
    #print('NUM METHYL LIGANDS: ', len(oxygens))
    for i, oxygen in enumerate(oxygens):
        for j, oxygen2 in enumerate(oxygens):
            if i != j:
                distance = np.linalg.norm(oxygen - oxygen2)
                d.append(np.linalg.norm(oxygen))
                dists[i * num_oxygens + j] = distance

    [counts, bins] = np.histogram(dists, bins=numbins, range=(0.0, cutoff))
    dr = bins[1] - bins[0]
    area = 4*np.pi*(np.mean(dists)/2)**2
    dareas = 2 * np.pi * bins[1:] * dr
    norm_factor =  (len(dists)/2)/area  # Update normalization factor for spherical polar coordinates
    counts[0] = 0
    return ((counts/dareas)/norm_factor,bins)

def get_rdf_inter(atoms, numbins, num_mols, cutoff):
    num_oxygens = int(num_mols / 2)
    num_methyls = int(num_mols / 2)
    oxygens = [atom[1] for atom in atoms if atom[0] == 'oh']
    methyls = [atom[1] for atom in atoms if atom[0] == 'ch3']
    oxygens = np.array(oxygens)
    methyls = np.array(methyls)
    dists = np.zeros(num_oxygens * num_methyls)
    distances = []
    d = []
    #print('NUM OXYGEN AND ALKANE LIGANDS: ', len(oxygens) + len(methyls))
    for i, oxygen in enumerate(oxygens):
        for j, methyl in enumerate(methyls):
            if j != i:
                distance = np.linalg.norm(oxygen - methyl)
                distances.append(distance)
                d.append(np.linalg.norm(oxygen))
                d.append(np.linalg.norm(methyl))
                dists[i * num_oxygens + j] = distance

    [counts, bins] = np.histogram(dists, bins=numbins, range=(0.0, cutoff))
    dr = bins[1] - bins[0]
    area = 4*np.pi*(np.mean(dists)/2)**2
    dareas = 2 * np.pi * bins[1:] * dr
    norm_factor =  (len(dists)/2)/area  # Update normalization factor for spherical polar coordinates
    counts[0] = 0
    return ((counts/dareas)/norm_factor,bins)

def get_rdf_anchors_methyl(atoms, numbins, num_mols, cutoff):
    oxygens = [atom[1] for atom in atoms if atom[0] == 'ch3']
    oxygens = np.array(oxygens)
    num_oxygens = int(num_mols / 2)
    dists = np.zeros(num_oxygens ** 2)
    d = []
    #print('NUM METHYL LIGANDS: ', len(oxygens))
    for i, oxygen in enumerate(oxygens):
        for j, oxygen2 in enumerate(oxygens):
            if i != j:
                distance = np.linalg.norm(oxygen - oxygen2)
                d.append(np.linalg.norm(oxygen))
                dists[i * num_oxygens + j] = distance

    [counts, bins] = np.histogram(dists, bins=numbins, range=(0.0, cutoff))
    dr = bins[1] - bins[0]
    area = 4*np.pi*(np.mean(dists)/2)**2
    dareas = 2 * np.pi * bins[1:] * dr
    norm_factor =  (len(dists)/2)/area  # Update normalization factor for spherical polar coordinates
    counts[0] = 0
    return ((counts/dareas)/norm_factor,bins)

def get_rdf_anchors_oxygen(atoms, numbins, num_mols, cutoff):
    oxygens = [atom[1] for atom in atoms if atom[0] == 'oh']
    oxygens = np.array(oxygens)
    num_oxygens = int(num_mols / 2)
    dists = np.zeros(num_oxygens ** 2)
    d = []
    #print('NUM ALCOHOL LIGANDS: ', len(oxygens))
    for i, oxygen in enumerate(oxygens):
        for j, oxygen2 in enumerate(oxygens):
            if i != j:
                distance = np.linalg.norm(oxygen - oxygen2)
                d.append(np.linalg.norm(oxygen))
                dists[i * num_oxygens + j] = distance

    [counts, bins] = np.histogram(dists, bins=numbins, range=(0.0, cutoff))
    dr = bins[1] - bins[0]
    area = 4*np.pi*(np.mean(dists)/2)**2
    dareas = 2 * np.pi * bins[1:] * dr
    norm_factor =  (len(dists)/2)/area  # Update normalization factor for spherical polar coordinates
    counts[0] = 0
    return ((counts/dareas)/norm_factor,bins)

def get_rdf_anchors(atoms, numbins, num_mols, cutoff):
    num_oxygens = int(num_mols / 2)
    num_methyls = int(num_mols / 2)
    oxygens = [atom[1] for atom in atoms if atom[0] == 'oh']
    methyls = [atom[1] for atom in atoms if atom[0] == 'ch3']
    oxygens = np.array(oxygens)
    methyls = np.array(methyls)
    dists = np.zeros(num_oxygens * num_methyls)
    distances = []
    d = []
    print('NUM OXYGEN AND ALKANE LIGANDS: ', len(oxygens) + len(methyls))
    for i, oxygen in enumerate(oxygens):
        for j, methyl in enumerate(methyls):
            if j != i:
                distance = np.linalg.norm(oxygen - methyl)
                distances.append(distance)
                d.append(np.linalg.norm(oxygen))
                d.append(np.linalg.norm(methyl))
                dists[i * num_oxygens + j] = distance

    [counts, bins] = np.histogram(dists, bins=numbins, range=(0.0, cutoff))
    dr = bins[1] - bins[0]
    area = 4*np.pi*(np.mean(dists)/2)**2
    dareas = 2 * np.pi * bins[1:] * dr
    norm_factor =  (len(dists)/2)/area  # Update normalization factor for spherical polar coordinates
    counts[0] = 0
    return ((counts/dareas)/norm_factor, bins)

def read_trajectory_file(filepath, num_configs, num_ligands_1, num_ligands_2, num_ligands_1_len, num_ligands_2_len, anchor_type):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():
                columns = line.split()
                if len(columns) == 4 and columns[0] != '1':
                    data.append(columns)

    total_lines = len(data)
    data = np.array(data[(total_lines - num_configs*(num_ligands_1*num_ligands_1_len + num_ligands_2*num_ligands_2_len)):], dtype=float)
    start_line = 0  # Update based on your logic
    end_line = start_line + (num_ligands_1 * num_ligands_1_len + num_ligands_2 * num_ligands_2_len)
    lig_sites = data[start_line:end_line, 1:4]
    type_sites = data[start_line:end_line, 0]

    oh_sites, ch3_sites, ch3_s_sites, oh_s_sites = [], [], [], []
    count_s = 0
    for k in range(len(type_sites)):
        if type_sites[k] == anchor_type:
            count_s += 1
            if count_s <= num_ligands_1:
                ch3_s_sites.append(lig_sites[k])
            elif count_s > num_ligands_1:
                oh_s_sites.append(lig_sites[k])
        if type_sites[k] == 4:
            ch3_sites.append(lig_sites[k])
        if type_sites[k] == 6:
            oh_sites.append(lig_sites[k])

    oh_sites = np.array(oh_sites)
    ch3_sites = np.array(ch3_sites)
    oh_s_sites = np.array(oh_s_sites)
    ch3_s_sites = np.array(ch3_s_sites)

    return oh_sites, ch3_sites, oh_s_sites, ch3_s_sites



def plot_rdf(bin_edges, rdf_data_ohs, rdf_data_ch3s, rdf_data_both, morph, sizes, anchor_type):
    plt.figure(figsize=(15, 5))

    for k, size in enumerate(sizes):
        plt.plot(bin_edges[:-1], rdf_data_ohs[k], label=f'Size {size}', alpha=0.2)

    plt.xlabel('r (angstroms)')
    plt.ylabel('g(r)')
    plt.xlim(0.0, 15.0)
    plt.title(f'OH-OH RDFs - Morph: {morph}, Anchor Type: {anchor_type}')
    plt.legend()
    plt.savefig(f'OH_OH_RDF_{morph}_{anchor_type}.png', format='png', dpi=300)
    plt.close()

    plt.figure(figsize=(15, 5))

    for k, size in enumerate(sizes):
        plt.plot(bin_edges[:-1], rdf_data_ch3s[k], label=f'Size {size}', alpha=0.2)

    plt.xlabel('r (angstroms)')
    plt.ylabel('g(r)')
    plt.xlim(0.0, 15.0)
    plt.title(f'CH3-CH3 RDFs - Morph: {morph}, Anchor Type: {anchor_type}')
    plt.legend()
    plt.savefig(f'CH3_CH3_RDF_{morph}_{anchor_type}.png', format='png', dpi=300)
    plt.close()

    plt.figure(figsize=(15, 5))

    for k, size in enumerate(sizes):
        plt.plot(bin_edges[:-1], rdf_data_both[k], label=f'Size {size}', alpha=0.2)

    plt.xlabel('r (angstroms)')
    plt.ylabel('g(r)')
    plt.xlim(0.0, 15.0)
    plt.title(f'OH-CH3 RDFs - Morph: {morph}, Anchor Type: {anchor_type}')
    plt.legend()
    plt.savefig(f'OH_CH3_RDF_{morph}_{anchor_type}.png', format='png', dpi=300)
    plt.close()

def get_rdf(data_anchor, data_head, nbins, num_mols, r_max):
    rdf_data_ohs, bin_edges = get_rdf_oxygens(data_head, nbins, num_mols, r_max)
    rdf_data_ch3s, bin_edges = get_rdf_methyls(data_head, nbins, num_mols, r_max)
    rdf_data_both, bin_edges = get_rdf_inter(data_head, nbins, num_mols, r_max)
    rdf_anchor_data_ohs, bin_edges = get_rdf_anchors_oxygen(data_anchor, nbins, num_mols, r_max)
    rdf_anchor_data_ch3s, bin_edges = get_rdf_anchors_methyl(data_anchor, nbins, num_mols, r_max)
    rdf_anchor_data_both, bin_edges = get_rdf_anchors(data_anchor, nbins, num_mols, r_max)

    return rdf_data_ohs, rdf_data_ch3s, rdf_data_both, rdf_anchor_data_ohs, rdf_anchor_data_ch3s, rdf_anchor_data_both, bin_edges
