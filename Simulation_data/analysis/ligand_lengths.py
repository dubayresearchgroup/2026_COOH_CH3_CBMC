import numpy as np
from scipy import stats

def get_lengths(data, num_type1, num_type2, num_ligands_1_len, num_ligands_2_len, exclude_types, anchor_type=5):
    """
    data: array of atoms [atom_type, x, y, z, ...]
    Returns: (average_length_mismatch, stdev_length_mismatch)
    """
    ligand_1_coords = []
    ligand_2_coords = []
    count = 0

    if exclude_types is None:
            exclude_types = set()
    elif isinstance(exclude_types, (int, np.integer)):
        exclude_types = {int(exclude_types)}
    else:
        exclude_types = set(int(x) for x in exclude_types)

    for i, atom in enumerate(data):
        if int(atom[0]) == anchor_type:
            if count < num_type1:
                end_idx = i + num_ligands_1_len
                if end_idx > len(data):
                    raise ValueError(f"Not enough atoms for ligand 1 at index {i}")
                ligand_atoms = [atom for atom in data[i:end_idx] if int(atom[0]) not in exclude_types]
                if len(ligand_atoms) == 0:
                    raise ValueError(f"All atoms excluded for ligand at index {i}")
                coords = np.array([[float(x) for x in atom[1:4]] for atom in ligand_atoms])
                ligand_1_coords.append(coords)
                count += 1
            elif len(ligand_2_coords) < num_type2:
                end_idx = i + num_ligands_2_len
                if end_idx > len(data):
                    raise ValueError(f"Not enough atoms for ligand 2 at index {i}")
                ligand_atoms = [atom for atom in data[i:end_idx] if int(atom[0]) not in exclude_types]
                if len(ligand_atoms) == 0:
                    raise ValueError(f"All atoms excluded for ligand at index {i}")
                coords = np.array([[float(x) for x in atom[1:4]] for atom in ligand_atoms])
                ligand_2_coords.append(coords)
    ds1 = []
    for ligand in ligand_1_coords:
        anchor = ligand[0]
        dists = [np.linalg.norm(anchor - atom) for atom in ligand[1:]]
        ds1.append(max(dists) if dists else 0.0)
    ds2 = []
    for ligand in ligand_2_coords:
        anchor = ligand[0]
        dists = [np.linalg.norm(anchor - atom) for atom in ligand[1:]]
        ds2.append(max(dists) if dists else 0.0)


    return ds1, ds2

if __name__ == "__main__":
    anchor_type = 5
    num_configs = 5
    replicates = 3
    w = [12, 12, 10, 8, 12, 12, 12, 8]
    num_ligands_1 = [265, 173, 260, 250, 175, 214, 174, 208]
    num_ligands_2 = [265, 173, 260, 250, 175, 214, 174, 208]
    num_ligands_1_len = [13, 13, 7, 13, 13, 13, 7, 13]
    num_ligands_2_len = [7, 10, 15, 15, 7, 10, 15, 15]
    monolayers = ["ddt_apt", "ddt_aht", "ht_aut", "ddt_aut", "ddt_mpa", "ddt_mha", "ht_mua", "ddt_mua"]
    lig_1 = ["ddt", "ddt", "ht", "ddt", "ddt", "ddt", "ht", "ddt"]
    lig_2 = ["apt", "aht", "aut", "aut", "mpa", "mha", "mua", "mua"]
    conf_types = ["janus", "random", "striped"]
    trials = 3
    with open("../datafiles/ligand_lengths.txt", "w") as file:
        file.write("monolayer\taverage\terror\n")
        file.close()
    for i, monolayer in enumerate(monolayers):
        exclude_types = [7]
        distances1, distances2 = [], []
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
                    ds1, ds2 = get_lengths(data[start_line:end_line], num_ligands_1[i], num_ligands_2[i], num_ligands_1_len[i], num_ligands_2_len[i], exclude_types=exclude_types, anchor_type=5)
                    distances1.extend(ds1)
                    distances2.extend(ds2)
        avg1, avg2 = np.mean(distances1), np.mean(distances2)
        sem1 = stats.sem(distances1) if len(distances1) > 1 else 0
        sem2 = stats.sem(distances2) if len(distances2) > 1 else 0
        average_length_mismatch = np.abs(avg1 - avg2)
        stdev_length_mismatch = np.sqrt(sem1**2 + sem2**2)
        with open("../datafiles/ligand_lengths.txt", "a") as file:
            file.write(f"{monolayer}\t{average_length_mismatch}\t{stdev_length_mismatch}\n")
            file.close()
