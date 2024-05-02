import os.path
import numpy as np
import sys

import extract_pdb_features

data_dir = os.path.join(os.path.dirname(__file__), 'data/')


feature_cache = {}  # path: (features, mask)
def encoder_features(pdb_path, virt_cb):
    """
    Calculate 3D descriptors for each residue of a PDB file.
    """
    feat = feature_cache.get(pdb_path, None)
    if feat is not None:
        return feat

    coords, valid_mask = extract_pdb_features.get_coords_from_pdb(pdb_path, full_backbone=True)
    coords = extract_pdb_features.move_CB(coords, virt_cb=virt_cb)
    partner_idx = extract_pdb_features.find_nearest_residues(coords, valid_mask)
    features, valid_mask2 = extract_pdb_features.calc_angles_forloop(coords, partner_idx, valid_mask)

    seq_dist = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)

    vae_features = np.hstack([features, log_dist])
    feature_cache[pdb_path] = vae_features, valid_mask2

    return vae_features, valid_mask2


if __name__ == '__main__':
    pdb_dir = sys.argv[1]
    out = sys.argv[2]
    a = int(sys.argv[3])
    b = int(sys.argv[4])
    c = float(sys.argv[5])
    virtual_center = (a, b, c)

    with open(data_dir + 'pdbs_train.txt') as file:
        pdbs_train = set(file.read().splitlines())

    # Find PDB files in the directory
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

    # Collect features for each PDB file
    features_per_residue = []
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        if os.path.basename(pdb_file) in pdbs_train:
            features, _ = encoder_features(pdb_path, virtual_center)
            features_per_residue.append(features)

    # Stack features into a single matrix
    all_features = np.vstack(features_per_residue)

    # Write features to disc
    np.save(out, all_features)