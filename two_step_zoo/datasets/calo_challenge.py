import h5py
from pathlib import Path
import numpy as np
import torch

from two_step_zoo.datasets.supervised_dataset import SupervisedDataset

def get_raw_data_tensors(cfg: dict, name: str, root: str):
    filenames = {"photons1":['dataset_1_photons_1.hdf5','dataset_1_photons_2.hdf5'],
                 "pions1":['dataset_1_pions_1.hdf5'],
                 "electrons2":['dataset_2_1.hdf5','dataset_2_2.hdf5'],
                 "electrons3":['dataset_3_1.hdf5','dataset_3_2.hdf5','dataset_3_3.hdf5','dataset_3_4.hdf5']}
    
    data_path = lambda x: Path(root) / x
    
    # NOTE: Data loading and preprocessing uses code from https://github.com/ViniciusMikuni/CaloScore
    showers = []
    energies = []
    for dataset in filenames[name]:
        with h5py.File(data_path(dataset), "r") as h5f:
            # rescale to GeV
            energy = h5f['incident_energies'][:].astype(np.float32) / 1000.0
            shower = h5f['showers'][:].astype(np.float32) / 1000.0
            
            shower = np.reshape(shower,(shower.shape[0], -1))
            # Rescale by incident energy times a dataset-dependent constant
            # to ensure that voxel energies are in [0,1]
            shower = shower / (cfg["max_deposited_ratio"] * energy)
            
            if cfg["normalized_deposited_energy"]:
                # Reshape datasets
                shower_padded = np.zeros([shower.shape[0], cfg["padded_shape"][1]], dtype=np.float32)
                # Compute total energy deposited in each shower
                deposited_energy = np.sum(shower, -1, keepdims=True)
                # Normalize voxel data by deposited energy (some showers deposit zero energy)
                shower = np.ma.divide(shower, np.sum(shower, -1, keepdims=True)).filled(0)
                # Add total deposited energy as a feature
                shower = np.concatenate((shower, deposited_energy), -1)
                shower_padded[:, :shower.shape[1]] += shower
                shower = shower_padded
            
            shower = shower.reshape(cfg["padded_shape"])

            # Transform voxel energies to logspace
            alpha = 1e-6 # for numerical stability of log
            x = alpha + (1 - 2*alpha) * shower
            shower = np.ma.log(x / (1 - x)).filled(0)  

            if cfg["logspace_incident_energies"]:        
                energy = np.log10(energy / cfg["energy_min"]) / np.log10(cfg["energy_max"] / cfg["energy_min"])
            else:
                energy = (energy - cfg["energy_min"]) / (cfg["energy_max"] - cfg["energy_min"])

            showers.append(shower)
            energies.append(energy)
            
    showers = np.reshape(showers, cfg["padded_shape"])
    energies = np.reshape(energies, (-1, 1))
    
    return showers, energies


def get_datasets_fn(cfg, dataset_name, data_root, valid_fraction):
    showers, energies = get_raw_data_tensors(cfg, dataset_name, data_root)

    perm = torch.randperm(showers.shape[0])
    shuffled_showers = showers[perm]
    shuffled_energies = energies[perm]

    valid_size = int(valid_fraction * showers.shape[0])
    valid_showers = torch.tensor(shuffled_showers[:valid_size], dtype=torch.get_default_dtype())
    valid_energies = torch.tensor(shuffled_energies[:valid_size], dtype=torch.get_default_dtype())
    train_showers = torch.tensor(shuffled_showers[valid_size:], dtype=torch.get_default_dtype())
    train_energies = torch.tensor(shuffled_energies[valid_size:], dtype=torch.get_default_dtype())
    
    train_dset = SupervisedDataset(dataset_name, "train", train_showers, train_energies)
    valid_dset = SupervisedDataset(dataset_name, "valid", valid_showers, valid_energies)

    # Currently the same as valid
    test_dset = SupervisedDataset(dataset_name, "test", valid_showers, valid_energies)

    return train_dset, valid_dset, test_dset


def get_physics_datasets(cfg, name, data_root, make_valid_dset):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.2 if make_valid_dset else 0
  
    if not name in ["photons1", "pions1", "electrons2", "electrons3"]:
        raise ValueError(f"Unknown dataset {name}")

    return get_datasets_fn(cfg, name, data_root, valid_fraction)
        