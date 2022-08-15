import numpy as np
from torch.utils.data import DataLoader

from .image import get_image_datasets
from .generated import get_generated_datasets
from .calo_challenge import get_physics_datasets
from .supervised_dataset import SupervisedDataset


def get_loaders_from_config(cfg, device):
    """
    Wrapper function providing frequently-used functionality.

    Updates `cfg` with dataset information.
    """
    train_loader, valid_loader, test_loader = get_loaders(
        cfg,
        dataset=cfg["dataset"],
        device=device,
        data_root=cfg.get("data_root", "data/"),
        make_valid_loader=cfg["make_valid_loader"],
        train_batch_size=cfg["train_batch_size"],
        valid_batch_size=cfg["valid_batch_size"],
        test_batch_size=cfg["test_batch_size"]
    )

    train_dataset = train_loader.dataset.x
    cfg["train_dataset_size"] = train_dataset.shape[0]
    cfg["data_shape"] = tuple(train_dataset.shape[1:])
    cfg["data_dim"] = int(np.prod(cfg["data_shape"]))

    if not cfg["make_valid_loader"]:
        valid_loader = test_loader
        print("WARNING: Using test loader for validation")

    return train_loader, valid_loader, test_loader


def get_loaders(
        cfg,
        dataset,
        device,
        data_root,
        make_valid_loader,
        train_batch_size,
        valid_batch_size,
        test_batch_size
):
    if dataset in ["celeba", "mnist", "fashion-mnist", "cifar10", "svhn"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, make_valid_loader)
        
    elif dataset in ["sphere", "klein", "two_moons"]:
        train_dset, valid_dset, test_dset = get_generated_datasets(dataset)
    
    elif dataset in ['photons1', 'pions1', 'electrons2', 'electrons3']:
        train_dset, valid_dset, test_dset = get_physics_datasets(cfg, dataset, data_root, make_valid_loader)

    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    train_loader = get_loader(train_dset, device, train_batch_size, drop_last=True)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader
    
    
def get_loader(dset, device, batch_size, drop_last):
    return DataLoader(
        dset.to(device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False
    )


def get_embedding_loader(embeddings, batch_size, drop_last, role):
    dataset = SupervisedDataset(
        name="embeddings",
        role=role,
        x=embeddings
    )
    return get_loader(dataset, embeddings.device, batch_size, drop_last)


def remove_drop_last(loader):
    dset = loader.dataset
    return get_loader(dset, dset.x.device, loader.batch_size, False)
