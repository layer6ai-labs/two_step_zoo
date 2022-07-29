import torch
import numpy as np
from pytorch_fid import fid_score
import prdc

from .metrics_helpers import InceptionHelper
from .ood_helpers import ood_acc


def fid(module, eval_loader=None, train_loader=None, gen_samples=50000, gen_batch_size=64,
        cache=None):
    """
    Following Heusel et al. (2017), compute FID from the training set if provided.
    """
    dataloader = eval_loader if train_loader is None else train_loader
    inception = InceptionHelper(module, dataloader, gen_samples, gen_batch_size)

    gen_mu, gen_sigma = inception.compute_inception_stats()

    if cache is None:
        gt_mu, gt_sigma = inception.compute_inception_stats(dataloader)
    elif "gt_feats" not in cache:
        gt_feats = inception.get_inception_features(dataloader)
        cache["gt_feats"] = gt_feats
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        cache["gt_stats"] = gt_mu, gt_sigma
    elif "gt_stats" not in cache:
        gt_feats = cache["gt_feats"]
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        cache["gt_stats"] = gt_mu, gt_sigma    
    else:
        gt_mu, gt_sigma = cache["gt_stats"]

    return fid_score.calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)


def precision_recall_density_coverage(module, eval_loader=None, train_loader=None, gen_samples=50000, gen_batch_size=64, nearest_k=5,
        cache=None):
    """
    Following Naaem et al. (2020), compute Precision, Recall, Density, Coverage from the training set if provided.
    """
    dataloader = eval_loader if train_loader is None else train_loader
    inception = InceptionHelper(module, dataloader, gen_samples, gen_batch_size)
    
    gen_feats = inception.get_inception_features()

    if cache is None:
        gt_feats = inception.get_inception_features(dataloader)
    elif "gt_feats" not in cache:
        gt_feats = inception.get_inception_features(dataloader)
        cache["gt_feats"] = gt_feats
    else:
        gt_feats = cache["gt_feats"]
        
    return prdc.compute_prdc(gt_feats, gen_feats, nearest_k)


def tabular_fid(module, eval_loader=None, train_loader=None, cache=None):
    """
    Following Heusel et al. (2017), compute FID from the training set if provided.
    """
    dataloader = eval_loader if train_loader is None else train_loader
    # TODO: implement
    pass


def log_likelihood(module, dataloader, cache=None):
    with torch.no_grad():
        return module.log_prob(dataloader).mean()


def l2_reconstruction_error(module, dataloader, cache=None):
    with torch.no_grad():
        return module.rec_error(dataloader).mean()


def loss(module, dataloader, cache=None):
    with torch.no_grad():
        return module.loss(dataloader).mean()


def null_metric(module, dataloader, cache=None):
    return 0


def likelihood_ood_acc(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        cache=None,
    ):
    return ood_acc(
        module, is_test_loader, oos_test_loader, is_train_loader, oos_train_loader, savedir,
        low_dim=False, cache=cache
    )


def likelihood_ood_acc_low_dim(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        cache=None,
    ):
    return ood_acc(
        module, is_test_loader, oos_test_loader, is_train_loader, oos_train_loader, savedir,
        low_dim=True, cache=cache
    )
