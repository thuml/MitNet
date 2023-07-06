import torch


def factual_mse_loss(factual, pred, **kwargs):
    return ((factual - pred) ** 2).mean()


def mutual_info(joint, marginal, prop, **kwargs):
    sample_wise_max = marginal.max(dim=0)[0]
    mi = joint.mean() - (torch.log(torch.exp(marginal - sample_wise_max.unsqueeze(0)).mean(dim=0)) + sample_wise_max) @ prop
    return torch.sqrt(2 * mi + 1) - 1


def pehe_sqrt(pred_eff, effect, prop, **kwargs):
    prop_new = torch.tensor([prop[i + 1] / (1 - prop[0]) for i in range(3)], device=effect.device)
    return (((effect - pred_eff) ** 2).mean(dim=0) @ prop_new).sqrt()


def ate_bias(pred_eff, effect, prop, **kwargs):
    prop_new = torch.tensor([prop[i + 1] / (1 - prop[0]) for i in range(3)], device=effect.device)
    return (pred_eff - effect).mean(dim=0) @ prop_new


def ate_err(pred_eff, effect, prop, **kwargs):
    return ate_bias(pred_eff, effect, prop, **kwargs).abs()
