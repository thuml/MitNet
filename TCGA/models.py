from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class MLP(nn.Sequential):
    def __init__(self, dims, relu=True):
        super().__init__()
        self.add_module("fc_0", nn.Linear(dims[0], dims[1]))
        if relu or len(dims) > 2:
            self.add_module("act_0", nn.ReLU())
        for i in range(1, len(dims) - 1):
            self.add_module(f"fc_{i}", nn.Linear(dims[i], dims[i+1]))
            if relu or i < len(dims) - 2:
                self.add_module(f"act_{i}", nn.ReLU())


class Head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ctrl = MLP([hidden_size, hidden_size, 1], relu=False)
        self.trtd = nn.ModuleList([MLP([2 * hidden_size, hidden_size, 1], relu=False) for _ in range(3)])

    def forward(self, f, f_dose, treatment):
        y_ctrl = self.ctrl(f)
        y_trtd = [trtd(x) for trtd, x in zip(self.trtd, f_dose)]
        y_eval = torch.cat([y_ctrl] + y_trtd, dim=1)
        y = y_eval[torch.arange(f.shape[0]), treatment.squeeze(1)].unsqueeze(1)
        return y, y_eval


class TARNet(nn.Module):
    def __init__(self, input_dim, prop, hidden_size=256):
        super().__init__()
        self.prop = prop
        self.repr = MLP([input_dim] + [hidden_size] * 3)
        self.pred = Head(hidden_size)

    def forward(self, inputs, treatment, eval_dose, **kwargs):
        f = self.repr(inputs)
        f_dose = [torch.cat([f, eval_dose[:, (idx,)] * f], dim=1) for idx in range(3)]
        pred, pred_eval = self.pred(f, f_dose, treatment)
        pred_eff = torch.stack(
            [pred_eval[:, i + 1] - pred_eval[:, 0] for i in range(3)], dim=1
        )
        return {"pred": pred, "pred_eff": pred_eff, "prop": self.prop}


class MitNet(nn.Module):
    def __init__(self, input_dim, prop, hidden_size=256):
        super().__init__()
        self.prop = prop
        self.repr = MLP([input_dim] + [hidden_size] * 3)
        self.pred = Head(hidden_size)
        self.stat = Head(hidden_size)
        self.grl = GradientReverseLayer()

    def forward(self, inputs, treatment, eval_dose, marginal_dose, **kwargs):
        f = self.repr(inputs)
        f_dose = [torch.cat([f, eval_dose[:, (idx,)] * f], dim=1) for idx in range(3)]
        f_marginal = [torch.cat([f, marginal_dose[:, (idx,)] * f], dim=1) for idx in range(3)]
        pred, pred_eval = self.pred(f, f_dose, treatment)
        pred_eff = torch.stack(
            [pred_eval[:, i + 1] - pred_eval[:, 0] for i in range(3)], dim=1
        )
        joint, _ = self.stat(f, f_dose, treatment)
        _, marginal = self.stat(f, f_marginal, treatment)
        return {
            "pred": pred,
            "pred_eff": pred_eff,
            "joint": joint,
            "marginal": marginal,
            "prop": self.prop,
        }
