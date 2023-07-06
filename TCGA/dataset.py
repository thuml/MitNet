import numpy as np
import torch
from torch.utils.data import Dataset


class TCGA(Dataset):
    def __init__(self, data_path, normalize=True):
        data = dict(np.load(data_path))
        self.x = torch.from_numpy(data["feature"]).float()
        self.t = torch.from_numpy(data["t"]).long()
        self.eval_y = torch.from_numpy(data["y"]).float()
        self.eval_d = torch.from_numpy(data["d"]).float()
        self.eval_e = torch.cat([
            self.eval_y[:, (i + 1,)] - self.eval_y[:, (0,)] for i in range(3)
        ], dim=1)
        self.length = self.x.shape[0]

        if normalize:
            x_mean = self.x.mean(dim=0, keepdim=True)
            x_std = self.x.std(dim=0, keepdim=True)
            self.x = (self.x - x_mean) / (x_std + 1e-6)

        self.d = torch.tensor([
            0.0 if self.t[i, 0] == 0 else self.eval_d[i, self.t[i, 0] - 1]
            for i in range(self.length)
        ]).unsqueeze(1)

        self.y = torch.tensor([self.eval_y[i, self.t[i, 0]] for i in range(self.length)]).unsqueeze(1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == list:
            rand_idx = torch.randint(self.length, (len(idx), 3))
            marginal_dose = torch.stack([self.eval_d[rand_idx[i], torch.arange(3)] for i in range(len(idx))], dim=0)
        else:
            rand_idx = torch.randint(self.length, (3,))
            marginal_dose = self.eval_d[rand_idx, torch.arange(3)]
        sample = {
            "treatment": self.t[idx],
            "dose": self.d[idx],
            "factual": self.y[idx],
            "eval_y": self.eval_y[idx],
            "eval_dose": self.eval_d[idx],
            "marginal_dose": marginal_dose,
            "inputs": self.x[idx],
            "effect": self.eval_e[idx],
        }
        return sample
