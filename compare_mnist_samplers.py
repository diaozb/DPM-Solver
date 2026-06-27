import os
import torch
from torchvision.utils import save_image, make_grid

from mindiffusion.ddpm import DDPM
from mindiffusion.unet import NaiveUnet
from mindiffusion.dpm_solver_sampler import DPMSolverSampler


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("contents/hw4", exist_ok=True)

    n_T = 1000

    ddpm = DDPM(
        eps_model=NaiveUnet(1, 1, n_feat=64),
        betas=(1e-4, 0.02),
        n_T=n_T,
    ).to(device)

    ckpt = torch.load("checkpoints/ddpm_mnist_32.pth", map_location=device)
    ddpm.load_state_dict(ckpt)
    ddpm.eval()

    sampler = DPMSolverSampler(ddpm)

    methods = [
        "ddim",
        "dpm_solver_2",
        "dpm_solver_3",
    ]

    nfes = [5, 10, 20, 50]

    for nfe in nfes:
        for method in methods:
            # same initial noise for fair visual comparison
            torch.manual_seed(1234)

            x = sampler.sample(
                n_sample=64,
                size=(1, 32, 32),
                device=device,
                nfe=nfe,
                method=method,
            )

            grid = make_grid(
                x,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )

            save_image(
                grid,
                f"contents/hw4/{method}_nfe_{nfe}.png",
            )

            print(f"Saved {method}, NFE={nfe}")

    print("All samples saved to contents/hw4/")


if __name__ == "__main__":
    main()
