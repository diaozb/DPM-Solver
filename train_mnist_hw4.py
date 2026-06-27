import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.ddpm import DDPM
from mindiffusion.unet import NaiveUnet


def train_mnist(
    n_epoch=30,
    batch_size=128,
    n_T=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs("contents/hw4", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    model = NaiveUnet(in_channels=1, out_channels=1, n_feat=64)
    ddpm = DDPM(
        eps_model=model,
        betas=(1e-4, 0.02),
        n_T=n_T,
    ).to(device)

    tf = transforms.Compose([
        transforms.Pad(2),          # MNIST: 28x28 -> 32x32, compatible with NaiveUnet
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for epoch in range(n_epoch):
        ddpm.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        loss_ema = None

        for x, _ in pbar:
            x = x.to(device)
            optim.zero_grad()

            loss = ddpm(x)
            loss.backward()
            optim.step()

            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"Epoch {epoch} | loss {loss_ema:.4f}")

        ddpm.eval()
        with torch.no_grad():
            samples = ddpm.sample(16, (1, 32, 32), device)
            grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            save_image(grid, f"contents/hw4/ddpm_train_epoch_{epoch:03d}.png")

        torch.save(ddpm.state_dict(), "checkpoints/ddpm_mnist_32.pth")

    print("Saved to checkpoints/ddpm_mnist_32.pth")


if __name__ == "__main__":
    train_mnist()
