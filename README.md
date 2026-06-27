# DPM-Solver Sampler on MNIST

This repository is based on the starter code from `cloneofsimo/minDiffusion` and extends it for Homework 4. The main goal is to implement and compare different deterministic samplers for diffusion models on MNIST:

* DDIM, which is equivalent to DPM-Solver-1
* DPM-Solver-2
* DPM-Solver-3

The implementation uses the same trained DDPM noise prediction model for all samplers. Only the sampling algorithm is changed during comparison.

## Project Structure

```text
DPM-Solver/
├── mindiffusion/
│   ├── ddpm.py
│   ├── unet.py
│   └── dpm_solver_sampler.py
├── train_mnist_hw4.py
├── compare_mnist_samplers.py
├── contents/
│   └── hw4/
│       ├── ddim_nfe_5.png
│       ├── dpm_solver_2_nfe_5.png
│       ├── dpm_solver_3_nfe_5.png
│       └── ...
└── checkpoints/
    └── ddpm_mnist_32.pth
```

## Code Logic

### 1. DDPM Training

The training script is:

```text
train_mnist_hw4.py
```

This script trains a DDPM model on MNIST. The MNIST images are padded from 28×28 to 32×32 so that they are compatible with the U-Net architecture used in the starter code. The images are normalized from `[0, 1]` to `[-1, 1]`.

The model uses:

```text
NaiveUnet(in_channels=1, out_channels=1, n_feat=64)
```

The diffusion process uses 1000 timesteps:

```text
n_T = 1000
```

After training, the checkpoint is saved to:

```text
checkpoints/ddpm_mnist_32.pth
```

This checkpoint is shared by all samplers in the comparison.

### 2. DDPM Schedule

The DDPM noise schedule is implemented in:

```text
mindiffusion/ddpm.py
```

The schedule stores:

```text
alpha_t
alphabar_t
sqrtab
sqrtmab
sqrt_beta_t
oneover_sqrta
mab_over_sqrtmab
```

The important detail is that:

```text
alphabar_t[0] = 1
```

This makes the indexing cleaner for DPM-Solver, because the sampler can use timestep indices from `1` to `n_T` while treating timestep `0` as the clean data endpoint.

### 3. DPM-Solver Sampler

The main sampler implementation is:

```text
mindiffusion/dpm_solver_sampler.py
```

The class `DPMSolverSampler` implements:

```text
dpm_solver_1_step
dpm_solver_2_step
dpm_solver_3_step
sample
```

The sampler uses the half-log-SNR variable:

```text
lambda = log(alpha / sigma)
```

For the VP diffusion setting:

```text
alpha^2 + sigma^2 = 1
```

so alpha and sigma can be computed directly from lambda.

The sampler first builds a lambda table from the trained DDPM schedule. During sampling, it uniformly splits the lambda interval and solves the diffusion ODE from the noisy endpoint to the clean endpoint.

### 4. DDIM / DPM-Solver-1

DDIM is implemented as the first-order DPM-Solver step:

```text
dpm_solver_1_step
```

It uses one model evaluation per solver step.

### 5. DPM-Solver-2

DPM-Solver-2 is implemented as a midpoint second-order solver:

```text
dpm_solver_2_step
```

Each step uses two model evaluations:

1. Evaluate the noise model at the starting point.
2. Estimate an intermediate midpoint.
3. Evaluate the noise model at the midpoint.
4. Use the midpoint prediction to update the sample.

### 6. DPM-Solver-3

DPM-Solver-3 is implemented in:

```text
dpm_solver_3_step
```

It uses two intermediate points with:

```text
r1 = 1 / 3
r2 = 2 / 3
```

Each third-order step uses three model evaluations. When the requested NFE is not divisible by 3, the sampler uses as many third-order steps as possible and then uses a lower-order step for the remaining budget.

### 7. NFE Logic

NFE means number of function evaluations, i.e. the number of calls to the noise prediction model.

The implemented order schedule is:

```text
DDIM / DPM-Solver-1:
    one model evaluation per step

DPM-Solver-2:
    two model evaluations per second-order step

DPM-Solver-3:
    three model evaluations per third-order step
```

For example:

```text
NFE = 10

DDIM:
    10 first-order steps

DPM-Solver-2:
    5 second-order steps

DPM-Solver-3:
    3 third-order steps + 1 first-order step
```

This makes the comparison fair because all samplers use the same total number of model evaluations.

## How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision tqdm matplotlib
```

### 2. Prepare MNIST

MNIST should be placed under:

```text
data/MNIST/
```

If the raw MNIST files already exist in:

```text
data/MNIST/raw/
```

then the training script can directly use them.

### 3. Train the DDPM Model

Run:

```bash
python train_mnist_hw4.py
```

After training, the checkpoint will be saved as:

```text
checkpoints/ddpm_mnist_32.pth
```

The training script also saves intermediate DDPM samples to:

```text
contents/hw4/
```

These images are useful for checking whether the trained DDPM model is reasonable.

### 4. Compare Samplers

Run:

```bash
python compare_mnist_samplers.py
```

This script loads:

```text
checkpoints/ddpm_mnist_32.pth
```

and compares:

```text
ddim
dpm_solver_2
dpm_solver_3
```

under different NFE values:

```text
[5, 10, 15, 20, 30, 50]
```

For fair visual comparison, the script fixes the random seed before each sampling run so that different samplers start from the same initial noise.

### 5. View Results

The generated images are saved to:

```text
contents/hw4/
```

The output filenames follow this format:

```text
{sampler_name}_nfe_{nfe}.png
```

Examples:

```text
contents/hw4/ddim_nfe_10.png
contents/hw4/dpm_solver_2_nfe_10.png
contents/hw4/dpm_solver_3_nfe_10.png
```

To view the results, open the PNG files in the `contents/hw4/` directory.

## Qualitative Results

The comparison shows the following qualitative behavior.

At very low NFE, especially NFE=5, the samplers are not equally stable. DDIM can still produce some recognizable digits, but the images are noisy. DPM-Solver-2 also produces noisy samples. DPM-Solver-3 can become unstable at this extremely low NFE because each solver step is very large.

At NFE=10, the generated digits become much clearer. DPM-Solver-2 produces cleaner and more recognizable samples than DDIM in this experiment. DPM-Solver-3 also improves significantly compared with its NFE=5 result, although it is not always visually better than DPM-Solver-2.

At NFE=20, NFE=30, and NFE=50, all three samplers generate recognizable MNIST digits. The gap between DDIM, DPM-Solver-2, and DPM-Solver-3 becomes smaller as NFE increases.

Overall, DPM-Solver-2 gives the most stable qualitative result in this implementation, especially under medium NFE. DPM-Solver-3 is effective when the NFE is not too small, but it can be unstable at extremely low NFE. This is acceptable for the homework because the comparison is qualitative and the goal is to implement and analyze DPM-Solver-2 and DPM-Solver-3 rather than proving that they always outperform DDIM under every NFE.

## Notes

DPM-Solver is a training-free sampler. After the DDPM noise prediction model is trained, no additional model training is needed for DPM-Solver-2 or DPM-Solver-3. The sampler only changes the numerical method used to solve the diffusion ODE during generation.

Therefore, the workflow is:

```text
Train one DDPM model on MNIST.
Load the same checkpoint.
Use DDIM, DPM-Solver-2, and DPM-Solver-3 for sampling.
Compare generated images under the same NFE.
```
