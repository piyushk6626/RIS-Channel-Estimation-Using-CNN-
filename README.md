# RIS-Assisted mmWave Dataset Creation And CNN Training

This repository contains a **Python dataset generator** and a **PyTorch CNN training pipeline** for a **single-user, narrowband, RIS-assisted mmWave channel estimation** problem.

The goal of this code is to create a **clean, reproducible, physically motivated synthetic dataset** and then use it to train and evaluate:

- Least Squares (LS)
- compressed sensing style baselines such as OMP
- CNN-based channel estimators

The implementation follows the scope of the project brief:

- one BS
- one RIS
- one user
- narrowband mmWave channel
- pilot-based channel estimation
- reduced-pilot experiments

It also adds realism beyond a toy simulator by including:

- 3D geometry
- sparse geometric channels
- LoS and NLoS path components
- 28 GHz close-in path loss
- RIS phase codebooks
- phase quantization
- reproducible random seeding

## 1. What This Repository Currently Does

The current codebase implements two stages:

1. **dataset creation**
2. **CNN training and evaluation**

The dataset generator creates synthetic pairs of:

- **input**: noisy pilot observations seen at the BS after RIS reflection
- **label**: the true cascaded BS-RIS-user channel

For each pilot length `Q`, the generator writes:

- `train.npz`
- `val.npz`
- `test.npz`

and also a dataset-level:

- `manifest.json`

The default setup produces datasets for:

- pilot lengths `{8, 12, 16, 24, 32}`
- SNR values `{0, 5, 10, 15, 20}` dB

With the default split sizes in [configs/dataset_small.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_small.yaml):

- train = `8000`
- val = `1000`
- test = `1000`

that means:

- `10000` samples per pilot length
- `50000` samples total across all five pilot-length folders

The training pipeline then reads one pilot folder at a time, trains a compact CNN, compares it with the LS baseline, and saves:

- model checkpoints
- CSV and JSON summaries
- CNN vs LS evaluation metrics
- publication-style plots for the run

## 2. Repository Structure

- [src/ris_dataset/config.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/config.py)
  Loads YAML config files into structured dataclasses and validates inputs.
- [src/ris_dataset/geometry.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/geometry.py)
  Samples UE positions and computes distances and geometric angles.
- [src/ris_dataset/channels.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/channels.py)
  Builds BS-RIS and RIS-UE sparse geometric channels.
- [src/ris_dataset/pilots.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/pilots.py)
  Builds the RIS pilot phase codebook and applies phase quantization.
- [src/ris_dataset/generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/generator.py)
  Generates samples, splits, whole datasets, and the LS sanity-check estimator.
- [src/ris_dataset/io.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/io.py)
  Converts complex tensors into real/imag channel format and saves `.npz` archives.
- [scripts/generate_dataset.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/scripts/generate_dataset.py)
  Command-line entry point.
- [src/ris_training/config.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/config.py)
  Loads the CNN training configuration and applies CLI overrides.
- [src/ris_training/data.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/data.py)
  Loads `.npz` splits, standardizes them, and prepares PyTorch datasets.
- [src/ris_training/model.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/model.py)
  Defines the compact CNN used for channel estimation.
- [src/ris_training/metrics.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/metrics.py)
  Computes MSE, NMSE, and grouped SNR summaries for CNN and LS.
- [src/ris_training/plotting.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/plotting.py)
  Creates saved figures for losses, NMSE, SNR sweeps, histograms, and channel heatmaps.
- [src/ris_training/trainer.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/trainer.py)
  Runs end-to-end training, checkpointing, evaluation, and report generation.
- [scripts/train_cnn.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/scripts/train_cnn.py)
  Training CLI for one pilot length or all configured pilot lengths.
- [configs/training_cnn.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/training_cnn.yaml)
  Default training preset tuned for Apple Silicon-friendly experimentation.
- [configs/dataset_small.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_small.yaml)
  Default experiment preset: BS `2x4`, RIS `4x4`.
- [configs/dataset_large.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_large.yaml)
  Larger preset: BS `4x4`, RIS `4x8`.
- [tests/test_generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/tests/test_generator.py)
  Smoke, reproducibility, physics, and LS sanity checks.
- [tests/test_training.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/tests/test_training.py)
  Loader, model-shape, metrics, and training smoke tests.

## 3. System Model

We model the communication path:

`BS -> RIS -> User`

The direct BS-user path is intentionally disabled in this first version to keep the estimation target focused and consistent with the project scope.

### 3.1 Nodes

- BS with `M` antennas
- RIS with `N` passive reflecting elements
- single-antenna user

For the default small preset:

- `M = 8`
- `N = 16`

### 3.2 Geometry

The default geometry is:

- BS at `(0, 0, 8)` meters
- RIS at `(20, 0, 5)` meters
- UE height fixed at `1.5` meters
- UE horizontal position sampled in a sector around the RIS

Default UE sampling sector:

- radius from `5 m` to `25 m`
- azimuth from `-60 deg` to `+60 deg`

This is a practical compromise:

- fixed BS and RIS give a stable reference geometry
- random UE placement gives channel diversity
- path lengths vary enough to create realistic power variation

## 4. Mathematical Model

This section explains the actual equations behind the generator.

### 4.1 Uniform Planar Array Response

Both the BS and RIS are modeled as **uniform planar arrays (UPAs)**.

For an array with `N_r` rows and `N_c` columns, the steering vector used in the code is:

```math
\mathbf{a}(\phi,\theta)
=
\frac{1}{\sqrt{N_r N_c}}
\exp\left(
j 2 \pi \frac{d}{\lambda}
\left(
r \sin(\phi)\cos(\theta)
+ c \sin(\theta)
\right)
\right)
```

In index form, for row index `r` and column index `c`:

```math
[\mathbf{a}(\phi,\theta)]_{r,c}
=
\frac{1}{\sqrt{N_r N_c}}
\exp\left(
j 2 \pi \frac{d}{\lambda}
\left(
r \sin(\phi)\cos(\theta)
+ c \sin(\theta)
\right)
\right)
```

where:

- `phi` is azimuth
- `theta` is elevation
- `d/lambda` is the element spacing in wavelengths

In this repository, the default is:

- `d/lambda = 0.5`

which means **half-wavelength spacing**.

Implementation:

- [src/ris_dataset/channels.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/channels.py)

### 4.2 Sparse mmWave BS-RIS Channel

The BS-RIS link is generated as a sparse geometric MIMO channel:

```math
\mathbf{G}
=
\sum_{\ell=1}^{L_{BR}}
\alpha_{\ell}
\mathbf{a}_{BS}(\phi^{rx}_{\ell}, \theta^{rx}_{\ell})
\mathbf{a}_{RIS}(\phi^{tx}_{\ell}, \theta^{tx}_{\ell})^H
```

where:

- `G` has shape `M x N`
- `L_BR` is the number of BS-RIS paths
- `alpha_l` is the complex gain of path `l`

Default BS-RIS path count:

- `1` LoS path
- `2` NLoS paths

So in the default config:

```math
L_{BR} = 3
```

### 4.3 Sparse mmWave RIS-UE Channel

The RIS-UE link is generated as:

```math
\mathbf{h}_{RU}
=
\sum_{p=1}^{L_{RU}}
\beta_p
\mathbf{a}_{RIS}(\phi_p,\theta_p)
```

where:

- `h_RU` has shape `N x 1` in math
- in code it is stored as a length-`N` complex vector

Default RIS-UE path count:

- `1` LoS path
- `1` NLoS path

So in the default config:

```math
L_{RU} = 2
```

### 4.4 Cascaded Channel

The channel label used for learning is the **cascaded channel**:

```math
\mathbf{H}_c = \mathbf{G} \operatorname{diag}(\mathbf{h}_{RU})
```

Shape:

- `G`: `M x N`
- `diag(h_RU)`: `N x N`
- `H_c`: `M x N`

This is exactly what the code computes in [src/ris_dataset/generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/generator.py):

```python
cascaded_channel = g_br * h_ru[np.newaxis, :]
```

This works because multiplying each column of `G` by the corresponding RIS-user coefficient is equivalent to multiplying by `diag(h_RU)`.

### 4.5 Pilot Observation Model

The clean received pilot matrix is generated as:

```math
\mathbf{Y}_{clean} = \mathbf{W}^H \mathbf{H}_c \mathbf{\Omega}
```

In this v1 dataset:

- the BS combiner is fixed as `W = I_M`
- pilot symbols are fixed to `1`

so the equation reduces to:

```math
\mathbf{Y}_{clean} = \mathbf{H}_c \mathbf{\Omega}
```

where:

- `Y_clean` has shape `M x Q`
- `Omega` has shape `N x Q`
- `Q` is the pilot length

Noisy observations are then:

```math
\mathbf{Y} = \mathbf{Y}_{clean} + \mathbf{N}
```

with complex Gaussian noise:

```math
\mathbf{N}_{m,q} \sim \mathcal{CN}(0, \sigma_n^2)
```

### 4.6 Noise Variance and SNR Control

For each generated sample, the code measures the clean observation power:

```math
P_s = \frac{1}{MQ}\|\mathbf{Y}_{clean}\|_F^2
```

and then computes the noise variance needed for the requested SNR:

```math
\sigma_n^2 = \frac{P_s}{10^{\text{SNR}_{dB}/10}}
```

Noise is generated as circularly symmetric complex Gaussian noise:

```math
\mathbf{N}
=
\sqrt{\sigma_n^2/2}
\left(
\mathbf{N}_R + j\mathbf{N}_I
\right)
```

where `N_R` and `N_I` are i.i.d. standard normal.

This is important because it means the dataset does **not** use a fixed global noise variance. Instead, every sample is normalized to match the requested SNR relative to that sample's clean pilot power.

### 4.7 Pilot Codebook

The RIS phase control matrix `Omega` is deterministic and DFT-based.

If `Q <= N`, the generator uses:

```math
\mathbf{\Omega} = \text{first } Q \text{ columns of } \mathbf{F}_N
```

where `F_N` is the normalized `N x N` DFT matrix.

If `Q > N`, the generator uses:

```math
\mathbf{\Omega} = \text{first } N \text{ rows of } \mathbf{F}_Q
```

This follows the planned rule:

- use DFT columns when pilots are fewer than RIS elements
- use DFT rows when pilots exceed RIS size

Implementation:

- [src/ris_dataset/pilots.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/pilots.py)

### 4.8 RIS Phase Quantization

By default, RIS phases are **2-bit quantized**.

That means each RIS coefficient phase is rounded to one of:

```math
\left\{
0,\frac{\pi}{2},\pi,\frac{3\pi}{2}
\right\}
```

The code supports ideal continuous phase too:

- set `pilot_quantization.ideal_continuous_phase: true`

in the YAML config.

## 5. Channel Realism Choices

This repository tries to stay realistic without becoming too large for a B.Tech-level project.

### 5.1 LoS vs NLoS Structure

For each link:

- LoS angles are derived from actual geometry
- NLoS angles are sampled randomly

This means:

- the main dominant path is geometry-consistent
- the weaker paths add multipath richness

### 5.2 Large-Scale Path Loss

The code uses a **close-in (CI) reference-distance path loss model**:

```math
PL(d) = FSPL(1m) + 10n \log_{10}(d) + X_{\sigma}
```

with:

```math
FSPL(1m) = 32.4 + 20 \log_{10}(f_{GHz})
```

where:

- `d` is distance in meters
- `n` is the path-loss exponent
- `X_sigma` is log-normal shadowing in dB

Default values:

- LoS exponent = `2.1`
- LoS shadowing std = `3.6 dB`
- NLoS exponent = `3.4`
- NLoS shadowing std = `9.7 dB`

These values are chosen to reflect 28 GHz urban microcell trends and are consistent with the literature referenced at the end of this file.

### 5.3 Small-Scale Fading Power

Path gains use:

- `sigma_los^2 = 1.0`
- `sigma_nlos^2 = 0.01`

Interpretation:

- LoS paths are dominant
- NLoS paths are much weaker

For LoS:

- amplitude is deterministic up to phase, scaled by path loss

For NLoS:

- amplitude is complex Gaussian, scaled by path loss and reduced variance

### 5.4 NLoS Distance Stretch

The NLoS distance is not forced to equal the direct geometric distance.

Instead, the generator multiplies the direct path length by a random factor:

- between `1.05` and `1.5`

This is a practical approximation to reflect the fact that reflected or scattered paths are usually longer than the direct path.

## 6. What Exactly Is Saved In The Dataset

Each `.npz` file stores:

- `observations`
- `channel`
- `omega`
- `snr_db`
- `user_xyz`
- `distances`
- `channel_norm`
- `seed`

### 6.1 Stored Shapes

For a fixed pilot length `Q`:

- `observations`: `[num_samples, Q, M, 2]`
- `channel`: `[num_samples, M, N, 2]`
- `omega`: `[num_samples, N, Q, 2]`
- `snr_db`: `[num_samples]`
- `user_xyz`: `[num_samples, 3]`
- `distances`: `[num_samples, 2]`
- `channel_norm`: `[num_samples]`
- `seed`: `[num_samples]`

The last dimension of size `2` always means:

- `[..., 0] = real part`
- `[..., 1] = imaginary part`

### 6.2 Why `observations` Is Stored As `[Q, M, 2]`

The mathematical observation matrix is generated as `M x Q`.

Before saving, the generator transposes it to `Q x M` so that:

- time/pilot index comes first
- antenna index comes second

This is often convenient for CNN preprocessing because it makes the pilot dimension explicit as a spatial axis.

## 7. Default Configurations

### 7.1 Small Configuration

File:

- [configs/dataset_small.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_small.yaml)

Values:

- carrier frequency = `28 GHz`
- spacing = `0.5 lambda`
- BS array = `2 x 4` -> `8` antennas
- RIS array = `4 x 4` -> `16` elements
- direct path = disabled

### 7.2 Large Configuration

File:

- [configs/dataset_large.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_large.yaml)

Values:

- carrier frequency = `28 GHz`
- spacing = `0.5 lambda`
- BS array = `4 x 4` -> `16` antennas
- RIS array = `4 x 8` -> `32` elements
- same geometry, path, and SNR defaults

## 8. Generation Pipeline

For each sample, the code does the following:

1. Sample a UE position from the configured sector.
2. Compute BS-RIS and RIS-UE distances.
3. Build the BS-RIS sparse channel `G`.
4. Build the RIS-UE sparse channel `h_RU`.
5. Form the cascaded channel `H_c = G diag(h_RU)`.
6. Build the RIS phase codebook `Omega` for the current pilot length.
7. Compute clean observations `Y_clean = H_c Omega`.
8. Measure clean pilot power.
9. Compute noise variance from requested SNR.
10. Add complex Gaussian noise.
11. Save input/label tensors and metadata.

For each split:

1. The split gets a deterministic seed derived from:
   - master seed
   - pilot length
   - split name
2. Sample-level seeds are generated from that split seed.
3. SNR values are assigned evenly across all requested SNR points.
4. Samples are shuffled across SNR values.

This makes the dataset:

- balanced
- reproducible
- easy to regenerate exactly

## 9. How To Run

### 9.1 Create a Virtual Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

For macOS and Apple Silicon, install PyTorch from the official PyTorch guide:

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)

For current macOS stable pip installs, PyTorch documents `pip3 install torch torchvision`, and the MPS backend is available through `torch.backends.mps.is_available()` on supported Apple Silicon systems.

Install the rest of the local dependencies after PyTorch:

```bash
.venv/bin/python -m pip install numpy PyYAML matplotlib pytest
```

### 9.2 Generate the Default Dataset

```bash
.venv/bin/python scripts/generate_dataset.py \
  --config configs/dataset_small.yaml \
  --out data/ris_mmwave_v1 \
  --seed 2026
```

### 9.3 Generate the Larger Dataset

```bash
.venv/bin/python scripts/generate_dataset.py \
  --config configs/dataset_large.yaml \
  --out data/ris_mmwave_v1_large \
  --seed 2026
```

### 9.4 Train One Pilot Length

This uses the default training config in [configs/training_cnn.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/training_cnn.yaml) and trains only `Q = 16`:

```bash
.venv/bin/python scripts/train_cnn.py \
  --config configs/training_cnn.yaml \
  --data-root data/ris_mmwave_v1 \
  --pilot-length 16 \
  --device auto
```

### 9.5 Train All Pilot Lengths

```bash
.venv/bin/python scripts/train_cnn.py \
  --config configs/training_cnn.yaml \
  --data-root data/ris_mmwave_v1 \
  --pilot-length all \
  --device auto
```

Important defaults in the training pipeline:

- device selection: `auto` -> `mps` first, then `cpu`
- dtype: `float32`
- batch size: `128`
- epochs: `60`
- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- early stopping patience: `8`
- `num_workers = 0` to stay friendly to MacBook memory and process limits

## 10. Output Directory Layout

After generation, the directory looks like this:

```text
data/ris_mmwave_v1/
├── manifest.json
├── pilots_8/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── pilots_12/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── pilots_16/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── pilots_24/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
└── pilots_32/
    ├── train.npz
    ├── val.npz
    └── test.npz
```

`manifest.json` stores:

- generation timestamp
- global config
- pilot lengths
- split counts
- output tensor shapes
- SNR histograms

The training pipeline writes a separate run directory:

```text
data/runs/
└── cnn_baseline/
    └── 20260421-153000/
        ├── pilot_length_summary.csv
        ├── pilot_length_vs_nmse.png
        ├── experiment_summary.md
        ├── pilots_8/
        │   ├── best.pt
        │   ├── last.pt
        │   ├── history.csv
        │   ├── metrics.json
        │   ├── normalization.json
        │   ├── predictions.npz
        │   └── plots/
        │       ├── channel_examples.png
        │       ├── error_histogram.png
        │       ├── loss_curve.png
        │       ├── nmse_curve.png
        │       └── snr_vs_nmse.png
        └── pilots_16/
            └── ...
```

Per-pilot run outputs:

- `best.pt`: checkpoint with the best validation NMSE
- `last.pt`: checkpoint from the final training epoch
- `history.csv`: epoch-by-epoch train and validation loss/NMSE
- `metrics.json`: validation and test summaries for CNN and LS
- `normalization.json`: train-split standardization statistics
- `predictions.npz`: saved test targets plus CNN and LS predictions
- `plots/*.png`: figures ready to use in the report

## 11. Verification And Tests

The repository includes tests in:

- [tests/test_generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/tests/test_generator.py)
- [tests/test_training.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/tests/test_training.py)

The tests cover:

- dataset smoke generation
- exact reproducibility with same seed
- different outputs with different seeds
- finite numerical values
- stronger channels for closer users on average
- channel rank bounded by path structure
- requested SNR matching measured SNR within tolerance
- LS doing better with fully observed pilots than with short pilots
- balanced SNR distribution in the manifest and `.npz` files
- training loader tensor shapes and normalization restoration
- CNN forward-shape checks for multiple pilot lengths
- metrics grouping by SNR
- one-epoch end-to-end training smoke run with saved artifacts

Run tests with:

```bash
.venv/bin/pytest -q
```

## 12. Least Squares Baseline Used For Sanity Check

This repository includes a simple LS estimator that is reused in both the generator sanity checks and the CNN evaluation pipeline:

```math
\hat{\mathbf{H}}_{LS}
=
\mathbf{Y}
\mathbf{\Omega}^H
\left(
\mathbf{\Omega}\mathbf{\Omega}^H
\right)^{\dagger}
```

Implementation:

- [src/ris_dataset/generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/generator.py)

This baseline is used to confirm that:

- longer pilot lengths should improve identifiability
- underdetermined short-pilot setups are harder
- the trained CNN can be compared against a transparent classical reference

## 13. Assumptions And Limits Of This v1 Dataset

This dataset is intentionally focused.

Current assumptions:

- single-user only
- narrowband only
- direct BS-user path disabled
- RIS is passive
- BS combiner fixed to identity
- pilot symbols fixed to `1`
- no hardware impairments
- no frequency selectivity
- no mobility over time

This is a strong starting point for a course project because it keeps the estimation target clear:

- learn the cascaded channel from limited noisy pilot observations

Future extensions could include:

- direct BS-user path
- wideband OFDM channels
- multi-user training
- hybrid beamforming
- time-varying channels
- imperfect RIS elements
- spatial correlation models

## 14. Important Implementation Notes

### 14.1 This Is Inspired By The Literature, Not A Paper Reproduction

The dataset generator is **research-informed**, but it is **not a line-by-line reproduction** of any one paper's experimental setup.

What we borrowed from the literature:

- the cascaded RIS channel viewpoint
- sparse geometric mmWave modeling
- pilot-overhead reduction framing
- DFT-style RIS observation matrices
- close-in path loss motivation for 28 GHz links

What we simplified for project practicality:

- single-user only
- narrowband only
- identity BS combiner
- no direct path
- fixed scene geometry except UE position

### 14.2 Why This Is Still A Good Dataset For The Assignment

This dataset is well suited for your assignment because it supports the exact research question:

> Can a learning-based estimator recover the RIS-assisted cascaded channel accurately even when the number of pilots is reduced?

With this repository, experiments can directly produce:

- NMSE vs SNR
- NMSE vs number of pilots
- LS vs CNN comparisons
- training and validation loss curves
- per-sample error histograms
- channel heatmap examples for qualitative inspection

## 15. References Used For This Implementation

These are the main references that informed the dataset design.

1. Jiguang He, Henk Wymeersch, Marco Di Renzo, and Markku Juntti, *Learning to Estimate RIS-Aided mmWave Channels*, IEEE Wireless Communications Letters, vol. 11, no. 4, pp. 841-845, 2022.
   Link: [https://doi.org/10.1109/LWC.2022.3147250](https://doi.org/10.1109/LWC.2022.3147250)
   Open repository page: [https://oulurepo.oulu.fi/handle/10024/34050](https://oulurepo.oulu.fi/handle/10024/34050)

   How it influenced this repository:
   - single-user RIS-aided channel estimation framing
   - cascaded channel observation model
   - reduced training overhead motivation

2. Asmaa Abdallah, Abdulkadir Celik, Mohammad M. Mansour, and Ahmed M. Eltawil, *RIS-Aided mmWave MIMO Channel Estimation Using Deep Learning and Compressive Sensing*, IEEE Transactions on Wireless Communications, vol. 22, no. 5, pp. 3503-3521, 2023.
   Link: [https://doi.org/10.1109/TWC.2022.3219140](https://doi.org/10.1109/TWC.2022.3219140)
   Open repository page: [https://scholarworks.aub.edu.lb/handle/10938/27506](https://scholarworks.aub.edu.lb/handle/10938/27506)

   How it influenced this repository:
   - RIS/mmWave sparse geometric channel structure
   - pilot-overhead viewpoint
   - learning plus baseline comparison mindset

3. George R. MacCartney, Junhong Zhang, Shuai Nie, and Theodore S. Rappaport, *Path Loss Models for 5G Millimeter Wave Propagation Channels in Urban Microcells*, in IEEE GLOBECOM 2013, pp. 3948-3953, 2013.
   Link: [https://doi.org/10.1109/GLOCOM.2013.6831690](https://doi.org/10.1109/GLOCOM.2013.6831690)
   NYU page: [https://nyuscholars.nyu.edu/en/publications/path-loss-models-for-5g-millimeter-wave-propagation-channels-in-u](https://nyuscholars.nyu.edu/en/publications/path-loss-models-for-5g-millimeter-wave-propagation-channels-in-u)

   How it influenced this repository:
   - the 28 GHz close-in path loss perspective
   - realistic LoS/NLoS path-loss exponent motivation

## 16. Summary

This repository now provides a complete RIS-assisted mmWave experimentation pipeline for a course project:

- mathematically grounded synthetic dataset generation
- configurable and reproducible split creation
- LS baseline evaluation
- a compact PyTorch CNN estimator
- saved checkpoints, metrics, and plots
- tests for both dataset generation and training

That lets you build the assignment around:

- NMSE vs SNR
- NMSE vs pilot length
- LS vs CNN comparisons
- reduced-pilot learning performance
