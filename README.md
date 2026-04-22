# RIS-Assisted mmWave Channel Estimation Using CNN

This repository implements a complete, reproducible pipeline for a focused RIS-assisted mmWave channel estimation project:

1. generate a synthetic narrowband single-user RIS dataset,
2. train a compact CNN to estimate the cascaded channel,
3. compare the CNN against a Least Squares (LS) baseline,
4. study how performance changes when the pilot length is reduced.

The code is intentionally scoped to match a practical B.Tech/M.Tech assignment:

- one BS,
- one RIS,
- one single-antenna user,
- narrowband channel,
- pilot-based estimation,
- reduced-pilot experiments.

The main research question supported by this codebase is:

> Can a compact CNN recover the RIS-assisted cascaded channel better than LS when the number of pilots is limited?

## Latest Stored Results

This repository already includes a full reference run at [`data/runs/cnn_baseline/20260421-221441`](data/runs/cnn_baseline/20260421-221441). That run is based on the dataset in [`data/ris_mmwave_v1/manifest.json`](data/ris_mmwave_v1/manifest.json) and uses the larger `4 x 4` BS / `4 x 8` RIS setting, i.e. `M = 16` antennas and `N = 32` RIS elements.

The top-line outcome is strong and very specific:

- the best CNN model occurs at pilot length `Q = 12`,
- the CNN reaches `-9.238 dB` test NMSE,
- LS reaches `26.912 dB` test NMSE at the same pilot length,
- the learned model therefore improves over LS by `36.150 dB`.

| Pilot length `Q` | Best epoch | CNN test NMSE (dB) | LS test NMSE (dB) | CNN gain over LS (dB) |
| --- | ---: | ---: | ---: | ---: |
| `8` | `43` | `-9.153` | `20.281` | `29.434` |
| `12` | `36` | `-9.238` | `26.912` | `36.150` |
| `16` | `31` | `-9.193` | `19.765` | `28.958` |
| `24` | `19` | `-8.637` | `13.615` | `22.253` |
| `32` | `20` | `-8.001` | `-8.448` | `-0.447` |

Technical interpretation:

- For `Q < N = 32`, the LS estimator is underdetermined in the RIS domain and performs poorly.
- The CNN stays near `-9 dB` NMSE for `Q = 8, 12, 16`, which shows that it is learning a strong structural prior over the synthetic channel family.
- LS only becomes competitive at `Q = 32`, where the pilot budget reaches the full RIS dimension; at that point the mean gain slightly turns negative.

Detailed, figure-by-figure analysis of this run is available in [RESULTS.md](RESULTS.md).

![Pilot Length vs Test NMSE](data/runs/cnn_baseline/20260421-221441/pilot_length_vs_nmse.png)

![Pilot Length vs CNN Gain](data/runs/cnn_baseline/20260421-221441/pilot_length_vs_gain.png)

![All-Pilot SNR Comparison](data/runs/cnn_baseline/20260421-221441/pilot_length_snr_comparison.png)

## 1. What This Repository Actually Implements

Implemented now:

- physically motivated synthetic dataset generation,
- balanced SNR sampling,
- deterministic RIS pilot/codebook generation,
- RIS phase quantization,
- PyTorch CNN training,
- LS baseline evaluation,
- plots, checkpoints, and experiment summaries,
- tests for both data generation and training.

Not implemented yet:

- OMP or other compressed sensing baselines,
- direct BS-user path,
- wideband/OFDM channels,
- multi-user setups,
- hybrid beamforming,
- hardware impairment modeling.

That means the current repository is best described as:

> a clean RIS-assisted mmWave dataset generator plus a compact CNN-vs-LS estimation benchmark.

## 2. Repository Structure

- [README.md](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/README.md)
  Project overview, math, training logic, and exact hyperparameters.
- [RESULTS.md](RESULTS.md)
  In-depth explanation of the stored reference experiment in `data/runs/cnn_baseline/20260421-221441`, including plots, metrics, and technical interpretation.
- [Brif.md](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/Brif.md)
  Assignment brief and project framing notes.
- [configs/dataset_small.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_small.yaml)
  Default small dataset preset.
- [configs/dataset_large.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_large.yaml)
  Larger array preset.
- [configs/training_cnn.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/training_cnn.yaml)
  Default CNN training preset.
- [scripts/generate_dataset.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/scripts/generate_dataset.py)
  CLI entry point for dataset generation.
- [scripts/train_cnn.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/scripts/train_cnn.py)
  CLI entry point for CNN training and evaluation.
- [src/ris_dataset/config.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/config.py)
  Dataset configuration dataclasses and validation.
- [src/ris_dataset/geometry.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/geometry.py)
  User placement and geometry utilities.
- [src/ris_dataset/channels.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/channels.py)
  Sparse geometric BS-RIS and RIS-UE channel generation.
- [src/ris_dataset/pilots.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/pilots.py)
  RIS phase codebook generation and phase quantization.
- [src/ris_dataset/generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/generator.py)
  Split generation, sample generation, and LS estimator.
- [src/ris_dataset/io.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_dataset/io.py)
  Complex-to-real serialization helpers.
- [src/ris_training/config.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/config.py)
  Training config parsing and CLI overrides.
- [src/ris_training/data.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/data.py)
  `.npz` loading, normalization, and PyTorch datasets.
- [src/ris_training/model.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/model.py)
  CNN model definition.
- [src/ris_training/trainer.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/trainer.py)
  Training loop, checkpointing, prediction, and evaluation.
- [src/ris_training/metrics.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/metrics.py)
  MSE/NMSE computation and SNR-wise summaries.
- [src/ris_training/plotting.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/plotting.py)
  Publication-style experiment plots.
- [tests/test_generator.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/tests/test_generator.py)
  Dataset generation and LS sanity tests.
- [tests/test_training.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/tests/test_training.py)
  Loader, shape, metric, and training smoke tests.

## 3. Problem Setup

The modeled communication path is:

`BS -> RIS -> User`

The direct BS-user path is disabled on purpose in this version so that the learning target stays clean:

- the model only needs to estimate the RIS-assisted cascaded channel,
- the pilot observations directly reflect the BS-RIS-user cascade,
- the reduced-pilot study is easier to interpret.

System nodes:

- BS with `M` antennas,
- RIS with `N` passive reflecting elements,
- one single-antenna user.

For the default small preset:

- `M = 8`,
- `N = 16`.

## 4. Exact Dataset and Channel Model

### 4.1 Array Model

Both the BS and RIS are modeled as uniform planar arrays (UPAs).

For an array with `N_r` rows and `N_c` columns, the steering vector used by the generator is:

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

where:

- `phi` is azimuth,
- `theta` is elevation,
- `d / lambda` is element spacing in wavelengths.

Default value:

- `element_spacing_lambda = 0.5`

So the arrays use half-wavelength spacing.

### 4.2 Geometry

Default geometry from [configs/dataset_small.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_small.yaml):

- BS position: `(0.0, 0.0, 8.0)` m
- RIS position: `(20.0, 0.0, 5.0)` m
- UE height: `1.5` m
- UE radius range: `5.0` m to `25.0` m
- UE azimuth range: `-60.0 deg` to `60.0 deg`

This means:

- BS and RIS stay fixed,
- the user position changes from sample to sample,
- geometry-driven LoS path lengths and angles also change from sample to sample.

### 4.3 Sparse BS-RIS Channel

The BS-RIS link is generated as a sparse geometric MIMO channel:

```math
\mathbf{G}
=
\sum_{\ell=1}^{L_{BR}}
\alpha_{\ell}
\mathbf{a}_{BS}(\phi^{rx}_{\ell}, \theta^{rx}_{\ell})
\mathbf{a}_{RIS}(\phi^{tx}_{\ell}, \theta^{tx}_{\ell})^H
```

Default BS-RIS path counts:

- LoS paths: `1`
- NLoS paths: `2`
- total: `3`

### 4.4 Sparse RIS-UE Channel

The RIS-UE link is generated as:

```math
\mathbf{h}_{RU}
=
\sum_{p=1}^{L_{RU}}
\beta_p
\mathbf{a}_{RIS}(\phi_p,\theta_p)
```

Default RIS-UE path counts:

- LoS paths: `1`
- NLoS paths: `1`
- total: `2`

### 4.5 Cascaded Channel Label

The learning target is the cascaded channel:

```math
\mathbf{H}_c = \mathbf{G} \operatorname{diag}(\mathbf{h}_{RU})
```

Shapes:

- `G`: `M x N`
- `diag(h_RU)`: `N x N`
- `H_c`: `M x N`

In code this is computed efficiently as element-wise column scaling:

```python
cascaded_channel = g_br * h_ru[np.newaxis, :]
```

because multiplying each column of `G` by one RIS-user coefficient is equivalent to right-multiplying by `diag(h_RU)`.

### 4.6 Pilot Observation Model

The clean received pilot matrix is:

```math
\mathbf{Y}_{clean} = \mathbf{W}^H \mathbf{H}_c \mathbf{\Omega}
```

In this repository:

- `W = I_M`,
- pilot symbols are fixed to `1`.

So the implemented observation equation becomes:

```math
\mathbf{Y}_{clean} = \mathbf{H}_c \mathbf{\Omega}
```

where:

- `Y_clean` has shape `M x Q`,
- `Omega` has shape `N x Q`,
- `Q` is the pilot length.

Noisy observations are:

```math
\mathbf{Y} = \mathbf{Y}_{clean} + \mathbf{N}
```

with circular complex Gaussian noise.

### 4.7 Per-Sample SNR Control

For every sample, the code measures the clean observation power:

```math
P_s = \frac{1}{MQ}\|\mathbf{Y}_{clean}\|_F^2
```

and sets the noise variance as:

```math
\sigma_n^2 = \frac{P_s}{10^{\text{SNR}_{dB}/10}}
```

So the dataset does not use one fixed global noise variance. Instead:

- each sample gets its own noise variance,
- the requested SNR is enforced relative to that sample's clean observation power.

### 4.8 RIS Codebook and Quantization

The RIS control matrix `Omega` is deterministic and DFT-based.

If `Q <= N`:

- use the first `Q` columns of the normalized `N x N` DFT matrix.

If `Q > N`:

- use the first `N` rows of the normalized `Q x Q` DFT matrix.

Phase control defaults:

- `ideal_continuous_phase = false`
- `bits = 2`

So by default the RIS phases are 2-bit quantized to:

```math
\left\{
0,\frac{\pi}{2},\pi,\frac{3\pi}{2}
\right\}
```

## 5. Exact Dataset Hyperparameters

### 5.1 Default Small Dataset

From [configs/dataset_small.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_small.yaml):

| Setting | Exact value |
| --- | --- |
| Carrier frequency | `28000000000.0 Hz` (`28 GHz`) |
| Element spacing | `0.5 lambda` |
| BS array | `2 x 4` |
| BS antennas | `8` |
| RIS array | `4 x 4` |
| RIS elements | `16` |
| Direct path | `false` |
| BS-RIS LoS paths | `1` |
| BS-RIS NLoS paths | `2` |
| RIS-UE LoS paths | `1` |
| RIS-UE NLoS paths | `1` |
| LoS path-loss exponent | `2.1` |
| LoS shadowing std | `3.6 dB` |
| NLoS path-loss exponent | `3.4` |
| NLoS shadowing std | `9.7 dB` |
| LoS gain variance | `sigma_los_sq = 1.0` |
| NLoS gain variance | `sigma_nlos_sq = 0.01` |
| NLoS distance scale min | `1.05` |
| NLoS distance scale max | `1.5` |
| Pilot lengths | `[8, 12, 16, 24, 32]` |
| SNR values (dB) | `[0, 5, 10, 15, 20]` |
| Train samples | `8000` |
| Validation samples | `1000` |
| Test samples | `1000` |

Per pilot length, the default dataset size is:

- total samples: `10000`
- train: `8000`
- val: `1000`
- test: `1000`

Across all five pilot lengths, the total number of saved examples is:

- `50000`

### 5.2 Default Large Dataset

From [configs/dataset_large.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/dataset_large.yaml):

- BS array: `4 x 4` -> `16` antennas
- RIS array: `4 x 8` -> `32` elements
- same pilot lengths, SNR grid, geometry style, and path modeling structure

## 6. What Is Stored in Each `.npz`

Each split file stores:

- `observations`
- `channel`
- `omega`
- `snr_db`
- `user_xyz`
- `distances`
- `channel_norm`
- `seed`

For pilot length `Q`, the saved tensor shapes are:

- `observations`: `[num_samples, Q, M, 2]`
- `channel`: `[num_samples, M, N, 2]`
- `omega`: `[num_samples, N, Q, 2]`
- `snr_db`: `[num_samples]`
- `user_xyz`: `[num_samples, 3]`
- `distances`: `[num_samples, 2]`
- `channel_norm`: `[num_samples]`
- `seed`: `[num_samples]`

The last dimension always means:

- `[..., 0] = real part`
- `[..., 1] = imaginary part`

Important layout detail:

- the physical observation is formed as an `M x Q` matrix,
- before saving, it is transposed to `[Q, M]`,
- this makes the pilot axis and BS antenna axis easy to treat as 2D spatial dimensions for the CNN.

## 7. Complete CNN Logic in Depth

This is the core training logic implemented by the code.

### 7.1 Learning Task

The CNN learns a direct regression:

```math
f_{\theta} : \mathbf{Y} \mapsto \hat{\mathbf{H}}_c
```

where:

- input = noisy pilot observation tensor,
- output = estimated cascaded BS-RIS-user channel.

The model does not estimate `G` and `h_RU` separately. It learns the full cascaded channel `H_c` directly.

### 7.2 Input and Target Tensors Seen by the CNN

Raw saved tensors:

- observations: `[batch, Q, M, 2]`
- channel labels: `[batch, M, N, 2]`

Before training, [src/ris_training/data.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/data.py) transposes them to channels-first format:

- input to CNN: `[batch, 2, Q, M]`
- target for CNN: `[batch, 2, M, N]`

So:

- channel `0` is the real part,
- channel `1` is the imaginary part.

For the default small configuration:

- input shape for `Q = 8`: `[batch, 2, 8, 8]`
- input shape for `Q = 12`: `[batch, 2, 12, 8]`
- input shape for `Q = 16`: `[batch, 2, 16, 8]`
- input shape for `Q = 24`: `[batch, 2, 24, 8]`
- input shape for `Q = 32`: `[batch, 2, 32, 8]`
- target shape for every pilot length: `[batch, 2, 8, 16]`

### 7.3 Normalization Logic

Normalization is fit only on the training split.

For both observations and channel labels, the code computes:

- mean over axes `(samples, height, width)`,
- standard deviation over axes `(samples, height, width)`,
- separately for the real and imaginary channels.

That means the normalization statistics are:

- one mean for the real part,
- one mean for the imaginary part,
- one standard deviation for the real part,
- one standard deviation for the imaginary part.

The standardized tensors are:

```math
\tilde{x} = \frac{x - \mu_x}{\sigma_x}, \qquad
\tilde{h} = \frac{h - \mu_h}{\sigma_h}
```

Important implementation detail:

- if a standard deviation is numerically tiny, the code replaces it with `1.0`,
- this avoids division-by-zero or exploding normalization.

Also important:

- training loss is computed on normalized targets,
- NMSE is computed after denormalizing predictions back to the original channel scale.

### 7.4 Exact CNN Architecture

The model is defined in [src/ris_training/model.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/model.py).

It is a compact convolutional regressor with:

- 3 convolution blocks,
- no pooling,
- no stride greater than 1,
- no residual connections,
- one hidden fully connected layer,
- one final linear output layer.

Each convolution block is:

1. `Conv2d(kernel_size=3, padding=1, bias=False)`
2. `BatchNorm2d`
3. `ReLU(inplace=True)`
4. `Dropout2d`

Exact default architecture hyperparameters:

- `conv_channels = (32, 64, 64)`
- `hidden_dim = 256`
- `dropout = 0.1`

Because padding is `1` and the kernel is `3 x 3`, every convolution preserves spatial resolution.

So the feature extractor does not shrink the input map:

- pilot dimension stays `Q`,
- BS antenna dimension stays `M`.

### 7.5 Layer-by-Layer Tensor Flow

For the default small config with `Q = 16`, `M = 8`, `N = 16`:

1. Input observation tensor:
   `[batch, 2, 16, 8]`
2. Conv block 1:
   `[batch, 32, 16, 8]`
3. Conv block 2:
   `[batch, 64, 16, 8]`
4. Conv block 3:
   `[batch, 64, 16, 8]`
5. Flatten:
   `[batch, 64 x 16 x 8] = [batch, 8192]`
6. Fully connected hidden layer:
   `[batch, 256]`
7. Output linear layer:
   `[batch, 2 x 8 x 16] = [batch, 256]`
8. Reshape:
   `[batch, 2, 8, 16]`

General formulas:

- input size to CNN = `2 x Q x M`
- output size from CNN = `2 x M x N`
- flattened feature size after conv stack = `64 x Q x M`

### 7.6 Exact Parameter Count

For the default CNN architecture, the parameter count depends on `Q`, `M`, and `N` because the fully connected layers depend on the flattened feature size and output size.

General parameter-count formula for this exact implementation:

```math
\text{params}
= 56448 + 16384QM + 514MN
```

where:

- `Q` = pilot length,
- `M` = number of BS antennas,
- `N` = number of RIS elements.

For the default small dataset (`M = 8`, `N = 16`):

| Pilot length `Q` | Exact parameter count |
| --- | ---: |
| `8` | `1,170,816` |
| `12` | `1,695,104` |
| `16` | `2,219,392` |
| `24` | `3,267,968` |
| `32` | `4,316,544` |

This happens because:

- the convolutional part is fixed,
- the flattened feature vector grows linearly with pilot length.

### 7.7 What the CNN Is Trying to Learn

Conceptually, the CNN is learning this mapping:

- local spatial patterns across nearby pilots,
- spatial correlations across nearby BS antennas,
- relationships between real and imaginary parts,
- structured distortions introduced by noise and reduced pilot observations,
- how to convert the observation map into the full `M x N` cascaded channel.

Why the model is set up this way:

- the input is naturally a 2D map: pilot index by BS antenna,
- the target is naturally a 2D map: BS antenna by RIS element,
- convolutions can exploit local structure before the dense layers reconstruct the full channel.

### 7.8 What This CNN Adds Beyond Plain LS

The main addition of the CNN is not a new analytical estimator formula. The addition is:

- it learns from many synthetic examples,
- it can exploit the structure of the training distribution,
- it can reduce the error caused by underdetermined or noisy pilot observations,
- it can outperform LS especially when pilot length is small.

LS treats each test sample independently through a pseudoinverse. The CNN instead uses:

- dataset-level statistical regularities,
- sparse-channel structure implicitly present in the training data,
- learned nonlinear denoising and reconstruction.

That is the central research improvement in this project.

## 8. Exact Training Logic

### 8.1 Dataloaders

For each pilot length:

- one train dataloader is created,
- one validation dataloader is created,
- one test dataloader is created.

Exact defaults:

- `batch_size = 128`
- `shuffle = True` for train
- `shuffle = False` for val/test
- `num_workers = 0`

### 8.2 Optimizer, Loss, and Device

Exact defaults from [configs/training_cnn.yaml](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/configs/training_cnn.yaml):

| Setting | Exact value |
| --- | --- |
| Experiment name | `cnn_baseline` |
| Data root | `data/dataset_small` |
| Output root | `data/runs` |
| Pilot lengths | `[8, 12, 16, 24, 32]` |
| Device | `auto` |
| Batch size | `128` |
| Num workers | `0` |
| Epochs | `60` |
| Seed | `2026` |
| Optimizer | `AdamW` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Early stopping patience | `8` |
| Early stopping min delta | `0.0` |
| Conv channels | `[32, 64, 64]` |
| Hidden dimension | `256` |
| Dropout | `0.1` |
| Plot examples | `3` |

Exact device-selection logic:

- if `--device cpu`, use CPU,
- if `--device mps`, require Apple Metal/MPS,
- if `--device auto`, use MPS when available, otherwise CPU.

### 8.3 Loss and Metric

Training loss:

- `nn.MSELoss()` on normalized target tensors.

Training/validation metric tracked every epoch:

- NMSE computed after denormalizing predictions and targets back to physical channel scale.

Batch NMSE is computed as:

```math
\text{NMSE}
=
\frac{\|\hat{H} - H\|_F^2}{\|H\|_F^2}
```

with safe clamping to avoid division by zero.

### 8.4 Epoch Logic

For each epoch, [src/ris_training/trainer.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/src/ris_training/trainer.py) does:

1. run one full training pass,
2. run one full validation pass,
3. record:
   - learning rate,
   - train loss,
   - val loss,
   - train NMSE,
   - val NMSE,
   - train NMSE in dB,
   - val NMSE in dB,
4. save `last.pt`,
5. if validation NMSE improves by more than `min_delta`, save `best.pt`,
6. update early stopping counter,
7. stop if no improvement is seen for `patience` consecutive epochs.

Important exact behavior:

- there is no learning-rate scheduler,
- there is no mixed precision,
- there is no gradient clipping,
- there is no data augmentation,
- there is no warmup,
- model selection is based on validation NMSE, not validation loss.

### 8.5 Early Stopping Rule

Improvement rule:

```python
if best_val_nmse - val_nmse > min_delta:
```

With the default config:

- `min_delta = 0.0`

So any strict improvement in validation NMSE resets patience.

Stopping rule:

- stop once `epochs_without_improvement >= 8`.

### 8.6 Evaluation After Training

After training stops:

1. reload `best.pt`,
2. predict validation and test sets with the CNN,
3. denormalize predictions,
4. compute CNN MSE/NMSE/NMSE(dB),
5. compute LS predictions on the same splits,
6. compute LS MSE/NMSE/NMSE(dB),
7. group results by SNR,
8. save plots and summary files.

The final comparison is therefore:

- CNN on validation and test,
- LS on validation and test,
- mean metrics and per-SNR metrics for both.

## 9. Least Squares Baseline

The repository uses a transparent LS estimator as the classical baseline:

```math
\hat{\mathbf{H}}_{LS}
=
\mathbf{Y}
\mathbf{\Omega}^H
\left(
\mathbf{\Omega}\mathbf{\Omega}^H
\right)^{\dagger}
```

In the evaluation code, the implementation handles:

- one shared `Omega` for all samples when possible,
- or per-sample `Omega` if needed.

Why LS is useful here:

- it is easy to understand,
- it gives a deterministic baseline,
- it shows how much performance the CNN gains over a classical pseudoinverse estimator.

## 10. End-to-End Pipeline Summary

For one sample, the full pipeline is:

1. sample a user location,
2. compute geometry-derived LoS quantities,
3. generate BS-RIS sparse channel `G`,
4. generate RIS-UE sparse channel `h_RU`,
5. form cascaded channel `H_c`,
6. build the DFT-based RIS pilot matrix `Omega`,
7. compute clean pilots `Y_clean = H_c Omega`,
8. compute noise variance from requested SNR,
9. add complex Gaussian noise,
10. save observation and channel tensors,
11. normalize train/val/test using train-split statistics,
12. train CNN on normalized data,
13. select best checkpoint using validation NMSE,
14. evaluate CNN and LS on held-out data,
15. save plots and summaries.

## 11. Commands to Run

### 11.1 Create Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Install PyTorch separately using the correct command for your platform from the official guide:

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)

Then install the remaining dependencies:

```bash
.venv/bin/python -m pip install numpy PyYAML matplotlib pytest
```

### 11.2 Generate Default Dataset

```bash
.venv/bin/python scripts/generate_dataset.py \
  --config configs/dataset_small.yaml \
  --out data/ris_mmwave_v1 \
  --seed 2026
```

### 11.3 Generate Large Dataset

```bash
.venv/bin/python scripts/generate_dataset.py \
  --config configs/dataset_large.yaml \
  --out data/ris_mmwave_v1_large \
  --seed 2026
```

### 11.4 Train One Pilot Length

Example for `Q = 16`:

```bash
.venv/bin/python scripts/train_cnn.py \
  --config configs/training_cnn.yaml \
  --data-root data/ris_mmwave_v1 \
  --pilot-length 16 \
  --device auto
```

### 11.5 Train All Pilot Lengths

```bash
.venv/bin/python scripts/train_cnn.py \
  --config configs/training_cnn.yaml \
  --data-root data/ris_mmwave_v1 \
  --pilot-length all \
  --device auto
```

Useful CLI overrides supported by [scripts/train_cnn.py](/Users/piyush/dev/project/Wirless Communiaction/RIS-Channel-Estimation-Using-CNN-/scripts/train_cnn.py):

- `--epochs`
- `--batch-size`
- `--seed`
- `--lr`
- `--patience`
- `--device`
- `--data-root`
- `--output-root`

## 12. Output Directory Layout

After dataset generation:

```text
data/ris_mmwave_v1/
├── manifest.json
├── pilots_8/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── pilots_12/
├── pilots_16/
├── pilots_24/
└── pilots_32/
```

After training:

```text
data/runs/
└── cnn_baseline/
    └── YYYYMMDD-HHMMSS/
        ├── pilot_length_summary.csv
        ├── pilot_length_snr_comparison.png
        ├── pilot_length_vs_gain.png
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

Per-pilot outputs:

- `best.pt`: best validation-NMSE checkpoint,
- `last.pt`: final-epoch checkpoint,
- `history.csv`: epoch-wise train/val loss and NMSE,
- `metrics.json`: CNN and LS summaries,
- `normalization.json`: mean/std used for normalization,
- `predictions.npz`: saved target and predictions,
- `plots/*.png`: report-ready figures.

Cross-pilot outputs:

- `pilot_length_summary.csv`
- `pilot_length_vs_nmse.png`
- `pilot_length_vs_gain.png`
- `pilot_length_snr_comparison.png`
- `experiment_summary.md`

## 13. Tests and Verification

The tests check:

- dataset smoke generation,
- reproducibility with fixed seed,
- finite numerical outputs,
- SNR consistency,
- LS sanity behavior,
- normalization correctness,
- CNN forward output shapes,
- metric grouping by SNR,
- training smoke run artifact creation,
- multi-pilot comparison artifact creation.

Run them with:

```bash
.venv/bin/pytest -q
```

## 14. Assumptions and Limits

Current assumptions:

- single-user only,
- narrowband only,
- no direct BS-user path,
- passive RIS,
- identity BS combiner,
- unit pilot symbols,
- no mobility,
- no frequency selectivity,
- no hardware impairments.

These are simplifications, but they are deliberate. They keep the problem narrow enough that:

- the dataset is easy to regenerate,
- the estimator target is clear,
- CNN-vs-LS comparisons are interpretable,
- reduced-pilot experiments are easy to present in a report.

## 15. Why This Is a Good Project-Scale CNN

This CNN is intentionally small and practical.

Why this architecture makes sense for the assignment:

- it is easy to explain,
- it has a clear input-output mapping,
- it uses standard deep learning blocks only,
- it is strong enough to learn denoising and channel reconstruction,
- it is still light enough to train on a laptop-sized setup.

In other words:

- the contribution is not architectural novelty,
- the contribution is showing that a compact learned estimator can beat LS under reduced pilot budgets on a realistic synthetic RIS/mmWave dataset.

## 16. References That Informed the Design

These references informed the dataset design and project framing. The code is inspired by them, but it is not a line-by-line reproduction of any single paper.

1. Jiguang He, Henk Wymeersch, Marco Di Renzo, and Markku Juntti, *Learning to Estimate RIS-Aided mmWave Channels*, IEEE Wireless Communications Letters, vol. 11, no. 4, pp. 841-845, 2022.
   Link: [https://doi.org/10.1109/LWC.2022.3147250](https://doi.org/10.1109/LWC.2022.3147250)

2. Asmaa Abdallah, Abdulkadir Celik, Mohammad M. Mansour, and Ahmed M. Eltawil, *RIS-Aided mmWave MIMO Channel Estimation Using Deep Learning and Compressive Sensing*, IEEE Transactions on Wireless Communications, vol. 22, no. 5, pp. 3503-3521, 2023.
   Link: [https://doi.org/10.1109/TWC.2022.3219140](https://doi.org/10.1109/TWC.2022.3219140)

3. George R. MacCartney, Junhong Zhang, Shuai Nie, and Theodore S. Rappaport, *Path Loss Models for 5G Millimeter Wave Propagation Channels in Urban Microcells*, IEEE GLOBECOM 2013, pp. 3948-3953, 2013.
   Link: [https://doi.org/10.1109/GLOCOM.2013.6831690](https://doi.org/10.1109/GLOCOM.2013.6831690)

## 17. Final One-Paragraph Summary

This repository generates a reproducible RIS-assisted mmWave dataset, trains a compact CNN to estimate the cascaded BS-RIS-user channel from noisy reduced-pilot observations, and compares that learned estimator against LS using NMSE and SNR-wise analysis. The README now documents the full CNN logic, the exact tensor shapes, the exact optimizer and early-stopping settings, and the exact dataset/training hyperparameters used by the code.
