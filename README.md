# MSST Separator

An experimental audio stem separator, forked from [ZFTurbo's Music-Source-Separation-Training repo](https://github.com/ZFTurbo/Music-Source-Separation-Training).

Will separate an audio file to the following stems:

- `vocals`: Just the vocals
- `inst`: Everything except the vocals, i.e. the "instrumental" mix
- `drums`: Just the drums
- `bass`: Just the bass
- `other`: Everything except vocals, drums, bass
- `residual`: Leftover audio that isn't included in the stems above, usually contains noise or hard-to-classify audio elements

## Install

Installing with rye is recommended.

First, add a `cu121` PyTorch soruce to Rye's config at `~/.rye/config.toml`:

```toml
[[sources]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
```

Then install this using Rye as a tool:

```bash
rye install --git https://github.com/jamesWalker55/Music-Source-Separation-Training.git music-source-separation-training

# Installed scripts:
#   - msst
```

Now create a config file at your local data directory. On Windows it should be `~/AppData/Roaming/jamesWalker55/MSST/config.ini`. For other systems, run the tool and it will print out the path to the config file. My example configuration:

```ini
[paths]
; Output directory
out_dir=D:/Audio Samples/_Acapella/MSST

; Vocal model: BS Roformer (viperx edition)
[vocal_model]
type=bs_roformer
config=$msst/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml
checkpoint=C:/Programs/Music-Source-Separation-Training/results/model_bs_roformer_ep_317_sdr_12.9755.ckpt

; Single stem model: BS Roformer (viperx edition)
[other_model]
type=bs_roformer
config=$msst/configs/viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml
checkpoint=C:/Programs/Music-Source-Separation-Training/results/model_bs_roformer_ep_937_sdr_10.5309.ckpt

; Single stem model: HTDemucs4 FT Drums
[drums_model]
type=htdemucs
config=$msst/configs/config_musdb18_htdemucs.yaml
checkpoint=C:/Programs/Music-Source-Separation-Training/results/f7e0c4bc-ba3fe64a.th

; Single stem model: HTDemucs4 FT Bass
[bass_model]
type=htdemucs
config=$msst/configs/config_musdb18_htdemucs.yaml
checkpoint=C:/Programs/Music-Source-Separation-Training/results/d12395a8-e57c48e6.th
```

## Usage

```
> msst -h
Reading config from: C:\Users\James\AppData\Roaming\jamesWalker55\MSST\config.ini
usage: msst [-h] [--out-dir OUT_DIR] [-s] [-i] input [input ...]

positional arguments:
  input              input files to process

options:
  -h, --help         show this help message and exit
  --out-dir OUT_DIR  output directory
  -s, --skip-stems   skip extracting drums, bass, and other stems
  -i, --no-vocals    if the input tracks don't have vocals, use this option to skip vocal extraction
```
