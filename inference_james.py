import argparse
import glob
import os
from pathlib import Path
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, NamedTuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import demix_track, demix_track_demucs, get_model_from_config


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:0")
    else:
        print("CUDA is not avilable. Run inference on CPU. It will be very slow...")
        return "cpu"


device = get_device()


@dataclass
class Model:
    type: str
    config_path: str
    checkpoint_path: str

    # I don't know how to type these
    # model: ???
    # config: ???

    def load_model(self):
        print(f"Loading {self.type} model: {self.checkpoint_path}")
        model, config = get_model_from_config(self.type, self.config_path)

        state_dict = torch.load(self.checkpoint_path)
        if self.type == "htdemucs":
            # Fix for htdemucs pround etrained models
            if "state" in state_dict:
                state_dict = state_dict["state"]
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        self.model = model
        self.config = config

    def demix(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        mix = torch.tensor(mix.T, dtype=torch.float32)
        if self.type == "htdemucs":
            res = demix_track_demucs(self.config, self.model, mix, device)
        else:
            res = demix_track(self.config, self.model, mix, device)

        for k in res:
            res[k] = res[k].T

        return res


# Vocal model: BS Roformer (viperx edition)
VOCAL_MODEL = Model(
    "bs_roformer",
    "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    "results/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
)

# Single stem model: BS Roformer (viperx edition)
OTHER_MODEL = Model(
    "bs_roformer",
    "configs/viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml",
    "results/model_bs_roformer_ep_937_sdr_10.5309.ckpt",
)

# Single stem model: HTDemucs4 FT Drums
DRUMS_MODEL = Model(
    "htdemucs", "configs/config_musdb18_htdemucs.yaml", "results/f7e0c4bc-ba3fe64a.th"
)

# Single stem model: HTDemucs4 FT Bass
BASS_MODEL = Model(
    "htdemucs", "configs/config_musdb18_htdemucs.yaml", "results/d12395a8-e57c48e6.th"
)


@contextmanager
def measure_time(text: str):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{text}: {elapsed_time:.2f} sec")


def load_audio(path: str):
    # mix, sr = sf.read(path)
    mix, sr = librosa.load(path, sr=44100, mono=False)
    mix = mix.T

    # Convert mono to stereo if needed
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=-1)

    return mix, sr


def save_audio(path: str, mix: np.ndarray, sr):
    path = str(path)
    subtype = "FLOAT" if path.lower().endswith("wav") else None
    sf.write(path, mix, sr, subtype=subtype)


def main():
    input_paths = [
        R"D:\Audio Samples\_Acapella\MSST\「3時12分 _ TAKU INOUE & 星街すいせい」MUSIC VIDEO LYFciXBcXIQ.opus",
        R"D:\Audio Samples\_Acapella\MSST\【original anime MV】III【hololive⧸宝鐘マリン＆こぼ・かなえる】 lUDPjyfmJrs.opus",
        R"D:\Audio Samples\_Acapella\MSST\05. インターネットが遅いさん (Super-Slow-Internet-san).mp3",
        R"D:\Audio Samples\_Acapella\MSST\Anomalie didn't know I copied him F9YelbEAsww.opus",
        R"D:\Audio Samples\_Acapella\MSST\LiSA 『ハルシネイト』 MUSiC CLiP ABy-evbZtw4.opus",
    ]

    print("Total files found: {}".format(len(input_paths)))

    input_paths = tqdm(input_paths)

    with measure_time("Load models"):
        VOCAL_MODEL.load_model()
        OTHER_MODEL.load_model()
        DRUMS_MODEL.load_model()
        BASS_MODEL.load_model()

    with measure_time("Elapsed time"):
        for path in input_paths:
            path = Path(path)

            input_paths.set_postfix({"track": path.name})

            try:
                mix, sr = load_audio(path)
            except Exception as e:
                print(f"Can't read track: {path}")
                print(f"Error message: {e}")
                continue

            vocals = VOCAL_MODEL.demix(mix)["vocals"]

            output_path = path.with_stem(path.stem + "_vocals").with_suffix(".flac")
            save_audio(output_path, vocals, sr)

            inst = mix - vocals
            other = OTHER_MODEL.demix(inst)["other"]

            output_path = path.with_stem(path.stem + "_other").with_suffix(".flac")
            save_audio(output_path, other, sr)

            drum_and_bass = inst - other
            bass = BASS_MODEL.demix(drum_and_bass)["bass"]

            output_path = path.with_stem(path.stem + "_drum_and_bass").with_suffix(
                ".flac"
            )
            save_audio(output_path, drum_and_bass, sr)

            output_path = path.with_stem(path.stem + "_bass").with_suffix(".flac")
            save_audio(output_path, bass, sr)

            drums = DRUMS_MODEL.demix(drum_and_bass - bass)["drums"]
            residual = drum_and_bass - bass - drums

            output_path = path.with_stem(path.stem + "_drums").with_suffix(".flac")
            save_audio(output_path, drums, sr)

            output_path = path.with_stem(path.stem + "_residual").with_suffix(".flac")
            save_audio(output_path, residual, sr)


if __name__ == "__main__":
    main()
