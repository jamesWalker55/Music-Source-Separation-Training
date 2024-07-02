# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import argparse
import time
from typing import Literal
import librosa
from tqdm import tqdm
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import demix_track, demix_track_demucs, get_model_from_config

import warnings

warnings.filterwarnings("ignore")


MODEL_TYPE_CHOICES = (
    "mdx23c",
    "htdemucs",
    "segm_models",
    "mel_band_roformer",
    "bs_roformer",
    "swin_upernet",
    "bandit",
)
ModelType = Literal[
    "mdx23c",
    "htdemucs",
    "segm_models",
    "mel_band_roformer",
    "bs_roformer",
    "swin_upernet",
    "bandit",
]


def run_folder(
    model,
    config,
    *,
    input_paths: str = None,
    store_dir: str = None,
    model_type: ModelType = None,
    extract_instrumental: bool = False,
    device=None,
    verbose=False,
):
    start_time = time.time()
    model.eval()
    all_mixtures_path = input_paths
    print("Total files found: {}".format(len(all_mixtures_path)))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    for path in all_mixtures_path:
        if not verbose:
            all_mixtures_path.set_postfix({"track": os.path.basename(path)})
        try:
            # mix, sr = sf.read(path)
            mix, sr = librosa.load(path, sr=44100, mono=False)
            mix = mix.T
        except Exception as e:
            print("Can read track: {}".format(path))
            print("Error message: {}".format(str(e)))
            continue

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if model_type == "htdemucs":
            res = demix_track_demucs(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)
        for instr in instruments:
            sf.write(
                "{}/{}_{}.wav".format(store_dir, os.path.basename(path)[:-4], instr),
                res[instr].T,
                sr,
                subtype="FLOAT",
            )

        if "vocals" in instruments and extract_instrumental:
            instrum_file_name = "{}/{}_{}.wav".format(
                store_dir, os.path.basename(path)[:-4], "instrumental"
            )
            sf.write(instrum_file_name, mix - res["vocals"].T, sr, subtype="FLOAT")

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=MODEL_TYPE_CHOICES,
        help=f"One of {', '.join(MODEL_TYPE_CHOICES)}",
        required=True,
    )
    parser.add_argument(
        "--config_path", type=str, help="path to config file", required=True
    )
    parser.add_argument(
        "--start_check_point",
        type=str,
        help="Initial checkpoint to valid weights",
        required=True,
    )
    parser.add_argument(
        "inputs",
        type=str,
        help="paths to mixtures to process",
        nargs="+",
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        help="path to store results as wav file",
        required=True,
    )
    parser.add_argument(
        "--device_ids", nargs="+", type=int, default=0, help="list of gpu ids"
    )
    parser.add_argument(
        "--extract_instrumental",
        action="store_true",
        help="invert vocals to get instrumental if provided",
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != "":
        print("Start from checkpoint: {}".format(args.start_check_point))
        state_dict = torch.load(args.start_check_point)
        if args.model_type == "htdemucs":
            # Fix for htdemucs pround etrained models
            if "state" in state_dict:
                state_dict = state_dict["state"]
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids) == int:
            device = torch.device(f"cuda:{device_ids}")
            model = model.to(device)
        else:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = "cpu"
        print("CUDA is not avilable. Run inference on CPU. It will be very slow...")
        model = model.to(device)

    run_folder(
        model,
        config,
        input_paths=args.inputs,
        store_dir=args.store_dir,
        model_type=args.model_type,
        extract_instrumental=args.extract_instrumental,
        device=device,
        verbose=False,
    )


if __name__ == "__main__":
    main()
