import argparse
import configparser
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from string import Template

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utils import demix_track, demix_track_demucs, get_model_from_config

BASE_DIR = Path(__file__).absolute().parent


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:0")
    else:
        print("CUDA is not avilable. Run inference on CPU. It will be very slow...")
        return "cpu"


device = get_device()


@dataclass
class Config:
    out_dir: Path

    vocal_model_type: str
    vocal_model_config: Path
    vocal_model_checkpoint: Path

    other_model_type: str
    other_model_config: Path
    other_model_checkpoint: Path

    drums_model_type: str
    drums_model_config: Path
    drums_model_checkpoint: Path

    bass_model_type: str
    bass_model_config: Path
    bass_model_checkpoint: Path

    @staticmethod
    def config_path():
        import platformdirs

        data_dir = platformdirs.user_data_dir(
            "MSST",
            "jamesWalker55",
            roaming=True,
            ensure_exists=True,
        )

        return Path(data_dir) / "config.ini"

    @staticmethod
    def default_config_path():
        path = BASE_DIR / "config.default.ini"
        assert path.exists()
        return path

    @staticmethod
    def resolve_path(path: str) -> Path:
        return Path(Template(path).safe_substitute({"msst": BASE_DIR}))

    @classmethod
    def load_config(cls):
        config_path = cls.config_path()
        default_config_path = cls.default_config_path()
        if not config_path.exists():
            print(f"Creating config file at: {config_path}")
            shutil.copyfile(default_config_path, config_path)

        print(f"Reading config from: {config_path}")
        c = configparser.ConfigParser()
        c.read(default_config_path)
        c.read(config_path)

        return cls(
            cls.resolve_path(c["paths"]["out_dir"]),
            c["vocal_model"]["type"],
            cls.resolve_path(c["vocal_model"]["config"]),
            cls.resolve_path(c["vocal_model"]["checkpoint"]),
            c["other_model"]["type"],
            cls.resolve_path(c["other_model"]["config"]),
            cls.resolve_path(c["other_model"]["checkpoint"]),
            c["drums_model"]["type"],
            cls.resolve_path(c["drums_model"]["config"]),
            cls.resolve_path(c["drums_model"]["checkpoint"]),
            c["bass_model"]["type"],
            cls.resolve_path(c["bass_model"]["config"]),
            cls.resolve_path(c["bass_model"]["checkpoint"]),
        )


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


def parse_args(config: Config | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", type=Path, help="input files to process")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.out_dir,
        help="output directory",
    )
    parser.add_argument(
        "-s",
        "--skip-stems",
        action="store_true",
        help="skip extracting drums, bass, and other stems",
    )
    parser.add_argument(
        "-i",
        "--no-vocals",
        action="store_true",
        help="if the input tracks don't have vocals, use this option to skip vocal extraction",
    )
    args = parser.parse_args()

    input_paths: list[Path] = args.input
    out_dir: Path = args.out_dir
    skip_stems: bool = args.skip_stems
    no_vocals: bool = args.no_vocals

    return (input_paths, out_dir, skip_stems, no_vocals)


def main():
    config = Config.load_config()

    input_paths, out_dir, skip_stems, no_vocals = parse_args(config)

    print("Total files found: {}".format(len(input_paths)))

    input_paths = tqdm(input_paths)

    # Vocal model: BS Roformer (viperx edition)
    vocal_model = Model(
        config.vocal_model_type,
        config.vocal_model_config,
        config.vocal_model_checkpoint,
    )
    # Single stem model: BS Roformer (viperx edition)
    other_model = Model(
        config.other_model_type,
        config.other_model_config,
        config.other_model_checkpoint,
    )
    # Single stem model: HTDemucs4 FT Drums
    drums_model = Model(
        config.drums_model_type,
        config.drums_model_config,
        config.drums_model_checkpoint,
    )
    # Single stem model: HTDemucs4 FT Bass
    bass_model = Model(
        config.bass_model_type,
        config.bass_model_config,
        config.bass_model_checkpoint,
    )

    with measure_time("Load models"):
        if not no_vocals:
            vocal_model.load_model()
        if not skip_stems:
            other_model.load_model()
            drums_model.load_model()
            bass_model.load_model()

    with measure_time("Elapsed time"):
        for path in input_paths:
            input_paths.set_postfix({"track": path.name})

            try:
                mix, sr = load_audio(path)
            except Exception as e:
                print(f"Can't read track: {path}")
                print(f"Error message: {e}")
                continue

            def save_audio_to_out_dir(name: str, mix: np.ndarray):
                output_name = f"{path.stem}_{name}.flac"
                output_path = out_dir / output_name
                save_audio(output_path, mix, sr)

            if not no_vocals:
                vocals = vocal_model.demix(mix)["vocals"]
                save_audio_to_out_dir("vocals", vocals)
                inst = mix - vocals
            else:
                inst = mix

            save_audio_to_out_dir("inst", inst)

            if skip_stems:
                continue

            other = other_model.demix(inst)["other"]

            save_audio_to_out_dir("other", other)

            drum_and_bass = inst - other
            bass = bass_model.demix(drum_and_bass)["bass"]

            save_audio_to_out_dir("bass", bass)

            drums = drums_model.demix(drum_and_bass - bass)["drums"]
            residual = drum_and_bass - bass - drums

            save_audio_to_out_dir("drums", drums)
            save_audio_to_out_dir("residual", residual)


if __name__ == "__main__":
    main()
