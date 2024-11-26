import argparse
import configparser
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from string import Template
import tomllib
from typing import NamedTuple

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


class ModelConfig(NamedTuple):
    type: str
    config: Path
    checkpoint: Path

    @classmethod
    def from_dict(cls, x: dict[str, str]):
        return cls(
            x["type"],
            Config.resolve_path(x["config"]),
            Config.resolve_path(x["checkpoint"]),
        )


class ModelOutputConfig(NamedTuple):
    instrument: str | None
    output_name: str

    @classmethod
    def from_dict(cls, x: dict[str, str]):
        return cls(
            x["instrument"],
            x["output_name"],
        )


@dataclass
class Config:
    out_dir: Path

    vocal_model: ModelConfig
    other_model: ModelConfig
    drums_model: ModelConfig
    bass_model: ModelConfig
    extra_models: dict[str, tuple[ModelConfig, list[ModelOutputConfig]]]

    @staticmethod
    def config_path():
        import platformdirs

        data_dir = platformdirs.user_data_dir(
            "MSST",
            "jamesWalker55",
            roaming=True,
            ensure_exists=True,
        )

        return Path(data_dir) / "config.toml"

    @staticmethod
    def default_config_path():
        path = BASE_DIR / "config.default.toml"
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
        with open(config_path, "rb") as f:
            c = tomllib.load(f)

        extra_models_dict: dict[str, tuple[ModelConfig, list[ModelOutputConfig]]] = {}
        assert isinstance(
            c["extra_models"], list
        ), "extra_models should be a list of model configs"
        for x in c["extra_models"]:
            key = x["key"]
            assert isinstance(key, str), "model key must be a string"
            assert key not in extra_models_dict, f"duplicate extra model key: {key!r}"
            assert key != "demix", "cannot name model as 'demix' as it is reserved"
            model_config = ModelConfig.from_dict(x)
            output_configs = [ModelOutputConfig.from_dict(y) for y in x["outputs"]]
            if len(output_configs) == 0:
                raise Exception(f"Model config has no outputs: {key!r}")
            if len(set(_.output_name for _ in output_configs)) != len(output_configs):
                raise Exception(f"Model config has duplicate output names: {key!r}")
            if len(set(_.instrument for _ in output_configs)) != len(output_configs):
                raise Exception(
                    f"Model config has duplicate output instruments: {key!r}"
                )
            extra_models_dict[key] = (model_config, output_configs)

        return cls(
            cls.resolve_path(c["paths"]["out_dir"]),
            ModelConfig.from_dict(c["vocal_model"]),
            ModelConfig.from_dict(c["other_model"]),
            ModelConfig.from_dict(c["drums_model"]),
            ModelConfig.from_dict(c["bass_model"]),
            extra_models_dict,
        )


@dataclass
class Model:
    type: str
    config_path: str
    checkpoint_path: str

    # I don't know how to type these
    # model: ???
    # config: ???

    @classmethod
    def from_config(cls, x: ModelConfig):
        return cls(x.type, str(x.config), str(x.checkpoint))

    def _load_model(self):
        print(f"Loading {self.type} model: {self.checkpoint_path}")
        with measure_time("Loaded in"):
            model, config = get_model_from_config(self.type, self.config_path)
            assert model

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

    def load_model_if_not_loaded(self):
        if not hasattr(self, "model"):
            self._load_model()

    def demix(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        self.load_model_if_not_loaded()

        mix = torch.tensor(mix.T, dtype=torch.float32)  # type: ignore
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


def load_audio(path: str | Path):
    # mix, sr = sf.read(path)
    mix, sr = librosa.load(path, sr=44100, mono=False)
    mix = mix.T

    # Convert mono to stereo if needed
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=-1)

    return mix, sr


def save_audio(path: str | Path, mix: np.ndarray, sr):
    path = str(path)
    subtype = "FLOAT" if path.lower().endswith("wav") else None
    sf.write(path, mix, sr, subtype=subtype)


def parse_args(config: Config):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.out_dir,
        help="output directory",
    )
    parser.add_argument(
        "-w",
        "--wav",
        action="store_true",
        help="save using WAV format instead of FLAC",
    )

    subparsers = parser.add_subparsers(required=True)

    sp = subparsers.add_parser("demix", help="classic 4-stem demixing mode")
    sp.set_defaults(key="demix")
    sp.add_argument("input", nargs="+", type=Path, help="input files to process")
    sp.add_argument(
        "-s",
        "--skip-stems",
        action="store_true",
        help="skip extracting drums, bass, and other stems",
    )
    sp.add_argument(
        "-i",
        "--no-vocals",
        action="store_true",
        help="if the input tracks don't have vocals, use this option to skip vocal extraction",
    )

    for key in config.extra_models.keys():
        sp = subparsers.add_parser(key)
        sp.set_defaults(key=key)
        sp.add_argument("input", nargs="+", type=Path, help="input files to process")

    args = parser.parse_args()

    input_paths: list[Path] = args.input
    out_dir: Path = args.out_dir
    save_wav: bool = args.wav
    model_key: str = args.key
    if model_key == "demix":
        skip_stems: bool = args.skip_stems
        no_vocals: bool = args.no_vocals
    else:
        skip_stems: bool = False
        no_vocals: bool = False

    return (input_paths, out_dir, save_wav, model_key, skip_stems, no_vocals)


def main():
    config = Config.load_config()

    input_paths, out_dir, save_wav, model_key, skip_stems, no_vocals = parse_args(
        config
    )

    def save_audio_to_out_dir(name: str, mix: np.ndarray):
        if save_wav:
            output_name = f"{path.stem}_{name}.wav"
        else:
            output_name = f"{path.stem}_{name}.flac"
        output_path = out_dir / output_name
        save_audio(output_path, mix, sr)

    print("Total files found: {}".format(len(input_paths)))

    input_paths = tqdm(input_paths)

    if model_key == "demix":
        # Vocal model: BS Roformer (viperx edition)
        vocal_model = Model.from_config(config.vocal_model)
        # Single stem model: BS Roformer (viperx edition)
        other_model = Model.from_config(config.other_model)
        # Single stem model: HTDemucs4 FT Drums
        drums_model = Model.from_config(config.drums_model)
        # Single stem model: HTDemucs4 FT Bass
        bass_model = Model.from_config(config.bass_model)

        with measure_time("Load models"):
            if not no_vocals:
                vocal_model.load_model_if_not_loaded()
            if not skip_stems:
                other_model.load_model_if_not_loaded()
                drums_model.load_model_if_not_loaded()
                bass_model.load_model_if_not_loaded()

        with measure_time("Elapsed time"):
            for path in input_paths:
                input_paths.set_postfix({"track": path.name})

                try:
                    mix, sr = load_audio(path)
                except Exception as e:
                    print(f"Can't read track: {path}")
                    print(f"Error message: {e}")
                    continue

                if not no_vocals:
                    vocals = vocal_model.demix(mix)["vocals"]
                    save_audio_to_out_dir("vocals", vocals)
                    inst = mix - vocals
                    save_audio_to_out_dir("inst", inst)
                else:
                    inst = mix

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
    else:
        model_config, model_outputs = config.extra_models[model_key]
        model = Model.from_config(model_config)
        model.load_model_if_not_loaded()

        with measure_time("Elapsed time"):
            for path in input_paths:
                input_paths.set_postfix({"track": path.name})

                try:
                    mix, sr = load_audio(path)
                except Exception as e:
                    print(f"Can't read track: {path}")
                    print(f"Error message: {e}")
                    continue

                demixed = model.demix(mix)
                residual_output = None

                for output in model_outputs:
                    if output.instrument is None:
                        residual_output = output
                        continue

                    if output.instrument not in demixed:
                        raise Exception(
                            f"Model config instrument {output.instrument!r} does not exist in demixed result (choose from {sorted(demixed.keys())})"
                        )

                    save_audio_to_out_dir(
                        output.output_name,
                        demixed[output.instrument],
                    )

                if residual_output is not None:
                    # calculate residual output
                    residual = mix
                    for output in model_outputs:
                        if output.instrument is None:
                            continue

                        if output.instrument not in demixed:
                            raise Exception(
                                f"Model config instrument {output.instrument!r} does not exist in demixed result (choose from {sorted(demixed.keys())})"
                            )

                        residual = residual - demixed[output.instrument]

                    save_audio_to_out_dir(
                        residual_output.output_name,
                        residual,
                    )


if __name__ == "__main__":
    main()
