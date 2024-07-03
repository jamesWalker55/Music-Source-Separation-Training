# https://github.com/ZFTurbo/Music-Source-Separation-Training

STORE_DIR := "D:/Audio Samples/_Acapella/MSST"

# Vocal model: BS Roformer (viperx edition)
# this will also create an instrumental
vocals path:
    rye run python inference_single.py \
        "{{path}}" \
        --flac \
        --extract_instrumental \
        --model_type bs_roformer \
        --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
        --start_check_point results/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
        --store_dir "{{STORE_DIR}}"

# Single stem model: BS Roformer (viperx edition)
# this removes drums and bass only, keeping the rest
# so you should remove vocals first!
other path:
    rye run python inference_single.py \
        "{{path}}" \
        --flac \
        --model_type bs_roformer \
        --config_path configs/viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml \
        --start_check_point results/model_bs_roformer_ep_937_sdr_10.5309.ckpt \
        --store_dir "{{STORE_DIR}}"

# Single stem model: HTDemucs4 FT Drums
drums path:
    rye run python inference_single.py \
        "{{path}}" \
        --flac \
        --model_type htdemucs \
        --config_path configs/config_musdb18_htdemucs.yaml \
        --start_check_point results/f7e0c4bc-ba3fe64a.th \
        --target_instrument drums \
        --store_dir "{{STORE_DIR}}"

# Single stem model: HTDemucs4 FT Bass
bass path:
    rye run python inference_single.py \
        "{{path}}" \
        --flac \
        --model_type htdemucs \
        --config_path configs/config_musdb18_htdemucs.yaml \
        --start_check_point results/d12395a8-e57c48e6.th \
        --target_instrument bass \
        --store_dir "{{STORE_DIR}}"
