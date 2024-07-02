# https://github.com/ZFTurbo/Music-Source-Separation-Training

# Vocal model: BS Roformer (viperx edition)
vocals:
    # Processing files in ./input/
    rye run python inference.py \
        --model_type bs_roformer \
        --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
        --start_check_point results/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
        --input_folder input/ \
        --store_dir separation_results/

# Single stem model: BS Roformer (viperx edition)
# this removes drums and bass only, keeping the rest
# so you should remove vocals first!
other:
    # Processing files in ./input/
    rye run python inference.py \
        --model_type bs_roformer \
        --config_path configs/viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml \
        --start_check_point results/model_bs_roformer_ep_937_sdr_10.5309.ckpt \
        --input_folder input/ \
        --store_dir separation_results/
