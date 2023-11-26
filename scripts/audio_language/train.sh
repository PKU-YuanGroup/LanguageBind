
CACHE_DIR="path/to/pretrained/weight"
ANNOTATION="path/to/data"
# this script is for 1024 total batch_size (n(16) GPUs * batch_size(16) * accum_freq(4))
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=2 --nproc_per_node 8 \
    -m main  \
    --train-data ${ANNOTATION} \
    --train-num-samples 4800000 \
    --clip-type "al" --num_mel_bins 126 --target_length 1036 --audio_sample_rate 16000 --audio_mean -4.2677393 --audio_std 4.5689974 \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 16 \
    --lr 1e-3 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.1 \
    --epochs 16 --batch-size 16 --accum-freq 4 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval --do_train \
    --val_a_cls_data "ESC50" "VGGSound" "Audioset" \
    --val_al_ret_data "Clotho" "Audiocaps"
