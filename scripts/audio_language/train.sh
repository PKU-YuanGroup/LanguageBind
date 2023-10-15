
CACHE_DIR="path/to/pretrained/weight"
TRAIN_DATA="path/to/data"
# this script is for 512 total batch_size (n(16) GPUs * batch_size(32) * accum_freq(1))
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=2 --nproc_per_node 8 \
    -m main  \
    --train-data ${TRAIN_DATA} \
    --train-num-samples 413639 \
    --clip-type "al" --num_mel_bins 112 --target_length 1008 --audio_sample_rate 16000 \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 16 \
    --lr 5e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.3 \
    --epochs 8 --batch-size 32 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval --do_train \
    --val_a_cls_data "ESC50"
