CACHE_DIR="path/to/pretrained/weight"
ANNOTATION="path/to/data"
# this script is for 1024 total batch_size (n(64) GPUs * batch_size(16) * accum_freq(1))
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP \
    -m main  \
    --train-data ${ANNOTATION} \
	--train-num-samples 10076613 \
    --clip-type "vl_new" --add-time-attn \
    --do_train \
    --lock-text --lock-image --text-type "mix" \
    --init-temp 0.07 --learn-temp --grad-checkpointing \
    --model "ViT-H-14" --cache-dir ${CACHE_DIR} \
    --lr 1e-4 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --tube-size 1 --force-patch-dropout 0.3 \
    --epochs 6 --batch-size 16 --accum-freq 1 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval \
    --val_vl_ret_data "msrvtt" "msvd" "activity" "didemo"