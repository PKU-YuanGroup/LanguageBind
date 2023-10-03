We provide the **off-the-shelf** scripts in the [scripts folder](scripts).

## Training LanguageBind 

For example, to **train** LanguageBind on **Depth-Language** with 16 GPUs (2 nodes x 8 GPUs).
* First download the [cache of pretrained weight](https://github.com/PKU-YuanGroup/LanguageBind#-model-zoo) and specify ```CACHE_DIR```.
* The second step is to develop a path to ```TRAIN_DATA``` according to the [dataset preparation](https://github.com/PKU-YuanGroup/LanguageBind#-vidal-10m).
* Then you can run

```bash
CACHE_DIR="path/to/pretrained/weight"
TRAIN_DATA="path/to/data"
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=1 --nproc_per_node 8 \
    -m main  \
    --train-data ${TRAIN_DATA} \
    --train-num-samples 3020000 \
    --clip-type "dl" --max-depth 10 \
    --do_train \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 2 \
    --lr 5e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.5 \
    --epochs 1 --batch-size 128 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval \
    --val_d_cls_data "NYUV2"
```


## Validating LanguageBind 

For example, to **validate** LanguageBind on **Depth-Language** with 1 GPUs.
* First specify ```RESUME```.
* The second step is to Simply remove ```--do_train``` from the script of train.
* Then you can run

```bash
CACHE_DIR="path/to/pretrained/weight"
RESUME="thermal_language.pt"
TRAIN_DATA="path/to/data"
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nproc_per_node 1 \
    -m main  \
    --train-data ${TRAIN_DATA} \
    --train-num-samples 3020000 \
    --clip-type "dl" --max-depth 10 \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 2 \
    --lr 5e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.5 \
    --epochs 1 --batch-size 128 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume ${RESUME} \
    --do_eval \
    --val_d_cls_data "NYUV2"
```
