We provide the **off-the-shelf** scripts in the [scripts folder](scripts).

## Training LanguageBind 


<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Cache of pretrained weight</th><th>Baidu Yun</th><th>Google Cloud</th><th>Peking University Yun</th>
    </tr>
    <tr align="center">
        <td>Large</td><td><a href="https://pan.baidu.com/s/1co46bkuUJXr8ePPKp1WWgA?pwd=ofm6">Link</a></td><td><a href="https://drive.google.com/drive/folders/1VQYZlqfKmCMuHffypf5F96odyMCEI87H?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/9CA764E6307790B01D2D4F7E314E8E43">Link</a></td>
    </tr>
    <tr align="center">
        <td>Huge</td><td><a href="https://pan.baidu.com/s/1QLpyXEYunoXS-oqGsvzKKA?pwd=vgo2">Link</a></td><td>-</td><td><a href="https://disk.pku.edu.cn:443/link/720A77A7DB9EFD167C5AC8E3FC4B6068">Link</a></td>
    </tr>
</table>
</div>


For example, to **train** LanguageBind on **Depth-Language** with 8 GPUs (1 nodes x 8 GPUs).
* First download the cache of pretrained weight above. and specify `CACHE_DIR=path/to/LanguageBind`.
* The second step is to develop a path to `ANNOTATION` and `DATA` [here](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/data/base_datasets.py#L37) according to the [dataset preparation](https://github.com/PKU-YuanGroup/LanguageBind#-vidal-10m).
* Then you can run

```bash
CACHE_DIR="/path/to/LanguageBind"
ANNOTATION="path/to/data"
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=1 --nproc_per_node 8 \
    -m main  \
    --train-data ${ANNOTATION} \
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
* The second step is to prepare the [downstream dataset](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/TRAIN_AND_VALIDATE.md#downstream-datasets).
* Then you can run

```bash
CACHE_DIR="/path/to/LanguageBind"
RESUME="thermal_language.pt"
ANNOTATION="path/to/data"
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nproc_per_node 1 \
    -m main  \
    --train-data ${ANNOTATION} \
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

## Downstream datasets

### Depth
NYU V2 dataset is downloaded from [this repo](https://github.com/TUI-NICR/nicr-scene-analysis-datasets/tree/main/nicr_scene_analysis_datasets/datasets/nyuv2) and we reformat them to conform to the standard ImageNet format. We also provide data as follows. Change the ```data_root``` [here](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/data/build_datasets.py#L221).

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Yun</th><th>Google Cloud</th><th>Peking University Yun</th>
    </tr>
    <tr align="center">
        <td>NYU</td><td><a href="https://pan.baidu.com/s/1AGOG8U3F7W8AvJiEmuzs-A?pwd=1dsg">Link</a></td><td><a href="https://drive.google.com/file/d/1CltzrTBLFqLxJzpztSIN-5ZosZpXQQ6u/view?usp=sharing">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/7D7B164DEA64059793D3C3E3A65C0F64">Link</a></td>
    </tr>
</table>
</div>

### Video
Video datasets are downloaded from [this repo](https://github.com/jpthu17/HBI) and we show the folder structure. Change the ```data_root``` [here](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/data/build_datasets.py#L74).

### Audio
Audio datasets are downloaded from [this repo](https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md#audio) and Audioset from [here](https://github.com/qiuqiangkong/audioset_tagging_cnn#1-download-dataset).We reformat them to conform to the standard ImageNet format. Change the ```data_root``` [here1](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/data/build_datasets.py#L144) and [here2](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/data/build_datasets.py#L159).

### Infrared (Thermal)
We download LLVIP from [official website](https://bupt-ai-cz.github.io/LLVIP/), and FLIR from [here](https://www.flir.com/oem/adas/adas-dataset-form/). We reformat them to conform to the standard ImageNet format. Change the ```data_root``` [here](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/data/build_datasets.py#L233). We also provide the processed data as follows.

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Yun</th><th>Google Cloud</th><th>Peking University Yun</th>
    </tr>
    <tr align="center">
        <td>LLVIP</td><td><a href="https://pan.baidu.com/s/15HPVr016F7eO9005NDRJTg?pwd=46fh">Link</a></td><td><a href="https://drive.google.com/file/d/1RfKNR8q6dHiAHB4OlYecnkUSx-ghLuEO/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/30D592EA37AC7C411264801A74994376">Link</a></td>
    </tr>
    <tr align="center">
        <td>FLIR V1</td><td><a href="https://pan.baidu.com/s/1ZDSo5VPxJ4SA7wS_rNk0uQ?pwd=l491">Link</a></td><td><a href="https://drive.google.com/file/d/1CezCLJ4GUfPMFimitPfK40OV2j2Kr8t8/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/AD89D6ADE2CAC2407B00650870CBBDEC">Link</a></td>
    </tr>
    <tr align="center">
        <td>FLIR V2</td><td><a href="https://pan.baidu.com/s/16xdr2aQkHo3zJ4KbaTmO3Q?pwd=tj9f">Link</a></td><td><a href="https://drive.google.com/file/d/1Z2ThG5QH-9biFI2-Z8k2fBKSA6Nrees6/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/E06C010970B0ED51926700D2F7A21EA8">Link</a></td>
    </tr>
</table>
</div>

### Folder structure
```bash
downstream_datasets
├── Audio
│   ├── audiocaps
│   │   └── audio
│   │       ├── test
│   │       ├── train
│   │       └── val
│   ├── audioset
│   │   ├── balanced_train_segments
│   │   ├── eval_segments
│   │   └── unbalanced_train_segments
│   │       ├── unbalanced_train_segments_part00
│   │       ├── unbalanced_train_segments_part01
│   │       ├── ...
│   │       └── unbalanced_train_segments_part40
│   ├── clotho
│   │   ├── CLOTHO_retrieval_dataset
│   │   └── evaluation
│   ├── esc50
│   │   └── test
│   │       ├── airplane
│   │       ├── breathing
│   │       ├── ...
│   │       └── wind
├── laionaudio
│   │   ├── audios
│   │   ├── freesound_no_overlap
│   │   └── jsons
├── vggsound
│       └── test
│           ├── air\ conditioning\ noise
│           ├── air\ horn
│           ├── ...
│           └── zebra\ braying
├── Depth
│   ├── nyuv2
│   │   ├── data
│   │   │   └── val
│   │   │       ├── bathroom
│   │   │       ├── bedroom
│   │   │       ├── bookstore
│   │   │       ├── classroom
│   │   │       ├── dining_room
│   │   │       ├── home_office
│   │   │       ├── kitchen
│   │   │       ├── living_room
│   │   │       ├── office
│   │   │       └── others
├── Thermal
│   ├── flirv1
│   │   └── val
│   │       ├── bicycle
│   │       ├── car
│   │       ├── dog
│   │       └── person
│   ├── flirv2
│   │   └── val
│   │       ├── bike
│   │       ├── bus
│   │       ├── car
│   │       ├── hydrant
│   │       ├── light
│   │       ├── motor
│   │       ├── other\ vehicle
│   │       ├── person
│   │       ├── sign
│   │       ├── skateboard
│   │       ├── stroller
│   │       └── truck
│   ├── llvip
│   │   ├── train
│   │   │   ├── background
│   │   │   └── person
│   │   └── val
│   │       ├── background
│   │       └── person
└── VideoTextRetrieval
    ├── vtRetdata
    │   ├── ActivityNet
    │   │   └── Videos
    │   │       └── Activity_Videos
    │   ├── Didemo
    │   │   └── videos
    │   ├── MSRVTT
    │   │   └── MSRVTT_Videos
    │   └── MSVD
    │       └── MSVD_Videos
```

