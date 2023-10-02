<!---
Copyright 2023 The OFA-Sys Team. 
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->



<p align="center">
    <img src="assets/logo.png" width="250" />
<p>
<h2 align="center"> LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment </h2>

<h5 align="center"> If you like our project, please give us a star ✨ on Github for latest update.  </h2>

[//]: # (<p align="center">)

[//]: # (        📖 <a href="https://arxiv.org/abs/2305.11172">Paper</a>&nbsp&nbsp｜ &nbsp<a href="datasets.md">Datasets</a>)

[//]: # (</p>)
<br>

* LanguageBind is a **language-centric** multimodal pretraining approach, **taking the language as the bind across different modalities** because the language modality is well-explored and contains rich semantics. 
* We thus propose **VIDAL-10M**, **10 Million data** with **V**ideo, **I**nfrared, **D**epth, **A**udio and their corresponding **L**anguage, which greatly expands the data beyond visual modalities.

The following figure shows the architecture of LanguageBind. LanguageBind can be easily extended to segmentation, detection tasks, and potentially to unlimited modalities.
<p align="center">
<img src="assets/languagebind.jpg" width=100%>
</p>

<br>


# News
* **2023.10.02:** Released the code. Training & validating scripts and checkpoints.
<br></br>
# Online Demo
Coming soon...

# Models and Results
## Model Zoo
We list the parameters and pretrained checkpoints of LanguageBind below. Note that LanguageBind can be disassembled into different branches to handle different tasks.
The cache comes from OpenCLIP, which we downloaded from HuggingFace. Note that the original cache for pretrained weights is the Image-Language weights, just a few more HF profiles.
We additionally trained Video-Language with the LanguageBind method, which is stronger than on CLIP4Clip framework.
<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Ckpt</th><th>Params</th><th>Modality Hidden size</th><th>Modality Layers</th><th>Language Hidden size</th><th>Language Layers</th>
    </tr>
    <tr align="center">
        <td>Video-Language</td><td>TODO</td><td>330M</td><td>1024</td><td>24</td><td>768</td><td>12</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Audio-Language</td><td><a href="https://pan.baidu.com/s/1PFN8aGlnzsOkGjVk6Mzlfg?pwd=sisz">BaiDu</a></td><td>330M</td><td>1024</td><td>24</td><td>768</td><td>12</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Depth-Language</td><td><a href="https://pan.baidu.com/s/1YWlaxqTRhpGvXqCyBbmhyg?pwd=olom">BaiDu</a></td><td>330M</td><td>1024</td><td>24</td><td>768</td><td>12</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Thermal(Infrared)-Language</td><td><a href="https://pan.baidu.com/s/1luUyyKxhadKKc1nk1wizWg?pwd=raf5">BaiDu</a></td><td>330M</td><td>1024</td><td>24</td><td>768</td><td>12</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Image-Language</td><td><a href="https://pan.baidu.com/s/1VBE4OjecMTeIzU08axfFHA?pwd=7j0m">BaiDu</a></td><td>330M</td><td>1024</td><td>24</td><td>768</td><td>12</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Cache for pretrained weight</td><td><a href="https://pan.baidu.com/s/1Tytx5MDSo96rwUmQZVY1Ww?pwd=c7r0">BaiDu</a></td><td>330M</td><td>1024</td><td>24</td><td>768</td><td>12</td>
    </tr>

</table>
<br>

## Results
Zero-shot Video-Text Retrieval Performance on MSR-VTT and MSVD datasets. We focus on reporting the parameters of the vision
encoder. Our experiments are based on 3 million video-text pairs of VIDAL-10M, and we train on the CLIP4Clip framework.. 
<p align="center">
<img src="assets/res1.jpg" width=100%>
</p>
Infrared-Language, Depth-Language, and Audio-Language zero-shot classification. We report the top-1 classification accuracy for all datasets.
<p align="center">
<img src="assets/res2.jpg" width=100%>
</p>


<br></br>

# Requirements and Installation
* Python >= 3.8
* Pytorch >= 1.13.0
* CUDA Version >= 10.2 (recommend 11.6)
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/LanguageBind
cd LanguageBind
pip install -r requirements.txt
```

<br></br>

# VIDAL-10M
Release the dataset after publication...

<br></br>

# Training & Inference
Release run scripts, details coming soon...

<br></br>

# Downstream datasets
Coming soon...

<br></br>

# Acknowledgement
* [OpenCLIP](https://github.com/mlfoundations/open_clip) An open source pretraining framework.

<br></br>


# Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

<br></br>

```BibTeX

```
