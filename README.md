


<p align="center">
    <img src="assets/logo.png" width="250" />
<p>
<h2 align="center"> LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment </h2>
<h5 align="center"> If you like our project, please give us a star ✨ on Github for latest update.  </h2>

[//]: # (<p align="center">)

[//]: # (        📖 <a href="https://arxiv.org/abs/2305.11172">Paper</a>&nbsp&nbsp｜ &nbsp<a href="datasets.md">Datasets</a>)

[//]: # (</p>)



The following figure shows the architecture of LanguageBind. LanguageBind can be easily extended to segmentation, detection tasks, and potentially to unlimited modalities.
<p align="center">
<img src="assets/languagebind.jpg" width=100%>
</p>



## 📰 News
**[2023.10.02]**  Code and pre-trained models are available now! <br>

## 🤗 Online Demo
Coming soon...

## 🔥 Highlights

### 😮 High performance, but NO intermediate modality required
LanguageBind is a **language-centric** multimodal pretraining approach, **taking the language as the bind across different modalities** because the language modality is well-explored and contains rich semantics. 

### ⚡️ A multimodal, fully aligned and voluminous dataset
We propose **VIDAL-10M**, **10 Million data** with **V**ideo, **I**nfrared, **D**epth, **A**udio and their corresponding **L**anguage, which greatly expands the data beyond visual modalities.

## 📍 Model Zoo
* We list the pretrained checkpoints of LanguageBind below. Note that LanguageBind can be disassembled into different branches to handle different tasks.
* The cache comes from OpenCLIP, which we downloaded from HuggingFace. Note that the original cache for pretrained weights is the Image-Language weights, just a few more HF profiles.
* We additionally trained Video-Language with the LanguageBind method, which is stronger than on CLIP4Clip framework.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>BaiDu</th><th>Google</th><th>PKU disk</th>
    </tr>
    <tr align="center">
        <td>Video-Language</td><td>TODO</td><td>TODO</td><td>TODO</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Audio-Language</td><td><a href="https://pan.baidu.com/s/1PFN8aGlnzsOkGjVk6Mzlfg?pwd=sisz">Link</a></td><td>TODO</td><td>TODO</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Depth-Language</td><td><a href="https://pan.baidu.com/s/1YWlaxqTRhpGvXqCyBbmhyg?pwd=olom">Link</a></td><td>TODO</td><td>TODO</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Thermal(Infrared)-Language</td><td><a href="https://pan.baidu.com/s/1luUyyKxhadKKc1nk1wizWg?pwd=raf5">Link</a></td><td>TODO</td><td>TODO</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Image-Language</td><td><a href="https://pan.baidu.com/s/1VBE4OjecMTeIzU08axfFHA?pwd=7j0m">Link</a></td><td>TODO</td><td>TODO</td>
    </tr>
    </tr>
    <tr align="center">
        <td>Cache for pretrained weight</td><td><a href="https://pan.baidu.com/s/1Tytx5MDSo96rwUmQZVY1Ww?pwd=c7r0">Link</a></td><td>TODO</td><td>TODO</td>
    </tr>
</table>
</div>

## 🚀 Main Results

### ✨ Video-Language
We focus on reporting the parameters of the vision encoder. Our experiments are based on 3 million video-text pairs of VIDAL-10M, and we train on the CLIP4Clip framework.. 
<p align="center">
<img src="assets/res1.jpg" width=80%>
</p>

### ✨ Multiple Modalities
Infrared-Language, Depth-Language, and Audio-Language zero-shot classification. We report the top-1 classification accuracy for all datasets.
<p align="center">
<img src="assets/res2.jpg" width=70%>
</p>

## 🔨 Requirements and Installation
* Python >= 3.8
* Pytorch >= 1.13.0
* CUDA Version >= 10.2 (recommend 11.6)
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/LanguageBind
cd LanguageBind
pip install -r requirements.txt
```

## 🎉 VIDAL-10M
Release the dataset after publication...

## ⤴️ Training & Inference
Release run scripts, details coming soon...

## 👀 Downstream datasets
Coming soon...

## ☎️ Contact 
Zhu Bin: binzhu@stu.pku.edu.cn

## 👍 Acknowledgement
* [OpenCLIP](https://github.com/mlfoundations/open_clip) An open source pretraining framework.
* [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) An open source Video-Text retrieval framework.

## 🔒 License
* The majority of this project is released under the MIT license as found in the [LICENSE](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/LICENSE) file.
* The dataset of this project is released under the CC-BY-NC 4.0 license as found in the [DATASET_LICENSE](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/DATASET_LICENSE) file. 

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil::

```BibTeX

```
