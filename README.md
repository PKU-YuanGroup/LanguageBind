


<p align="center">
    <img src="assets/logo.png" width="250" />
<p>
<h2 align="center"> <a href="https://arxiv.org/pdf/2310.01852.pdf">LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment</a></h2>
<h5 align="center"> If you like our project, please give us a star ✨ on Github for latest update.  </h2>

<p align="center">
📖 <a href="https://arxiv.org/pdf/2310.01852.pdf">Paper</a>
    &nbsp｜&nbsp
🤗<a href="https://huggingface.co/spaces/lb203/LanguageBind">Demo</a>
    &nbsp&nbsp|&nbsp&nbsp
🤖 <a href="https://github.com/PKU-YuanGroup/LanguageBind#-model-zoo">Model zoo</a>
    &nbsp&nbsp|&nbsp&nbsp
📄<a href="TRAIN_AND_VALIDATE.md">Instruction</a>
    &nbsp｜
💥<a href="DATASETS">Datasets</a>
</p>



* The following first figure shows the architecture of LanguageBind. LanguageBind can be easily extended to segmentation, detection tasks, and potentially to unlimited modalities. 
* The second figure shows our proposed VIDAL-10M dataset, which includes five modalities: video, infrared, depth, audio, and language.
<p align="center">
<img src="assets/languagebind.jpg" width=100%>
</p>
<p align="center">
<img src="assets/iclr_dataset_sample.jpg" width=99%>
</p>


## 📰 News
**[2023.10.04]**  📍 Code, checkpoints and demo are available now! Welcome to **watch** this repository for the latest updates.

## 🤗 Demo

* **Local demo.** Highly recommend trying out our web demo, which incorporates all features currently supported by LanguageBind.
```python
python gradio_app.py --languagebind_weight LanguageBind.pt
```

* **Online demo.** We provide the [online demo](https://huggingface.co/spaces/lb203/LanguageBind) in Huggingface Spaces. In this demo, you can calculate the similarity of modalities to language, such as audio-to-language, video-to-language, and depth-to-image.


## 😮 Highlights

### 💡 High performance, but NO intermediate modality required
LanguageBind is a **language-centric** multimodal pretraining approach, **taking the language as the bind across different modalities** because the language modality is well-explored and contains rich semantics. 

### ⚡️ A multimodal, fully aligned and voluminous dataset
We propose **VIDAL-10M**, **10 Million data** with **V**ideo, **I**nfrared, **D**epth, **A**udio and their corresponding **L**anguage, which greatly expands the data beyond visual modalities.

### 🔥 Multi-view enhanced description for training
We make multi-view enhancements to language. We produce multi-view description that combines **meta-data**, **spatial**, and **temporal** to greatly enhance the semantic information of the language. In addition we further **enhance the language with ChatGPT** to create a good semantic space for each modality aligned language.

## 🤖 Model Zoo
* We list the pretrained checkpoints of LanguageBind below. Note that LanguageBind can be disassembled into different branches to handle different tasks.
* The cache comes from OpenCLIP, which we downloaded from HuggingFace. Note that the original cache for pretrained weights is the Image-Language weights, just a few more HF profiles.
* We additionally trained Video-Language with the LanguageBind method, which is stronger than on CLIP4Clip framework.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Baidu Yun</th><th>Google Cloud</th><th>Peking University Yun</th>
    </tr>
    </tr>
    <tr align="center">
        <td>LanguageBind</td><td><a href="https://pan.baidu.com/s/14e-bB-kmsroZHU0mlD_Rtg?pwd=vi1m">Link</a></td><td><a href="https://drive.google.com/file/d/1h1alryrhxeWRadrQRB5eN6diEODpHwNb/view?usp=drive_link">Link</a></td><td>TODO</a></td>
    </tr>
    <tr align="center">
        <td>Video-Language (LanguageBind)</td><td><a href="https://pan.baidu.com/s/1vpzY43Rt8L9VB2nr10Z9wQ?pwd=yc1y">Link</a></td><td><a href="https://drive.google.com/file/d/1JlPJV1BUIygQxM5IkCyiXKdvi7v-T9-6/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/1F74489352C16DAFC10A66E3240BDD50">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Video-Language (CLIP4Clip)</td><td><a href="https://pan.baidu.com/s/1YQB_mQ9ADxLb60sbpPReNQ?pwd=1q17">Link</a></td><td><a href="https://drive.google.com/file/d/1HoyL1rnWsDiZrZ9Yr5iFdS4w4U1NcDeQ/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/0DF72C877A7B0227F516D7EEE94AB514">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Audio-Language</td><td><a href="https://pan.baidu.com/s/1PFN8aGlnzsOkGjVk6Mzlfg?pwd=sisz">Link</a></td><td><a href="https://drive.google.com/file/d/10eA-OxGn-0M-7I3KEOVx7VbnBCQ9ZgDU/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/C2DE5036286C2A0DD06A01A960309CA4">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Depth-Language</td><td><a href="https://pan.baidu.com/s/1YWlaxqTRhpGvXqCyBbmhyg?pwd=olom">Link</a></td><td><a href="https://drive.google.com/file/d/1d8I8LWy927osjoRJqB92Qd6AENJ52mD8/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/C64DA9DDB9CC9B8E4BACDC2007A66D84">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Thermal(Infrared)-Language</td><td><a href="https://pan.baidu.com/s/1luUyyKxhadKKc1nk1wizWg?pwd=raf5">Link</a></td><td><a href="https://drive.google.com/file/d/1ks4MlhFqR1b_AIAP4E_uCTIOfy1N71KI/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/0F4D7034428C0476E339811000DF83A2">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Image-Language</td><td><a href="https://pan.baidu.com/s/1VBE4OjecMTeIzU08axfFHA?pwd=7j0m">Link</a></td><td><a href="https://drive.google.com/file/d/1I6WIDKZrHXSeVK6lvp1l-isszMv6t1JZ/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/FE902FAE21FF3490046E02DB5C2564FA">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Cache for pretrained weight</td><td><a href="https://pan.baidu.com/s/1Tytx5MDSo96rwUmQZVY1Ww?pwd=c7r0">Link</a></td><td><a href="https://drive.google.com/drive/folders/1VQYZlqfKmCMuHffypf5F96odyMCEI87H?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/9CA764E6307790B01D2D4F7E314E8E43">Link</a></td>
    </tr>
</table>
</div>

## 🚀 Main Results

### ✨ Video-Language
We focus on reporting the parameters of the vision encoder. Our experiments are based on 3 million video-text pairs of VIDAL-10M, and we train on the CLIP4Clip framework.
<p align="center">
<img src="assets/res1.jpg" width=80%>
</p>

### ✨ Multiple Modalities
Infrared-Language, Depth-Language, and Audio-Language zero-shot classification. We report the top-1 classification accuracy for all datasets.
<p align="center">
<img src="assets/res2.jpg" width=70%>
</p>

## 🛠️ Requirements and Installation
* Python >= 3.8
* Pytorch >= 1.13.0
* CUDA Version >= 10.2 (recommend 11.6)
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/LanguageBind
cd LanguageBind
pip install -r requirements.txt
```

## Usage
**We open source all modal preprocessing code.** Here is a simple script for multi-modal inference with LanguageBind.
```python
modality_transform = {
	'language': get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir),
	'video': get_video_transform(args),
	'audio': get_audio_transform(args),
	'depth': get_depth_transform(args),
	'thermal': get_thermal_transform(args),
	'image': get_image_transform(args),
}

image = ['assets/zHSOYcZblvY_resize256/0.jpg', 'assets/zlmxeeMOGVQ_resize256/0.jpg']
audio = ['assets/zHSOYcZblvY.wav', 'assets/zlmxeeMOGVQ.wav']
video = ['assets/zHSOYcZblvY.mp4', 'assets/zlmxeeMOGVQ.mp4']
depth = ['assets/zHSOYcZblvY_depth/0.png', 'assets/zlmxeeMOGVQ_depth/0.png']
thermal = ['assets/zHSOYcZblvY_thermal/0.jpg', 'assets/zlmxeeMOGVQ_thermal/0.jpg']
language = ["Training a bird to climb up a ladder.", 'A man riding a motorcycle.']

inputs = {
			 'image': stack_dict([load_and_transform_image(i, modality_transform['image']) for i in image], device),
			 'video': stack_dict([load_and_transform_video(i, modality_transform['video']) for i in video], device),
			 'audio': stack_dict([load_and_transform_audio(i, modality_transform['audio']) for i in audio], device),
			 'thermal': stack_dict([load_and_transform_thermal(i, modality_transform['thermal']) for i in thermal], device),
			 'depth': stack_dict([load_and_transform_depth(i, modality_transform['depth']) for i in depth], device),
			 'language': stack_dict([load_and_transform_text(i, modality_transform['language']) for i in language], device)
}

with torch.no_grad():
	embeddings = model(inputs)

print("Video x Text: \n", torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
print("Image x Text: \n", torch.softmax(embeddings['image'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
print("Depth x Text: \n", torch.softmax(embeddings['depth'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
print("Audio x Text: \n", torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
print("Thermal x Text: \n", torch.softmax(embeddings['thermal'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
```
More details are in inference.py. Run the following command to start.
```python
python inference.py --languagebind_weight LanguageBind.pt
```

## 💥 VIDAL-10M
The datasets is in [DATASETS.md](DATASETS.md).

## 💡 Training & Validating
The training & validating instruction is in [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md).

## 👍 Acknowledgement
* [OpenCLIP](https://github.com/mlfoundations/open_clip) An open source pretraining framework.
* [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) An open source Video-Text retrieval framework.

## 🔒 License
* The majority of this project is released under the MIT license as found in the [LICENSE](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/LICENSE) file.
* The dataset of this project is released under the CC-BY-NC 4.0 license as found in the [DATASET_LICENSE](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/DATASET_LICENSE) file. 

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@misc{zhu2023languagebind,
      title={LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment}, 
      author={Bin Zhu and Bin Lin and Munan Ning and Yang Yan and Jiaxi Cui and Wang HongFa and Yatian Pang and Wenhao Jiang and Junwu Zhang and Zongwei Li and Cai Wan Zhang and Zhifeng Li and Wei Liu and Li Yuan},
      year={2023},
      eprint={2310.01852},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
