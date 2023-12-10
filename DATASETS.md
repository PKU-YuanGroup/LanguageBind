## Sample data
We are releasing sample data here so that individuals who are interested can further modify the code to train it on their own data, which includes videos, text from various sources, depth, and infrared.

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th></th><th>Baidu Yun</th><th>Google Cloud</th><th>Peking University Yun</th>
    </tr>
    <tr align="center">
        <td>DATA</td><td><a href="https://pan.baidu.com/s/1MnQUO6xrMPE5HAwveAdtZA?pwd=5ug9">Link</a></td><td><a href="https://drive.google.com/file/d/1p7y_0H3c84VbWpI-zx_m_mgn84uTZTdO/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/B6BDBDDCC616D47126DD0FF568CAF6CD">Link</a></td>
    </tr>
    <tr align="center">
        <td>ANNOTATION</td><td><a href="https://pan.baidu.com/s/1uxxx_67VWy25q7CDilLsHA?pwd=37j3">Link</a></td><td><a href="https://drive.google.com/file/d/1WWVkt9LdbGK0VeQh-g7xy1gUGBwzwVah/view?usp=drive_link">Link</a></td><td><a href=https://disk.pku.edu.cn:443/link/67D836DE504E96457554455A597DC57F"">Link</a></td>
    </tr>
</table>
</div>

## VIDAL-10M

### Text and Video
Due to policy restrictions, we are unable to directly release the videos. However, we provide the YouTube IDs, which can be used to download the videos independently. All textual sources and YouTube IDs can be downloaded from [Google Disk](https://drive.google.com/file/d/1qgm3rO9JugazLJ6KRsAKZfLIagHu3PJ-/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/13gY-IcFSFIuDZ-q0hMTx0g?pwd=gum9).

The organization format of `ANNOTATION` is as follows:
```Bash
{
  "ImkVYKWqlDU": {
    "folder": "coco_vat_9",
    "mplug": "This video describes a group of scuba divers rolling backwards off a boat while playing an instrument. They are having fun and enjoying their time in the water.",
    "polish_mplug": "scuba divers are seen rolling backwards off a boat while playing an instrument, displaying enjoyment and having a good time in the water.",
    "ofa": [
      " a man in a wet suit and a helmet on a boat",
      " a man in a scuba suit on a boat",
      " a person in a boat holding a diver helmet",
      " a man in a wetsuit on a jet ski",
      " a picture of a body of water with the words boats on it",
      " a person in the water with the words if they rolled",
      " a person in the water with a paddle",
      " a person in the water with a scooter"
    ],
    "sound_mplug": "scuba divers rolling backwards off a boat while playing an instrument showcases exuberant laughter, splashing water, and cheery melodies blending with the gentle waves.",
    "raw": "WHY SCUBA DIVERS ROLL BACKWARDS OFF BOAT #shorts"
  },
  "id": {
    "folder": "video_folder",
    "mplug": "mplug_caption",
    "polish_mplug": "polish_mplug_caption",
    "ofa": [
      "ofa_caption_0",
      "ofa_caption_1",
      "ofa_caption_2",
      "ofa_caption_3",
      "ofa_caption_4",
      "ofa_caption_5",
      "ofa_caption_6",
      "ofa_caption_7"
    ],
    "sound_mplug": "sound_mplug_caption",
    "raw": "raw_caption#hashtags"
  },
  ...
}
```

### Depth and Thermal (Infrared)

We are uploading data to [Hugging Face](https://huggingface.co/datasets/LanguageBind/VIDAL-Depth-Thermal), but based on a conservative estimate, it's approximately **20T**. Please be patient as we work on it.
