import sys

import gradio as gr
import argparse
import numpy as np
import torch
from torch import nn

from data.process_image import load_and_transform_image, get_image_transform
from main import SET_GLOBAL_VALUE
from model.build_model import create_vat_model
from data.process_audio import load_and_transform_audio, get_audio_transform
from data.process_video import load_and_transform_video, get_video_transform
from data.process_depth import load_and_transform_depth, get_depth_transform
from data.process_thermal import load_and_transform_thermal, get_thermal_transform
from data.process_text import load_and_transform_text
from model.languagebind import stack_dict, LanguageBind
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from training.params import parse_args

code_highlight_css = (
"""
#chatbot .hll { background-color: #ffffcc }
#chatbot .c { color: #408080; font-style: italic }
#chatbot .err { border: 1px solid #FF0000 }
#chatbot .k { color: #008000; font-weight: bold }
#chatbot .o { color: #666666 }
#chatbot .ch { color: #408080; font-style: italic }
#chatbot .cm { color: #408080; font-style: italic }
#chatbot .cp { color: #BC7A00 }
#chatbot .cpf { color: #408080; font-style: italic }
#chatbot .c1 { color: #408080; font-style: italic }
#chatbot .cs { color: #408080; font-style: italic }
#chatbot .gd { color: #A00000 }
#chatbot .ge { font-style: italic }
#chatbot .gr { color: #FF0000 }
#chatbot .gh { color: #000080; font-weight: bold }
#chatbot .gi { color: #00A000 }
#chatbot .go { color: #888888 }
#chatbot .gp { color: #000080; font-weight: bold }
#chatbot .gs { font-weight: bold }
#chatbot .gu { color: #800080; font-weight: bold }
#chatbot .gt { color: #0044DD }
#chatbot .kc { color: #008000; font-weight: bold }
#chatbot .kd { color: #008000; font-weight: bold }
#chatbot .kn { color: #008000; font-weight: bold }
#chatbot .kp { color: #008000 }
#chatbot .kr { color: #008000; font-weight: bold }
#chatbot .kt { color: #B00040 }
#chatbot .m { color: #666666 }
#chatbot .s { color: #BA2121 }
#chatbot .na { color: #7D9029 }
#chatbot .nb { color: #008000 }
#chatbot .nc { color: #0000FF; font-weight: bold }
#chatbot .no { color: #880000 }
#chatbot .nd { color: #AA22FF }
#chatbot .ni { color: #999999; font-weight: bold }
#chatbot .ne { color: #D2413A; font-weight: bold }
#chatbot .nf { color: #0000FF }
#chatbot .nl { color: #A0A000 }
#chatbot .nn { color: #0000FF; font-weight: bold }
#chatbot .nt { color: #008000; font-weight: bold }
#chatbot .nv { color: #19177C }
#chatbot .ow { color: #AA22FF; font-weight: bold }
#chatbot .w { color: #bbbbbb }
#chatbot .mb { color: #666666 }
#chatbot .mf { color: #666666 }
#chatbot .mh { color: #666666 }
#chatbot .mi { color: #666666 }
#chatbot .mo { color: #666666 }
#chatbot .sa { color: #BA2121 }
#chatbot .sb { color: #BA2121 }
#chatbot .sc { color: #BA2121 }
#chatbot .dl { color: #BA2121 }
#chatbot .sd { color: #BA2121; font-style: italic }
#chatbot .s2 { color: #BA2121 }
#chatbot .se { color: #BB6622; font-weight: bold }
#chatbot .sh { color: #BA2121 }
#chatbot .si { color: #BB6688; font-weight: bold }
#chatbot .sx { color: #008000 }
#chatbot .sr { color: #BB6688 }
#chatbot .s1 { color: #BA2121 }
#chatbot .ss { color: #19177C }
#chatbot .bp { color: #008000 }
#chatbot .fm { color: #0000FF }
#chatbot .vc { color: #19177C }
#chatbot .vg { color: #19177C }
#chatbot .vi { color: #19177C }
#chatbot .vm { color: #19177C }
#chatbot .il { color: #666666 }
""")
#.highlight  { background: #f8f8f8; }

title_markdown = ("""
<h1 align="center"><a href="https://github.com/PKU-YuanGroup/LanguageBind"><img src="https://z1.ax1x.com/2023/10/04/pPOBSL6.png", alt="LanguageBind🚀" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>

<h2 align="center"> LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment </h2>

<h5 align="center"> If you like our project, please give us a star ✨ on Github for latest update.  </h2>

<div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/PKU-YuanGroup/LanguageBind'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        <a href="https://arxiv.org/pdf/2310.01852.pdf"><img src="https://img.shields.io/badge/Arxiv-2310.01852-red"></a>
        <a href='https://github.com/PKU-YuanGroup/LanguageBind/stargazers'><img src='https://img.shields.io/github/stars/PKU-YuanGroup/LanguageBind.svg?style=social'></a>
    </div>
</div>
""")
css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""


def image_to_language(image, language):
    inputs = {}
    inputs['image'] = stack_dict([load_and_transform_image(image, modality_transform['image'])], device)
    inputs['language'] = stack_dict([load_and_transform_text(language, modality_transform['language'])], device)
    with torch.no_grad():
        embeddings = model(inputs)
    return (embeddings['image'] @ embeddings['language'].T).item()


def video_to_language(video, language):
    inputs = {}
    inputs['video'] = stack_dict([load_and_transform_video(video, modality_transform['video'])], device)
    inputs['language'] = stack_dict([load_and_transform_text(language, modality_transform['language'])], device)
    with torch.no_grad():
        embeddings = model(inputs)
    return (embeddings['video'] @ embeddings['language'].T).item()


def audio_to_language(audio, language):
    inputs = {}
    inputs['audio'] = stack_dict([load_and_transform_audio(audio, modality_transform['audio'])], device)
    inputs['language'] = stack_dict([load_and_transform_text(language, modality_transform['language'])], device)
    with torch.no_grad():
        embeddings = model(inputs)
    return (embeddings['audio'] @ embeddings['language'].T).item()


def depth_to_language(depth, language):
    inputs = {}
    inputs['depth'] = stack_dict([load_and_transform_depth(depth.name, modality_transform['depth'])], device)
    inputs['language'] = stack_dict([load_and_transform_text(language, modality_transform['language'])], device)
    with torch.no_grad():
        embeddings = model(inputs)
    return (embeddings['depth'] @ embeddings['language'].T).item()


def thermal_to_language(thermal, language):
    inputs = {}
    inputs['thermal'] = stack_dict([load_and_transform_thermal(thermal, modality_transform['thermal'])], device)
    inputs['language'] = stack_dict([load_and_transform_text(language, modality_transform['language'])], device)
    with torch.no_grad():
        embeddings = model(inputs)
    return (embeddings['thermal'] @ embeddings['language'].T).item()

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args.device = 'cuda:0'
    args.cache_dir = 'tokenizer_cache'
    device = torch.device(args.device)
    model = LanguageBind(args, no_temp=True)
    ckpt = torch.load(args.languagebind_weight, map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()


    modality_transform = {
        'language': get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir),
        'video': get_video_transform(args),
        'audio': get_audio_transform(args),
        'depth': get_depth_transform(args),
        'thermal': get_thermal_transform(args),
        'image': get_image_transform(args),
    }

    with gr.Blocks(title="LanguageBind🚀", css=css) as demo:
        gr.Markdown(title_markdown)
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath", height=224, width=224, label='Image Input')
                language_i = gr.Textbox(lines=2, label='Text Input')
                out_i = gr.Textbox(label='Similarity of Image to Text')
                b_i = gr.Button("Calculate similarity of Image to Text")
            with gr.Column():
                video = gr.Video(type="filepath", height=224, width=224, label='Video Input')
                language_v = gr.Textbox(lines=2, label='Text Input')
                out_v = gr.Textbox(label='Similarity of Video to Text')
                b_v = gr.Button("Calculate similarity of Video to Text")
            with gr.Column():
                audio = gr.Audio(type="filepath", label='Audio Input')
                language_a = gr.Textbox(lines=2, label='Text Input')
                out_a = gr.Textbox(label='Similarity of Audio to Text')
                b_a = gr.Button("Calculate similarity of Audio to Text")
        with gr.Row():
            with gr.Column():
                depth = gr.File(height=224, width=224, label='Depth Input, need a .png file, 16 bit, with values ranging from 0-10000 (representing 0-10 metres, but 1000 times)')
                language_d = gr.Textbox(lines=2, label='Text Input')
                out_d = gr.Textbox(label='Similarity of Depth to Text')
                b_d = gr.Button("Calculate similarity of Depth to Text")
            with gr.Column():
                thermal = gr.Image(type="filepath", height=224, width=224, label='Thermal Input, you should first convert to RGB')
                language_t = gr.Textbox(lines=2, label='Text Input')
                out_t = gr.Textbox(label='Similarity of Thermal to Text')
                b_t = gr.Button("Calculate similarity of Thermal to Text")

        b_i.click(image_to_language, inputs=[image, language_i], outputs=out_i)
        b_a.click(audio_to_language, inputs=[audio, language_a], outputs=out_a)
        b_v.click(video_to_language, inputs=[video, language_v], outputs=out_v)
        b_d.click(depth_to_language, inputs=[depth, language_d], outputs=out_d)
        b_t.click(thermal_to_language, inputs=[thermal, language_t], outputs=out_t)

    demo.launch()
