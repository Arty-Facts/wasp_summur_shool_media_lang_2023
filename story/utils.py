import torch 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torchaudio
from functools import cache
import glob, json

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices, load_audio
import gdown


@cache
def load_tts(use_deepspeed = True,
            kv_cache = True,
            half = True,
            model_dir = MODELS_DIR, 
            load_custom_voices = True):
    if torch.backends.mps.is_available():
        use_deepspeed = False
    if load_custom_voices:
        url = "https://drive.google.com/drive/folders/1wwi2ZiIdvVEYjkTbMpJv09WiIkTpogXh"
        out = "tortoise-tts/tortoise/"
        gdown.download_folder(url, output=out, quiet=False)
    return TextToSpeech(models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)

def text_to_speech(text, 
                   voice, 
                   index=0,
                   preset="high_quality", # "ultra_fast", "fast", "standard", "high_quality"
                   output_path="results/", 
                   seed=None, 
                   cvvp_amount=0.0, 
                   use_deepspeed=False,
                   kv_cache=True,
                   half=True,
                   model_dir=MODELS_DIR, 
                   load_custom_voices=True,
                   ):
    os.makedirs(output_path, exist_ok=True)
    tts = load_tts(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half, model_dir=model_dir, load_custom_voices=load_custom_voices)

    if '&' in voice:
        voice_sel = voice.split('&')
    else:
        voice_sel = [voice]

    voice_samples = []
    for name in voice_sel:
        clip_paths = glob.glob(f"tortoise-tts/tortoise/voices/{name}/*.*")
        if len(clip_paths) == 0:
            raise Exception(f"Voice {name} not found")
        for clip_path in clip_paths:
            voice_samples.append(load_audio(clip_path, 22050))


    gen, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples,
                                preset=preset, use_deterministic_seed=seed, return_deterministic_state=True, cvvp_amount=cvvp_amount)

    output = os.path.join(output_path, f'{index}_{voice}.wav')
    gen = gen.squeeze(0).cpu()
    torchaudio.save(output, gen, 24000)
    return gen, output

def parse_story(story_path, good_as="freeman"):
    with open(story_path, "r") as f:
        story = json.load(f)
    timeline = []
    index = 0
    for scenario in story:
        scenario_text = scenario["scenario"]
        timeline.append((index, good_as.lower(),  scenario_text))
        index += 1
        for dialog in scenario["dialogue"]:
            character = dialog["character"].lower()
            line = dialog["line"]
            timeline.append((index, character, line))
            index += 1
    return timeline
