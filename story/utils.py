from typing import Any
import torch 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torchaudio
from functools import cache
import glob, json
import requests
from pathlib import Path
import time
import threading
import logging

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices, load_audio
import gdown


@cache
def load_tts(use_deepspeed = True,
            kv_cache = True,
            half = True,
            model_dir = MODELS_DIR, 
            load_custom_voices = True, 
            device = None
            ):
    if torch.backends.mps.is_available():
        use_deepspeed = False
    if load_custom_voices:
        url = "https://drive.google.com/drive/folders/1wwi2ZiIdvVEYjkTbMpJv09WiIkTpogXh"
        out = "tortoise-tts/tortoise/"
        gdown.download_folder(url, output=out, quiet=False)
    return TextToSpeech(models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half, device=device)

def text_to_speech(text, 
                   voice, 
                   index=0,
                   preset="high_quality", # "ultra_fast", "fast", "standard", "high_quality"
                   output_path="results/", 
                   seed=None, 
                   cvvp_amount=0.0, 
                   use_deepspeed=False,
                   kv_cache=True,
                   half=False,
                   model_dir=MODELS_DIR, 
                   load_custom_voices=True,
                   device=None
                   ):
    os.makedirs(output_path, exist_ok=True)
    tts = load_tts(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half, model_dir=model_dir, load_custom_voices=load_custom_voices, device=device)

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

    conditioning_latents = tts.get_conditioning_latents(voice_samples)
    gen, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples,conditioning_latents=conditioning_latents,
                                preset=preset, use_deterministic_seed=seed, return_deterministic_state=True, cvvp_amount=cvvp_amount)

    output = os.path.join(output_path, f'{index}_{voice}.wav')
    gen = gen.squeeze(0).cpu()
    torchaudio.save(output, gen, 24000)
    return gen, output

def parse_story(story_path, god_as="freeman", default_style="Neutral"):
    with open(story_path, "r") as f:
        story = json.load(f)
    timeline = []
    index = 0
    for scenario in story:
        scenario_text = scenario["scenario"]
        timeline.append((index, god_as.lower(), default_style,  scenario_text))
        index += 1
        for dialog in scenario["dialogue"]:
            character = dialog["character"].lower()
            sentiment = default_style
            if "sentiment" in dialog:
                sentiment = dialog["sentiment"]
            text = dialog["line"]
            timeline.append((index, character, sentiment,  text))
            index += 1
    return timeline


def load_content(content):
    if isinstance(content, bytes):
        return "tmp", content
    elif isinstance(content, str):
        path = Path(content)
        return path.name, path.read_bytes()
    elif isinstance(content, Path):
        return content.name, content.read_bytes()
    else:
        raise Exception("Unknown content type")
    
def critical_log(response):
    logging.critical('Failed to make request')
    logging.critical(f'Status code: {response.status_code}')
    logging.critical(f'Response: {response.text}')
    
def dispatch_generate_bvh(
        audio,
        style="Neutral", 
        base_url='http://129.192.81.237', 
        seed=1337,
        temperature=0.9,
        pose="pose_0",
    ):
    # The URL to make the POST request to
    url = f'{base_url}/generate_bvh/'

    # Prepare the multipart/form-data payload
    name, data = load_content(audio)

    assert style in  ["Agreement", "Angry", "Disagreement", "Distracted", "Flirty", "Happy", "Laughing", "Neutral", "Old", "Pensive", "Relaxed", "Sad", "Sarcastic", "Scared", "Sneaky", "Speech", "Still", "Threatening", "Tired"]
    # todo validate pose

    files = {
        'audio': (name, data,  'audio/wav'), 
    }
    data = {
        'pose': pose,
        'style': style,
        'temperature': f'{temperature}',
        'seed': f'{seed}'
    }

    # Make the POST request
    response = requests.post(url, files=files, data=data)

    # Check the status code and print the response
    if response.status_code != 202:
        critical_log(response)
        raise Exception("Failed to make request")
    return response.json()

def dispatch_generate_fbx(
        bvh,
        base_url='http://129.192.81.237', 
    ):
    # The URL to make the POST request to
    url = f'{base_url}/export_fbx/'

    # Prepare the multipart/form-data payload
    name, data = load_content(bvh)

    files = {
        'motion': (name, data,  'application/octet-stream'), 
    }

    # Make the POST request
    response = requests.post(url, files=files)

    # Check the status code and print the response
    if response.status_code != 202:
        critical_log(response)
        raise Exception("Failed to make request")
    return response.json()

def dispatch_generate_mp4(
        bvh,
        audio,
        base_url='http://129.192.81.237', 
    ):
    # The URL to make the POST request to
    url = f'{base_url}/visualise/'

    # Prepare the multipart/form-data payload

    name_bvh, data_bvh = load_content(bvh)
    name_audio, data_audio = load_content(audio)


    files = {
        'motion': (name_bvh, data_bvh,  'application/octet-stream'),
        'audio': (name_audio, data_audio,  'audio/wav'),
    }

    # Make the POST request
    response = requests.post(url, files=files)

    # Check the status code and print the response
    if response.status_code != 202:
        critical_log(response)
        raise Exception("Failed to make request")
    return response.json()

def job_done(
        job_id, 
        base_url='http://129.192.81.237', 
    ):
    url = f'{base_url}/job_id/{job_id}/'
    response = requests.get(url)
    state = "PENDING"
    if response.status_code == 200:
        obj = response.json()
        state = obj["state"]
    else:
        critical_log(response)
        raise Exception("Failed to make request")
    return state == "SUCCESS"

def get_data(
        job_id, 
        base_url='http://129.192.81.237', 
    ):
    url = f'{base_url}/get_files/{job_id}/'
    response = requests.get(url)
    if response.status_code != 200:
        critical_log(response)
        raise Exception("Failed to make request")    
    return response.content

def save_data(data, path):
    with open(path, 'wb') as f:
        f.write(data)

def wait_and_get(job_id, base_url='http://129.192.81.237', ):
    while not job_done(job_id, base_url=base_url):
        time.sleep(1)
    return get_data(job_id, base_url=base_url)



class Worker:
    def __init__(self, index, voice, sentiment, text, **kvargs):
        self.index = index
        self.voice = voice
        self.sentiment = sentiment
        self.text = text
        self.audio = None
        self.bvh = None
        self.fbx = None
        self.mp4 = None
        self.state = "NOT_STARTED"
        self.error = None
        self.worker = None
        self.kvargs = kvargs

    def dispatch(self, audio):
        try:
            self.state = "RUNNING"
            bvh_id = dispatch_generate_bvh(audio, style=self.sentiment)
            logging.info(f"index {self.index} - bvh_id {bvh_id}")
            self.bvh = wait_and_get(bvh_id)
            logging.info(f"index {self.index} - bvh done")

            fbx_id = dispatch_generate_fbx(self.bvh)
            logging.info(f"index {self.index} - fbx_id {fbx_id}")
            self.fbx = wait_and_get(fbx_id)
            logging.info(f"index {self.index} - fbx done")

            mp4_id = dispatch_generate_mp4(self.bvh, self.audio)
            logging.info(f"index {self.index} - mp4_id {mp4_id}")
            self.mp4 = wait_and_get(mp4_id)
            logging.info(f"index {self.index} - mp4 done")

            logging.info(f"index {self.index} - done")
            self.state = "SUCCESS"
        except Exception as e:
            self.state = "FAILURE"
            self.error = e


    def __call__(self, device) -> Any:
        try:
            self.audio, _ = text_to_speech(self.text, self.voice, index=self.index, device=device, **self.kvargs)
            # do dispatch in a thread
            self.worker = threading.Thread(target=self.dispatch, args=(self.audio,))
            self.worker.start()
            
        except Exception as e:
            self.state = "FAILURE"
            self.error = e
            raise e
        
    def join(self):
        if self.state == "RUNNING" and self.worker is not None:
            self.worker.join()

    def save_audio(self, path):
        save_data(self.audio, path)

    def save_bvh(self, path):
        self.join()
        save_data(self.bvh, path)

    def save_fbx(self, path):
        self.join()
        save_data(self.fbx, path)
    
    def save_mp4(self, path):
        self.join()
        save_data(self.mp4, path)

