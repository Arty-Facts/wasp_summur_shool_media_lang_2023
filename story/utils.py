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
import random
import zipfile

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices, load_audio
import gdown
from moviepy.editor import VideoFileClip, concatenate_videoclips

style_to_pose = {
    "Agreement": "pose_11",
    "Angry": "pose_3",
    "Disagreement": "pose_4",
    "Distracted": "pose_8",
    "Flirty": "pose_1",
    "Happy": "pose_2",
    "Laughing": "pose_2",
    "Neutral": "pose_6",
    "Old": "pose_12",
    "Pensive": "pose_0",
    "Relaxed": "pose_5",
    "Sad": "pose_7",
    "Sarcastic": "pose_3",
    "Scared": "pose_8",
    "Sneaky": "pose_12",
    "Speech": "pose_0",
    "Still": "pose_14",
    "Threatening": "pose_10",
    "Tired": "pose_13",
}


@cache
def load_tts(use_deepspeed = True,
            kv_cache = True,
            half = True,
            model_dir = MODELS_DIR, 
            load_custom_voices = True, 
            device = None,
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
        raise Exception(f"Unknown content type {type(content)}")
    
def critical_log(response):
    logger = logging.getLogger("story_error")
    logger.critical('Failed to make request')
    logger.critical(f'Status code: {response.status_code}')
    logger.critical(f'Response: {response.text}')
    
def dispatch_generate_bvh(
        wav,
        style="Neutral", 
        base_url='http://129.192.81.237', 
        seed=None,
        temperature=0.5,
    ):
    if seed is None:
        seed = random.randint(0, 2**32-1)

    # The URL to make the POST request to
    url = f'{base_url}/generate_bvh/'

    # Prepare the multipart/form-data payload
    name, data = load_content(wav)

    if style not in ["Agreement", "Angry", "Disagreement", "Distracted", "Flirty", "Happy", "Laughing", "Neutral", "Old", "Pensive", "Relaxed", "Sad", "Sarcastic", "Scared", "Sneaky", "Speech", "Still", "Threatening", "Tired"]:
        logging.error(f"Unknown style {style}")
        style = "Neutral"
    # todo validate pose

    files = {
        'audio': (name, data,  'audio/wav'), 
    }
    data = {
        'pose': style_to_pose[style],
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
        wav,
        base_url='http://129.192.81.237', 
    ):
    # The URL to make the POST request to
    url = f'{base_url}/visualise/'

    # Prepare the multipart/form-data payload

    name_bvh, data_bvh = load_content(bvh)
    name_wav, data_wav = load_content(wav)


    files = {
        'motion': (name_bvh, data_bvh,  'application/octet-stream'),
        'audio': (name_wav, data_wav, 'audio/wav'),
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

def combine_mp4s(mp4s, output_path, output_name="story"):
    # Load each video clip
    video_clips = [VideoFileClip(file) for file in mp4s]

    # Concatenate the video clips
    final_clip = concatenate_videoclips(video_clips)

    # Write the concatenated clip to an output file
    output_file = 'output.mp4'
    path = os.path.join(output_path, f"{output_name}.mp4")
    final_clip.write_videofile(path, codec='libx264')
    # Close the video clips
    for clip in video_clips:
        clip.close()
    return path



class Worker:
    def __init__(self, index, voice, sentiment, text, output_path,logger=None, **kvargs):
        self.index = index
        self.voice = voice
        self.sentiment = sentiment
        self.text = text
        self.wav = None
        self.bvh = None
        self.fbx = None
        self.mp4 = None
        self.wav_path = None
        self.bvh_path = None
        self.fbx_path = None
        self.mp4_path = None
        self.state = "NOT_STARTED"
        self.error = None
        self.worker = None
        self.output_path = output_path
        self.kvargs = kvargs
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

    def dispatch(self, wav):
        try:
            self.state = "RUNNING"
            bvh_id = dispatch_generate_bvh(wav, style=self.sentiment)
            self.logger.info(f"index {self.index} - bvh_id {bvh_id}")
            self.bvh = wait_and_get(bvh_id)
            self.logger.info(f"index {self.index} - bvh done")

            fbx_id = dispatch_generate_fbx(self.bvh)
            self.logger.info(f"index {self.index} - fbx_id {fbx_id}")

            mp4_id = dispatch_generate_mp4(self.bvh, wav)
            self.logger.info(f"index {self.index} - mp4_id {mp4_id}")

            self.fbx = wait_and_get(fbx_id)
            self.logger.info(f"index {self.index} - fbx done")

            self.save_fbx(sync=False)

            self.mp4 = wait_and_get(mp4_id)
            self.logger.info(f"index {self.index} - mp4 done")


            self.logger.info(f"index {self.index} - done")
            self.state = "SUCCESS"
        except Exception as e:
            self.state = "FAILURE"
            self.error = e
            raise e


    def __call__(self, device="cuda:0") -> Any:
        try:
            self.logger.info(f"index {self.index} - tts")
            self.wav, wav_path = text_to_speech(self.text, self.voice, index=self.index, device=device, output_path=self.output_path, **self.kvargs)
            self.logger.info(f"index {self.index} - wav done")
            # self.dispatch(wav_path)
            # do dispatch to server in a thread
            self.worker = threading.Thread(target=self.dispatch, args=(wav_path,))
            self.worker.start()
            
        except Exception as e:
            self.state = "FAILURE"
            self.error = e
            raise e
        
    def join(self):
        if self.worker is not None:
            try:
                self.worker.join()
            except Exception as e:
                self.logger.error(f"index {self.index} - worker join failed")

    def save_wav(self, sync=True):
        self.wav_path = os.path.join(self.output_path, f"{self.index}_{self.voice}.wav")
        self.logger.info(f"index {self.index} - saving wav")
        if sync:
            self.join()
        torchaudio.save(self.wav_path, self.wav, 24000)
        return self.wav_path

    def save_bvh(self, sync=True):
        self.bvh_path = os.path.join(self.output_path, f"{self.index}_{self.voice}.bvh")
        self.logger.info(f"index {self.index} - saving bvh")
        if sync:
            self.join()
        save_data(self.bvh, self.bvh_path)
        return self.bvh_path

    def save_fbx(self, sync=True):
        self.fbx_path = os.path.join(self.output_path, f"{self.index}_{self.voice}.fbx")
        self.logger.info(f"index {self.index} - saving fbx")
        if sync:
            self.join()
        save_data(self.fbx, self.fbx_path)
        return self.fbx_path
    
    def save_mp4(self, sync=True):
        self.mp4_path = os.path.join(self.output_path, f"{self.index}_{self.voice}.mp4")
        self.logger.info(f"index {self.index} - saving mp4")
        if sync:
            self.join()
        save_data(self.mp4, self.mp4_path)
        return self.mp4_path

    def get_bvh(self):
        self.join()
        return self.bvh
    
    def get_fbx(self):
        self.join()
        return self.fbx
    
    def get_mp4(self):
        self.join()
        return self.mp4
    
    def get_wav(self):
        return self.wav


def zip_story(path, exts, name="story", alias=None):
    # List of files to be included in the ZIP archive
    files = []
    for ext in exts:
        files.extend(glob.glob(f"{path}/*.{ext}"))

    # Name of the output ZIP file
    output_zip = f'{path}/{name}.zip'

    # Open the output ZIP file in write mode
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for original_file in files:
            # Add each file to the ZIP archive with a new name
            name = Path(original_file).stem.split('_')[-1]
            new_file = Path(original_file).name
            if alias is not None:
                if name in alias:
                    new_file = Path(original_file.replace(name, alias[name])).name
            print(f"Adding {original_file} as {new_file}")
            zipf.write(original_file, arcname=new_file)

    print(f'ZIP archive "{output_zip}" created successfully.')

    return output_zip


