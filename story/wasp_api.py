import requests
import random
import time
import logging
from pathlib import Path



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
        pose="pose_6",
    ):
    if seed is None:
        seed = random.randint(0, 2**32-1)

    # The URL to make the POST request to
    url = f'{base_url}/generate_bvh/'

    # Prepare the multipart/form-data payload
    name, data = load_content(wav)

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

def wav_to_fbx(wav, 
               base_url='http://129.192.81.237', 
               ):
    filename = Path(wav).name
    ext = filename.split(".")[-1]
    bvh_id = dispatch_generate_bvh(wav, base_url=base_url)
    bvh = wait_and_get(bvh_id, base_url=base_url)
    fbx_id = dispatch_generate_fbx(bvh, base_url=base_url)
    fbx = wait_and_get(fbx_id, base_url=base_url)
    path = Path(wav).parent / f"{filename.replace(ext, 'fbx')}"
    save_data(fbx, path)
    return path



