from story.utils import text_to_speech, parse_story, Worker
import argparse
import torch
import torchaudio
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to story json file', default="story.json")
    parser.add_argument('--output_path', type=str, help='path to output folder', default="results/")
    parser.add_argument('--output_name', type=str, help='name of output file', default="story")
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='high_quality')  # "ultra_fast", "fast", "standard", "high_quality"
  
    args = parser.parse_args()
    all_workers = []
    for index, voice, sentiment, text in parse_story(args.file):
        audio, path = text_to_speech(text, voice, index=index, preset=args.preset, output_path=args.output_path)
        worker = Worker(index, voice, sentiment, text, preset=args.preset, output_path=args.output_path)
        all_workers.append(worker)
        #fbx files 
        

    full_audio = torch.cat(all_parts, dim=-1)
    torchaudio.save(os.path.join(args.output_path, f"{args.output_name}.wav"), full_audio, 24000)

if __name__ == '__main__':
    main()