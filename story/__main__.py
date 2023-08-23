from story.utils import text_to_speech, parse_story, combine_mp4s, Worker
import argparse
import torch
import torchaudio
import os
import logging
import tqdm
import time
import threading
import multiprocessing
from functools import partial
import shutil

class dummy:
    def is_alive():
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to story json file', default="story.json")
    parser.add_argument('--output_path', type=str, help='path to output folder', default="results/")
    parser.add_argument('--output_name', type=str, help='name of output file', default="story")
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='high_quality')  # "ultra_fast", "fast", "standard", "high_quality"
  
    args = parser.parse_args()
    all_workers = []
    # if ctl+c is pressed, join all workers
    the_story = parse_story(args.file)

    #copy file to output folder
    os.makedirs(args.output_path, exist_ok=True)
    shutil.copy(args.file, args.output_path)
    start = time.time()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{args.output_path}/{args.output_name}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])
    logging.info("Starting story generation.")

    logging.info(f"Story has {len(the_story)} parts.")
    try:
        for index, voice, sentiment, text in the_story:
            logging.info(f"Dialog {index}/{len(the_story)} time passed: {int(time.time() - start)/60} min {int(time.time() - start)%60} seconds.")
            worker = Worker(index, voice, sentiment, text, preset=args.preset, output_path=args.output_path)
            worker()
            all_workers.append(worker)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt, joining all workers.")
        for w in all_workers:
            w.join()
        logging.info("All workers joined.")
    print("Fetching all from compute server ...")
    for w in tqdm.tqdm(all_workers):
        # w.join()
        w.save_mp4()

    #combine all wavs
    print("Combining all wavs...")
    all_wav = [worker.get_wav() for worker in all_workers]
    full_audio = torch.cat(all_wav, dim=-1)
    torchaudio.save(os.path.join(args.output_path, f"{args.output_name}.wav"), full_audio, 24000)
    
    #combine all mp4s
    print("Combining all mp4s...")
    all_mp4s = [ w.mp4_path for w in all_workers]
    combine_mp4s(all_mp4s, args.output_path, args.output_name)
    tot_time = int(time.time() - start)
    logging.info(f"Finished story generation in {tot_time//60} min  {tot_time%60} seconds.")
    print(f"Finished story generation in {tot_time//60} min  {tot_time%60} seconds.")

if __name__ == '__main__':
    main()