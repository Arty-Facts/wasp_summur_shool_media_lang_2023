from story.utils import text_to_speech, parse_story, Worker
import argparse
import torch
import torchaudio
import os
import logging
import tqdm
def main():
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("story.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])
    logging.info("Starting story generation.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to story json file', default="story.json")
    parser.add_argument('--output_path', type=str, help='path to output folder', default="results/")
    parser.add_argument('--output_name', type=str, help='name of output file', default="story")
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='ultra_fast')  # "ultra_fast", "fast", "standard", "high_quality"
  
    args = parser.parse_args()
    all_workers = []
    # if ctl+c is pressed, join all workers

    try:
        for index, voice, sentiment, text in parse_story(args.file):
            worker = Worker(index, voice, sentiment, text, preset=args.preset, output_path=args.output_path)
            worker()
            all_workers.append(worker)
    except KeyboardInterrupt:
        print("Keyboard Interrupt, joining all workers.")
        for w in all_workers:
            w.join()
        logging.info("All workers joined.")
    print("Finalizing all workers...")
    for w in tqdm.tqdm(all_workers):
        # w.join()
        w.save_mp4()

    all_parts = [worker.get_wav() for worker in all_workers]
    full_audio = torch.cat(all_parts, dim=-1)
    torchaudio.save(os.path.join(args.output_path, f"{args.output_name}.wav"), full_audio, 24000)


if __name__ == '__main__':
    main()