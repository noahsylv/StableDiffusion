import os
import cv2
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import argparse
from optimizedSD.optimized_img2img import optimized_image_two_image
import random
import shutil

def fixed_width_number(x, length=4):
    out = str(x)
    while len(out) < length:
        out = "0" + out
    return out

# python optimizedSD/sequence_generation.py --prompt "person"

# python optimizedSD/sequence_generation.py --prompt "snakes in the grass" --video-file "assets/videos/pa2.mp4" --fps 1/5 --strength 0.3 --W 940 --H 512

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file", type=str, help="dir to write results to", default="assets/videos/vid2.mp4")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--strength", type=float, default=.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    opt = parser.parse_args()
    test = 10

    # read in video
    video_name_encoded = opt.video_file.split("/")[-1].split(".")[0].replace(" ", "_").lower()
    prompt_encoded = opt.prompt.replace(" ", "_").lower()
    frame_path = f"assets/frames/{video_name_encoded}"
    outdir = f"outputs/sequence-generation/{video_name_encoded}/{prompt_encoded}_{opt.strength}_{opt.seed}_{opt.fps}_{opt.W}_{opt.H}"
    final_outdir = f"outputs/sequence-generation-processed/{video_name_encoded}/{prompt_encoded}_{opt.strength}_{opt.seed}_{opt.fps}_{opt.W}_{opt.H}"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    seed = int(random.random() * 1000)
    seed = opt.seed
    for i, frame in enumerate(get_frames_from_video(opt.video_file, opt.fps)):
        # if i > 3:
        #     break
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        frame_filename = f"{frame_path}/{fixed_width_number(i)}.jpg"
        cv2.imwrite(frame_filename, frame)
        optimized_image_two_image(    
            outdir=outdir,
            seed=seed+i, init_img=frame_filename, H=opt.H, W=opt.W, device='cuda',
            unet_bs=1, turbo=opt.turbo, precision="autocast", n_samples=1,
            n_rows=0, from_file=None, prompt=opt.prompt, strength=opt.strength,
            ddim_steps=50, n_iter=1, scale=10.0, ddim_eta=0.0,
            sampler="ddim", format="png"
        )
    rename_sequence(outdir + "/" + prompt_encoded, final_outdir)

def rename_sequence(directory, out_directory):
    files = os.listdir(directory)

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # copy to new directory
    for f in files:
        new_name = f.split("_")[-1]
        print(f)
        shutil.copy(directory + "/" + f, out_directory + "/" + new_name)



def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def get_frames_from_video(video_file, saving_frames_per_second):
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # saving_frames_per_second = min(fps, 10)
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break
        frame_duration = count / fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            break
        if frame_duration >= closest_duration:
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            yield frame
            # cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1
    return frame

if __name__ == '__main__':
    main()