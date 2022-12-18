from flask import Flask, json, send_file, request
import random
import os
from optimizedSD.optimized_txt2img import run_txt2img, get_prompt_path

api = Flask(__name__)

# python optimizedSD/optimized_txt2img.py --prompt "a photo of a dog" 
# --H 512 --W 512 --seed 37 --n_iter 5 --n_samples 20 --ddim_steps 50 
# --ddim_eta 0.0 --scale 10.0


def get_filename_from_directory(path):
    if not os.path.exists(path):
        return None
    try:
        return path + "/" + os.listdir(path)[0]
    except IndexError:
        return None
def get_image_from_prompt(prompt):
    # get single image from prompt
    # prompt_path = prompt.replace(" ", "_").lower()
    prompt_path = get_prompt_path("outputs/server-samples", prompt).replace("\\", "/")
    print(prompt_path)
    # path = f"outputs/server-samples/{prompt_path}"
    existing_img = get_filename_from_directory(prompt_path)
    if not existing_img:
        run_txt2img(
            outdir="outputs/server-samples", 
            ckpt="models/ldm/stable-diffusion-v1/model.ckpt",
            unet_bs=1,
            device="cuda",
            turbo=False,
            precision="autocast",
            fixed_code=False,
            n_samples=1,
            C=4,
            H=512,
            W=512,
            from_file=None,
            n_iter=1,
            scale=10.0,
            ddim_steps=50,
            ddim_eta=0.0,
            sampler="plms",
            prompt=prompt,
            n_rows=0,
            f=8,
            format="png",
            seed = 1
        )
    return get_filename_from_directory(prompt_path)

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

@api.route('/')
def index():
    with open("index.html", "r") as f:
        return f.read() 

@api.route('/companies', methods=['GET'])
def get_companies():
    return json.dumps(companies)

@api.route('/test', methods=['GET'])
def test():
    return json.dumps(companies)


@api.route('/generate_image', methods=['GET'])
def generate_image():
    prompt = request.args.get('prompt').replace("\"", "")
    print(prompt)
    if prompt is None:
        return f"Need prompt {random.random()}"
    else:
        filename = get_image_from_prompt(prompt)
        return send_file(filename, mimetype='image/gif')

if __name__ == '__main__':
    api.run()