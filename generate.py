#
# Created on Mon Jul 25 2023
#
# Copyright (c) 2023 rlsn
#
# !pip install diffusers transformers safetensors einops scipy

from tripper import Tripper, schedulers
import diffusers
import argparse
from attrdict import AttrDict
from utils import const_schedule, zoom, export_as_gif, timestr, interpolate_animation
import PIL

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help="filename of the running configuration", default="config.json")

    args = parser.parse_args()
    config = AttrDict.from_json(args.config_file)
    print(f"running with config:{config}")

    tripper = Tripper(config.model_path)
    tripper.set_scheduler(schedulers[config.scheduler])
    if config.generate_animation:
        config.init_image = PIL.Image.open(config.init_image)
        # strength schedule
        config.strength_schedule = const_schedule(config.strength,config.nframes)       
        config.transform_fn = lambda img,s: zoom(img, config.zoom)
        config.nsteps=int(config.nframes//config.diffusion_cadence)
        imgs = tripper.generate_animation(**config)
        imgs = interpolate_animation(imgs, config.diffusion_cadence)
        export_as_gif(f"{config.out_dir}/{timestr()}.gif", imgs, frames_per_second=config.fps)
    else:
        tripper.txt2img(**config)