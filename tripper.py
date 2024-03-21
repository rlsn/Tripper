#
# Created on Mon Jul 25 2023
# a stable diffusion pipline to generate transforming images based on text description
# Copyright (c) 2023 rlsn
#
import diffusers
from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)
import torch
from utils import load_lora_weights, convert_prompt_embeds, clean_prompt, timestr
import os,json

schedulers = {
    "euler": diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
    "euler a":diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
    "DDIM":diffusers.schedulers.scheduling_ddim.DDIMScheduler,
    "DDPM":diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
    "DPM++ 2M SDE Karras":diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
    "DPM++ 2M Karras":diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
}

class Tripper(object):
    def __init__(self, model_file):
        txt2img_pipe = StableDiffusionPipeline.from_single_file(model_file, torch_dtype=torch.float16)
        img2img_pipe = StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)
        txt2img_pipe.safety_checker = None
        img2img_pipe.safety_checker = None

        self.txt2img_pipe = txt2img_pipe.to('cuda')
        self.img2img_pipe = img2img_pipe.to("cuda")
        self.loras = dict()

    def scheduler(self):
        return self.txt2img_pipe.scheduler
    def show_schedulers(self):
        return self.txt2img_pipe.scheduler.compatibles
    def set_scheduler(self, scheduler_cls):
        self.txt2img_pipe.scheduler = scheduler_cls.from_config(self.txt2img_pipe.scheduler.config)
        self.img2img_pipe.scheduler = scheduler_cls.from_config(self.img2img_pipe.scheduler.config)

    def load_lora(self, lora_dict):
        for lora in lora_dict:
            if lora not in self.loras:
                self.txt2img_pipe = load_lora_weights(self.txt2img_pipe, lora, lora_dict[lora], 'cuda', torch.float32, load=True)
                self.loras[lora] = lora_dict[lora]
                print(f"loaded {lora}")
            else:
                print(f"already loaded {lora}")
    def unload_lora(self, lora_dict):
        for lora in lora_dict:
            if lora in self.loras:
                self.txt2img_pipe = load_lora_weights(self.txt2img_pipe, lora, lora_dict[lora], 'cuda', torch.float32, load=False)
                del self.loras[lora]
                print(f"unloaded {lora}")
            else:
                print(f"have not loaded {lora}")

    def txt2img(self, prompt, negative_prompt, lora_dict,
                width=512, height=768, num_img=6, guidance_scale=7, num_inference_steps=25,
                out_dir="preview", **kargs):
        os.makedirs(out_dir, exist_ok = True)

        self.load_lora(lora_dict)

        prompt = clean_prompt(prompt)
        prompt_embeds, negative_prompt_embeds = convert_prompt_embeds(self.txt2img_pipe, prompt, negative_prompt)
        images = self.txt2img_pipe(prompt_embeds=prompt_embeds,
                              negative_prompt_embeds=negative_prompt_embeds,
                              guidance_scale=guidance_scale,
                              num_images_per_prompt=num_img,
                              num_inference_steps=num_inference_steps,
                              height=height, width=width,
                              ).images
        for i,img in enumerate(images):
            fn = f"{out_dir}/{timestr()}_{i}.jpg"
            img.convert("RGB").save(fn)
        # self.unload_lora(lora_dict)

        return images

    def img2img(self, image, prompt, negative_prompt, lora_dict, strength=0.5,
                num_img=6, guidance_scale=7, num_inference_steps=25,
                out_dir="preview", **kargs):
        os.makedirs(out_dir, exist_ok = True)

        self.load_lora(lora_dict)

        prompt = clean_prompt(prompt)
        prompt_embeds, negative_prompt_embeds = convert_prompt_embeds(self.txt2img_pipe, prompt, negative_prompt)
        images = self.img2img_pipe(prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image=image,
                        strength = strength,
                        num_images_per_prompt=num_img,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps).images
        for i,img in enumerate(images):
            fn = f"{out_dir}/{timestr()}_{i}.jpg"
            img.convert("RGB").save(fn)

        self.unload_lora(lora_dict)
        return images

    def generate_video(self, init_image, prompt, negative_prompt, 
                        lora_dict, nsteps, strength_schedule,
                        transform_fn,
                        guidance_scale=7, 
                        num_inference_steps=40,
                        out_dir="preview", **kargs):

        os.makedirs(out_dir, exist_ok = True)

        with open(f"{out_dir}/config.json","w") as fp:
            config = {"prompt":prompt,
            "negative_prompt":negative_prompt,
            "loras":lora_dict,
            "guidance_scale":guidance_scale,
            "num_inference_steps":num_inference_steps,
            }
            json.dump(config,fp,indent=4)

        images = [init_image]
        self.load_lora(lora_dict)


        prompt = clean_prompt(prompt)
        prompt_embeds, negative_prompt_embeds = convert_prompt_embeds(self.txt2img_pipe, prompt, negative_prompt)
        for s in range(nsteps):
            print(f"{s}/{nsteps}")
            image = transform_fn(images[-1], s)
            images += self.img2img_pipe(prompt_embeds=prompt_embeds,
                               negative_prompt_embeds=negative_prompt_embeds,
                              image=image,
                              strength = strength_schedule[s],
                              guidance_scale=guidance_scale,
                              num_inference_steps=num_inference_steps).images

            fn = out_dir+"/%06d.jpg"%s
            images[-1].convert("RGB").save(fn)

        # self.unload_lora(lora_dict)
        return images