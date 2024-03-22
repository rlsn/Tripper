#
# Created on Mon Jul 25 2023
#
# 2023 rlsn
#
from collections import defaultdict
from einops import einsum
import torch
import PIL
from PIL import Image
import numpy as np
import time
from safetensors.torch import load_file
from scipy.stats import norm

def timestr():
    return time.strftime('%Y%m%d%H%M%S', time.localtime())

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )

def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype, load=True):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            x = einsum(weight_up.expand(-1,-1,weight_down.size(2), weight_down.size(3)),weight_down,"c1 k h w, k c2 h w -> c1 c2 h w")
            if load:
                curr_layer.weight.data += multiplier * alpha * x
                # curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data -= multiplier * alpha * x
                # curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            if load:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
            else:
                curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


def convert_prompt_embeds(pipe, prompt,negative_prompt):
    max_length = pipe.tokenizer.model_max_length

    input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids

    negative_ids = pipe.tokenizer(negative_prompt, return_tensors="pt").input_ids
    if input_ids.shape[-1]>negative_ids.shape[-1]:
        negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
    elif input_ids.shape[-1]<negative_ids.shape[-1]:
        input_ids = pipe.tokenizer(prompt, truncation=False, padding="max_length", max_length=negative_ids.shape[-1], return_tensors="pt").input_ids

    input_ids = input_ids.to("cuda")
    negative_ids = negative_ids.to("cuda")

    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    return prompt_embeds, negative_prompt_embeds


def clean_prompt(s):
    tokens = s.split(',')
    clean = []
    for t in tokens:
        t = t.strip()
        if t.startswith("<lora"):
            continue
        if t not in clean:
              clean.append(t)
    s = ", ".join(clean)
    return s

def cos_schedule(low,high,phase,cycles,steps):
    amp = (high-low)/2
    offset = (high+low)/2
    return offset+np.cos(np.linspace(phase,phase+2*np.pi*cycles,steps))*amp

def const_schedule(v,steps):
    return np.array([v]*steps)

def zoom(im,ratio):
    if ratio<1:
        s = im.size
        w,h = im.size[0]*ratio, im.size[1]*ratio
        m = (s[0]-w)/2,(s[1]-h)/2
        nim = im.crop((m[0], m[1], s[0]-m[0], s[1]-m[1]))
        return nim.resize(s)
    elif ratio>1:
        s = im.size
        r=ratio-1
        m = int((s[0]*r)//2), int((s[1]*r)//2)
        im = np.array(im)
        nim = np.pad(im, ((m[1], m[1]), (m[0], m[0]), (0, 0)), mode='symmetric') 
        nim = Image.fromarray(nim)
        return nim.resize(s)
    else:
        return im

def impulse_schedule(floor,ceiling,impulse,width,steps):
    x = np.arange(steps)
    Y=[]
    for imp in impulse:
        y = norm.pdf(x,imp, width)
        y*=(ceiling-floor)/y.max()
        Y.append(y)
    Y=np.array(Y).sum(0)
    print(Y.shape)
    return Y+floor

def interpolation(img1, img2, num_frame=1):
    img1 = np.array(img1,dtype=float)
    img2 = np.array(img2,dtype=float)
    d = (img2 - img1)/(num_frame+1)
    imgs = []
    for i in range(1,num_frame+1):
        im = img1+d*i
        imgs.append(Image.fromarray(im.astype(np.uint8)))
    return imgs

def interpolate_video(imgs, cadence=2):
    if cadence<=1:
        return imgs
    else:
        result = []
        for i in range(len(imgs)-1):
            result+=[imgs[i]]+interpolation(imgs[i],imgs[i+1],cadence-1)
        result+=[imgs[-1]]
    return result