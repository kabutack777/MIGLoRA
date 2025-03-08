import torch
from safetensors.torch import load_file
import pdb
import json
import os
import sys
import PIL
import random
import numpy as np
from mask_encoder import ControlNetConditioningEmbedding
from safetensors.torch import load_file
from diffusers import StableDiffusionLoRAMultiLayoutPipeline, DPMSolverMultistepScheduler,UniPCMultistepScheduler

base_model_path = "models/realisticVisionV51_v51VAE"
pid = int(sys.argv[1])
tols = int(sys.argv[2])
unet_flag = int(sys.argv[3])
cfg = float(sys.argv[4])
ckpt_in = sys.argv[5]
base_model = sys.argv[6]
schd = sys.argv[7]
save_dir_base = sys.argv[8]
lora_path = sys.argv[9]
mask_encoder_path = sys.argv[10]


state_dict = load_file(mask_encoder_path, device="cpu")

mask_encoder = ControlNetConditioningEmbedding()
mask_encoder.load_state_dict(state_dict)
mask_encoder.to('cuda')
mask_encoder.eval()

pipe = StableDiffusionLoRAMultiLayoutPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16,safety_checker = None,)
pipe.load_lora_weights(lora_path)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if schd == "UniPCM":
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
elif schd == "DPM":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
else:
    raise ValueError("Scheduler setup error.")

pipe.enable_model_cpu_offload()
pipe.to('cuda')


def load_image(image_path):
    with open(image_path, 'rb') as f:
        with PIL.Image.open(f) as image:
            image = image.convert('RGB')
    return image
def calculate_weights(bounding_boxes, base_weight=1.0, smoothing_factor=1e-5):
    def calculate_box_area(box):
        xmin, ymin, xmax, ymax = box
        width = max(xmax - xmin, 0)  
        height = max(ymax - ymin, 0)  
        area = width * height
        return area

    areas = [calculate_box_area(box) for box in bounding_boxes]
    weights = [base_weight / (area + smoothing_factor) for area in areas]
        
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    weights_list = weights.tolist()
    weights_list.insert(0, 1.0)
    weights = torch.tensor(weights_list, dtype=torch.float32)
        
    return weights


save_dir = "%s/image_grit_7k_%s_%s_%s_cfg%s_512P-reshape_mask"  % (save_dir_base, base_model, ckpt_in, schd, cfg)
save_dir_bbox = "%s/image_grit_7k_%s_%s_%s_cfg%s_512P-bbox-reshape_mask"  % (save_dir_base, base_model, ckpt_in, schd, cfg)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(save_dir_bbox):
    os.mkdir(save_dir_bbox)


file_json = ""

with open(file_json, encoding='utf-8') as f:
    json_data = json.load(f)

cnt = 0
for v in json_data:
    cnt += 1
    if cnt % tols != pid:
        continue
    base_info, caption, obj_nums, img_size, path_img, list_bbox_info, crop_location = v
    img_id = base_info["id"]

    image = load_image(path_img)

    obj_bbox = [obj[1] for obj in list_bbox_info]
    obj_bbox = np.array(obj_bbox)
    obj_class = [obj[0] for obj in list_bbox_info]


    W, H = image.size
    r_image = image
    r_obj_bbox = obj_bbox
    r_obj_class = obj_class


    if W != 512 and H != 512:
        print ("image size is not 512." % img_id)
        continue

    cond_image = np.zeros_like(r_image, dtype=np.uint8)
    mask_images = []
    list_cond_image = []

    for iit in range(0, len(r_obj_bbox)): 
        dot_bbox = r_obj_bbox[iit]
        dx1, dy1, dx2, dy2 = [int(xx) for xx in dot_bbox]
        cond_image = np.zeros_like(r_image, dtype=np.uint8)
        cond_image[dy1:dy2, dx1:dx2] = 1
        list_cond_image.append(cond_image)

        mask_image = torch.from_numpy(cond_image).to(torch.float32)
        mask_image = mask_image.permute(2, 0, 1)
        mask_images.append(mask_image)

    obj_cond_image = np.stack(list_cond_image, axis=0)

    obj_mask_image = torch.stack(mask_images,dim = 0).to('cuda')

    mask_cond = mask_encoder(obj_mask_image)


    list_bbox_info = [r_obj_class, obj_cond_image]

    layo_prompt = r_obj_class
    layo_bbox = torch.FloatTensor(r_obj_bbox)
    layo_cond = torch.FloatTensor(obj_cond_image)

    if unet_flag:
        prompt = caption
    else:
        prompt = ""

    seed = -1
    if seed == -1:
        seed = int(random.randrange(4294967294))
    generator = torch.manual_seed(seed)
    
    list_cond_image_pil = [PIL.Image.fromarray(dot_cond).convert('RGB') for dot_cond in list_cond_image]

    image = pipe(
        prompt, layo_prompt,mask_type ='reshape', guess_mode=False, guidance_scale=cfg,
        num_inference_steps=50, image=list_cond_image_pil,mask_cond = mask_cond,
        width=512, height=512, generator=generator,
    ).images[0]

    image.save("%s/%s.png" % (save_dir, img_id))
