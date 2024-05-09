# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace
import random

import datetime
import copy
from PIL import Image
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler
from drag_pipeline import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from drag_utils import drag_diffusion_update

import base64
from base64 import b64encode

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers.utils import load_image

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

chosen_index = 0
def selected_gallery(event_data: gr.EventData):
    global chosen_index
    chosen_index = event_data._data["index"]
    return
def set_as_input(gallery):
    filename = gallery[chosen_index]["name"]
    res =cv2.imread(filename)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res

# initialize the stable diffusion model
diffusion_model_path = "runwayml/stable-diffusion-v1-5"
# diffusion_model_path = "stable-diffusion-v1-5"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
# diffusion_model_path = "stable-diffusion-v1-5"
model = DragPipeline.from_pretrained(diffusion_model_path, scheduler=scheduler).to(device)
# call this function to override unet forward function,
# so that intermediate features are returned after forward
model.modify_unet_forward()

def preprocess_image(image, device, resolution=512):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = F.interpolate(image, (resolution, resolution))
    image = image.to(device)
    return image

def mask_image(image, mask, color=[255,0,0], alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    contours = cv2.findContours(np.uint8(deepcopy(mask)), cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return out

def inference(source_image,
              image_with_clicks,
              mask,
              prompt,
              points,
              lora_path,
              n_actual_inference_step=40,
              lam=0.1,
              n_pix_step=40,
              save_dir="./results"
    ):

    seed = 42 # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    # args.n_inference_step = n_inference_step
    args.n_inference_step = 50
    args.n_actual_inference_step = n_actual_inference_step
    
    # args.guidance_scale = guidance_scale
    args.guidance_scale = 1.0

    # unet_feature_idx = unet_feature_idx.split(" ")
    # unet_feature_idx = [int(k) for k in unet_feature_idx]
    # args.unet_feature_idx = unet_feature_idx
    args.unet_feature_idx = [2]

    # args.sup_res = sup_res
    args.sup_res = 256

    # args.r_m = r_m
    # args.r_p = r_p
    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    # args.lr = lr
    args.lr = 0.01

    args.n_pix_step = n_pix_step
    print(args)
    full_h, full_w = source_image.shape[:2]

    if diffusion_model_path == 'stabilityai/stable-diffusion-2-1':
        source_image = preprocess_image(source_image, device, resolution=768)
        image_with_clicks = preprocess_image(image_with_clicks, device, resolution=768)
    else:
        source_image = preprocess_image(source_image, device, resolution=512)
        image_with_clicks = preprocess_image(image_with_clicks, device, resolution=512)

    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=n_actual_inference_step)

    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res, args.sup_res), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1] / full_h, point[0] / full_w]) * args.sup_res
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    init_code = invert_code
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    updated_init_code, updated_text_emb = drag_diffusion_update(model, init_code, t,
        handle_points, target_points, mask, args)

    # inference the synthesized image
    gen_image = model(prompt,
        prompt_embeds=updated_text_emb,
        latents=updated_init_code,
        guidance_scale=1.0,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=n_actual_inference_step
        )

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image * 0.5 + 0.5,
        torch.ones((1,3,512,25)).cuda(),
        image_with_clicks * 0.5 + 0.5,
        torch.ones((1,3,512,25)).cuda(),
        gen_image[0:1]
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image

# order: target point, handle point
# colors = [(0, 0, 255), (255, 0, 0)]

# False: 使用ControlNet生成，True：使用DragDiff
flag = False

with gr.Blocks() as demo:
    # with gr.Row():
    #     gr.Markdown("""
    #     # Official Implementation of [DragDiffusion](https://arxiv.org/abs/2306.14435)
    #     """)

    with gr.Tab(label="Image"):
        with gr.Row():
            # input image
            original_image = gr.State(value=None) # store original image
            mask = gr.State(value=None) # store mask
            selected_points = gr.State([]) # store points
            length = 480
            # with gr.Column():
            #     gr.Markdown("""<p style="text-align: center; font-size: 25px">Draw Mask</p>""")
            #     canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask", show_label=True).style(height=length, width=length) # for inpainting
            #     gr.Markdown(
            #         """
            #         Instructions: 1. Draw a mask; 2. Click points;

            #         3. Input prompt and LoRA path; 4. Run results.
            #         """
            #     )
            # with gr.Column():
            #     gr.Markdown("""<p style="text-align: center; font-size: 25px">Click Points</p>""")
            #     input_image = gr.Image(type="numpy", label="Click Points", show_label=True).style(height=length, width=length) # for points clicking
            #     undo_button = gr.Button('Undo point')
            # with gr.Column():
            #     gr.Markdown("""<p style="text-align: center; font-size: 25px">Editing Results</p>""")
            #     output_image = gr.Image(type="numpy", label="Editing Results", show_label=True).style(height=length, width=length)
            #     run_button = gr.Button("Run")
            with gr.Column():
                main_image = gr.Image(visible=True, tool="sketch", label="main", show_label=True, scale=10)
                hidden_image = gr.Image(visible=False, label="hidden", show_label=True, scale=10)
                stored_image = gr.Image(visible=False, scale=1)
                with gr.Row():
                    drag_button = gr.Button("选择拖拽点")
                    undo_button = gr.Button("删除所有拖拽点")
                    switch_button = gr.Button("切换显示")
                with gr.Row():
                    strength = gr.Slider(minimum=0.0,maximum=1.0,value=0.6)
                    prompt = gr.Textbox(label="prompt")
                    lora_path = gr.Textbox(value="", label="lora path")
                with gr.Row():
                    generate_button = gr.Button("生成图像")
                    SDXL_button = gr.Button("使用SDXL生成")
                    matrix_button = gr.Button("生成图像矩阵")
                    setinput_button = gr.Button("设置矩阵图像到输入")
                    SD15_button = gr.Button("使用SD1.5生成")
                matrix_gallery = gr.Gallery()
                matrix_gallery.style(grid=8)

        # Parameters
        # with gr.Accordion(label='Parameters', open=True):
        #     with gr.Row():
        #         prompt = gr.Textbox(label="prompt")
        #         lora_path = gr.Textbox(value="", label="lora path")

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, evt: gr.SelectData):
        # collect the selected point
        sel_pix.append(evt.index)
        # draw points
        points = []
        for idx, point in enumerate(sel_pix):
            if idx % 2 == 0:
                # draw a red circle at the handle point
                cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
            else:
                # draw a blue circle at the handle point
                cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
            points.append(tuple(point))
            # draw an arrow from handle point to target point
            if len(points) == 2:
                cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
                points = []
        return img if isinstance(img, np.ndarray) else np.array(img)
    
    hidden_image.select(
        get_point,
        [hidden_image, selected_points],
        [hidden_image],
    )

    # clear all handle/target points
    def undo_points(original_image, mask):
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = original_image.copy()
        return masked_img, []

    undo_button.click(
        undo_points,
        [original_image, mask],
        [hidden_image, selected_points]
    )

    def SDXLapi(main_image, prompt, strength):
        copy_img = main_image["image"]
        copy_img = cv2.resize(copy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        stored_img = copy_img.copy()
        init_image = Image.fromarray(main_image['image'])
        init_image = init_image.resize((512, 512))
        model_key_refiner = "/data0/group1/SD_XL/SDXL0.9/stable-diffusion-xl-refiner-0.9"
        pipe = DiffusionPipeline.from_pretrained(
            model_key_refiner, torch_dtype=torch.float16
        ).to("cuda")
        image = pipe(prompt=prompt, image=init_image, num_inference_steps=20, strength=strength).images[0]
        return image, stored_img

    SDXL_button.click(
        fn=SDXLapi,
        inputs=[main_image, prompt, strength],
        outputs=[main_image, stored_image]
    )
    
    def SD15api(main_image, prompt, strength):
        copy_img = main_image["image"]
        copy_img = cv2.resize(copy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        stored_img = copy_img.copy()
        init_image = Image.fromarray(main_image['image'])
        init_image = init_image.resize((512, 512))
        model_key_sd21 = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_key_sd21, torch_dtype=torch.float16
        ).to("cuda")
        image = pipe(prompt=prompt, image=init_image, num_inference_steps=20, strength=strength).images[0]
        return image, stored_img
    
    SD15_button.click(
        fn=SD15api,
        inputs=[main_image, prompt, strength],
        outputs=[main_image, stored_image]
    )

    def controlnetapi(main_image, prompt, mask):
        init_image = Image.fromarray(main_image['image'])
        mask_image = Image.fromarray(main_image['mask'])
        init_image = init_image.resize((512, 512))
        mask_image = mask_image.resize((512, 512))
        control_image = make_inpaint_condition(init_image, mask_image)
        controlnet_seed = int(random.randrange(4294967294))
        generator = torch.Generator(device="cuda").manual_seed(controlnet_seed)
        image = np.array(init_image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        canny_image = Image.fromarray(image)
        canny_image = canny_image.resize((512, 512))

        controlnet_inpaint = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        ).to("cuda")
        controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
        ).to("cuda")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=MultiControlNetModel([controlnet_inpaint, controlnet_canny]),
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        image = pipe(
            prompt,
            num_inference_steps=40,
            generator=generator,
            eta=1.0,
            image=init_image,
            mask_image=mask_image,
            control_image=[control_image,canny_image],
            strength=1.0
        ).images[0]

        return image
    
    # ControlNet Inpaint+Lineart，需要后台开一个7864 sd端口
    import requests, json
    def submit_post(url: str, data: dict):
        return requests.post(url, data=json.dumps(data))
    def save_encoded_image(b64_image: str, output_path: str):
        with open(output_path, "wb") as image_file:
            image_file.write(base64.b64decode(b64_image))
    def controlnetrequest(main_image, prompt, weight=0.8):
        revert_image = cv2.cvtColor(main_image['image'], cv2.COLOR_RGB2BGR)
        retval, buffer = cv2.imencode('.jpg', revert_image)
        init_image = b64encode(buffer).decode("utf-8")
        retval, buffer2 = cv2.imencode('.jpg', main_image['mask'])
        mask_image = b64encode(buffer2).decode("utf-8")
        txt2img_url = 'http://127.0.0.1:7864/sdapi/v1/txt2img'
        payload = {
            "prompt": prompt,
            "negative_prompt": "",
            "batch_size": 1,
            "steps": 20,
            "cfg_scale": 7,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": init_image,
                            "mask": '',
                            "module": "canny",
                            # "model": "control_v11p_sd15_lineart [43d4be0d]",
                            "model": "control_v11p_sd15_canny [d14c016b]",
                            "weight": weight,
                            "control_mode": 1,
                        },
                        {
                            "input_image": init_image,
                            "mask": mask_image,
                            "module": "inpaint_only",
                            "model": "control_v11p_sd15_inpaint [ebff9138]",
                            "weight": 1,
                            "control_mode": 1,
                        }
                    ]
                }
            }
        }
        response = submit_post(txt2img_url, payload)
        save_encoded_image(response.json()['images'][0],'genImage.png')
        output = cv2.imread('genImage.png')
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output


    def generate(original_image, hidden_image, mask, prompt, selected_points, lora_path, main_image):
        copy_img = main_image["image"]
        copy_img = cv2.resize(copy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        stored_img = copy_img.copy()
        global flag
        if flag == True:
            flag = False
            ret_image = inference(original_image, hidden_image, mask, prompt, selected_points, lora_path)
        else:
            ret_image = controlnetapi(main_image, prompt, mask)
            # ret_image = controlnetrequest(main_image, prompt)
        return ret_image, gr.Image.update(visible=False), gr.Image.update(visible=True), stored_img

    generate_button.click(
        generate,
        [original_image, hidden_image, mask, prompt, selected_points, lora_path, main_image],
        [main_image, hidden_image, main_image, stored_image]
    )

    def switch(main_image, stored_image):
        main_copy = main_image["image"]
        # stored_copy = stored_image["image"]
        stored_copy = stored_image
        main_copy = cv2.resize(main_copy, (512, 512), interpolation=cv2.INTER_LINEAR)
        stored_copy = cv2.resize(stored_copy, (512, 512), interpolation=cv2.INTER_LINEAR)
        main_image_copy = main_copy.copy()
        stored_image_copy = stored_copy.copy()
        return stored_image_copy, main_image_copy
    
    switch_button.click(
        switch,
        [main_image, stored_image],
        [main_image, stored_image]
    )

    def store_img(img):
        global flag
        flag = True
        image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
        # resize the input to 512x512
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = image.copy()
        # when new image is uploaded, `selected_points` should be empty
        return image, [], masked_img, mask, gr.Image.update(visible=False), gr.Image.update(visible=True)
    
    drag_button.click(
        store_img,
        [main_image],
        [original_image, selected_points, hidden_image, mask, main_image, hidden_image]
    )

    def generate_matrix(main_image, prompt):
        ret_images = []
        for i in range(0,5):
            weight = 2-i*0.3
            ret_image = controlnetrequest(main_image, prompt, weight=weight)
            ret_images.append(ret_image)
        return ret_images
    
    matrix_button.click(
        generate_matrix,
        [main_image, prompt],
        [matrix_gallery]
    )

    matrix_gallery.select(
        fn=selected_gallery,
        inputs=[],outputs=[])
    setinput_button.click(
        fn=set_as_input,
        inputs=[matrix_gallery],
        outputs=[main_image])

demo.queue().launch(share=True, debug=True, enable_queue=True)
