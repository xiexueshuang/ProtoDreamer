import html
import json
import math
import mimetypes
import os
import platform
import random
import sys
import tempfile
import time
import traceback
from functools import partial, reduce
import warnings

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import sd_hijack, sd_models, localization, script_callbacks, ui_extensions, deepbooru, sd_vae, extra_networks, postprocessing, ui_components, ui_common, ui_postprocessing
from modules.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML
from modules.paths import script_path, data_path

from modules.shared import opts, cmd_opts, restricted_opts

import modules.codeformer_model
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.scripts
import modules.shared as shared
import modules.styles
import modules.textual_inversion.ui
from modules import prompt_parser
from modules.images import save_image
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
from modules.textual_inversion import textual_inversion
import modules.hypernetworks.ui
from modules.generation_parameters_copypaste import image_from_url_text
import modules.extras

################### 在此插入import #########################
import cv2
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace
import random
import datetime
import copy
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, AutoencoderKL
from modules.drag_pipeline import DragPipeline
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from modules.drag_utils import drag_diffusion_update
from modules.lora_utils import train_lora
import base64
from base64 import b64encode
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image

warnings.filterwarnings("default" if opts.show_warnings else "ignore", category=UserWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

#################### Overall Parameters ##########################3
pos_prompt = [", real , extremely detailed, high tech, light and shadow, only one object, highres, 4k, Industrial design, if design award, full view, complete view, white background, studio lighting, blender, C4D, oc render, futuristic",
              ", real , extremely detailed, high tech, light and shadow, only one object, highres, 4k, full view, complete view, futuristic design, sense of design, digital design, contrast of volume and lines, glass, metal, texture, neon light, transparent plastic, optics, surrealism, composition, fluorescent, bright, ultra details, white background, 3d art, c4d, behance, hyperreal, blender, octane render, behavior, studio lighting, sk, hd.",
              ", (The background is black cloth:1.2),black background,complete view,emauromin style,Industrial Products,Product Design,minimalistic futuristic design,finely detailed,8k,purism,ue 5,minimalism,",
              ", full view, black background, complete view, emauromin style, minimalistic futuristic design, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
              ", real , extremely detailed, high tech, light and shadow, only one object, highres, 4k, full view, complete view, A high-end future design, product photography, realistic style, product layout design image, pearlescent plastic and metallic textured material, trendy art station, pure dark background, film lighting, high detail, ultra-realistic,8k, no text in image",
              ", real , extremely detailed, high tech, light and shadow, only one object, highres, 4k, full view, complete view, A high-end future design, product photography, realistic style, product layout design image, pearlescent plastic and metallic textured material, trendy art station, pure dark background, film lighting, high detail, ultra-realistic,8k, no text in image"]
neg_prompt = ["incomplete, more than one object in the image",
              "incomplete, more than one object in the image",
              "(worst quality:1.4),Nothing in the background,Sparks,flame,Cloud,(low quality:1.4),(normal quality:1.5),lowres,((monochrome)),((grayscale)),cropped,text,jpeg artifacts,signature,watermark,username,sketch,cartoon,drawing,anime,duplicate,blurry,semi-realistic,out of frame,ugly,deformed,weird colors,EasyNegative,flame",
              "(worst quality:1.8), (low quality:1.8), (normal quality:2), lowres, ((monochrome)), ((grayscale)), cropped, text, jpeg artifacts, signature, watermark, username, sketch, cartoon, drawing, anime, duplicate, blurry, semi-realistic, out of frame, ugly, deformed, weird colors, EasyNegative",
              "incomplete, more than one object in the image",
              "incomplete, more than one object in the image"]
drag_flag = False
is_left_flag = False
chosen_index = 0 
chosen_style = 0

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_region
        )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅


def plaintext_to_html(text):
    return ui_common.plaintext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])

def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]

    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    shared.prompt_styles.save_styles(shared.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(2)]


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    from modules import processing, devices

    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)

    with devices.autocast():
        p.init([""], [0], [0])

    return f"resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def apply_styles(prompt, prompt_neg, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, left + ".txt"), 'a'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def create_seed_inputs(target_interface):
    with FormRow(elem_id=target_interface + '_seed_row', variant="compact", visible=False):
        seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1, elem_id=target_interface + '_seed', )
        seed.style(container=False)
        random_seed = ToolButton(random_symbol, elem_id=target_interface + '_random_seed')
        reuse_seed = ToolButton(reuse_symbol, elem_id=target_interface + '_reuse_seed')

        seed_checkbox = gr.Checkbox(label='Extra', elem_id=target_interface + '_subseed_show', value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with FormRow(visible=False, elem_id=target_interface + '_subseed_row') as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=target_interface + '_subseed')
        subseed.style(container=False)
        random_subseed = ToolButton(random_symbol, elem_id=target_interface + '_random_subseed')
        reuse_subseed = ToolButton(reuse_symbol, elem_id=target_interface + '_reuse_subseed')
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=target_interface + '_subseed_strength')

    with FormRow(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=target_interface + '_seed_resize_from_w')
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=target_interface + '_seed_resize_from_h')

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox



def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )


def update_token_counter(text, steps):
    try:
        text, _ = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact", visible=False):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6, visible=False):
            with gr.Row(visible=False):
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3, value='Industrial design, unmanned aerial vehicle, if design award, white background, studio lighting, blender, CAD, oc render')
                        # print("Prompt:", prompt)placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)",

            with gr.Row(visible=False):
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_classes="interrogate-col", visible=False):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column",visible=False):
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                interrupt = gr.Button('中止', elem_id=f"{id_part}_interrupt", elem_classes="generate-box-interrupt" )
                skip = gr.Button('跳过', elem_id=f"{id_part}_skip", elem_classes="generate-box-skip")
                submit = gr.Button('生成图片', elem_id=f"{id_part}_generate", variant='primary')
                submit_by_SDXL = gr.Button('使用SDXL生成')

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            with gr.Row(elem_id=f"{id_part}_tools",visible=False):
                paste = ToolButton(value=paste_symbol, elem_id="paste")
                clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                extra_networks_button = ToolButton(value=extra_networks_symbol, elem_id=f"{id_part}_extra_networks")
                prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"{id_part}_style_apply")
                save_style = ToolButton(value=save_style_symbol, elem_id=f"{id_part}_style_create")

                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")

                clear_prompt_button.click(
                    fn=lambda *x: x,
                    _js="confirm_clear_prompt",
                    inputs=[prompt, negative_prompt],
                    outputs=[prompt, negative_prompt],
                )

            with gr.Row(elem_id=f"{id_part}_styles_row", visible=False):
                prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True)
                create_refresh_button(prompt_styles, shared.prompt_styles.reload, lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]}, f"refresh_{id_part}_styles")

    return prompt, prompt_styles, negative_prompt, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button, submit_by_SDXL


def setup_progressbar(*args, **kwargs):
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


def create_output_panel(tabname, outdir):
    return ui_common.create_output_panel(tabname, outdir)


def create_sampler_and_steps_selection(choices, tabname):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}", visible=False):
            sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}", visible=False):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            sampler_index = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")

    return steps, sampler_index


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder.split(","))}

    for i, category in sorted(enumerate(shared.ui_reorder_categories), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def get_value_for_setting(key):
    value = getattr(opts, key)

    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision'}}

    return gr.update(value=value, **args)


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=len(x) > 0),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown


def create_ui():
    import modules.img2img
    import modules.txt2img

    reload_javascript()

    parameters_copypaste.reset()

    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button, submit_by_SDXL = create_toprow(is_img2img=False)

        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks, extra_networks_button, 'txt2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):
                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers, "txt2img")

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="txt2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="txt2img_height")

                            with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn")

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "cfg":
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=24, elem_id="txt2img_cfg_scale")

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('txt2img')

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="txt2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="txt2img_tiling")
                            enable_hr = gr.Checkbox(label='Hires. fix', value=False, elem_id="txt2img_enable_hr")
                            hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)

                    elif category == "hires_fix":
                        with FormGroup(visible=False, elem_id="txt2img_hires_fix") as hr_options:
                            with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                            with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = modules.scripts.scripts_txt2img.setup_ui()

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
            for input in hr_resolution_preview_inputs:
                input.change(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )
                input.change(
                    None,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress=False,
                )

            txt2img_gallery, generation_info, html_info, html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    txt2img_prompt_styles,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    hr_scale,
                    hr_upscaler,
                    hr_second_pass_steps,
                    hr_resize_x,
                    hr_resize_y,
                    override_settings,
                ] + custom_inputs,

                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            txt2img_prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            txt_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    txt_prompt_img
                ],
                outputs=[
                    txt2img_prompt,
                    txt_prompt_img
                ]
            )

            enable_hr.change(
                fn=lambda x: gr_show(x),
                inputs=[enable_hr],
                outputs=[hr_options],
                show_progress = False,
            )

            txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d),
                (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None,
            ))

            txt2img_preview_params = [
                txt2img_prompt,
                txt2img_negative_prompt,
                steps,
                sampler_index,
                cfg_scale,
                seed,
                width,
                height,
            ]

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)


    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    
    ######################### 在此插入前置函数 #############################
    def make_inpaint_condition(image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image
    def selected_gallery(event_data: gr.EventData):
        global chosen_index
        chosen_index = event_data._data["index"]
        return
    def selected_style(event_data: gr.EventData):
        global chosen_style
        chosen_style = event_data._data["index"]
        # print("Chosen style image:",chosen_style)
        return
    def selected_history(event_data: gr.EventData):
        global chosen_history
        chosen_history = event_data._data["index"]
        return

    # # initialize the stable diffusion model
    # diffusion_model_path = "runwayml/stable-diffusion-v1-5"
    # # diffusion_model_path = "stable-diffusion-v1-5"
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
    #                         beta_schedule="scaled_linear", clip_sample=False,
    #                         set_alpha_to_one=False, steps_offset=1)
    # # diffusion_model_path = "stable-diffusion-v1-5"
    # model = DragPipeline.from_pretrained(diffusion_model_path, scheduler=scheduler).to(device)
    # # call this function to override unet forward function,
    # # so that intermediate features are returned after forward
    # model.modify_unet_forward()

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

    # def inference(source_image,
    #             image_with_clicks,
    #             mask,
    #             prompt,
    #             points,
    #             lora_path,
    #             n_actual_inference_step=40,
    #             lam=0.1,
    #             n_pix_step=40,
    #             save_dir="./results"
    #     ):
    #     lora_path = "/data0/group1/webui/modules/lora/lora_ckpt/lora_temp"
    #     seed = 42 # random seed used by a lot of people for unknown reason
    #     seed_everything(seed)

    #     args = SimpleNamespace()
    #     args.prompt = prompt
    #     args.points = points
    #     # args.n_inference_step = n_inference_step
    #     args.n_inference_step = 50
    #     args.n_actual_inference_step = n_actual_inference_step
        
    #     # args.guidance_scale = guidance_scale
    #     args.guidance_scale = 1.0

    #     # unet_feature_idx = unet_feature_idx.split(" ")
    #     # unet_feature_idx = [int(k) for k in unet_feature_idx]
    #     # args.unet_feature_idx = unet_feature_idx
    #     args.unet_feature_idx = [2]

    #     # args.sup_res = sup_res
    #     args.sup_res = 256

    #     # args.r_m = r_m
    #     # args.r_p = r_p
    #     args.r_m = 1
    #     args.r_p = 3
    #     args.lam = lam

    #     # args.lr = lr
    #     args.lr = 0.01

    #     args.n_pix_step = n_pix_step
    #     print(args)
    #     full_h, full_w = source_image.shape[:2]

    #     if diffusion_model_path == 'stabilityai/stable-diffusion-2-1':
    #         source_image = preprocess_image(source_image, device, resolution=768)
    #         image_with_clicks = preprocess_image(image_with_clicks, device, resolution=768)
    #     else:
    #         source_image = preprocess_image(source_image, device, resolution=512)
    #         image_with_clicks = preprocess_image(image_with_clicks, device, resolution=512)

    #     # set lora
    #     if lora_path == "":
    #         print("applying default parameters")
    #         model.unet.set_default_attn_processor()
    #     else:
    #         print("applying lora: " + lora_path)
    #         model.unet.load_attn_procs(lora_path)

    #     # invert the source image
    #     # the latent code resolution is too small, only 64*64
    #     invert_code = model.invert(source_image,
    #                             prompt,
    #                             guidance_scale=args.guidance_scale,
    #                             num_inference_steps=args.n_inference_step,
    #                             num_actual_inference_steps=n_actual_inference_step)

    #     mask = torch.from_numpy(mask).float() / 255.
    #     mask[mask > 0.0] = 1.0
    #     mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    #     mask = F.interpolate(mask, (args.sup_res, args.sup_res), mode="nearest")

    #     handle_points = []
    #     target_points = []
    #     # here, the point is in x,y coordinate
    #     for idx, point in enumerate(points):
    #         cur_point = torch.tensor([point[1] / full_h, point[0] / full_w]) * args.sup_res
    #         cur_point = torch.round(cur_point)
    #         if idx % 2 == 0:
    #             handle_points.append(cur_point)
    #         else:
    #             target_points.append(cur_point)
    #     print('handle points:', handle_points)
    #     print('target points:', target_points)

    #     init_code = invert_code
    #     model.scheduler.set_timesteps(args.n_inference_step)
    #     t = model.scheduler.timesteps[args.n_inference_step - n_actual_inference_step]

    #     # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    #     # update according to the given supervision
    #     updated_init_code, updated_text_emb = drag_diffusion_update(model, init_code, t,
    #         handle_points, target_points, mask, args)

    #     # inference the synthesized image
    #     gen_image = model(prompt,
    #         prompt_embeds=updated_text_emb,
    #         latents=updated_init_code,
    #         guidance_scale=1.0,
    #         num_inference_steps=args.n_inference_step,
    #         num_actual_inference_steps=n_actual_inference_step
    #         )

    #     # save the original image, user editing instructions, synthesized image
    #     save_result = torch.cat([
    #         source_image * 0.5 + 0.5,
    #         torch.ones((1,3,512,25)).cuda(),
    #         image_with_clicks * 0.5 + 0.5,
    #         torch.ones((1,3,512,25)).cuda(),
    #         gen_image[0:1]
    #     ], dim=-1)

    #     if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)
    #     save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    #     save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    #     out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    #     out_image = (out_image * 255).astype(np.uint8)
    #     return out_image
    
    def inference(source_image,
              image_with_clicks,
              mask,
              prompt,
              points,
              lora_path,
            #   model_path="runwayml/stable-diffusion-v1-5",
              model_path="/data0/group1/webui/models/Stable-diffusion/stable-diffusion-v1-5",
              vae_path="default",
              n_actual_inference_step=40,
              lam=0.1,
              n_pix_step=40,
              save_dir="./results"
    ):
        lora_path = "/data0/group1/webui/modules/lora/lora_ckpt/lora_temp"
        # initialize model
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                            beta_schedule="scaled_linear", clip_sample=False,
                            set_alpha_to_one=False, steps_offset=1)
        model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
        # call this function to override unet forward function,
        # so that intermediate features are returned after forward
        model.modify_unet_forward()

        # set vae
        if vae_path != "default":
            model.vae = AutoencoderKL.from_pretrained(
                vae_path
            ).to(model.vae.device, model.vae.dtype)

        # initialize parameters
        seed = 42 # random seed used by a lot of people for unknown reason
        seed_everything(seed)

        args = SimpleNamespace()
        args.prompt = prompt
        args.points = points
        args.n_inference_step = 50
        args.n_actual_inference_step = n_actual_inference_step
        args.guidance_scale = 1.0

        args.unet_feature_idx = [2]

        args.sup_res = 256

        args.r_m = 1
        args.r_p = 3
        args.lam = lam

        args.lr = 0.01

        args.n_pix_step = n_pix_step
        print(args)
        full_h, full_w = source_image.shape[:2]

        source_image = preprocess_image(source_image, device)
        image_with_clicks = preprocess_image(image_with_clicks, device)

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

    def train_lora_interface(original_image, prompt, progress=gr.Progress()):
        train_image = original_image["image"]
        # model_path = "runwayml/stable-diffusion-v1-5"
        model_path = "/data0/group1/webui/models/Stable-diffusion/stable-diffusion-v1-5"
        vae_path = "default"
        my_lora_path = "/data0/group1/webui/modules/lora/lora_ckpt/lora_temp"
        train_lora(train_image,prompt,model_path,vae_path,my_lora_path,200,0.0002,16,progress)
        return "Training LoRA Done!"

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        
        ########################## 在此插入UI ############################
        with gr.Tab(label=" "):
            with gr.Row():
                original_image = gr.State(value=None) # store original image
                mask = gr.State(value=None) # store mask
                selected_points = gr.State([]) # store points
                length = 480
            with gr.Row():
                # 主要图像
                with gr.Column(scale=11):
                    main_image = gr.Image(visible=True, source="upload", tool="sketch", label="Image", show_label=False).style(height=643)
                    # main_image = gr.Image(visible=True, source="webcam", tool="sketch", label="Image", mirror_webcam=False, show_label=False, interactive=True).style(height=643)
                    hidden_image = gr.Image(visible=False, label="hidden", show_label=False).style(height=643)
                    stored_image = gr.Image(visible=False, scale=1)
                    matrix_gallery = gr.Gallery(visible=False, allow_preview=False).style(grid=3,height=643)
                    with gr.Row():
                        left_button = gr.Button("last scheme",interactive=False)
                        right_button = gr.Button("next scheme",interactive=False)
                    setmain_button = gr.Button("Choose",visible=False)
                with gr.Column(scale=6):
                    # 生成区域Tab
                    # with gr.Column():
                    with gr.Tab(label="Create Scheme"):
                        prompt = gr.Textbox(label="Prompt",lines=8)
                        ref_gallery = gr.Gallery(label="Render Style",allow_preview=False).style(grid=3,height=220)
                        show_gallery = gr.Button("Load Gallery")
                        mode_radio = gr.Radio(["Rapid Creation", "Controllable Creation"], label="Mode")
                        # color_prompt = gr.Textbox(label="prompt",lines=5,show_label=False)
                        strength_slider = gr.Slider(label="Denoising strength", value=0.6, minimum=0.0, maximum=1.0, step=.01)
                        create_button = gr.Button("Create",variant='primary',scale=1)
                        test_button = gr.Button("Test Matrix",visible=False)
                        SD15_button = gr.Button("Generate Img2Img",visible=False)
                        colormask_button = gr.Button("Generate Color Mask",visible=False)
                        colorinpaint_button = gr.Button("Color Inpaint",visible=False)
                    # 优化区域Tab
                    # with gr.Column():
                    with gr.Tab(label="Refine Scheme"):
                        mask_button = gr.Button("Mask Local Area")
                        local_prompt = gr.Textbox(label="Local Refine",lines=23)
                        with gr.Row():
                            drag_button = gr.Button("Add Anchor").style(width=180)
                            undo_button = gr.Button("Delete All Anchor").style(width=180)
                        with gr.Row():
                            SDXL_button = gr.Button("SDXL Refine",visible=False)
                            generate_button = gr.Button("Refine",variant='primary')
                        train_lora_button = gr.Button("Train Lora")
                        lora_status_bar = gr.Textbox(label="display LoRA training status")
                        with gr.Row():
                            strength = gr.Slider(minimum=0.0,maximum=1.0,value=0.6,visible=False)
                            lora_path = gr.Textbox(value="", label="lora path",visible=False)
                        with gr.Row(visible=False):
                            matrix_button = gr.Button("生成图像矩阵")
                            setinput_button = gr.Button("设置矩阵图像到输入")
            # matrix_gallery = gr.Gallery()
            # matrix_gallery.style(grid=8)
        with gr.Row(label="Generate History"):
            with gr.Column():
                sethistory_input = gr.Button("Set as input")
                history_gallery = gr.Gallery(label="Generate History", allow_preview=False).style(grid=6,height=643)
            

        ########################## 在此插入函数 ############################
        def set_history_as_input(gallery,input_image):
            if chosen_history != -1:
                filename = gallery[chosen_history]["name"]
                res =cv2.imread(filename)
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                return res
            else:
                return input_image
        sethistory_input.click(set_history_as_input,[history_gallery,main_image],[main_image])
        
        def SD21api(input_image,pos_prompt,neg_prompt):
            reverted_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            retval2, buffer2 = cv2.imencode('.jpg', reverted_img)
            input_image = b64encode(buffer2).decode("utf-8")
            IMG2IMG_URL = 'http://127.0.0.1:7866/sdapi/v1/img2img'
            body = {
                    "Model":'v2-1_512-ema-pruned', 
                    "init_images": [input_image],
                    "prompt": pos_prompt,
                    "negative_prompt": neg_prompt,
                    "denoising_strength": 0.58,
                    "steps": 50,
                    "cfg_scale": 7
                }
            response = requests.post(IMG2IMG_URL, json.dumps(body))
            save_encoded_image(response.json()['images'][0],'genImage.png')
            output = cv2.imread('genImage.png')
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return output
        def SDXL10api(input_image,pos_prompt,neg_prompt, strength_slider):
            print("strength:",strength_slider)
            reverted_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            retval2, buffer2 = cv2.imencode('.jpg', reverted_img)
            input_image = b64encode(buffer2).decode("utf-8")
            IMG2IMG_URL = 'http://127.0.0.1:7870/sdapi/v1/img2img'
            # body = {
            #         "Model":'sd_xl_refiner_1.0', 
            #         "init_images": [input_image],
            #         "prompt": pos_prompt,
            #         # "resize mode": "Crop and resize",
            #         "sampler_name": "DPM++ 2M SDE Karras",
            #         "negative_prompt": neg_prompt,
            #         "denoising_strength": strength_slider,
            #         "steps": 50,
            #         "cfg_scale": 21,
            #         "width": 1024,
            #         "height": 1024,
            #     }
            body = {
                    "Model":'sd_xl_base_1.0', 
                    "init_images": [input_image],
                    "prompt": pos_prompt,
                    # "resize mode": "Crop and resize",
                    "sampler_name": "Restart",
                    "negative_prompt": neg_prompt,
                    "denoising_strength": strength_slider,
                    "steps": 50,
                    "cfg_scale": 21,
                    "width": 1024,
                    "height": 1024,
                }
            response = requests.post(IMG2IMG_URL, json.dumps(body))
            save_encoded_image(response.json()['images'][0],'genImage.png')
            output = cv2.imread('genImage.png')
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return output
        
        def create_single(input_image, prompt_init, history_gallery, strength_slider):
            # Split prompt & color prompt
            two_prompt_list = prompt_init.split("\n\n")
            prompt = two_prompt_list[0]
            color_prompt = two_prompt_list[1]
            print(prompt)
            print(color_prompt)
            # Img Operations
            copy_img = input_image["image"]
            copy_img = cv2.resize(copy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            stored_img = copy_img.copy()
            # Generate Color Masks
            img = np.array(input_image["image"])
            blur = cv2.GaussianBlur(img,(5,5),0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
            blue_lowbound = np.array([75, 43, 46])
            blue_highbound = np.array([120, 255, 255])
            mask = cv2.inRange(hsv, blue_lowbound, blue_highbound)
            cv2.imwrite("/data0/group1/pt_blue.jpg", mask)
            yellow_lowbound = np.array([10, 43, 46])
            yellow_highbound = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, yellow_lowbound, yellow_highbound)
            cv2.imwrite("/data0/group1/pt_yellow.jpg", mask)
            red_lowbound = np.array([156, 43, 46])
            red_highbound = np.array([180, 255, 255])
            red_lowbound2 = np.array([0, 80, 80])
            red_highbound2 = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, red_lowbound, red_highbound)
            mask += cv2.inRange(hsv, red_lowbound2, red_highbound2)
            cv2.imwrite("/data0/group1/pt_red.jpg", mask)
            green_lowbound = np.array([40, 43, 46])
            green_highbound = np.array([77, 255, 255])
            mask = cv2.inRange(hsv, green_lowbound, green_highbound)
            cv2.imwrite("/data0/group1/pt_green.jpg", mask)
            
            # Generate Img2Img
            nega_prompt = ""
            global chosen_style
            print(chosen_style)
            if chosen_style != -1:
                prompt = prompt + pos_prompt[chosen_style]
                nega_prompt = neg_prompt[chosen_style]
            init_image = Image.fromarray(input_image['image'])
            # init_image = init_image.resize((512, 512))
            # - Generate
            # model_key_sd21 = "runwayml/stable-diffusion-v1-5"
            # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            #     model_key_sd21, torch_dtype=torch.float16
            # ).to("cuda")
            # image = pipe(prompt=prompt, negative_prompt=nega_prompt, image=init_image, num_inference_steps=30, strength=0.7).images[0]
            
            # - Generate by SDXL API
            width, height = init_image.size
            crop_size = min(width, height)
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2

            # 中心裁剪
            img_cropped = init_image.crop((left, top, right, bottom))

            # resize
            init_image = img_cropped.resize((1024,1024))
            # init_image = np.array(init_image)
            # cv2.imwrite("/data0/group1/init.jpg", init_image)
            
            init_image = np.array(init_image)
            image = SDXL10api(init_image, prompt, nega_prompt, strength_slider)
            cv2.imwrite("/data0/group1/pt_SDXL10output.jpg", image)
            image = Image.fromarray(image)
            
            # - Refine
            # model_key_refiner = "/data0/group1/SD_XL/SDXL0.9/stable-diffusion-xl-refiner-0.9"
            # pipe = DiffusionPipeline.from_pretrained(
            #     model_key_refiner, torch_dtype=torch.float16
            # ).to("cuda")
            # image = pipe(prompt=prompt, negative_prompt=nega_prompt, image=image, num_inference_steps=20, strength=0.5).images[0]
            
            # Inpaint
            # init_image = np.array(image)
            # clist = color_prompt.split(", ")
            # for i in clist:
            #     # get color and discription
            #     if i == "":
            #         break
            #     llist = i.split(" is ")
            #     color = llist[0]
            #     discription = llist[1]
            #     print(color, discription)
            #     # generate with masks
            #     mask_path = "/data0/group1/pt_" + color + ".jpg"
            #     mask_image = np.array(Image.open(mask_path))
            #     init_image = controlnetapi(init_image, mask_image, discription)
            #     init_image = np.array(init_image)
            
            # Inpaint For Matrix
            init_image = np.array(image)
            clist = color_prompt.split(", ")
            ret_image_list = []
            pair1 = clist[0].split(" is ") # [color1, discription1]
            pair2 = clist[1].split(" is ") # [color2, discription2]
            pair1[1] = pair1[1] + ", real "
            pair2[1] = pair2[1] + ", real "
            mask_path1 = "/data0/group1/pt_" + pair1[0] + ".jpg"
            mask_path2 = "/data0/group1/pt_" + pair2[0] + ".jpg"
            mask_image1 = np.array(Image.open(mask_path1))
            mask_image2 = np.array(Image.open(mask_path2))
            ret, thresh = cv2.threshold(mask_image1, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((20,20),np.uint8)
            mask_image_1 = cv2.dilate(thresh, kernel, iterations = 1)
            cv2.imwrite("/data0/group1/webui/outputs/mask/mask_image_1.jpg", mask_image_1)
            
            ret, thresh = cv2.threshold(mask_image2, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((20,20),np.uint8)
            mask_image_2 = cv2.dilate(thresh, kernel, iterations = 1)
            cv2.imwrite("/data0/group1/webui/outputs/mask/mask_image_2.jpg", mask_image_2)
            
            strength_list1 = [0.7,0.85,1.0]
            strength_list2 = [0.7,0.85,1.0]
            for i in strength_list1:
                # strength1 = 0.4*i+0.2
                strength1 = i
                print(pair1[1],strength1)
                mid_image = controlnetapi(init_image, mask_image1, pair1[1], strength1)
                mid_image = np.array(mid_image)
                for j in strength_list2:
                    # strength2 = 0.4*j+0.2
                    strength2 = j
                    print(pair2[1],strength2)
                    final_image = controlnetapi(mid_image, mask_image2, pair2[1], strength2)
                    final_image = np.array(final_image)
                    ret_image_list.append(final_image)
            
            # Store to history
            ret_gallery = []
            for i in range(len(history_gallery)):
                res = cv2.imread(history_gallery[i]["name"])
                ret_gallery.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            for single_image in ret_image_list:
                hist_image = Image.fromarray(single_image)
                ret_gallery.append(hist_image)
            
            return stored_img, gr.Button.update(interactive=False), gr.Button.update(interactive=True), ret_image_list, ret_gallery, \
                gr.Image.update(visible=False),gr.Gallery.update(visible=True), \
                gr.Button.update(visible=False),gr.Button.update(visible=False),gr.Button.update(visible=True)
                
        create_button.click(
            create_single,[main_image, prompt, history_gallery, strength_slider], 
            [stored_image,right_button,left_button,matrix_gallery,history_gallery,main_image,matrix_gallery,left_button,right_button,setmain_button])
        
        def generate_color_mask(input_image):
            # generate masks
            img = np.array(input_image["image"])
            blur = cv2.GaussianBlur(img,(5,5),0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
            blue_lowbound = np.array([75, 43, 46])
            blue_highbound = np.array([120, 255, 255])
            mask = cv2.inRange(hsv, blue_lowbound, blue_highbound)
            cv2.imwrite("/data0/group1/pt_blue.jpg", mask)
            yellow_lowbound = np.array([26, 43, 46])
            yellow_highbound = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, yellow_lowbound, yellow_highbound)
            cv2.imwrite("/data0/group1/pt_yellow.jpg", mask)
            red_lowbound = np.array([156, 43, 46])
            red_highbound = np.array([180, 255, 255])
            red_lowbound2 = np.array([0, 80, 80])
            red_highbound2 = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, red_lowbound, red_highbound)
            mask += cv2.inRange(hsv, red_lowbound2, red_highbound2)
            cv2.imwrite("/data0/group1/pt_red.jpg", mask)
            green_lowbound = np.array([7, 43, 46])
            green_highbound = np.array([77, 255, 255])
            mask = cv2.inRange(hsv, green_lowbound, green_highbound)
            cv2.imwrite("/data0/group1/pt_green.jpg", mask)         
        colormask_button.click(generate_color_mask, [main_image], [])
        
        def controlnetapi(init_image, mask_image, prompt, m_strength=1.0):
            ret, thresh = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((20,20),np.uint8)
            mask_image = cv2.dilate(thresh, kernel, iterations = 1)
            # cv2.imwrite("/data0/group1/webui/outputs/mask/mask1", mask)
            init_image = Image.fromarray(init_image)
            mask_image = Image.fromarray(mask_image)
            init_image = init_image.resize((512, 512))
            mask_image = mask_image.resize((512, 512))
            control_image = make_inpaint_condition(init_image, mask_image)
            controlnet_seed = int(random.randrange(4294967294))
            generator = torch.Generator(device="cuda").manual_seed(controlnet_seed)
            image = np.array(init_image)
            low_threshold = 100
            high_threshold = 200
            canny_image = cv2.Canny(image, low_threshold, high_threshold)
            canny_image = Image.fromarray(canny_image)
            canny_image = canny_image.resize((512, 512))
            # controlnet_inpaint = ControlNetModel.from_pretrained(
            #     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            # ).to("cuda")
            # controlnet_canny = ControlNetModel.from_pretrained(
            #     "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
            # ).to("cuda")
            
            # SDXL canny + sd1.5 inpainting 版本
            # controlnet_inpaint = ControlNetModel.from_pretrained(
            #     "/data0/group1/webui/models/ControlNet/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            # ).to("cuda")
            # controlnet_canny = ControlNetModel.from_pretrained(
            #      "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
            # ).to("cuda")
            # pipe1 = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            #     # "runwayml/stable-diffusion-v1-5",
            #     "/data0/group1/webui/models/Stable-diffusion/stable-diffusion-v1-5",
            #     # controlnet=controlnet_inpaint,
            #     controlnet= controlnet_inpaint,
            #     torch_dtype=torch.float16
            # ).to("cuda")
            # pipe1.scheduler = UniPCMultistepScheduler.from_config(pipe1.scheduler.config)
            # image1 = pipe1(
            #     prompt,
            #     num_inference_steps=40,
            #     generator=generator,
            #     eta=0,
            #     image=init_image,
            #     mask_image=mask_image,
            #     control_image=control_image,
            #     # control_image=control_image,
            #     strength=m_strength
            # ).images[0]
            
            # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
            # pipe2 = StableDiffusionXLControlNetPipeline.from_pretrained(
            #     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_canny, vae=vae, torch_dtype=torch.float16, use_safetensors=True
            # )
            # pipe2.scheduler = UniPCMultistepScheduler.from_config(pipe2.scheduler.config)
            # pipe2.enable_model_cpu_offload()
            
            # image2 = pipe2(
            #     prompt,
            #     num_inference_steps=40,
            #     generator=generator,
            #     eta=0,
            #     image=image1,
            #     mask_image=mask_image,
            #     control_image=[control_image],
            #     # control_image=control_image,
            #     strength=m_strength
            # ).images[0]
            
            # sd1.5 canny + inpainting 版本
            controlnet_inpaint = ControlNetModel.from_pretrained(
                "/data0/group1/webui/models/ControlNet/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            ).to("cuda")
            controlnet_canny = ControlNetModel.from_pretrained(
                "/data0/group1/webui/models/ControlNet/control_v11p_sd15_canny", torch_dtype=torch.float16
            ).to("cuda")
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                # "runwayml/stable-diffusion-v1-5",
                "/data0/group1/webui/models/Stable-diffusion/stable-diffusion-v1-5",
                # controlnet=controlnet_inpaint,
                controlnet=MultiControlNetModel([controlnet_inpaint, controlnet_canny]),
                torch_dtype=torch.float16
            ).to("cuda")
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            image = pipe(
                prompt,
                num_inference_steps=40,
                generator=generator,
                eta=0,
                image=init_image,
                mask_image=mask_image,
                control_image=[control_image,canny_image],
                # control_image=control_image,
                strength=m_strength
            ).images[0]
            return image
        
        # def color_inpaint(input_image, color_prompt):
        #     init_image = np.array(input_image["image"])
        #     clist = color_prompt.split(", ")
        #     for i in clist:
        #         # get color and discription
        #         llist = i.split(" is ")
        #         color = llist[0]
        #         discription = llist[1]
        #         print(color, discription)
        #         # generate with masks
        #         mask_path = "/data0/group1/pt_" + color + ".jpg"
        #         mask_image = np.array(Image.open(mask_path))
        #         # init_image = cv2.cvtColor(init_image, cv2.COLOR_RGB2BGR)
        #         init_image = controlnetapi(init_image, mask_image, discription)
        #         init_image = np.array(init_image)
        #         # init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
        #     return init_image   
        # colorinpaint_button.click(color_inpaint, [main_image, color_prompt], [main_image])
        
        train_lora_button.click(
            train_lora_interface,
            [main_image,local_prompt],
            [lora_status_bar]
        )
        
        def test_matrix_func(input_image):
            image = input_image["image"].copy()
            return [image,image,image,image,image,image,image,image,image,image,image,image,image,image,image,image], \
                    gr.Image.update(visible=False),gr.Gallery.update(visible=True), \
                    gr.Button.update(visible=False),gr.Button.update(visible=False),gr.Button.update(visible=True)
        test_button.click(test_matrix_func, [main_image], 
            [matrix_gallery,main_image,matrix_gallery,left_button,right_button,setmain_button])
        
        def load_style_images():
            style1 = Image.open("/data0/group1/webui/style_images/Style1.png")
            style2 = Image.open("/data0/group1/webui/style_images/Style2.png")
            style3 = Image.open("/data0/group1/webui/style_images/Style3.png")
            style4 = Image.open("/data0/group1/webui/style_images/Style4.png")
            style5 = Image.open("/data0/group1/webui/style_images/Style5.png")
            style6 = Image.open("/data0/group1/webui/style_images/Style6.png")
            return [style1, style2, style3, style4, style5, style6],gr.Button.update(visible=False)       
        show_gallery.click(load_style_images,[],[ref_gallery,show_gallery])
        
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

        # clear all handle/target points, make it back to mask stage
        def undo_points(original_image, mask):
            global drag_flag
            drag_flag = False
            if mask.sum() > 0:
                mask = np.uint8(mask > 0)
                masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
            else:
                masked_img = original_image.copy()
            return masked_img, [], gr.Image.update(visible=False), gr.Image.update(visible=True)
        undo_button.click(
            undo_points,
            [original_image, mask],
            [hidden_image, selected_points, hidden_image, main_image]
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
            global chosen_style
            # print("expand:",chosen_style)
            expand_prompt = prompt + pos_prompt[chosen_style]
            expand_neg_prompt = neg_prompt[chosen_style]
            print(expand_prompt)
            image = pipe(prompt=expand_prompt, negative_prompt=expand_neg_prompt, image=init_image, num_inference_steps=20, strength=strength).images[0]
            global is_left_flag
            is_left_flag = False
            return image, stored_img
        SDXL_button.click(
            fn=SDXLapi,
            inputs=[main_image, prompt, strength],
            outputs=[main_image, stored_image]
        )
        
        def SD15api(main_image, prompt, strength):
            nega_prompt = ""
            global chosen_style
            print(chosen_style)
            if chosen_style != -1:
                prompt = prompt + pos_prompt[chosen_style]
                nega_prompt = neg_prompt[chosen_style]
            copy_img = main_image["image"]
            copy_img = cv2.resize(copy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            stored_img = copy_img.copy()
            init_image = Image.fromarray(main_image['image'])
            init_image = init_image.resize((512, 512))
            # Generate
            # model_key_sd21 = "runwayml/stable-diffusion-v1-5"
            model_key_sd21 = "/data0/group1/webui/models/Stable-diffusion/stable-diffusion-v1-5"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_key_sd21, torch_dtype=torch.float16
            ).to("cuda")
            image = pipe(prompt=prompt, negative_prompt=nega_prompt, image=init_image, num_inference_steps=30, strength=0.7).images[0]
            # Reifine
            model_key_refiner = "/data0/group1/SD_XL/SDXL0.9/stable-diffusion-xl-refiner-0.9"
            pipe = DiffusionPipeline.from_pretrained(
                model_key_refiner, torch_dtype=torch.float16
            ).to("cuda")
            image = pipe(prompt=prompt, negative_prompt=nega_prompt, image=image, num_inference_steps=20, strength=0.5).images[0]
            return image, stored_img
        SD15_button.click(
            fn=SD15api,
            inputs=[main_image, prompt, strength],
            outputs=[main_image, stored_image]
        )

        def controlnetrefineapi(main_image, prompt, mask):
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

            # controlnet_inpaint = ControlNetModel.from_pretrained(
            #     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            # ).to("cuda")
            controlnet_inpaint = ControlNetModel.from_pretrained(
                "/data0/group1/webui/models/ControlNet/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            ).to("cuda")
            controlnet_canny = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
            ).to("cuda")
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                # "runwayml/stable-diffusion-v1-5",
                "/data0/group1/webui/models/Stable-diffusion/stable-diffusion-v1-5",
                controlnet=MultiControlNetModel([controlnet_inpaint, controlnet_canny]),
                # controlnet=controlnet_inpaint,
                torch_dtype=torch.float16
            ).to("cuda")
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            image = pipe(
                prompt,
                num_inference_steps=50,
                generator=generator,
                eta=0,
                image=init_image,
                mask_image=mask_image,
                control_image=[control_image,canny_image],
                # control_image = control_image,
                strength=1.0
            ).images[0]
            # img is not a numpy array, neither a scalar
            # image = Image.fromarray(image)
            print(image)
            image.save("/data0/group1/inpainting.jpg", quality=95)
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
            txt2img_url = 'http://127.0.0.1:7863/sdapi/v1/txt2img'
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

        def generate(global_prompt, original_image, hidden_image, mask, prompt, selected_points, lora_path, main_image, history_gallery):
            copy_img = main_image["image"]
            copy_img = cv2.resize(copy_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            stored_img = copy_img.copy()
            global drag_flag
            if drag_flag == True:
                drag_flag = False
                ret_image = inference(original_image, hidden_image, mask, prompt, selected_points, lora_path)
            else:
                ret_image = controlnetrefineapi(main_image, prompt, mask)
                # ret_image = controlnetrequest(main_image, prompt)
                # ----------
                # refiner
                model_key_refiner = "/data0/group1/SD_XL/SDXL0.9/stable-diffusion-xl-refiner-0.9"
                pipe = DiffusionPipeline.from_pretrained(
                    model_key_refiner, torch_dtype=torch.float16
                ).to("cuda")
                global chosen_style
                chosen_style = 2
                expand_prompt = prompt
                if chosen_style != -1:
                    expand_prompt = prompt + pos_prompt[chosen_style]
                full_prompt = global_prompt + ", " + expand_prompt
                print(full_prompt)
                ret_image = pipe(prompt=full_prompt, image=ret_image, num_inference_steps=20, strength=0.4).images[0]
                # ----------
            # Set into history
            hist_image = ret_image
            ret_gallery = []
            for i in range(len(history_gallery)):
                res = cv2.imread(history_gallery[i]["name"])
                ret_gallery.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            ret_gallery.append(hist_image)
            return ret_image, gr.Image.update(visible=False), gr.Image.update(visible=True), stored_img, \
                gr.Button.update(interactive=False), gr.Button.update(interactive=True),ret_gallery
        generate_button.click(
            generate,
            [prompt, original_image, hidden_image, mask, local_prompt, selected_points, lora_path, main_image, history_gallery],
            [main_image, hidden_image, main_image, stored_image, right_button, left_button, history_gallery]
        )

        def switch(main_image, stored_image):
            main_copy = main_image["image"]
            stored_copy = stored_image
            main_image_copy = main_copy.copy()
            stored_image_copy = stored_copy.copy()
            return stored_image_copy, main_image_copy, gr.Button.update(interactive=False), gr.Button.update(interactive=True)
        left_button.click(switch,[main_image, stored_image],[main_image, stored_image, left_button, right_button])
        right_button.click(switch,[main_image, stored_image],[main_image, stored_image, right_button, left_button])

        def store_img(img):
            global drag_flag
            drag_flag = True
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
        ref_gallery.select(
            fn=selected_style,
            inputs=[],outputs=[]
        )
        history_gallery.select(
            fn=selected_history,
            inputs=[],outputs=[]
        )

        def setmain_func(gallery):
            global chosen_index
            filename = gallery[chosen_index]["name"]
            res =cv2.imread(filename)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            return res, gr.Gallery.update(visible=False), gr.Image.update(visible=True), gr.Button.update(visible=False), \
                gr.Button.update(visible=True), gr.Button.update(visible=True)
        setmain_button.click(setmain_func, [matrix_gallery],
            [main_image, matrix_gallery, main_image, setmain_button, left_button, right_button])

        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)
        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks, extra_networks_button, 'img2img')

        with FormRow().style(equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}", visible=False):
                        gr.HTML("Copy image to: ", elem_id=f"img2img_label_copy_to_{tab_name}")

                        for title, name in zip(['img2img', 'sketch', 'inpaint', 'inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue

                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    with gr.TabItem('输入', id='img2img', elem_id="img2img_img2img_tab",visible=False) as tab_img2img:
                        with gr.Column(scale=1, elem_id=f"mode_img2img",visible=False):
                            init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA").style(height=580, width=580)
                            add_copy_image_controls('img2img', init_img)

                img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, submit, img2img_interrogate, img2img_deepbooru, img2img_prompt_style_apply, img2img_save_style, img2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button, submit_by_SDXL = create_toprow(is_img2img=True)



                with gr.Tabs(visible=False, elem_id="mode_img2img"):
                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                        sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA").style(height=480)
                        add_copy_image_controls('sketch', sketch)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)
                        add_copy_image_controls('inpaint', init_img_with_mask)

                    with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA").style(height=480)
                        inpaint_color_sketch_orig = gr.State(None)
                        add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            f"<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running." +
                            f"<br>Use an empty output directory to save pictures normally instead of writing to the output directory." +
                            f"<br>Add inpaint batch mask directory to enable inpaint batch processing."
                            f"{hidden}</p>"
                        )
                        img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory (required for inpaint batch processing only)", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                def copy_image(img):
                    if isinstance(img, dict) and 'image' in img:
                        return img['image']

                    return img

                for button, name, elem in copy_image_buttons:
                    button.click(
                        fn=copy_image,
                        inputs=[elem],
                        outputs=[copy_image_destinations[name]],
                    )
                    button.click(
                        fn=lambda: None,
                        _js="switch_to_"+name.replace(" ", "_"),
                        inputs=[],
                        outputs=[],
                    )

                with FormRow(visible=False):
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Just resize")

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers_for_img2img, "img2img")

                    elif category == "dimensions":
                        with FormRow(visible=False):
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="img2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="img2img_height")

                            with gr.Column(elem_id="img2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="img2img_res_switch_btn")

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="img2img_column_batch", visible=False):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "cfg":
                        with FormGroup():
                            with FormRow():
                                cfg_scale = gr.Slider(value=24.0, minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', elem_id="img2img_cfg_scale", visible=False)
                                image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale', value=24, elem_id="img2img_image_cfg_scale", visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
                            denoising_strength = gr.Slider(value=0.55, minimum=0.0, maximum=1.0, step=0.01, label='图片相似度', elem_id="img2img_denoising_strength", visible=False)

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('img2img')

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact", visible=False):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="img2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="img2img_tiling")

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="img2img_column_batch", visible=False):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="img2img_override_settings_row", visible=False) as row:
                            override_settings = create_override_settings_dropdown('img2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container", visible=False):
                            custom_inputs = modules.scripts.scripts_img2img.setup_ui()

                    elif category == "inpaint":
                        with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                            with FormRow():
                                mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                                mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")

                            with FormRow():
                                inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")

                            with FormRow():
                                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")

                            with FormRow():
                                with gr.Column():
                                    inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")

                                with gr.Column(scale=4):
                                    inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                            def select_img2img_tab(tab):
                                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

                            for i, elem in enumerate([tab_img2img]):
                                for i, elem in enumerate([tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]):
                                    elem.select(
                                        fn=lambda tab=i: select_img2img_tab(tab),
                                        inputs=[],
                                        outputs=[inpaint_controls, mask_alpha],
                                    )

            img2img_gallery, generation_info, html_info, html_log = create_output_panel("img2img", opts.outdir_img2img_samples)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            img2img_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    img2img_prompt_img
                ],
                outputs=[
                    img2img_prompt,
                    img2img_prompt_img
                ]
            )

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    dummy_component,
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_prompt_styles,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    inpaint_color_sketch_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    steps,
                    sampler_index,
                    mask_blur,
                    mask_alpha,
                    inpainting_fill,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    image_cfg_scale,
                    denoising_strength,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    resize_mode,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpainting_mask_invert,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    img2img_batch_inpaint_mask_dir,
                    override_settings,
                ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    # generation_info,
                    # html_info,
                    # html_log,
                    # init_img
                ],
                show_progress=False,
            )

            # interrogate_args = dict(
            #     _js="get_img2img_tab_index",
            #     inputs=[
            #         dummy_component,
            #         # img2img_batch_input_dir,
            #         # img2img_batch_output_dir,
            #         init_img,
            #         # sketch,
            #         # init_img_with_mask,
            #         # inpaint_color_sketch,
            #         # init_img_inpaint,
            #     ],
            #     outputs=[img2img_prompt, dummy_component],
            # )

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)
            # submit_by_SDXL.click(fn="Img2ImgSDXL",inputs=[],outputs=[])
            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            # img2img_interrogate.click(
            #     fn=lambda *args: process_interrogate(interrogate, *args),
            #     **interrogate_args,
            # )

            # img2img_deepbooru.click(
            #     fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
            #     **interrogate_args,
            # )

            prompts = [(txt2img_prompt, txt2img_negative_prompt), (img2img_prompt, img2img_negative_prompt)]
            style_dropdowns = [txt2img_prompt_styles, img2img_prompt_styles]
            style_js_funcs = ["update_txt2img_tokens", "update_img2img_tokens"]

            for button, (prompt, negative_prompt) in zip([txt2img_save_style, img2img_save_style], prompts):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, prompt, negative_prompt],
                    outputs=[txt2img_prompt_styles, img2img_prompt_styles],
                )

            for button, (prompt, negative_prompt), styles, js_func in zip([txt2img_prompt_style_apply, img2img_prompt_style_apply], prompts, style_dropdowns, style_js_funcs):
                button.click(
                    fn=apply_styles,
                    _js=js_func,
                    inputs=[prompt, negative_prompt, styles],
                    outputs=[prompt, negative_prompt, styles],
                )

            token_button.click(fn=update_token_counter, inputs=[img2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[img2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)

            img2img_paste_fields = [
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (image_cfg_scale, "Image CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (mask_blur, "Mask blur"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=img2img_paste, tabname="img2img", source_text_component=img2img_prompt, source_image_component=None,
            ))

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()

    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with gr.Row(visible=False).style(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=image,
                    ))

        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )

    def update_interp_description(value):
        interp_description_css = "<p style='margin-bottom: 2.5em'>{}</p>"
        interp_descriptions = {
            "No interpolation": interp_description_css.format("No interpolation will be used. Requires one model; A. Allows for format conversion and VAE baking."),
            "Weighted sum": interp_description_css.format("A weighted sum will be used for interpolation. Requires two models; A and B. The result is calculated as A * (1 - M) + B * M"),
            "Add difference": interp_description_css.format("The difference between the last two models will be added to the first. Requires three models; A, B and C. The result is calculated as A + (B - C) * M")
        }
        return interp_descriptions[value]

    with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
        with gr.Row(visible=False).style(equal_height=False):
            with gr.Column(variant='compact'):
                interp_description = gr.HTML(value=update_interp_description("Weighted sum"), elem_id="modelmerger_interp_description")

                with FormRow(elem_id="modelmerger_models"):
                    primary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_primary_model_name", label="Primary model (A)")
                    create_refresh_button(primary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")

                    secondary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_secondary_model_name", label="Secondary model (B)")
                    create_refresh_button(secondary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_B")

                    tertiary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_tertiary_model_name", label="Tertiary model (C)")
                    create_refresh_button(tertiary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_C")

                custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="modelmerger_custom_name")
                interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Multiplier (M) - set to 0 to get model A', value=0.3, elem_id="modelmerger_interp_amount")
                interp_method = gr.Radio(choices=["No interpolation", "Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method", elem_id="modelmerger_interp_method")
                interp_method.change(fn=update_interp_description, inputs=[interp_method], outputs=[interp_description])

                with FormRow():
                    checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="ckpt", label="Checkpoint format", elem_id="modelmerger_checkpoint_format")
                    save_as_half = gr.Checkbox(value=False, label="Save as float16", elem_id="modelmerger_save_as_half")

                with FormRow():
                    with gr.Column():
                        config_source = gr.Radio(choices=["A, B or C", "B", "C", "Don't"], value="A, B or C", label="Copy config from", type="index", elem_id="modelmerger_config_method")

                    with gr.Column():
                        with FormRow():
                            bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                            create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")

                with FormRow():
                    discard_weights = gr.Textbox(value="", label="Discard weights with matching name", elem_id="modelmerger_discard_weights")

                with gr.Row():
                    modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant='primary')

            with gr.Column(variant='compact', elem_id="modelmerger_results_container"):
                with gr.Group(elem_id="modelmerger_results_panel"):
                    modelmerger_result = gr.HTML(elem_id="modelmerger_result", show_label=False)

    # with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Row().style(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>")

        with gr.Row(variant="compact").style(equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):

                with gr.Tab(label="Create embedding"):
                    new_embedding_name = gr.Textbox(label="Name", elem_id="train_new_embedding_name")
                    initialization_text = gr.Textbox(label="Initialization text", value="*", elem_id="train_initialization_text")
                    nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1, elem_id="train_nvpt")
                    overwrite_old_embedding = gr.Checkbox(value=False, label="Overwrite Old Embedding", elem_id="train_overwrite_old_embedding")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(value="Create embedding", variant='primary', elem_id="train_create_embedding")

                with gr.Tab(label="Create hypernetwork"):
                    new_hypernetwork_name = gr.Textbox(label="Name", elem_id="train_new_hypernetwork_name")
                    new_hypernetwork_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], elem_id="train_new_hypernetwork_sizes")
                    new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure", placeholder="1st and last digit must be 1. ex:'1, 2, 1'", elem_id="train_new_hypernetwork_layer_structure")
                    new_hypernetwork_activation_func = gr.Dropdown(value="linear", label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)", choices=modules.hypernetworks.ui.keys, elem_id="train_new_hypernetwork_activation_func")
                    new_hypernetwork_initialization_option = gr.Dropdown(value = "Normal", label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"], elem_id="train_new_hypernetwork_initialization_option")
                    new_hypernetwork_add_layer_norm = gr.Checkbox(label="Add layer normalization", elem_id="train_new_hypernetwork_add_layer_norm")
                    new_hypernetwork_use_dropout = gr.Checkbox(label="Use dropout", elem_id="train_new_hypernetwork_use_dropout")
                    new_hypernetwork_dropout_structure = gr.Textbox("0, 0, 0", label="Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15", placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'")
                    overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork", elem_id="train_overwrite_old_hypernetwork")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_hypernetwork = gr.Button(value="Create hypernetwork", variant='primary', elem_id="train_create_hypernetwork")

                with gr.Tab(label="Preprocess images"):
                    process_src = gr.Textbox(label='Source directory', elem_id="train_process_src")
                    process_dst = gr.Textbox(label='Destination directory', elem_id="train_process_dst")
                    process_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_process_width")
                    process_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_process_height")
                    preprocess_txt_action = gr.Dropdown(label='Existing Caption txt Action', value="ignore", choices=["ignore", "copy", "prepend", "append"], elem_id="train_preprocess_txt_action")

                    with gr.Row():
                        process_flip = gr.Checkbox(label='Create flipped copies', elem_id="train_process_flip")
                        process_split = gr.Checkbox(label='Split oversized images', elem_id="train_process_split")
                        process_focal_crop = gr.Checkbox(label='Auto focal point crop', elem_id="train_process_focal_crop")
                        process_multicrop = gr.Checkbox(label='Auto-sized crop', elem_id="train_process_multicrop")
                        process_caption = gr.Checkbox(label='Use BLIP for caption', elem_id="train_process_caption")
                        process_caption_deepbooru = gr.Checkbox(label='Use deepbooru for caption', visible=True, elem_id="train_process_caption_deepbooru")

                    with gr.Row(visible=False) as process_split_extra_row:
                        process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_split_threshold")
                        process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05, elem_id="train_process_overlap_ratio")

                    with gr.Row(visible=False) as process_focal_crop_row:
                        process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_face_weight")
                        process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_entropy_weight")
                        process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_edges_weight")
                        process_focal_crop_debug = gr.Checkbox(label='Create debug image', elem_id="train_process_focal_crop_debug")
                    
                    with gr.Column(visible=False) as process_multicrop_col:
                        gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
                        with gr.Row():
                            process_multicrop_mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384, elem_id="train_process_multicrop_mindim")
                            process_multicrop_maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768, elem_id="train_process_multicrop_maxdim")
                        with gr.Row():
                            process_multicrop_minarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area lower bound", value=64*64, elem_id="train_process_multicrop_minarea")
                            process_multicrop_maxarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area upper bound", value=640*640, elem_id="train_process_multicrop_maxarea")
                        with gr.Row():
                            process_multicrop_objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective", elem_id="train_process_multicrop_objective")
                            process_multicrop_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1, elem_id="train_process_multicrop_threshold")
   
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            with gr.Row():
                                interrupt_preprocessing = gr.Button("Interrupt", elem_id="train_interrupt_preprocessing")
                            run_preprocess = gr.Button(value="Preprocess", variant='primary', elem_id="train_run_preprocess")

                    process_split.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_split],
                        outputs=[process_split_extra_row],
                    )

                    process_focal_crop.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_focal_crop],
                        outputs=[process_focal_crop_row],
                    )

                    process_multicrop.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_multicrop],
                        outputs=[process_multicrop_col],
                    )

                def get_textual_inversion_template_names():
                    return sorted([x for x in textual_inversion.textual_inversion_templates])

                with gr.Tab(label="Train"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
                    with FormRow():
                        train_embedding_name = gr.Dropdown(label='Embedding', elem_id="train_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                        create_refresh_button(train_embedding_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")

                        train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork", choices=[x for x in shared.hypernetworks.keys()])
                        create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks, lambda: {"choices": sorted([x for x in shared.hypernetworks.keys()])}, "refresh_train_hypernetwork_name")

                    with FormRow():
                        embedding_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005", elem_id="train_embedding_learn_rate")
                        hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001", elem_id="train_hypernetwork_learn_rate")
                    
                    with FormRow():
                        clip_grad_mode = gr.Dropdown(value="disabled", label="Gradient Clipping", choices=["disabled", "value", "norm"])
                        clip_grad_value = gr.Textbox(placeholder="Gradient clip value", value="0.1", show_label=False)

                    with FormRow():
                        batch_size = gr.Number(label='Batch size', value=1, precision=0, elem_id="train_batch_size")
                        gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0, elem_id="train_gradient_step")

                    dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images", elem_id="train_dataset_directory")
                    log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion", elem_id="train_log_directory")

                    with FormRow():
                        template_file = gr.Dropdown(label='Prompt template', value="style_filewords.txt", elem_id="train_template_file", choices=get_textual_inversion_template_names())
                        create_refresh_button(template_file, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")

                    training_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_training_width")
                    training_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_training_height")
                    varsize = gr.Checkbox(label="Do not resize images", value=False, elem_id="train_varsize")
                    steps = gr.Number(label='Max steps', value=100000, precision=0, elem_id="train_steps")

                    with FormRow():
                        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_create_image_every")
                        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_save_embedding_every")

                    use_weight = gr.Checkbox(label="Use PNG alpha channel as loss weight", value=False, elem_id="use_weight")

                    save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True, elem_id="train_save_image_with_stored_embedding")
                    preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False, elem_id="train_preview_from_txt2img")

                    shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False, elem_id="train_shuffle_tags")
                    tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0, elem_id="train_tag_drop_out")

                    latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'], elem_id="train_latent_sampling_method")

                    with gr.Row():
                        train_embedding = gr.Button(value="Train Embedding", variant='primary', elem_id="train_train_embedding")
                        interrupt_training = gr.Button(value="Interrupt", elem_id="train_interrupt_training")
                        train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary', elem_id="train_train_hypernetwork")

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column(elem_id='ti_gallery_container',visible=False):
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)
                ti_gallery = gr.Gallery(label='Output', show_label=False, elem_id='ti_gallery',visible=False).style(grid=4)
                ti_progress = gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")

        create_embedding.click(
            fn=modules.textual_inversion.ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
            ],
            outputs=[
                train_embedding_name,
                ti_output,
                ti_outcome,
            ]
        )

        create_hypernetwork.click(
            fn=modules.hypernetworks.ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure
            ],
            outputs=[
                train_hypernetwork_name,
                ti_output,
                ti_outcome,
            ]
        )

        run_preprocess.click(
            fn=wrap_gradio_gpu_call(modules.textual_inversion.ui.preprocess, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                process_src,
                process_dst,
                process_width,
                process_height,
                preprocess_txt_action,
                process_flip,
                process_split,
                process_caption,
                process_caption_deepbooru,
                process_split_threshold,
                process_overlap_ratio,
                process_focal_crop,
                process_focal_crop_face_weight,
                process_focal_crop_entropy_weight,
                process_focal_crop_edges_weight,
                process_focal_crop_debug,
                process_multicrop,
                process_multicrop_mindim,
                process_multicrop_maxdim,
                process_multicrop_minarea,
                process_multicrop_maxarea,
                process_multicrop_objective,
                process_multicrop_threshold,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ],
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(modules.textual_inversion.ui.train_embedding, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        train_hypernetwork.click(
            fn=wrap_gradio_gpu_call(modules.hypernetworks.ui.train_hypernetwork, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

        interrupt_preprocessing.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)

        args = info.component_args() if callable(info.component_args) else info.component_args

        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise Exception(f'bad options item type: {str(t)} for key {key}')

        elem_id = "setting_"+key

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
            else:
                with FormRow():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                    create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
        else:
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))

        return res

    components = []
    component_dict = {}
    shared.settings_components = component_dict

    script_callbacks.ui_settings_callback()
    opts.reorder()

    def run_settings(*args):
        changed = []

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            assert comp == dummy_component or opts.same_type(value, opts.data_labels[key].default), f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if comp == dummy_component:
                continue

            if opts.set(key, value):
                changed.append(key)

        try:
            opts.save(shared.config_filename)
        except RuntimeError:
            return opts.dumpjson(), f'{len(changed)} settings changed without save: {", ".join(changed)}.'
        return opts.dumpjson(), f'{len(changed)} settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}.'

    def run_settings_single(value, key):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()

        if not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()

        opts.save(shared.config_filename)

        return get_value_for_setting(key), opts.dumpjson()

    with gr.Blocks(analytics_enabled=True) as settings_interface:
        with gr.Row(visible=True):
            with gr.Column(scale=6):
                settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            with gr.Column():
                restart_gradio = gr.Button(value='Reload UI', variant='primary', elem_id="settings_restart_gradio")

        result = gr.HTML(elem_id="settings_result")

        quicksettings_names = [x.strip() for x in opts.quicksettings.split(",")]
        quicksettings_names = {x: i for i, x in enumerate(quicksettings_names) if x != 'quicksettings'}

        quicksettings_list = []

        previous_section = None
        current_tab = None
        current_row = None
        with gr.Tabs(elem_id="settings", visible=True):
            for i, (k, item) in enumerate(opts.data_labels.items()):
                section_must_be_skipped = item.section[0] is None

                if previous_section != item.section and not section_must_be_skipped:
                    elem_id, text = item.section

                    if current_tab is not None:
                        current_row.__exit__()
                        current_tab.__exit__()

                    gr.Group()
                    current_tab = gr.TabItem(elem_id="settings_{}".format(elem_id), label=text)
                    current_tab.__enter__()
                    current_row = gr.Column(variant='compact')
                    current_row.__enter__()

                    previous_section = item.section

                if k in quicksettings_names and not shared.cmd_opts.freeze_settings:
                    quicksettings_list.append((i, k, item))
                    components.append(dummy_component)
                elif section_must_be_skipped:
                    components.append(dummy_component)
                else:
                    component = create_setting_component(k)
                    component_dict[k] = component
                    components.append(component)

            if current_tab is not None:
                current_row.__exit__()
                current_tab.__exit__()

            with gr.TabItem("Actions"):
                request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
                download_localization = gr.Button(value='Download localization template', elem_id="download_localization")
                reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary', elem_id="settings_reload_script_bodies")
                with gr.Row():
                    unload_sd_model = gr.Button(value='Unload SD checkpoint to free VRAM', elem_id="sett_unload_sd_model")
                    reload_sd_model = gr.Button(value='Reload the last SD checkpoint back into VRAM', elem_id="sett_reload_sd_model")

            with gr.TabItem("Licenses"):
                gr.HTML(shared.html("licenses.html"), elem_id="licenses")

            gr.Button(value="Show all pages", elem_id="settings_show_all_pages")
            

        def unload_sd_weights():
            modules.sd_models.unload_model_weights()

        def reload_sd_weights():
            modules.sd_models.reload_model_weights()

        unload_sd_model.click(
            fn=unload_sd_weights,
            inputs=[],
            outputs=[]
        )

        reload_sd_model.click(
            fn=reload_sd_weights,
            inputs=[],
            outputs=[]
        )

        request_notifications.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        download_localization.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='download_localization'
        )

        def reload_scripts():
            modules.scripts.reload_script_body_only()
            reload_javascript()  # need to refresh the html page

        reload_script_bodies.click(
            fn=reload_scripts,
            inputs=[],
            outputs=[]
        )

        def request_restart():
            shared.state.interrupt()
            shared.state.need_restart = True

        restart_gradio.click(
            fn=request_restart,
            _js='restart_reload',
            inputs=[],
            outputs=[],
        )

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, " ", "img2img"),
        # (extras_interface, "Extras", "extras"),
        # (pnginfo_interface, "PNG Info", "pnginfo"),
        # (modelmerger_interface, "Checkpoint Merger", "modelmerger"),
        # (train_interface, "Train", "ti"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "Settings", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        # print(label)
        shared.tab_names.append(label)

    with gr.Blocks(analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            for i, k, item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        parameters_copypaste.connect_paste_params_buttons()

        
        for interface, label, ifid in interfaces:
            if label in shared.opts.hidden_tabs:
                    continue
            if label=='txt2img':
                with gr.Tabs(elem_id="tabs", visible=False) as tabs:
                    with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                        interface.render()
            else:
                with gr.Tabs(elem_id="tabs") as tabs:
                    with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                        interface.render()

        if os.path.exists(os.path.join(script_path, "notification.mp3")):
            audio_notification = gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        footer = shared.html("footer.html")
        footer = footer.format(versions=versions_html())
        gr.HTML(footer, elem_id="footer")

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )

        for i, k, item in quicksettings_list:
            component = component_dict[k]
            info = opts.data_labels[k]

            component.change(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
                show_progress=info.refresh is not None,
            )

        text_settings.change(
            fn=lambda: gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit"),
            inputs=[],
            outputs=[image_cfg_scale],
        )

        button_set_checkpoint = gr.Button('Change checkpoint', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_checkpoint'], dummy_component],
            outputs=[component_dict['sd_model_checkpoint'], text_settings],
        )

        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys],
            queue=False,
        )

        def modelmerger(*args):
            try:
                results = modules.extras.run_modelmerger(*args)
            except Exception as e:
                print("Error loading/saving model file:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                modules.sd_models.list_models()  # to remove the potentially missing models from the list
                return [*[gr.Dropdown.update(choices=modules.sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
            return results

        modelmerger_merge.click(fn=lambda: '', inputs=[], outputs=[modelmerger_result])
        modelmerger_merge.click(
            fn=wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)]),
            _js='modelmerger',
            inputs=[
                dummy_component,
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                interp_method,
                interp_amount,
                save_as_half,
                custom_name,
                checkpoint_format,
                config_source,
                bake_in_vae,
                discard_weights,
            ],
            outputs=[
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                component_dict['sd_model_checkpoint'],
                modelmerger_result,
            ]
        )

    ui_config_file = cmd_opts.ui_config_file
    ui_settings = {}
    settings_count = len(ui_settings)
    error_loading = False

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception:
        error_loading = True
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    def loadsave(path, x):
        def apply_field(obj, field, condition=None, init_field=None):
            key = path + "/" + field

            if getattr(obj, 'custom_script_source', None) is not None:
              key = 'customscript/' + obj.custom_script_source + '/' + key

            if getattr(obj, 'do_not_save_to_config', False):
                return

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition and not condition(saved_value):
                pass

                # this warning is generally not useful;
                # print(f'Warning: Bad ui setting value: {key}: {saved_value}; Default value "{getattr(obj, field)}" will be used instead.')
            else:
                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number, gr.Dropdown] and x.visible:
            apply_field(x, 'visible')

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

        if type(x) == gr.Checkbox:
            apply_field(x, 'value')

        if type(x) == gr.Textbox:
            apply_field(x, 'value')

        if type(x) == gr.Number:
            apply_field(x, 'value')

        if type(x) == gr.Dropdown:
            def check_dropdown(val):
                if getattr(x, 'multiselect', False):
                    return all([value in x.choices for value in val])
                else:
                    return val in x.choices

            apply_field(x, 'value', check_dropdown, getattr(x, 'init_field', None))

    # visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")
    # visit(extras_interface, loadsave, "extras")
    # visit(modelmerger_interface, loadsave, "modelmerger")
    # visit(train_interface, loadsave, "train")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    # Required as a workaround for change() event not triggering when loading values from ui-config.json
    interp_description.value = update_interp_description(interp_method.value)

    return demo


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def javascript_html():
    script_js = os.path.join(script_path, "script.js")
    head = f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    inline = f"{localization.localization_js(shared.opts.localization)};"
    if cmd_opts.theme is not None:
        inline += f"set_theme('{cmd_opts.theme}');"

    for script in modules.scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    head += f'<script type="text/javascript">{inline}</script>\n'

    return head


def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        head += stylesheet(cssfile)

    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))

    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse


def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    short_commit = commit[0:8]

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
python: <span title="{sys.version}">{python_version}</span>
 • 
torch: {getattr(torch, '__long_version__',torch.__version__)}
 • 
xformers: {xformers_version}
 • 
gradio: {gr.__version__}
 • 
commit: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{short_commit}</a>
 • 
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""
