from future_diffusion.disco import * 
import subprocess
import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from CLIP import clip
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
from functools import partial
if is_colab:
    os.chdir('/content')
    from google.colab import files
else:
    os.chdir(f'{PROJECT_DIR}')
from IPython.display import Image as ipyimg
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import GPUtil
from pynvml import *

nvmlInit()

class DiscoModel():
    def __init__(self):
        self.intermediate_saves = 0#@param{type: 'raw'}
        self.intermediates_in_subfolder = True #@param{type: 'boolean'}

        self.diffusion_model = "512x512_diffusion_uncond_finetune_008100"
        self.model_config = model_config
        self.model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250, #No need to edit this, it is taken care of later.
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })


        model_default = self.model_config['image_size']

        #@markdown ####**Basic Settings:**
        self.batch_name = 'TimeToDisco' #@param{type: 'string'}
        self.steps = 250 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
        self.width_height = [512, 512]#@param{type: 'raw'}
        self.clip_guidance_scale = 5000 #@param{type: 'number'}
        self.tv_scale =  0#@param{type: 'number'}
        self.range_scale =   150#@param{type: 'number'}
        self.sat_scale =   0#@param{type: 'number'}
        self.cutn_batches = 1  #@param{type: 'number'}
        self.skip_augs = False#@param{type: 'boolean'}

        #@markdown ---

        #@markdown ####**Init Settings:**
        self.init_image = None #@param{type: 'string'}
        self.init_scale = 2000 #@param{type: 'integer'}
        self.skip_steps = 125 #@param{type: 'integer'}
        #@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*

        #Get corrected sizes
        self.side_x = (self.width_height[0]//64)*64;
        self.side_y = (self.width_height[1]//64)*64;
        if self.side_x != self.width_height[0] or self.side_y != self.width_height[1]:
          print(f'Changing output size to {self.side_x}x{self.side_y}. Dimensions must by multiples of 64.')

        #Update Model Settings
        self.timestep_respacing = f'ddim{self.steps}'
        self.diffusion_steps = (1000//self.steps)*self.steps if self.steps < 1000 else self.steps
        self.model_config.update({
            'timestep_respacing': self.timestep_respacing,
            'diffusion_steps': self.diffusion_steps,
        })

        #Make folder for batch
        self.batchFolder = f'{outDirPath}/{self.batch_name}'
        createPath(self.batchFolder)

        if type(self.intermediate_saves) is not list:
          if self.intermediate_saves:
            self.steps_per_checkpoint = math.floor((self.steps - self.skip_steps - 1) // (self.intermediate_saves+1))
            self.steps_per_checkpoint = self.steps_per_checkpoint if self.steps_per_checkpoint > 0 else 1
            print(f'Will save every {self.steps_per_checkpoint} steps')
          else:
            self.steps_per_checkpoint = self.steps+10
        else:
          self.steps_per_checkpoint = None

        if self.intermediate_saves and self.intermediates_in_subfolder is True:
          partialFolder = f'{self.batchFolder}/partials'
          createPath(partialFolder)

        self.args = None

        self.perlin_init = False  #@param{type: 'boolean'}
        self.perlin_mode = 'mixed' #@param ['mixed', 'color', 'gray']
        self.set_seed = 'random_seed' #@param{type: 'string'}
        self.eta = 0.8#@param{type: 'number'}
        self.clamp_grad = True #@param{type: 'boolean'}
        self.clamp_max = 0.05 #@param{type: 'number'}

        ### EXTRA ADVANCED SETTINGS:
        self.randomize_class = True
        self.clip_denoised = False
        self.fuzzy_prompt = False
        self.rand_mag = 0.05

        self.cut_overview = "[12]*400+[4]*600" #@param {type: 'string'}       
        self.cut_innercut ="[4]*400+[12]*600"#@param {type: 'string'}  
        self.cut_ic_pow = 1#@param {type: 'number'}  
        self.cut_icgray_p = "[0.2]*400+[0]*600"#@param {type: 'string'}

        self.text_prompts = {
            0: None
        }

        self.image_prompts = {
            # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
        }

        """# 4. Diffuse!"""

        #@title Do the Run!
        #@markdown `n_batches` ignored with animation modes.
        self.display_rate =  15 #@param{type: 'number'}
        self.n_batches =  1 #@param{type: 'number'}

        random.seed()
        self.seed = random.randint(0, 2**32)

        #Update Model Settings
        self.timestep_respacing = f'ddim{self.steps}'
        self.diffusion_steps = (1000//self.steps)*self.steps if self.steps < 1000 else self.steps
        self.model_config.update({
            'timestep_respacing': self.timestep_respacing,
            'diffusion_steps': self.diffusion_steps,
        })

        self.batch_size = 1 

        print('Prepping model...')
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(torch.load(f'{model_path}/{diffusion_model}.pt', map_location='cpu'))
        self.model.requires_grad_(False).eval().to(device)
        self.isBusy = False
        for name, param in self.model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if self.model_config['use_fp16']:
            self.model.convert_to_fp16()

        self.resume_run = False #@param{type: 'boolean'}
        self.run_to_resume = 'latest' #@param{type: 'string'}
        self.resume_from_frame = 'latest' #@param{type: 'string'}
        self.retain_overwritten_frames = False #@param{type: 'boolean'}
        if self.retain_overwritten_frames is True:
          self.retainFolder = f'{self.batchFolder}/retained'
          createPath(self.retainFolder)


        self.skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
        self.calc_frames_skip_steps = math.floor(self.steps * self.skip_step_ratio)

        self.base_init_path = "init_images"

        if self.steps <= self.calc_frames_skip_steps:
          sys.exit("ERROR: You can't skip more steps than your total steps")

    def move_files(self, start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{self.batch_name}({self.batchNum})_{i:04}.png'
            new_file = new_folder + f'/{self.batch_name}({self.batchNum})_{i:04}.png'
            os.rename(old_file, new_file)

  #@markdown ---

    def showUtilizationGPU(self):
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        print()
        print()
        GPUtil.showUtilization()

    def isModelBusy(self):
      return self.isBusy

    def settings(self, id, prompt, init_image,init_scale ,skip_steps ,width ,height ,clip_guidance_scale ,tv_scale ,sat_scale,range_scale, cutn_batches):
      import re
      import urllib

      urls = []

      print("INIT IMAHE:: ", init_image, type(init_image), init_image == "")

      if init_image != "":
          print("USING RE", init_image != "")
          urls = re.findall('https?://+', init_image)

      if urls==[]:
        init_image = None

      counter = 0 
      retrieved_image = False

      print("After all, init image:", init_image, urls)

      while True and counter!=3 and init_image:
        try:
          imgURL = init_image
          response = requests.get(imgURL)
          if response.status_code == 200:
              file_path = os.path.join(self.base_init_path, f"{id}.png")
              Path(file_path).touch()
              with open(file_path, 'wb') as f:
                  f.write(response.content)
              retrieved_image = True  
              break
          else:
            raise Exception
        except Exception as e:
          print("Failure to download {} : {}".format(imgURL, e))
          print("Retrying download: {}/3".format(counter))
          print()
          try:
            open(file_path, 'a').close()
          except Exception as e:
            print("opening file failed.. skipping.")
            pass

        counter+=1

      if not retrieved_image and init_image:
        raise Exception("Failed to retrieved init image... passing.")
        
      if init_image:
        init_image = f"init_images/{id}.png"

      image_prompts = {}

      self.text_prompts[0] = prompt

      start_frame = 0
      batchNum = len(glob(self.batchFolder+"/*.txt"))
      while os.path.isfile(f"{self.batchFolder}/{self.batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{self.batchFolder}/{self.batch_name}-{batchNum}_settings.txt") is True:
        batchNum += 1

      print(f'Starting Run: {self.batch_name}({batchNum}) at frame {start_frame}')

      print("SKIP_STEPS == ", skip_steps)

      self.args = {
          'batchNum': batchNum,
          'id' : id,
          'model' : self.model,
          'diffusion' : self.diffusion,
          'prompts_series':split_prompts(self.text_prompts) if self.text_prompts else None,
          'image_prompts_series':split_prompts(image_prompts) if image_prompts else None,
          'seed': self.seed,
          'display_rate':self.display_rate,
          'n_batches':self.n_batches if animation_mode == 'None' else 1,
          'batch_size':self.batch_size,
          'batch_name': self.batch_name,
          'steps': self.steps,
          'diffusion_sampling_mode': diffusion_sampling_mode,
          'width_height': [width,height],
          'clip_guidance_scale': clip_guidance_scale,
          'tv_scale': tv_scale,
          'range_scale': range_scale,
          'sat_scale': sat_scale,
          'cutn_batches': cutn_batches,
          'init_image': init_image,
          'init_scale': init_scale,
          'skip_steps': skip_steps,
          'side_x': (int(width)//64)*64,
          'side_y':(int(height)//64)*64,
          'timestep_respacing': self.timestep_respacing,
          'diffusion_steps': self.diffusion_steps,
          'animation_mode': animation_mode,
          'video_init_path': video_init_path,
          'extract_nth_frame': extract_nth_frame,
          'video_init_seed_continuity': video_init_seed_continuity,
          'key_frames': key_frames,
          'max_frames': max_frames if animation_mode != "None" else 1,
          'interp_spline': interp_spline,
          'start_frame': start_frame,
          'midas_depth_model': midas_depth_model,
          'midas_weight': midas_weight,
          'near_plane': near_plane,
          'far_plane': far_plane,
          'fov': fov,
          'padding_mode': padding_mode,
          'sampling_mode': sampling_mode,
          'frames_scale': frames_scale,
          'calc_frames_skip_steps': self.calc_frames_skip_steps,
          'skip_step_ratio': self.skip_step_ratio,
          'calc_frames_skip_steps': self.calc_frames_skip_steps,
          'text_prompts': self.text_prompts,
          'image_prompts': image_prompts,
          'cut_overview': eval(self.cut_overview),
          'cut_innercut': eval(self.cut_innercut),
          'cut_ic_pow': self.cut_ic_pow,
          'cut_icgray_p': eval(self.cut_icgray_p),
          'intermediate_saves': self.intermediate_saves,
          'intermediates_in_subfolder': self.intermediates_in_subfolder,
          'steps_per_checkpoint': self.steps_per_checkpoint,
          'perlin_init': self.perlin_init,
          'perlin_mode': self.perlin_mode,
          'set_seed': self.set_seed,
          'eta': self.eta,
          'clamp_grad': self.clamp_grad,
          'clamp_max': self.clamp_max,
          'skip_augs': self.skip_augs,
          'randomize_class': self.randomize_class,
          'clip_denoised': self.clip_denoised,
          'fuzzy_prompt': self.fuzzy_prompt,
          'rand_mag': self.rand_mag,
      }

      # print(self.args)

      # print("**", self.args['init_image'])

      self.args = SimpleNamespace(**self.args)
      
    def runModel(self):
        self.showUtilizationGPU()
        print("Run model.. setting is busy to True!")
        self.isBusy = True
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Running model with width: {self.args.width_height[0]} and height {self.args.width_height[0]}")

        try:
          print("About to run model..")
          do_run(self.args)
          print("Setting is busy to false!")
          self.isBusy = False
          res = subprocess.check_output(["nvidia-smi",  "-L"])
          print(res)
        except Exception as e:
            self.showUtilizationGPU()
            self.isBusy = False
            print(e)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            res = subprocess.check_output(["nvidia-smi",  "-L"])
            print(res)

        finally:
            self.showUtilizationGPU()
            self.isBusy = False
            print('Seed used:', self.seed)
            gc.collect()
            torch.cuda.empty_cache()
            res = subprocess.check_output(["nvidia-smi",  "-L"])
            print(res)