#@title 1.1 Check GPU Status
from future_diffusion.DiscoSetup import * 

import torch
import gc
import math
from IPython import display
import lpips
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from CLIP import clip
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from future_diffusion.DiscoUtils import *
from future_diffusion.Cutouts import *
# MakeCutouts, MakeCutoutsDango
from future_diffusion.Loss import *

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0


#@title 1.6 Define the secondary diffusion model

from future_diffusion.SecondaryModel import *

"""# 2. Diffusion and CLIP model settings"""

#@markdown ####**Models Settings:**
diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100"]
use_secondary_model = True #@param {type: 'boolean'}
# diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']  


use_checkpoint = True #@param {type: 'boolean'}
ViTB32 = True #@param{type:"boolean"}
ViTB16 = False #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = False #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}

#@markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False #@param{type:"boolean"}

from future_diffusion.ModelDownload import download_models

download_models(model_path, diffusion_model, use_secondary_model)

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
    secondary_model.eval().requires_grad_(False).to(device)


normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)


"""### Animation Settings"""
from future_diffusion.DiscoAnimate import *

from future_diffusion.Run.DiscoRun import *
