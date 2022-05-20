


#@title 1.1 Check GPU Status
from future_diffusion.DiscoSetup import * 
from future_diffusion.disco import * 

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
from future_diffusion.Loss import *
from future_diffusion.SecondaryModel import *
from future_diffusion.Run.DiscoCondition import *

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def saveImage(args, image, cur_t, j, filename):
  if args.steps_per_checkpoint is not None:
    if j % args.steps_per_checkpoint == 0 and j > 0:
      if args.intermediates_in_subfolder is True:
        image.save(f'{partialFolder}/{filename}')
      else:
        image.save(f'{batchFolder}/{filename}')
  else:
    if j in args.intermediate_saves:
      if args.intermediates_in_subfolder is True:
        image.save(f'{partialFolder}/{filename}')
      else:
        image.save(f'{batchFolder}/{filename}')
  if cur_t == -1:
    if True:
      pass
    image.save(f'{batchFolder}/{filename}')

def createSeed(seed):
  if seed is not None:
    np.random.seed(seed)
    print('seed', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def getInitImage(args):
  if args.init_image in ['','none', 'None', 'NONE']:
    init_image = None
  else:
    init_image = args.init_image

  init = None
  if init_image is not None:
      init = Image.open(fetch(init_image)).convert('RGB')
      init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
      init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

  return init

def createModelStats(args, frame_num = 0):
  target_embeds, weights = [], []

  if args.prompts_series is not None and frame_num >= len(args.prompts_series):
    frame_prompt = args.prompts_series[-1]
  elif args.prompts_series is not None:
    frame_prompt = args.prompts_series[frame_num]
  else:
    frame_prompt = []
  
  print(f'Frame {frame_num} Prompt: {frame_prompt}')

  model_stats = []
  for clip_model in args.clip_models:
    cutn = 16
    model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
    model_stat["clip_model"] = clip_model
    
    for prompt in frame_prompt:
        txt, weight = parse_prompt(prompt)
        txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
        model_stat["target_embeds"].append(txt)
        model_stat["weights"].append(weight)


    model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
    model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
    if model_stat["weights"].sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    model_stat["weights"] /= model_stat["weights"].sum().abs()
    model_stats.append(model_stat)

  return model_stats


# removed list:
# image prompt - useful for video???
# fuzzy prompt

def do_run(args):
  createSeed(args.seed)
  frame_num = 0
  # display.clear_output(wait=True)

  init = getInitImage(args)
  loss_values = []
  model_stats = createModelStats(args)

  cur_t = None
  condition_fn = buildConditionFunction(args, cur_t, model_stats, init, loss_values)

  if args.diffusion_sampling_mode == 'ddim':
      sample_fn = args.diffusion.ddim_sample_loop_progressive
  else:
      sample_fn = args.diffusion.plms_sample_loop_progressive


  image_display = Output()
  for i in range(args.n_batches):
      if args.animation_mode == 'None':
        # display.clear_output(wait=True)
        batchBar = tqdm(range(args.n_batches), desc ="Batches")
        batchBar.n = i
        batchBar.refresh()
      print('')
      display.display(image_display)
      gc.collect()
      torch.cuda.empty_cache()
      cur_t = args.diffusion.num_timesteps - args.skip_steps - 1
      total_steps = cur_t

      if args.perlin_init:
          init = regen_perlin()

      if args.diffusion_sampling_mode == 'ddim':
          samples = sample_fn(
              args.model,
              (args.batch_size, 3, args.side_y, args.side_x),
              clip_denoised=args.clip_denoised,
              model_kwargs={},
              cond_fn=condition_fn,
              progress=True,
              skip_timesteps=args.skip_steps,
              init_image=init,
              randomize_class=args.randomize_class,
              eta=args.eta,
          )
      else:
          samples = sample_fn(
              args.model,
              (args.batch_size, 3, args.side_y, args.side_x),
              clip_denoised=args.clip_denoised,
              model_kwargs={},
              cond_fn=condition_fn,
              progress=True,
              skip_timesteps=args.skip_steps,
              init_image=init,
              randomize_class=args.randomize_class,
              order=2,
          )
      
      print('samples', samples)
      # with run_display:
      for j, sample in enumerate(samples):    
        cur_t -= 1
        intermediateStep = False
        if args.steps_per_checkpoint is not None:
            if j % args.steps_per_checkpoint == 0 and j > 0:
              intermediateStep = True
        elif j in args.intermediate_saves:
          intermediateStep = True
        with image_display:
          if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
              for k, image in enumerate(sample['pred_xstart']):
                  # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                  current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                  percent = math.ceil(j/total_steps*100)
                  if args.n_batches > 0:
                    #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                    if cur_t == -1 and args.intermediates_in_subfolder is True:
                      save_num = f'{frame_num:04}' if args.animation_mode != "None" else i
                      filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                    else:
                      #If we're working with percentages, append it
                      if args.steps_per_checkpoint is not None:
                        filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                      # Or else, iIf we're working with specific steps, append those
                      else:
                        filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                  image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                  if j % args.display_rate == 0 or cur_t == -1:
                    image.save('progress.png')
                    # display.clear_output(wait=True)
                    display.display(display.Image('progress.png'))
                  saveImage(args, image, cur_t, j, filename)
      
      plt.plot(np.array(loss_values), 'r')