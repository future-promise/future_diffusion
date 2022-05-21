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
from future_diffusion.Run.Display import enumerateSamples
# from future_diffusion.Run.DiscoCondition import *

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

def getSamples(args, cond_fn, init):
  if args.diffusion_sampling_mode == 'ddim':
      sample_fn = args.diffusion.ddim_sample_loop_progressive
  else:
      sample_fn = args.diffusion.plms_sample_loop_progressive
  
  if args.diffusion_sampling_mode == 'ddim':
    samples = sample_fn(
        args.model,
        (args.batch_size, 3, args.side_y, args.side_x),
        clip_denoised=args.clip_denoised,
        model_kwargs={},
        cond_fn=cond_fn,
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
          cond_fn=cond_fn,
          progress=True,
          skip_timesteps=args.skip_steps,
          init_image=init,
          randomize_class=args.randomize_class,
          order=2,
      )
  return samples

def applyModel(args, cur_t, x, n):
  if args.use_secondary_model is True:
    alpha = torch.tensor(args.diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
    sigma = torch.tensor(args.diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
    cosine_t = alpha_sigma_to_t(alpha, sigma)
    out = secondary_model(x, cosine_t[None].repeat([n])).pred
    fac = args.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
    x_in = out * fac + x * (1 - fac)
    x_in_grad = torch.zeros_like(x_in)
  else:
    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
    out = args.diffusion.p_mean_variance(args.model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
    fac = args.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
    x_in = out['pred_xstart'] * fac + x * (1 - fac)
    x_in_grad = torch.zeros_like(x_in)
  return out, x_in, x_in_grad

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


def runModelStat(model_stat, t, args, n, x_in, x_in_grad, loss_values):
  for i in range(args.cutn_batches):
    t_int = int(t.item())+1 #errors on last step without +1, need to find source
    #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
    try:
        input_resolution=model_stat["clip_model"].visual.input_resolution
    except:
        input_resolution=224

    cuts = MakeCutoutsDango(input_resolution,args,
            Overview= args.cut_overview[1000-t_int], 
            InnerCrop = args.cut_innercut[1000-t_int], IC_Size_Pow=args.cut_ic_pow, IC_Grey_P = args.cut_icgray_p[1000-t_int]
            )
    clip_in = normalize(cuts(x_in.add(1).div(2)))
    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
    dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
    dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
    loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
    x_in_grad += torch.autograd.grad(losses.sum() * args.clip_guidance_scale, x_in)[0] / args.cutn_batches

def sumLosses(args,x_in, out, init):
  tv_losses = tv_loss(x_in)
  if args.use_secondary_model is True:
    range_losses = range_loss(out)
  else:
    range_losses = range_loss(out['pred_xstart'])
  sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
  loss = tv_losses.sum() * args.tv_scale + range_losses.sum() * args.range_scale + sat_losses.sum() * args.sat_scale
  if init is not None and args.init_scale:
      init_losses = lpips_model(x_in, init)
      loss = loss + init_losses.sum() * args.init_scale
  return loss


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

  # condition_fn = buildConditionFunction(args, cur_t, model_stats, init, loss_values)
  # print('condition fn ', condition_fn)
  cur_t = None

  def buildConditionFunction():
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]

            out, x_in, x_in_grad = applyModel(args, cur_t, x, n)

            # only has length 1
            for model_stat in model_stats:
              runModelStat(model_stat, t, args, n, x_in, x_in_grad, loss_values)

            loss = sumLosses(args, x_in, out, init)

            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any()==False:
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
              x_is_NaN = True
              grad = torch.zeros_like(x)
        if args.clamp_grad and x_is_NaN == False:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
        return grad
    return cur_t, cond_fn

  cur_t, condition_fn = buildConditionFunction()
  print('outside cur_t', cur_t)

  image_display = Output()
  i = 0
  if args.animation_mode == 'None':
    # display.clear_output(wait=True)
    batchBar = tqdm(range(args.n_batches), desc ="Batches")
    batchBar.n = i
    batchBar.refresh()
  display.display(image_display)
  gc.collect()
  torch.cuda.empty_cache()
  cur_t = args.diffusion.num_timesteps - args.skip_steps - 1
  total_steps = cur_t

  if args.perlin_init:
      init = regen_perlin()

  samples = getSamples(args, condition_fn, init)

  enumerateSamples(samples, args, cur_t, image_display, total_steps)
  
  plt.plot(np.array(loss_values), 'r')