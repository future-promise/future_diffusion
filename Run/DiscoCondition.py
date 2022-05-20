



#@title 1.1 Check GPU Status
from future_diffusion.DiscoSetup import * 
from future_diffusion.disco import * 

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from future_diffusion.DiscoUtils import *
from future_diffusion.Cutouts import *
from future_diffusion.Loss import *
from future_diffusion.SecondaryModel import *

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def buildConditionFunction(args, cur_t, model_stats, init, loss_values, frame_num =0):
  def cond_fn(x, t, y=None):
      with torch.enable_grad():
          x_is_NaN = False
          x = x.detach().requires_grad_()
          n = x.shape[0]
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
          for model_stat in model_stats:
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
          x_in_grad += torch.autograd.grad(loss, x_in)[0]
          if torch.isnan(x_in_grad).any()==False:
              grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
          else:
            # print("NaN'd")
            x_is_NaN = True
            grad = torch.zeros_like(x)
      if args.clamp_grad and x_is_NaN == False:
          magnitude = grad.square().mean().sqrt()
          return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
      return grad
  return cond_fn