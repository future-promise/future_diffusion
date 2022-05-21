


import math
from IPython import display
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from datetime import datetime

from future_diffusion.DiscoSetup import * 
from future_diffusion.disco import * 
from future_diffusion.DiscoUtils import *
from future_diffusion.Cutouts import *
from future_diffusion.Loss import *
from future_diffusion.SecondaryModel import *

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
    
def enumerateSamples(samples, args, cur_t, image_display, total_steps, i=0, frame_num = 0):
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
              print('what does this loop do', j)
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