import pandas as pd
import numpy as np

#@markdown ####**Animation Mode:**
animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input'] {type:'string'}

video_init_path = "training.mp4" #@param {type: 'string'}
extract_nth_frame = 2 #@param {type: 'number'}
video_init_seed_continuity = True #@param {type: 'boolean'}


key_frames = True #@param {type:"boolean"}
max_frames = 10000#@param {type:"number"}

interp_spline = 'Linear' #Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = "0:(0)"#@param {type:"string"}
zoom = "0: (1), 10: (1.05)"#@param {type:"string"}
translation_x = "0: (0)"#@param {type:"string"}
translation_y = "0: (0)"#@param {type:"string"}
translation_z = "0: (10.0)"#@param {type:"string"}
rotation_3d_x = "0: (0)"#@param {type:"string"}
rotation_3d_y = "0: (0)"#@param {type:"string"}
rotation_3d_z = "0: (0)"#@param {type:"string"}
midas_depth_model = "dpt_large"#@param {type:"string"}
midas_weight = 0.3#@param {type:"number"}
near_plane = 200#@param {type:"number"}
far_plane = 10000#@param {type:"number"}
fov = 40#@param {type:"number"}
padding_mode = 'border'#@param {type:"string"}
sampling_mode = 'bicubic'#@param {type:"string"}


turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10 # frames

#insist turbo be used only w 3d anim.
if turbo_mode and animation_mode != '3D':
    print('=====')
    print('Turbo mode only available with 3D animations. Disabling Turbo.')
    print('=====')
    turbo_mode = False


frames_scale = 1500 #@param{type: 'integer'}
frames_skip_steps = '60%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}


vr_mode = False 
vr_eye_angle = 0.5 
vr_ipd = 5.0

#insist VR be used only w 3d anim.
if vr_mode and animation_mode != '3D':
    print('=====')
    print('VR mode only available with 3D animations. Disabling VR.')
    print('=====')
    vr_mode = False

def split_prompts(prompts):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series
