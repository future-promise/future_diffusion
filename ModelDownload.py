import hashlib
from future_diffusion.DiscoSetup import wget
import os

def download_models(model_path, diffusion_model,use_secondary_model,fallback=False, check_model_SHA = False):
    model_256_downloaded = False
    model_512_downloaded = False
    model_secondary_downloaded = False

    model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
    model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
    model_secondary_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'

    model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
    model_512_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'

    model_256_link_fb = 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'
    model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link_fb = 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth'

    model_256_path = f'{model_path}/256x256_diffusion_uncond.pt'
    model_512_path = f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_path = f'{model_path}/secondary_model_imagenet_2.pth'

    if fallback:
        model_256_link = model_256_link_fb
        model_512_link = model_512_link_fb
        model_secondary_link = model_secondary_link_fb
    # Download the diffusion model
    if diffusion_model == '256x256_diffusion_uncond':
        if os.path.exists(model_256_path) and check_model_SHA:
            print('Checking 256 Diffusion File')
            with open(model_256_path,"rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest();
            if hash == model_256_SHA:
                print('256 Model SHA matches')
                model_256_downloaded = True
            else:
                print("256 Model SHA doesn't match, redownloading...")
                wget(model_256_link, model_path)
                if os.path.exists(model_256_path):
                    model_256_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
        elif os.path.exists(model_256_path) and not check_model_SHA or model_256_downloaded == True:
            print('256 Model already downloaded, check check_model_SHA if the file is corrupt')
        else:  
            wget(model_256_link, model_path)
            if os.path.exists(model_256_path):
                model_256_downloaded = True
            else:
                print('First URL Failed using FallBack')
                download_models(diffusion_model,True)
    elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        if os.path.exists(model_512_path) and check_model_SHA:
            print('Checking 512 Diffusion File')
            with open(model_512_path,"rb") as f:
                  bytes = f.read() 
                  hash = hashlib.sha256(bytes).hexdigest();
            if hash == model_512_SHA:
                print('512 Model SHA matches')
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
            else:  
                print("512 Model SHA doesn't match, redownloading...")
                wget(model_512_link, model_path)
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
        elif os.path.exists(model_512_path) and not check_model_SHA or model_512_downloaded:
            print('512 Model already downloaded, check check_model_SHA if the file is corrupt')
        else:  
            wget(model_512_link, model_path)
            model_512_downloaded = True
    # Download the secondary diffusion model v2
    if use_secondary_model:
        if os.path.exists(model_secondary_path) and check_model_SHA:
            print('Checking Secondary Diffusion File')
            with open(model_secondary_path,"rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest();
            if hash == model_secondary_SHA:
                print('Secondary Model SHA matches')
                model_secondary_downloaded = True
            else:  
                print("Secondary Model SHA doesn't match, redownloading...")
                wget(model_secondary_link, model_path)
                if os.path.exists(model_secondary_path):
                    model_secondary_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
        elif os.path.exists(model_secondary_path) and not check_model_SHA or model_secondary_downloaded:
            print('Secondary Model already downloaded, check check_model_SHA if the file is corrupt')
        else:  
            wget(model_secondary_link, model_path)
            if os.path.exists(model_secondary_path):
                model_secondary_downloaded = True
            else:
                print('First URL Failed using FallBack')
                download_models(diffusion_model,use_secondary_model,True)