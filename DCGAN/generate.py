import torch
import torchvision.utils as vutils
import numpy as np
from generator import Generator 
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def generate_images(args):
    # set up device
    device = torch.device('cuda:0' 
        if (torch.cuda.is_available() and args.ngpu>0)  
        else 'cpu')

    # load generator model
    netG = Generator(args).to(device)
    netG.load_state_dict(torch.load(args.netG))

    filename, ext = os.path.splitext(os.path.basename(args.output_path))
    
    for i in tqdm(range(args.n)):
        # create random noise
        noise = torch.randn(1, args.nz, 1, 1, device=device)

        with torch.no_grad():
            fake = netG(noise).detach().cpu()

        img_np = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(), (1, 2, 0))
        
        new_filename = f"{filename}_{str(i).zfill(3)}{ext}"
        new_output_path = os.path.join(os.path.dirname(args.output_path), new_filename)
        
        # save image
        plt.imsave(new_output_path, img_np)
