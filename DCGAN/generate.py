import torch
import torchvision.utils as vutils
import numpy as np
from generator import Generator 
import matplotlib.pyplot as plt

def generate_images(args):

	# set up device
	device = torch.device('cuda:0' 
		if (torch.cuda.is_available() and args.ngpu>0)  
		else 'cpu')

	# load generator model
	netG = Generator(args).to(device)
	netG.load_state_dict(torch.load(args.netG))

	# create random noise
	noise = torch.randn(args.n, args.nz, 1, 1, device=device)
	fake = netG(noise).detach().cpu()
	img = vutils.make_grid(fake, padding=2, normalize=True)

	# save image
	plt.axis("off")
	plt.imshow(np.transpose(img,(1,2,0)))
	plt.savefig(args.output_path)

