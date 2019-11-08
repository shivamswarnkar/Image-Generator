from utils.data import create_data_loader
from utils.weights import weights_init
from generator import Generator
from discriminator import Discriminator

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils

def train_gan(args):

	# prepare dataloader
	dataloader = create_data_loader(args)

	# set up device
	device = torch.device('cuda:0' 
		if (torch.cuda.is_available() and args.ngpu>0) 
		else 'cpu')

	# Create & setup generator
	netG = Generator(args).to(device)

	# handle multiple gpus
	if (device.type == 'cuda' and args.ngpu>1):
		netG = nn.DataParallel(netG, list(range(args.ngpu)))

	# load from checkpoint if available
	if args.netG:
		netG.load_state_dict(torch.load(args.netG))
	
	# initialize network with random weights 
	else:
		netG.apply(weights_init)

	# Create & setup discriminator
	netD = Discriminator(args).to(device)

	# handle multiple gpus
	if (device.type == 'cuda' and args.ngpu>1):
		netD = nn.DataParallel(netD, list(range(args.ngpu)))

	# load from checkpoint if available
	if args.netD:
		netD.load_state_dict(torch.load(args.netD))
	
	# initialize network with random weights 
	else:
		netD.apply(weights_init)


	# setup up loss & optimizers
	criterion = nn.BCELoss()
	optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
	optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

	# For input of generator in testing
	fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

	# convention for training
	real_label = 1
	fake_label = 0

	# training data for later analysis
	img_list= []
	G_losses = []
	D_losses = []
	iters = 0


	# epochs
	num_epochs = 150

	print('Starting Training Loop....')
	# For each epoch
	for e in range(args.num_epochs):
		# for each batch in the dataloader
			for i, data in enumerate(dataloader, 0):
				########## Training Discriminator ##########
				netD.zero_grad()

				# train with real data
				real_data = data[0].to(device)

				# make labels
				batch_size = real_data.size(0)
				labels = torch.full((batch_size,), real_label, device=device)

				# forward pass real data through D
				real_outputD = netD(real_data).view(-1)

				# calc error on real data
				errD_real = criterion(real_outputD, labels)

				# calc grad
				errD_real.backward()
				D_x = real_outputD.mean().item()

				# train with fake data
				noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
				fake_data = netG(noise)
				labels.fill_(fake_label)

				# classify fake
				fake_outputD = netD(fake_data.detach()).view(-1)

				# calc error on fake data
				errD_fake = criterion(fake_outputD, labels)

				# calc grad
				errD_fake.backward()
				D_G_z1 = fake_outputD.mean().item()

				# add all grad and update D
				errD = errD_real + errD_fake
				optimizerD.step()

				########################################
				########## Training Generator ##########
				netG.zero_grad()

				# since aim is fooling the netD, labels should be flipped
				labels.fill_(real_label)

				# forward pass with updated netD
				fake_outputD = netD(fake_data).view(-1)

				# calc error
				errG = criterion(fake_outputD, labels)

				# calc grad
				errG.backward()

				D_G_z2 = fake_outputD.mean().item()

				# update G
				optimizerG.step()

				########################################

				# output training stats
				if i%500==0:
					print(f'[{e+1}/{args.num_epochs}][{i+1}/{len(dataloader)}]\
						\tLoss_D:{errD.item():.4f}\
						\tLoss_G:{errG.item():.4f}\
						\tD(x):{D_x:.4f}\
						\tD(G(z)):{D_G_z1:.4f}/{D_G_z2:.4f}')

				# for later plot
				G_losses.append(errG.item())
				D_losses.append(errD.item())

				# generate fake image on fixed noise for comparison
				if ((iters % 500== 0) or 
					((e == args.num_epochs -1) and (i==len(dataloader)-1))):
					with torch.no_grad():
						fake = netG(fixed_noise).detach().cpu()
						img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
				iters +=1

			if e%args.save_every==0:
				# save at args.save_every epoch
				torch.save(netG.state_dict(), args.outputG)
				torch.save(netD.state_dict(), args.outputD)
				print(f'Made a New Checkpoint for {e+1}')

	torch.save(netG.state_dict(), args.outputG)
	torch.save(netD.state_dict(), args.outputD)
	print(f'Saved Final model at {args.outputG} & {args.outputD}')
	# return training data for analysis
	return img_list, G_losses, D_losses







		
	


