from utils.args import get_train_args, get_generate_args
from DCGAN.train import train_gan
from DCGAN.generate import generate_images

def train(dataroot, 
	netD=None, netG=None, workers=2, 
	batch_size=128, image_size=64, 
	nc=3, nz=100, ngf=64, ndf=64, 
	num_epochs=5, lr=0.0002, 
	beta1=0.5, ngpu=1, save_every=5, 
	outputD='checkpoints/netD.pth', 
	outputG='checkpoints/netG.pth'):

	# get default args
	args = get_train_args(base_args=True)

	# update args 
	args.netD = netD
	args.netG = netG
	args.dataroot = dataroot
	args.workers = workers
	args.batch_size = batch_size
	args.image_size = image_size
	args.nc = nc
	args.nz = nz
	args.ngf = ngf
	args.ndf = ndf
	args.num_epochs = num_epochs
	args.lr = lr
	args.beta1 = beta1
	args.ngpu =ngpu
	args.save_every = save_every
	args.outputD = outputD
	args.outputG = outputG

	# train DCGAN
	return train_gan(args)


def generate(netG, n=64, nc=3, nz=100, ngf=64, ngpu=1, output_path='output/fake.png'):

	# get default args
	args = get_generate_args(base_args=True)

	# update args
	args.netG = netG
	args.n = n
	args.nc = nc
	args.nz = nz
	args.ngf = ngf
	args.ngpu = ngpu
	args.output_path=output_path

	# generate image
	generate_images(args)

