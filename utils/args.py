import argparse


def get_train_args(base_args=False):
	parser = argparse.ArgumentParser('Training DCGAN for Image Generation')

	parser.add_argument('--netD', type=str, 
		default=None, 
		help='Path to pretrained/checkpoint of discriminator network file. If not provided, training will start from scratch.')

	parser.add_argument('--netG', type=str, 
		default=None, 
		help='Path to pretrained/checkpoint of generator network file. If not provided, training will start from scratch.')

	parser.add_argument('--dataroot', type=str, 
		help='Path of source image dataset')

	parser.add_argument('--workers', type=int, 
		default=2, 
		help='Number of workers for dataloading')

	parser.add_argument('--batch_size', type=int, 
		default=128, 
		help='Batch Size for GAN training')

	parser.add_argument('--image_size', type=int, 
		default=64, 
		help='Height-width of the generated image')

	parser.add_argument('--nc', type=int, 
		default=3, 
		help='Number of channels in output image')

	parser.add_argument('--nz', type=int, 
		default=100, 
		help='Size of latent vector z; output of generator')

	parser.add_argument('--ngf', type=int, 
		default=64, 
		help='Size of feature maps in generator')

	parser.add_argument('--ndf', type=int, 
		default=64, 
		help='Size of features maps in discriminator')

	parser.add_argument('--num_epochs', type=int, 
		default=5, 
		help='Number of Epochs for training')

	parser.add_argument('--lr', type=float, 
		default=0.0002, 
		help='Learning Rate')

	parser.add_argument('--beta1', type=float, 
		default=0.5, 
		help='Beta 1 value for Adam Optimizer')

	parser.add_argument('--ngpu', type=int, 
		default=1, 
		help='Number of GPUs to use')

	parser.add_argument('--save_every', type=int, 
		default=5, 
		help='Make a checkpoint after each n epochs')

	parser.add_argument('--outputD', type=str, 
		default='checkpoints/netD.pth', 
		help='Path where discriminator model will be saved/checkpoint')

	parser.add_argument('--outputG', type=str, 
		default='checkpoints/netG.pth', 
		help='Path where generator model will be saved/checkpoint')


	if base_args:
		args = parser.parse_args([])	
	else:
		args = parser.parse_args()

	return args


def get_generate_args(base_args=False):
	parser = argparse.ArgumentParser('Image Generation using DCGAN')

	parser.add_argument('--netG', type=str, 
		help='Path to pretrained/checkpoint of generator network file which will be used to generate images.')

	parser.add_argument('--n', type=int, 
		default=64, 
		help='Number of Images to be generated')

	parser.add_argument('--nc', type=int, 
		default=3, 
		help='Number of channels in output image')

	parser.add_argument('--ngpu', type=int, 
		default=1, 
		help='Number of GPUs to use')

	parser.add_argument('--nz', type=int, 
		default=100, 
		help='Size of latent vector z; output of generator')

	parser.add_argument('--ngf', type=int, 
		default=64, 
		help='Size of feature maps in generator')

	parser.add_argument('--output_path', type=str, 
		default='output/fake.png', 
		help='Path where generated images will be saved')

	if base_args:
		args = parser.parse_args([])	
	else:
		args = parser.parse_args()

	return args
