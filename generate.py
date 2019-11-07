from utils.args import get_generate_args
from DCGAN.generate import generate_images

if __name__ == '__main__':
	# read arguments from terminal
	args = get_generate_args()

	# train gan
	generate_images(args)