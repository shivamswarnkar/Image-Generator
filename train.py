from utils.args import get_train_args
from DCGAN.train import train_gan

if __name__ == '__main__':
	# read arguments from terminal
	args = get_train_args()

	# train gan
	train_gan(args)