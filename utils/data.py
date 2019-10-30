import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset


def create_data_loader(args):
	dataset = dset.ImageFolder(root=args.dataroot,
		transform=transforms.Compose(
			[
			transforms.Resize(args.image_size),
			transforms.CenterCrop(args.image_size),
			transforms.ToTensor(),
			transforms.Normalize(
				(0.5,0.5,0.5),
				(0.5, 0.5,0.5)
				)
			]))

	# setting up data loader
	dataloader = torch.utils.data.DataLoader(dataset, 
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers
		)

	return dataloader

