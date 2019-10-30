import torch.nn as nn

# Discriminator Network
class Discriminator(nn.Module):
  def __init__(self, args):
    super(Discriminator, self).__init__()
    self.ngpu = args.ngpu
    self.main = nn.Sequential(
        # input nc * 64 * 64
        nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        
        # second
        nn.Conv2d(args.ndf, args.ndf *2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ndf *2),
        nn.LeakyReLU(0.2, inplace=True),
        
        # third
        nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        
        # third
        nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        
        # output
        nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
    
  def forward(self, input):
    return self.main(input)