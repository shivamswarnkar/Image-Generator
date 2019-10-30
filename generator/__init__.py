import torch.nn as nn

# Generator Network
class Generator(nn.Module):
  def __init__(self, args):
    super(Generator, self).__init__()
    self.ngpu = args.ngpu
    self.main = nn.Sequential(
        # input Z; First
        nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(args.ngf * 8),
        nn.ReLU(True),
        
        #second
        nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ngf*4),
        nn.ReLU(True),
        
        #third
        nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ngf*2),
        nn.ReLU(True),
        
        # fourth
        nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ngf),
        nn.ReLU(True),
        
        # output nc * 64 * 64
        nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    
  def forward(self, input):
    return self.main(input)