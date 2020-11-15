
from torch.utils.data import DataLoader
from datasets import InputDataset  
import datetime
import time
import sys 

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from nets import * 

# Reference
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py

vgg19 =   models.vgg19(pretrained=True)
dec_w = torch.load('models/pretrained_models/autoencoder_vgg19/vgg19_4/feature_invertor_conv4_1.pth')
instrument_list = ['harpsichord', 'piano']
device_type = "cpu"
gpu_number = 0
lr = 2e-4
b1, b2 = 0, .9
weight_decay = 1e-4
epoch = 0
n_epochs = 2

# lambdas
lambda_style = 2
lambda_homo = 3
lambda_cycle = 3
lambda_id = 3
lambda_D_style = 4

# Loss functions
criterion_identity = torch.nn.L1Loss()
criterion_GAN = torch.nn.BCELoss()
criterion_style = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
G = Generator(vgg19, dec_w)
D = Discriminator(input_shape=(1,256,256), nclass=len(instrument_list))
#S = Siamese()
# gpu 
if (device_type == "gpu") and torch.has_cudnn:
    device = torch.device("cuda:{}".format(gpu_number))
else:
    device = torch.device("cpu")
G = G.to(device)
D = D.to(device)
#S = S.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
#optimizer_S = torch.optim.Adam(S.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

Tensor = torch.cuda.FloatTensor if device_type == "gpu" else torch.Tensor


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []
        self.label = []

    def push_and_pop(self, data, label):
        to_return_data = []
        to_return_label = []
        for i, element in enumerate(data.data):
            element = torch.unsqueeze(element, 0)
            element_label = torch.unsqueeze(label.data[i], 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                self.label.append(element_label)
                to_return_data.append(element)
                to_return_label.append(element_label)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return_data.append(self.data[i].clone())
                    to_return_label.append(self.label[i].clone())
                    self.data[i] = element
                    self.label[i] = element_label
                else:
                    to_return_data.append(element)
                    to_return_label.append(element_label)
        return Variable(torch.cat(to_return_data)), Variable(torch.cat(to_return_label))
    
# Buffers of previously generated samples
fake_buffer = ReplayBuffer()

# Training data loader
data = InputDataset(root='./data/spectrogram', instrument_list=instrument_list)
loader = DataLoader(data, batch_size=3, shuffle=True)

# ----------
#  Training
# ----------
prev_time = time.time()
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(loader):
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Labels        
        label_A = Variable(batch["lab_A"].type(Tensor), requires_grad=False)
        label_B = Variable(batch["lab_B"].type(Tensor), requires_grad=False)
        # Adversarial ground truths
        valid = Variable(Tensor(real_A.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(real_A.size(0), 1).fill_(0.0), requires_grad=False)

        
        # ------------------
        #  Train Generator
        # ------------------
        G.train()
        optimizer_G.zero_grad()
#        sys.stdout.write('Size of real_A:' + str(real_A.size()))
#        print('Size of real_A:', real_A.size())
#        print('Size of real_B:', real_B.size())
        # concatenate content A and style B
        real_AB = torch.cat([real_A, real_B], dim=1)
        real_BA = torch.cat([real_B, real_A], dim=1)
#        print(real_AB.size())
        
        # Homogeneous loss
        fake_B = G(real_AB) 
        loss_homo_real_B = criterion_identity(G(torch.cat([real_B, fake_B], dim=1)), real_B)
        fake_A = G(real_BA)
        loss_homo_real_A = criterion_identity(G(torch.cat([real_A, fake_A], dim=1)), real_A)
        
        loss_homo_fake_B = criterion_identity(G(torch.cat([fake_B, real_B], dim=1)), fake_B)
        loss_homo_fake_A = criterion_identity(G(torch.cat([fake_A, real_A], dim=1)), fake_A)
        
        loss_homo = (loss_homo_real_B + loss_homo_real_A + loss_homo_fake_B + loss_homo_fake_A) / 4        
        
        # GAN loss 
        print('Size of input to D:', fake_B.size())
        fake_B_disc, fake_B_style = D(fake_B) # fake/real and style
        loss_GAN_AB = criterion_GAN(fake_B_disc, valid) # Loss measures generator's ability to fool the discriminator
        loss_G_style_AB = criterion_style(fake_B_style, label_B)# Loss measures generator's ability to generate style of B
        
        fake_A_disc, fake_A_style = D(fake_A) # fake/real and style
        loss_GAN_BA = criterion_GAN(fake_A_disc, valid) # Loss measures generator's ability to fool the discriminator
        loss_G_style_BA = criterion_style(fake_A_style, label_A)# Loss measures generator's ability to generate style of B
        
        loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2
        loss_G_style = (loss_G_style_AB + loss_G_style_BA)/2
       
        # Cycle loss
        loss_cycle_real_A = criterion_identity(G(torch.cat([fake_B, real_A], dim=1)), real_A)
        loss_cycle_real_B = criterion_identity(G(torch.cat([fake_A, real_B], dim=1)), real_B)
        loss_cycle_fake_A = criterion_identity(G(torch.cat([real_B, fake_A], dim=1)), fake_A)
        loss_cycle_fake_B = criterion_identity(G(torch.cat([real_A, fake_B], dim=1)), fake_B)
        
        loss_cycle = (loss_cycle_real_A + loss_cycle_real_B + loss_cycle_fake_A + loss_cycle_fake_B)/4
        
        # Identity loss
        loss_id_B = criterion_identity(G(torch.cat([real_B, real_B], dim=1)), real_B)
        loss_id_A = criterion_identity(G(torch.cat([real_A, real_A], dim=1)), real_A)
        
        loss_id = (loss_id_B + loss_id_A)/2
        
        # Total loss
        loss_G = loss_GAN + lambda_style*loss_G_style + lambda_homo*loss_homo + lambda_cycle*loss_cycle + lambda_id*loss_id
        
        loss_G.backward()
        optimizer_G.step()
        
        # -----------------------
        #  Train Discriminator
        # -----------------------
        D.zero_grad()
        
        # Real loss
        real_A_disc, real_A_style = D(real_A)
        loss_real_A = criterion_GAN(real_A_disc, valid)
        real_B_disc, real_B_style = D(real_B)
        loss_real_B = criterion_GAN(real_B_disc, valid)
        loss_real = (loss_real_A + loss_real_B)/2
        
        # Fake loss
        
        fake_A_, fake_A_label_ = fake_buffer.push_and_pop(fake_A, label_A)
        fake_A_disc_, fake_A_style_ = D(fake_A_.detach())
        loss_fake_A = criterion_GAN(fake_A_disc_, fake)
        
        fake_B_, fake_B_label_ = fake_buffer.push_and_pop(fake_B)
        fake_B_disc_, fake_B_style_ = D(fake_B_.detach())
        loss_fake_B = criterion_GAN(fake_B_disc_, fake)
        
        loss_fake = (loss_fake_A + loss_fake_B)/2
        
        loss_real_fake = (loss_real + loss_fake)/2
        
        # D Style loss
        loss_D_style_A = criterion_style(fake_A_style_, fake_A_label_.detach())
        loss_D_style_B = criterion_style(fake_B_style_, fake_B_label_.detach())
        loss_D_style = (loss_D_style_A + loss_D_style_B)/2
        
        loss_D = loss_real_fake + lambda_D_style*loss_D_style
        
        loss_D.backward()
        optimizer_D.step()
        
        # --------------
        #  Log Progress
        # --------------
        
        # Determine approximate time left
        batches_done = epoch * len(loader) + i
        batches_left = n_epochs * len(loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        
        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, id: %f, homo: %f] ETA: %s"
            % (
                epoch,
                n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_id.item(),
                loss_homo.item(),
                time_left,
            )
        )
        
        # If at sample interval save image
#         if batches_done % opt.sample_interval == 0:
#             sample_images(batches_done)
        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
        
        
        
        
        
        
        
        
        
        