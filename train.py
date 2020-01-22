import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from model import Generator,Discriminator,DHead,QHead
from ECG_dataset import ECG_dataset
from utils import *
from config import params

from torch.utils.tensorboard import SummaryWriter

# set env settings
seed = 1122
random.seed(seed)
torch.manual_seed(seed)
print("random seed : ",seed)

device = "cuda:{}".format(params['GPU']) if (torch.cuda.is_available() and params['GPU'] is not None) else 'cpu'
device = torch.device(device)
print(device, ' is used.\n')

writer = SummaryWriter()

#Load dataset and dataloader
data_folder_dir = params['folder_dir']
filepathList = filepathList_gen(data_folder_dir,params['sample_run'])
ecg_dataset = ECG_dataset(filepathList,freq=params['freq'],length=params['length'],
                          norm=params['norm'],sample_step=params['sample_step'])
ecg_dataloader = DataLoader(ecg_dataset,params['batch_size'],True,num_workers=8,drop_last=True)
print('Finish data load')

# Initialize the network
netG = Generator(params['n_z']+params['n_con_c']+params['n_dis_c']*params['dis_c_dim']).to(device)
netG.apply(weight_init)
print(netG)

discriminator = Discriminator().to(device)
discriminator.apply(weight_init)
print(discriminator)

netD = DHead().to(device)
netD.apply(weight_init)
print(netD)

netQ = QHead().to(device)
netQ.apply(weight_init)
print(netQ)

#loss for discriminator real/fake
criterionD = nn.BCELoss()
#loss for discreate latent code
criterionQ_dis = nn.CrossEntropyLoss()
#loss for continous latent code
criterionQ_con = NormalNLLLoss()

# Adam optimiser is used.
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['learning_rate'],
                    betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'],
                    betas=(params['beta1'], params['beta2']))


fixed_noise,idx = noise_sample(params['n_dis_c'],params['dis_c_dim'],params['n_con_c'],params['n_z'],10,device)

real_label,fake_label = 1,0

# List variables to store results pf training.
img_list = []
G_losses = []
D_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format('ECG_dataset') % (params['num_epochs'], params['batch_size'], len(ecg_dataloader)))
print("-"*25)

start_time = time.time()
iters = 0

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, (data,mean,std) in enumerate(ecg_dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        # Updating discriminator and DHead
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size, ), real_label, device=device)
        output1 = discriminator(real_data)
        probs_real = netD(output1).view(-1)
        loss_real = criterionD(probs_real, label)
        # Calculate gradients.
        loss_real.backward()

        # Fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(params['n_dis_c'], params['dis_c_dim'], params['n_con_c'], params['n_z'], b_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())
        probs_fake = netD(output2).view(-1)
        loss_fake = criterionD(probs_fake, label)
        # Calculate gradients.
        loss_fake.backward()

        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()
        # Updating Generator and QHead
        optimG.zero_grad()

        # Fake data treated as real.
        output = discriminator(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output).view(-1)
        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(params['n_dis_c']):
            dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if (params['n_con_c'] != 0):
            con_loss = criterionQ_con(noise[:, (-1)*params['n_con_c']: ].view(-1, params['n_con_c']), q_mu, q_var)*0.1 #?

        # Net loss for generator.
        gen_loss*=100
        dis_loss*=10
        con_loss*=0.01

        weight = kl_anneal_function('logistic',iters)
        weight = weight if weight>0.5 else 0

        G_loss = gen_loss + weight*dis_loss + weight*con_loss
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        optimG.step()

        # Check progress of training.
        # if i != 0 and i%100 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f [%.1f+%.1f+%.1f]'
              % (epoch+1, params['num_epochs'], i, len(ecg_dataloader),
                D_loss.item(), G_loss.item(),gen_loss.item(),dis_loss.item(),con_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        writer.add_scalar('Loss/Gen',G_loss.item(),iters)
        writer.add_scalar('Loss/Dis',D_loss.item(),iters)

        iters += 1

        # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
        if i %25==0:
            with torch.no_grad():
                gen_data = netG(fixed_noise).detach().cpu()

            for j, gen in enumerate(gen_data):
                gen.squeeze_(0)
                image = gen_plot(gen.numpy(), j)
                writer.add_image('GEN/Image_{}/'.format(j), image)

            for j,real in enumerate(real_data):
                if j>=10: break
                real = real.detach().cpu().squeeze(0)
                image = gen_plot(real.numpy(),j)
                writer.add_image('REAL/Image_{}'.format(j),image)


    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))


    #img_list.append(plt.plot(gen_data[0].numpy()))

    # # Generate image to check performance of generator.
    # if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
    #     with torch.no_grad():
    #         gen_data = netG(fixed_noise).detach().cpu()
    #     plt.figure(figsize=(10, 10))
    #     plt.axis("off")
    #     plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
    #     plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
    #     plt.close('all')


# Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'discriminator' : discriminator.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'params' : params
            }, '../save/model_epoch_%d'.format('ECG_dataset') %(epoch+1))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)
