import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,noise_n):
        super().__init__()

        # start // torch.nn.ConvTranspose1d(10, 10, kernel_size=4, stride=1, padding=0, output_padding=0)
        # x2 // torch.nn.ConvTranspose1d(10, 10, kernel_size=4, stride=2, padding=1, output_padding=0)
        # x4 //  torch.nn.ConvTranspose1d(10, 10, kernel_size=8, stride=4, padding=2, output_padding=0)

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose1d(noise_n,noise_n*2,kernel_size=4,stride=1,bias=False), # 1->4
            nn.BatchNorm1d(noise_n*2)
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose1d(noise_n*2,512,kernel_size=8,stride=4,padding=2,output_padding=0,bias=False), #4->16
            nn.BatchNorm1d(512)
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose1d(512,256,kernel_size=8,stride=4,padding=2,output_padding=0,bias=False), #16->64
            nn.BatchNorm1d(256)
        )
        self.tconv4 = nn.ConvTranspose1d(256,128,kernel_size=8,stride=4,padding=2,output_padding=0) #64->256
        self.tconv5 = nn.ConvTranspose1d(128,64,kernel_size=8,stride=4,padding=2,output_padding=0)#256->1024
        self.tconv6 = nn.ConvTranspose1d(64,1,kernel_size=4,stride=2,padding=1,output_padding=0) #1024->2048
        #self.tconv_single = nn.ConvTranspose1d(32,1,1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.tconv1(x))
        x = F.leaky_relu(self.tconv2(x))
        x = F.leaky_relu(self.tconv3(x))
        x = self.tconv4(x)
        x = self.tconv5(x)
        x = self.tconv6(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=7,stride=4,padding=3), #2048->512
            nn.LeakyReLU(0.1),

            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3), #512->128
            nn.LeakyReLU(0.1),

            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3,bias=False), #128->32
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3,bias=False), #32->16
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3,bias=False), #16->8
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Conv1d(256, 512, kernel_size=7, stride=2, padding=3, bias=False),  #8->4
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),

            nn.Conv1d(512, 1024, kernel_size=7, stride=2, padding=3, bias=False),  #4->1
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_disc = nn.Conv1d(128, 10, 1)  # 10 is n_dis_c
        self.conv_mu = nn.Conv1d(128, 2, 1)  # 2 is n_con_c
        self.conv_var = nn.Conv1d(128, 2, 1)  # 2 is n_con_c

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var























