import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torchvision import datasets, transforms
from PIL import Image
from livelossplot import PlotLosses
from pathlib import Path

# Train a VAE on CelebaHQ images resized to 32x32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CelebAHQ(Dataset):
    def __init__(self, dataset_path):
        self.imgs_path = list(Path(dataset_path).glob('*.*'))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        length = len(self.imgs_path)
        return length

    def __getitem__(self, idx):
        image = Image.open(self.imgs_path[idx]).convert('RGB')
        image = self.transform(image).to(device)
        return image


class VAE(Module):

    def __init__(self, bottleneck_size, channel_size=1):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channel_size = channel_size
        self.conv1 = nn.Conv2d(self.channel_size, 32, kernel_size=3, padding=1,
                               stride=2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.fc_mu = nn.Linear(2048, self.bottleneck_size)
        self.fc_std = nn.Linear(2048, self.bottleneck_size)
        self.fc_decoder = nn.Linear(self.bottleneck_size, 2048)
        self.deconv1 = nn.ConvTranspose2d(
            512, 256, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm8 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm9 = nn.BatchNorm2d(32)
        self.convfinal = nn.Conv2d(32, out_channels=3, kernel_size=3,
                                   padding=1)

    def encoder(self, image):
        x = F.leaky_relu(self.batchnorm1(self.conv1(image)))
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = F.leaky_relu(self.batchnorm4(self.conv4(x)))
        x = F.leaky_relu(self.batchnorm5(self.conv5(x)))
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        std = torch.exp(self.fc_std(x))
        return mu, std

    def reparametrization(self, mu, std):
        N = torch.distributions.Normal(0, 1)
        z = mu + std * (N.sample(mu.shape)).to(device)

        return z

    def decoder(self, code):
        x = self.fc_decoder(code)
        x = x.view(-1, 512, 2, 2)
        x = F.leaky_relu(self.batchnorm6(self.deconv1(x)))
        x = F.leaky_relu(self.batchnorm7(self.deconv2(x)))
        x = F.leaky_relu(self.batchnorm8(self.deconv3(x)))
        x = F.leaky_relu(self.batchnorm9(self.deconv4(x)))
        decoded_image = F.sigmoid(self.convfinal(x))
        return decoded_image

    def forward(self, image):
        mu, std = self.encoder(image)
        z = self.reparametrization(mu, std)
        decoded_image = self.decoder(z)
        return decoded_image, mu, std


def reconstruction_loss(prediction, target):
    recon_loss = ((target - prediction) ** 2).mean()
    return recon_loss


def kl_divergence_loss(mu, std):
    kl_loss = (std ** 2 + mu ** 2 - torch.log(std) - 1 / 2).mean()
    return kl_loss


def train():
    celeba_hq_train_data_path = "celeba_hq_32/train"
    celeba_hq_train_dataset = CelebAHQ(celeba_hq_train_data_path)
    celeba_hq_validation_data_path = "celeba_hq_32/val"
    celeba_hq_validation_dataset = CelebAHQ(celeba_hq_validation_data_path)
    torch_train_celeba_hq = DataLoader(
        celeba_hq_train_dataset, shuffle=True, batch_size=512, num_workers=4)
    torch_validation_celeba_hq = DataLoader(
        celeba_hq_validation_dataset, shuffle=True, batch_size=512,
        num_workers=4)
    liveloss = PlotLosses()
    MAX_ITERATIONS = 50
    LEARNING_RATE = 1e-3
    celeba_hq_vae = VAE(bottleneck_size=256, channel_size=3).to(device)
    celeba_hq_vae_optimizer = optim.AdamW(
        celeba_hq_vae.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    alpha = 0.1

    for i in range(MAX_ITERATIONS):
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_val_loss = 0
        celeba_hq_vae.train()

        for input_images in torch_train_celeba_hq:
            decoded_images, mu, std = celeba_hq_vae(input_images)
            recon_loss = reconstruction_loss(input_images, decoded_images)
            kl_loss = kl_divergence_loss(mu, std)
            loss = recon_loss + alpha * kl_loss
            celeba_hq_vae_optimizer.zero_grad()
            loss.backward()
            celeba_hq_vae_optimizer.step()
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        celeba_hq_vae.eval()
        with torch.no_grad():

            with torch.no_grad():
                for input_images in torch_validation_celeba_hq:
                    decoded_images, mu, std = celeba_hq_vae(input_images)
                    loss = reconstruction_loss(input_images, decoded_images)
                    total_val_loss += loss.item()

        liveloss.update(
            {
                'total train loss': total_train_loss,
                'train kl loss': total_kl_loss,
                'train recon loss': total_recon_loss,
                'val loss': total_val_loss
            }
        )
        liveloss.send()


if __name__ == '__main__':
    train()
