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

# Train an AE on CelebaHQ images resized to 32x32

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


class AutoEncoder(Module):

    def __init__(self, bottleneck_size, channel_size=1):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channel_size = channel_size
        self.conv1 = nn.Conv2d(self.channel_size, 8, 3, padding="same")
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(8, 32, 3, padding="same")
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding="same")
        self.fc_encoder = nn.Linear(512, self.bottleneck_size)
        self.fc_decoder = nn.Linear(self.bottleneck_size, 512)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 8, 3, padding=1, stride=2)
        self.deconv4 = nn.ConvTranspose2d(
            8, self.channel_size, 2, padding=1, stride=2)

    def encoder(self, image):
        x = F.relu(self.pool1(self.conv1(image)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        code = self.fc_encoder(x)
        return code

    def decoder(self, code):
        x = F.relu(self.fc_decoder(code))
        x = x.reshape(-1, 128, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        decoded_image = torch.sigmoid(self.deconv4(x))
        return decoded_image

    def forward(self, image):
        code = self.encoder(image)
        decoded_image = self.decoder(code)
        return decoded_image


def reconstruction_loss(prediction, target):
    loss_fn = nn.MSELoss()
    recon_loss = loss_fn(prediction, target)
    return recon_loss


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
    MAX_ITERATIONS = 100
    LEARNING_RATE = 1e-3
    celeba_hq_autoencoder = AutoEncoder(
        bottleneck_size=256, channel_size=3).to(device)
    celeba_hq_optimizer = optim.Adam(
        celeba_hq_autoencoder.parameters(), lr=LEARNING_RATE)

    for i in range(MAX_ITERATIONS):

        total_train_loss = 0
        celeba_hq_autoencoder.train()

        for input_images in torch_train_celeba_hq:
            output_images = celeba_hq_autoencoder(input_images)
            loss = reconstruction_loss(input_images, output_images)
            celeba_hq_optimizer.zero_grad()
            loss.backward()
            celeba_hq_optimizer.step()
            total_train_loss += loss.detach()
        liveloss.update(
            {'train loss': total_train_loss.cpu().detach().numpy()})
        liveloss.send()

        if i % 10 == 0:
            celeba_hq_autoencoder.eval()
            total_val_loss = 0
            with torch.no_grad():
                for input_images in torch_validation_celeba_hq:
                    output_images = celeba_hq_autoencoder(input_images)
                    loss = reconstruction_loss(input_images, output_images)
                    total_val_loss += loss.detach()
            liveloss.update(
                {'val loss': total_val_loss.cpu().detach().numpy()})
            liveloss.send()


if __name__ == '__main__':
    train()
