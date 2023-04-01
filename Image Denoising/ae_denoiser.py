import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torchvision import datasets, transforms
from PIL import Image
from livelossplot import PlotLosses
from pathlib import Path

# Train an AE to denoise images
# Current architecture is to denoise 128x128 monochrome images,
# can be easily tweaked to denoise RGB images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LoadDenoisingDataset(Dataset):
    def __init__(self, input_imgs_path, target_imgs_path):
        super().__init__()
        self.input_imgs_path = list(input_imgs_path.glob('*.*'))
        self.target_imgs_path = list(target_imgs_path.glob('*.*'))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        length = len(self.target_imgs_path)
        return length

    def __getitem__(self, idx):
        crop_size = (0, 0, 128, 128)
        input_image = Image.open(self.input_imgs_path[idx]).convert('L').crop(
            crop_size)
        input_image = self.transform(input_image).to(device)
        target_image = Image.open(self.target_imgs_path[idx]).convert(
            'L').crop(crop_size)
        target_image = self.transform(target_image).to(device)
        return (input_image, target_image)


class Encoder(Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, high_res_image):
        x = self.relu(self.conv1(high_res_image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        low_res_image = self.relu(self.conv4(x))
        return low_res_image


class Decoder(Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, low_res_image):
        x = self.relu(self.deconv1(low_res_image))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        denoised_image = self.relu(self.deconv4(x))

        return denoised_image


def loss_function(prediction, target):
    loss_fn = nn.MSELoss()
    loss = loss_fn(prediction, target)
    return loss


def train():
    input_imgs_path = Path("noisy_input_images_folder")
    target_imgs_path = Path("target_images_folder")
    dataset = LoadDenoisingDataset(input_imgs_path, target_imgs_path)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    MAX_ITERATIONS = 100
    LEARNING_RATE = 1e-3
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    plotlosses = PlotLosses()
    encoder.train()
    decoder.train()
    BATCH_SIZE = 32
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    for i in range(MAX_ITERATIONS):
        total_loss = 0
        for input_images, target_images in data_loader:
            output_images = decoder(encoder(input_images))
            loss = loss_function(target_images, output_images)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.detach()
        plotlosses.update({'loss': total_loss.cpu().detach().numpy()})
        plotlosses.send()


if __name__ == '__main__':
    train()
