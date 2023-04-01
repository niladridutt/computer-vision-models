import torch
import torch.nn as nn
from torch.autograd import Variable

# Iterative FGSM attack on an autoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_function(prediction, target):
    loss_fn = nn.MSELoss()
    loss = loss_fn(prediction, target)
    return loss


def fgsm(images, targets, encoder, decoder):
    adversarial_image = torch.stack(images, dim=0).to(device)[-1:]
    adversarial_target = torch.stack(targets, dim=0).to(device)[-1:]
    # Iterative FGSM attack
    adversarial = Variable(adversarial_image, requires_grad=True)
    encoder.train()
    decoder.train()
    epsilon = 0.75
    MAX_ITERATIONS = 50
    alpha = 0.002
    for i in range(MAX_ITERATIONS):
        if adversarial.grad is not None:
            adversarial.grad.zero_()
        output_image = decoder(encoder(adversarial))
        loss = loss_function(adversarial_target, output_image)
        loss.backward()
        x_grad = alpha * torch.sign(adversarial.grad.data)
        adv_temp = adversarial.data + x_grad
        total_grad = adv_temp - adversarial_image  # total perturbation
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)
        x_adv = adversarial_image + total_grad  # add total perturbation to the original image
        adversarial.data = x_adv
    adversarial_image = torch.clamp(adversarial.data, 0, 1).detach()
    return adversarial_image


if __name__ == '__main__':
    # Load your own input images, targets, encoder, & decoder
    adversarial_image = fgsm(images, targets, encoder, decoder)
