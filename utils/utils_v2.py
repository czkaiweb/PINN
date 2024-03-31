import os
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from torchvision import transforms, models
from torchvision.transforms import functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from  torch.nn.functional import softmax
import matplotlib.pyplot as plt

class CustomTransform:
    def __call__(self, x):
        # x is a PyTorch tensor of the image

        # Ensure image is in the expected format [C, H, W]
        if x.ndim == 2:  # If the image is grayscale, unsqueeze to add a channel dimension
            x = x.unsqueeze(0)

        x = F.resize(x, [224, 224])
        
        # Step 1: Rotate the image by 180 degrees
        rotated_image = F.rotate(x, 180)

        # Step 2: Subtract the rotated image from the original
        subtracted_image = torch.sub(x, rotated_image)

        # Step 3: Rescale the subtracted result to the 0-1 range
        subtracted_image = subtracted_image - subtracted_image.min()
        subtracted_image = subtracted_image / subtracted_image.max()

        # Step 4: Make central reflection
        R90_image = F.rotate(x,90)
        R270_image = F.rotate(x, 270)

        # Step 5: Rescale the subtracted result to the 0-1 range
        quad_image = torch.add(x, rotated_image)
        quad_image = torch.sub(quad_image, R90_image)
        quad_image = torch.sub(quad_image, R270_image)
        quad_image = quad_image - quad_image.min()
        quad_image = quad_image / quad_image.max()

        # Ensure the subtracted_image has the same dimensions as the input, for grayscale images
        if subtracted_image.ndim == 2:
            subtracted_image = subtracted_image.unsqueeze(0)

        # Step 4: Append the rescaled result as a new channel to the input image
        result_image = torch.cat((x, subtracted_image, quad_image), dim=0)  # Assuming x is [C, H, W]
    
        mean = [0.0603, 0.4412, 0.4412]
        std = [0.1175, 0.1747, 0.1813]
        result_image = F.normalize(result_image, mean=mean, std=std)

        angle = random.randint(-180, 180)
        result_image = F.rotate(result_image, angle)
        
        return result_image


class LensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'A/B/C').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filepaths = []
        self.labels = []

        label_numeric = {
            "no": 0,
            "sphere": 1,
            "vort": 2
        }

        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            for filename in os.listdir(label_path):
                self.filepaths.append(os.path.join(label_path, filename))
                self.labels.append(label_numeric[label_dir])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.filepaths[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image).float()
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset(batch_size = 32):
    # Usage
    drive_dir = "./"
    train_dir = f'{drive_dir}/dataset/train'  # Change this path to your dataset's root directory

    dataset_fortrain = LensDataset(train_dir, transform=CustomTransform())
    total_fortrain = len(dataset_fortrain)


    train_len = int(total_fortrain * 0.9)
    val_len = total_fortrain - train_len

    dataset_train, dataset_val = random_split(dataset_fortrain, [train_len, val_len])

    
    #dataset_train = LensDataset(train_dir, transform=transform)
    sampler_train = RandomSampler(dataset_train)
    sampler_val = RandomSampler(dataset_val)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val)

    
    test_dir = f'{drive_dir}/dataset/val'  # Change this path to your dataset's root directory
    dataset_test = LensDataset(test_dir, transform=CustomTransform())
    #dataset_val = LensDataset(val_dir, transform=transform)
    sampler_test = RandomSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, sampler=sampler_test)

    # Class names for CIFAR-10 dataset
    classes = ("no", "sphere", "vort")

    return dataset_train, dataloader_train, dataset_val, dataloader_val, dataset_test, dataloader_test, classes

def train(model, trainloader, criterion, optimizer, device, isLogits=True):
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    # Switch to train mode
    model.train()
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        if isLogits:
            outputs = model(inputs)
        else:
            outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy

def test(model, testloader, criterion, device, isLogits=True):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            if isLogits:
                outputs = model(inputs)
            else:
                outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy

def infer(model, testloader, criterion, device, isLogits=True):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    test_prop = []
    test_label = []

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            if isLogits:
                outputs = model(inputs)
            else:
                outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            probabilities = softmax(outputs, dim=1)
            test_prop.append(probabilities)
            test_label.append(labels)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total
    return test_prop, test_label, test_total, test_correct

def train_epochs(model, trainloader, testloader, criterion, optimizer, scheduler, classes, device, num_epochs, save_interval=2, model_prefix = "resnet50", isLogits=True):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device, isLogits)
        test_loss, test_accuracy = test(model, testloader, criterion, device, isLogits)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')

        if (epoch + 1) % save_interval == 0:
          # Save the model and variables
          torch.save(model.state_dict(), f'{model_prefix}_lense_{epoch+1}.pth')
          checkpoint = {
              'epoch': epoch + 1,
              'train_losses': train_losses,
              'train_accuracies': train_accuracies,
              'test_losses': test_losses,
              'test_accuracies': test_accuracies,
              'classes': classes
          }
          torch.save(checkpoint, f'{model_prefix}_lense_variables_{epoch+1}.pth')
        scheduler.step(test_loss)

    return model, train_losses, train_accuracies, test_losses, test_accuracies

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    #plt.show()

def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    #plt.show()

def plot_image(dataset, model, classes):
    idx = random.randint(0, len(dataset))
    label = dataset[idx][1]
    img = dataset[idx][0].unsqueeze(0).to(device)  # Move the input image tensor to the GPU
    model.eval()
    #model.to(device)  # Move the model to the GPU
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    # Convert the image and show it
    img = img.squeeze().permute(1, 2, 0).cpu()  # Move the image tensor back to the CPU and adjust dimensions
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {classes[predicted]}, True: {classes[label]}')
    plt.savefig('predicted_image.png')
    #plt.show()
    print("Predicted label: ", classes[predicted[0].item()])
    print("Actual label: ", classes[label])
    
def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model for which to count the parameters.

    Returns:
        int: The total number of trainable parameters in the model.

    """
    # Count the number of trainable parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Finding the value in M
    num_parameters = num_parameters/1e6

    print(f"The model has {num_parameters:.2f}M trainable parameters.")

    return num_parameters