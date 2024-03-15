import os
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms, models
from torchvision.transforms import functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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

        # Step 3: Rescale the subtracted result to the 0-255 range
        subtracted_image = subtracted_image - subtracted_image.min()
        subtracted_image = subtracted_image / subtracted_image.max()

        # Step 4: Make central reflection
        flipped_image = F.hflip(x)
        flipped_image = F.rotate(flipped_image, 180)

        # Step 5: Rescale the subtracted result to the 0-255 range
        flipped_image = torch.sub(x, flipped_image)
        flipped_image = flipped_image - flipped_image.min()
        flipped_image = flipped_image / flipped_image.max()

        # Ensure the subtracted_image has the same dimensions as the input, for grayscale images
        if subtracted_image.ndim == 2:
            subtracted_image = subtracted_image.unsqueeze(0)

        # Step 4: Append the rescaled result as a new channel to the input image
        result_image = torch.cat((x, subtracted_image, flipped_image), dim=0)  # Assuming x is [C, H, W]
    
        mean = [0.0617, 0.5000, 0.5000]
        std = [0.1168, 0.0678, 0.0655]
        result_image = F.normalize(result_image, mean=mean, std=std)
        
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

def load_dataset():
    # Usage
    drive_dir = "./"
    train_dir = f'{drive_dir}/dataset/train'  # Change this path to your dataset's root directory

    dataset_train = LensDataset(train_dir, transform=CustomTransform())
    #dataset_train = LensDataset(train_dir, transform=transform)
    sampler_train = RandomSampler(dataset_train)

    dataloader_train = DataLoader(dataset_train, batch_size=100, sampler=sampler_train)

    val_dir = f'{drive_dir}/dataset/val'  # Change this path to your dataset's root directory
    dataset_val = LensDataset(val_dir, transform=CustomTransform())
    #dataset_val = LensDataset(val_dir, transform=transform)
    sampler_val = RandomSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=32, sampler=sampler_val)

    # Class names for CIFAR-10 dataset
    classes = ("no", "sphere", "vort")

    return dataset_train, dataloader_train, dataset_val, dataloader_val, classes

def train(model, trainloader, criterion, optimizer, device):
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
        outputs = model(inputs)
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

def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
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

def train_epochs(model, trainloader, testloader, criterion, optimizer, device, num_epochs, save_interval=2):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')

        if (epoch + 1) % save_interval == 0:
          # Save the model and variables
          torch.save(model.state_dict(), f'resnet50_lense_{epoch+1}.pth')
          checkpoint = {
              'epoch': epoch + 1,
              'train_losses': train_losses,
              'train_accuracies': train_accuracies,
              'test_losses': test_losses,
              'test_accuracies': test_accuracies,
              'classes': classes
          }
          torch.save(checkpoint, f'resnet50_lense_variables_{epoch+1}.pth')
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

if __name__ == "__main__":
    train_model = True
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Number of classes
    num_classes = 3

    # Import ResNet50 model pretrained on ImageNet
    model = models.resnet50(pretrained=True)
    #print("Network before modifying conv1:")
    #print(model)

    #Modify conv1 to suit CIFAR-10
    #model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Modify the final fully connected layer according to the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    #print("Network after modifying conv1:")
    #print(model)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    # Load the dataset
    trainset, trainloader, testset, testloader, classes = load_dataset()

    if train_model:
      # Train the model for 20 epochs, saving every 5 epochs
      num_epochs = 20
      save_interval =2
      model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(
          model, trainloader, testloader, criterion, optimizer, device,
          num_epochs, save_interval)

      # Save the final trained model
      torch.save(model.state_dict(), f'resnet50_lense_final_model_epochs_{num_epochs}.pth')

      # Plot and save the loss and accuracy plots
      plot_loss(train_losses, test_losses)
      plot_accuracy(train_accuracies, test_accuracies)
    else:
      # Load the pre-trained model
      model.load_state_dict(torch.load('resnet50_lense_final_model_epochs_20.pth'))
      # Load the variables
      checkpoint = torch.load("resnet50_cifar10_variables.pth")
      epoch = checkpoint['epoch']
      train_losses = checkpoint['train_losses']
      train_accuracies = checkpoint['train_accuracies']
      test_losses = checkpoint['test_losses']
      test_accuracies = checkpoint['test_accuracies']
      classes = checkpoint['classes']
      model.to(device)
      model.eval()

    # Plot and save an example image
    #plot_image(testset, model, classes)

