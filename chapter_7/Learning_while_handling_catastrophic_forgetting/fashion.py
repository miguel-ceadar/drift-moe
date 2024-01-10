from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd 
import models as models
# Set the seed for PyTorch
torch.manual_seed(42)
def auto_remap_target(target, class_list):
    """
    Automatically remap target values to the range [0, num_classes-1].

    Args:
    - target (torch.Tensor): Target tensor with original class indices.
    - class_list (list): List of class indices in your classification task.

    Returns:
    - torch.Tensor: Remapped target tensor.
    """
    remap_dict = {class_val: idx for idx, class_val in enumerate(class_list)}
    remapped_target = torch.tensor([remap_dict[val.item()] for val in target])
    return remapped_target

def inverse_remap_target(remapped_target, class_list):
    """
    Map remapped target values back to their original values.

    Args:
    - remapped_target (torch.Tensor): Remapped target tensor.
    - class_list (list): List of class indices in your classification task.

    Returns:
    - torch.Tensor: Original target tensor.
    """
    original_target = torch.tensor([class_list[idx] for idx in remapped_target])
    return original_target

def train(args, model, device, train_loader, optimizer, epoch, label_types=[0, 1, 2, 3, 4, 9]):
    """
    Train the neural network model using negative log-likelihood loss for multi classification.

    Args:
    - args: Commacnd-line arguments and configurations.
    - model: The neural network model to be trained.
    - device: The device to which data and model should be moved (e.g., "cuda" for GPU or "cpu").
    - train_loader: DataLoader providing training data.
    - optimizer: Optimizer for updating model parameters.
    - epoch: The current epoch number.

    Returns:
    - accuracy: Accuracy of the model on the training set.
    - average_loss: Average binary cross-entropy loss over the training set.
    """
    model.train()
    total_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        remapped_target = auto_remap_target(target, label_types)
        
        loss = F.nll_loss(output, remapped_target)
        
        total_loss += loss.item()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        pred = inverse_remap_target(pred, label_types)
        #print("train:",torch.unique(pred), torch.unique(target))
        #import pdb;pdb.set_trace()

        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    # Calculate and return the average loss and accuracy
    average_loss = total_loss / len(train_loader.dataset)
    accuracy = 100.*correct / len(train_loader.dataset)
    return accuracy, average_loss

def validate(model, device, dataloader, label_types=[0, 1, 2, 3, 4, 9]):
    """
    Evaluate the neural network model on a validation or test set using binary cross-entropy loss for binary classification.

    Args:
    - model: The neural network model to be evaluated.
    - device: The device to which data and model should be moved (e.g., "cuda" for GPU or "cpu").
    - test_loader: DataLoader providing validation or test data.

    Returns:
    - accuracy: Accuracy of the model on the validation or test set.
    - val_loss: Binary cross-entropy loss on the validation or test set.
    - all_predictions: Predicted labels for all instances in the validation or test set.
    - all_labels: True labels for all instances in the validation or test set.
    """
    model.eval()
    val_loss = 0
    correct = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
            for data, target in dataloader:
                
                data, target = data.to(device), target.to(device)
                output = model(data)
        
                remapped_target = auto_remap_target(target, label_types)
                
                val_loss = F.nll_loss(output, remapped_target, reduction='sum').item()
                #val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1)  # get the index of the max log-probability
                        
                pred = inverse_remap_target(pred, label_types)
                #print("test:",torch.unique(pred), torch.unique(target))
                correct += pred.eq(target.view_as(pred)).sum().item()
                all_predictions.append(pred.cpu().numpy())
                all_labels.append(target.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions).squeeze(0)
    all_labels = np.array(all_labels).squeeze(0)
    
    val_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print('\Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
    
    return accuracy, val_loss, all_predictions, all_labels
    
def train_and_validate(args, model, device, train_loader, val_loader, optimizer, scheduler, title = "", task1_classes=[0, 1, 2, 3, 4, 9]):
    """
    Train and validate a neural network model across multiple epochs.

    Args:
    - args: Command-line arguments and configurations.
    - model: The neural network model to be trained and validated.
    - device: The device to which data and model should be moved (e.g., "cuda" for GPU or "cpu").
    - train_loader: DataLoader providing training data.
    - val_loader: DataLoader providing validation or test data.
    - optimizer: Optimizer for updating model parameters.
    - scheduler: Learning rate scheduler.
    - title: Title for the plot (optional).

    Returns:
    - all_predictions: Predicted labels for all instances in the validation or test set.
    - all_labels: True labels for all instances in the validation or test set.
    """
    train_losses = [] 
    train_accuracies = [] 
    val_accuracies = [] 
    val_losses = [] 
    

    for epoch in range(1, args.epochs + 1):
        train_accuracy, train_loss = train(args, model, device, train_loader, optimizer, epoch,task1_classes)
        val_accuracy, val_loss, all_predictions, all_labels = validate(model, device, val_loader,task1_classes)
        scheduler.step()
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(f'Epoch {epoch}/{args.epochs}: Training Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
    # Plot loss and accuracy after all epochs
    plot_loss_and_accuracy(train_accuracies, train_losses, val_accuracies, val_losses, title)

    return all_predictions, all_labels

def save_model_state_dict(model_state_dict, filename, overwrite=True):
    """
    Save the state dictionary of a PyTorch model to a file.

    Args:
    - model_state_dict: The state dictionary of the PyTorch model.
    - filename: The name of the file to which the state dictionary will be saved.
    - overwrite: Boolean flag indicating whether to overwrite an existing file (default is True).

    Returns:
    - None
    """
    models_folder = "models"
    
    # Check if the models folder exists, if not, create it
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    file_path = os.path.join(models_folder, filename)

    # Check if the file already exists
    if os.path.exists(file_path) and not overwrite:
        user_input = input(f"The file '{filename}' already exists. Do you want to overwrite it? (yes/no): ")
        if user_input.lower() != 'yes':
            print("Model not saved.")
            return

    # Save the model state dict
    torch.save(model_state_dict, file_path)
    print(f"Model state dict saved to: {file_path}")
def plot_loss_and_accuracy(train_accuracies, train_losses, val_accuracies, val_losses, title = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Training Loss and Accuracy
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(train_accuracies, label='Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_title(f'{title} - training data')
    ax1.legend()

    # Plot Testing Loss and Accuracy
    ax2.plot(val_losses, label='Validation Loss')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_title(f'{title} - validation data')
    ax2.legend()
    
def get_data_loaders(dataset, train_size, val_size, train_kwargs, val_kwargs):
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, **train_kwargs)
    val_loader = DataLoader(val_set, **val_kwargs)
    return train_loader, val_loader
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                            help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                            help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                            help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
    parser.add_argument('--val_reduced', action='store_true', default=True,
                            help='For Saving the current Model')
    parser.add_argument('--device', action='store_true', default=torch.device("cpu"),
                            help='.')
    
    args = parser.parse_args()
    
    device = args.device
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.val_batch_size}

    if 'cuda' in str(device):
            cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            val_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    #import pdb;pdb.set_trace()
    if args.val_reduced:
            sampled_indices = torch.randperm(len(mnist_dataset))[:1000]
            mnist_dataset = Subset(mnist_dataset, sampled_indices)
    # Define task 1 labeling
    task1_classes = [0, 1, 2, 3, 4, 9]
    # Check the unique labels in your dataset to make sure they are integers
    unique_labels = set(label for _, label in mnist_dataset)
    print("Unique Labels in Dataset:", unique_labels)
    # Make sure labels are integers; if not, convert them to integers
    mnist_dataset = [(image, int(label)) for image, label in mnist_dataset]
    task1_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label in task1_classes]
    task1_dataset = Subset(mnist_dataset, task1_indices)
    print("Selected labels for Task 1:", set(label for _, label in task1_dataset))
    train_size_task1 = int(0.8 * len(task1_indices))
    val_size_task1 = len(task1_indices) - train_size_task1
    train_loader_task1, val_loader_task1 = get_data_loaders(task1_dataset, train_size_task1, val_size_task1,
                                                                train_kwargs, val_kwargs)
    model = models.Net_log_softmax(classes = len(task1_classes)).to(device)
    #model = models.Net_tunable(classes = len(task1_classes)).to(device)
    #model = models.AlexNet(classes = len(task1_classes)).to(device)
    # model = models.BaseModel(28 * 28, 100, 6).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    all_predictions, all_labels = train_and_validate(args, model, device, train_loader_task1, val_loader_task1, optimizer, scheduler,title="Task 1: ", task1_classes=task1_classes)
    model_state_dict_task1 = model.state_dict()
    save_model_state_dict(model_state_dict_task1, "mnist_cnn_task1.pt")
    print(f"results task 1 ".center(60,"-"))
    val_accuracy, val_loss, all_predictions, all_labels  = validate(model, device, val_loader_task1)
    # Define task 2 labeling
    task2_classes = [5, 6, 7, 8, 9, 1]
    task2_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label in task2_classes]
    task2_dataset = Subset(mnist_dataset, task2_indices)
    print("Selected labels for Task 2:", set(label for _, label in task2_dataset))
    train_size_task2 = int(0.8 * len(task2_indices))
    val_size_task2 = len(task2_indices) - train_size_task2

    model_task2 = models.Net_log_softmax(classes = len(task2_classes)).to(device)
    # model_task2 = models.BaseModel(28 * 28, 100, 6).to(device)
    # model_task2 = models.AlexNet(classes = 6).to(device)
    model_task2.load_state_dict(model_state_dict_task1.copy()) 
    optimizer_task2 = optim.Adadelta(model_task2.parameters(), lr=args.lr)
    # optimizer_task2 = optim.Adadelta(model.parameters(), lr=1)
    # scheduler_task2 = StepLR(optimizer_task2, step_size=1, gamma=0.7)
    scheduler_task2 = StepLR(optimizer_task2, step_size=1, gamma=args.gamma)

    print(f"MODEL TRAINING TASK #2".center(60, "-"))
    #import pdb;pdb.set_trace()
    train_loader_task2, val_loader_task2 = get_data_loaders(task2_dataset, train_size_task2, val_size_task2,
                                                                train_kwargs, val_kwargs)

    train_and_validate(args, model_task2, device, train_loader_task2, val_loader_task2, optimizer_task2, scheduler_task2, title="Task 2: ", task1_classes = task2_classes)

    model_state_dict_task2 = model_task2.state_dict()
    save_model_state_dict(model_state_dict_task2, "mnist_cnn_task2.pt")
if __name__ == '__main__':
    main()