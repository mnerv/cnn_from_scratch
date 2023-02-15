import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from model import Model

def evaluate(model, data_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to GPU if available
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 9.5e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # seed = 1337
    # torch.manual_seed(seed)

    # Download and load the MNIST dataset for numbers
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download dataset for training and test
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Convert to loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = Model().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model on MNIST dataset
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            # Move data to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update learning rate
        # scheduler.step()

        # Print training progress
        print('Epoch [{}/{}], Loss: {:.3E}'.format(epoch+1, num_epochs, loss.item()))

    # Evaluate the model on test set
    eval_result = evaluate(model, test_loader, device)
    print('Test Accuracy: {:.2f}%'.format(eval_result))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.pth')
