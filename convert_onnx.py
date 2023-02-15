import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
from model import Model
from train import evaluate
import torch.onnx


if __name__ == '__main__':
    batch_size = 32
    # Download and load the MNIST dataset for numbers
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2);

    model = Model()
    model.load_state_dict(torch.load('model.pth'))

    eval_result = evaluate(model, test_loader, 'cpu')
    print('Test Accuracy: {:.2f}%'.format(eval_result))

    # Convert 
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)

    torch.onnx.export(model,
                     dummy_input,
                     'model.onnx',
                     export_params=True,
                     opset_version=10,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={
                         'input': { 0 : 'batch_size' },
                         'output': { 0 : 'batch_size' }
                     }
                    )
