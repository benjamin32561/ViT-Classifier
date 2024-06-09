import torch

DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('vit_model.pt')

# Generate a random input tensor with the shape (N, 3, 224, 224)
# N is batch size
x = torch.rand(4, 3, 224, 224)
y = model(x)
print(y)
# Predicted classes: bicycle, bus, car, jeep, minibus, minivan, motorcycle, taxi, tender, truck