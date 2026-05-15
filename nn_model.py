import torch
import torch.nn as nn
import torch.optim as optim

# sample numeric data (for learning concept first)
X = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.0, 0.0]
    
])

y = torch.tensor([
    [1.0],
    [1.0],
    [0.0],
    [0.0]
    
])

# define model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# loss + optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, y)
    if epoch % 100 == 0:
        print(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# test
print(model(X))
print(loss)
