import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the target function
def f_1(x):
    return np.exp(-2 * x) * np.cos(5 * np.pi * x) + x

# Generate training data
N = 5000
x_train = np.linspace(0, 1, N).reshape(-1, 1)
y_train = f_1(x_train)

x_train_t = torch.from_numpy(x_train).float()
y_train_t = torch.from_numpy(y_train).float()

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
EPOCHS = 2000
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    y_pred = model(x_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss.item():.6f}")

# Evaluate on dense grid
x_test = np.linspace(0, 1, 400).reshape(-1, 1)
y_test = f_1(x_test)
x_test_t = torch.from_numpy(x_test).float()
with torch.no_grad():
    y_pred_test = model(x_test_t).cpu().numpy()

# Plot results
plt.figure(figsize=(7, 4))
plt.plot(x_test, y_test, label='True $f_1(x)$', lw=2)
plt.plot(x_test, y_pred_test, '--', label='NN prediction', lw=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Neural Network Fit to $f_1(x)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nn_fit_f1.svg')
print('Saved plot to nn_fit_f1.svg')
