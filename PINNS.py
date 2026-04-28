import torch
import torch.nn as nn
from Pinns_model import PINN
import matplotlib.pyplot as plt



# Collocation points

N_f = 1000
x_f = torch.rand(N_f,1,requires_grad=True)
t_f = torch.rand(N_f,1,requires_grad=True)

# Boundary points( at x=0,x=1)

N_b = 200
t_b = torch.rand(N_b,1)
x_b0 = torch.zeros(N_b,1)
x_b1 = torch.ones(N_b,1)

u_b0 = torch.zeros(N_b,1)
u_b1 = torch.zeros(N_b,1)

# Initial Points(at t=0)

N_i = 200
x_i = torch.rand(N_i,1)
t_i = torch.zeros(N_i,1)
u_i = torch.sin(torch.pi*x_i)


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)



train_losses = []


for epoch in range(1000):

    optimizer.zero_grad()   # clear old gradients

    # Forward pass 
    u_f = model(x_f, t_f)
    u_b0_pred = model(x_b0, t_b)
    u_b1_pred = model(x_b1, t_b)
    u_i_pred = model(x_i, t_i)

    # Derivatives
    u_f_t = torch.autograd.grad(u_f, t_f,
                               grad_outputs=torch.ones_like(u_f),
                               create_graph=True)[0]

    u_f_x = torch.autograd.grad(u_f, x_f,
                               grad_outputs=torch.ones_like(u_f),
                               create_graph=True)[0]

    u_f_xx = torch.autograd.grad(u_f_x, x_f,
                                grad_outputs=torch.ones_like(u_f_x),
                                create_graph=True)[0]

    # Residual
    f = u_f_t - u_f_xx

    # Loss
    loss_pde = torch.mean(f**2)
    loss_bc = torch.mean((u_b0_pred - u_b0)**2) + torch.mean((u_b1_pred - u_b1)**2)
    loss_ic = torch.mean((u_i_pred - u_i)**2)

    loss = loss_pde + loss_bc + loss_ic

    train_losses.append(loss.item()) 
    # Backprop
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Training Loss: {train_losses[epoch]}")


# Define Exact solution

def exact_solution(x, t):
    return torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)

# test points

# Create grid
x = torch.linspace(0, 1, 100)
t = torch.linspace(0, 1, 100)

X, T = torch.meshgrid(x, t, indexing='ij')

# Flatten for model input
x_test = X.reshape(-1, 1)
t_test = T.reshape(-1, 1)

# predictions

with torch.no_grad():
    u_pred = model(x_test, t_test)

# Get exact solution
u_exact = exact_solution(x_test, t_test)

# Error for test data
error = torch.abs(u_pred - u_exact)
mse = torch.mean((u_pred - u_exact)**2)

print(f"\nTest MSE Error: {mse.item()}")

# Reshape for plotting
U_pred = u_pred.reshape(100, 100).numpy()
U_exact = u_exact.reshape(100, 100).numpy()
Error = error.reshape(100, 100).numpy()

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Exact Solution 
im1 = axes[0].imshow(U_exact.T, extent=[0,1,0,1], origin='lower', aspect='auto')
axes[0].set_title("Exact Solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(im1, ax=axes[0])

# Predicted Solution 
im2 = axes[1].imshow(U_pred.T, extent=[0,1,0,1], origin='lower', aspect='auto')
axes[1].set_title("PINN Prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(im2, ax=axes[1])

# Error 
im3 = axes[2].imshow(Error.T, extent=[0,1,0,1], origin='lower', aspect='auto')
axes[2].set_title("Absolute Error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("pinn_results.png", dpi=300)
plt.show()
