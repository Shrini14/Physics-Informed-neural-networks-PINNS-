# Physics-Informed Neural Network (PINN) for 1D Heat Equation

## 📌 Overview

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **1D Heat Equation** without using traditional numerical methods like FDM or FEM.

Instead of solving on a grid, the neural network learns the solution by enforcing:

- The governing PDE (physics)
- Boundary conditions
- Initial condition

---

## 🧠 Problem Statement

We solve the heat equation:

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}
$$

Domain:
- \( x \in [0,1] \)
- \( t \in [0,1] \)

### Initial Condition:
$$
u(x,0) = \sin(\pi x)
$$

### Boundary Conditions:
$$
u(0,t) = 0, \quad u(1,t) = 0
$$

---

## ⚙️ Methodology

### 1. Neural Network

A fully connected neural network is used:

- Input: (x, t)
- Output: u(x, t)
- Activation: Tanh

---

### 2. Training Points

Three types of points are used:

| Type | Purpose |
|------|--------|
| Collocation Points | Enforce PDE |
| Boundary Points | Enforce BC |
| Initial Points | Enforce IC |

---

### 3. Physics Loss (PDE Residual)

The PDE is enforced using:

$$
f = \frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2}
$$

---

### 4. Loss Function

Total loss:

$$
L = L_{PDE} + L_{BC} + L_{IC}
$$

Where:

- \(L_{PDE}\): Physics loss (residual)
- \(L_{BC}\): Boundary loss
- \(L_{IC}\): Initial condition loss

---

### 5. Training

- Optimizer: Adam
- Backpropagation is used to minimize total loss
- Automatic differentiation computes derivatives

---

## 📊 Results

The model is evaluated by comparing:

- Predicted solution (PINN)
- Exact analytical solution

### Metrics:
- Mean Squared Error (MSE)

---

## 📈 Visualizations

The results include:

1. Exact Solution
2. PINN Prediction
3. Absolute Error

These are visualized using heatmaps over the space-time domain.

---

## ✅ Key Observations

- The PINN successfully learns the physics of the system
- The predicted solution closely matches the exact solution
- Small errors remain due to:
  - Limited training
  - Neural network approximation

---

## ⚠️ Limitations

- Training can be slow
- Sensitive to hyperparameters
- May struggle with complex PDEs

---


## 🛠️ Tech Stack

- Python
- PyTorch
- Matplotlib

---

## 📌 Key Concept

> PINNs learn solutions by minimizing **physics violations**, not by fitting data.

---

## 👤 Author

R. Shrinivass
