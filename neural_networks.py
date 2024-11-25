import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull


result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: self.activation(x) * (1 - self.activation(x))

        self.hidden_activations = None
        self.gradients = None


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for output
        self.hidden_activations = self.a1  # Store hidden activations for visualization
        return self.a2

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        output_error = self.a2 - y  # dL/dz2
        dW2 = np.dot(self.a1.T, output_error) / X.shape[0]
        db2 = np.sum(output_error, axis=0, keepdims=True) / X.shape[0]
        
        hidden_error = np.dot(output_error, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, hidden_error) / X.shape[0]
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / X.shape[0]

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # TODO: store gradients for visualization
        self.gradients = [np.abs(dW1).mean(), np.abs(dW2).mean()]

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def plot_hidden_space(ax, hidden_features, y, frame):
    from scipy.spatial import ConvexHull

    # Scatter the hidden features
    ax.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
               c=y.ravel(), cmap='bwr', alpha=0.7, edgecolor='k')

    # Add Convex Hull if possible
    if hidden_features.shape[0] >= 3:  # ConvexHull requires at least 3 points
        try:
            hull = ConvexHull(hidden_features[:, :2])  # Use only 2D projection
            for simplex in hull.simplices:
                # Plot the edges of the hull in 3D
                ax.plot(hidden_features[simplex, 0], hidden_features[simplex, 1], zs=0,
                        color='k', lw=0.8, alpha=0.7)  # Add edges in 2D projection
        except Exception as e:
            print(f"ConvexHull error: {e}")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title(f"Hidden Space at Step {frame}", fontsize=10)

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    # Plot Hidden Space with Convex Hull
    hidden_features = mlp.hidden_activations
    plot_hidden_space(ax_hidden, hidden_features, y, frame)

    # TODO: Hyperplane visualization in the hidden space
    if mlp.W2.shape[0] >= 2:  # Check if the weights support a hyperplane in 2D
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
        zz = -(mlp.W2[0, 0] * xx + mlp.W2[1, 0] * yy + mlp.b2[0, 0])  # 2D plane equation
        ax_hidden.contourf(xx, yy, zz, levels=50, cmap='coolwarm', alpha=0.3)

    # TODO: Distorted input space transformed by the hidden layer
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))  # Higher resolution grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid)  # Forward pass for the grid
    predictions = predictions.reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=np.linspace(predictions.min(), predictions.max(), 100), cmap='bwr', alpha=0.6)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame}", fontsize=10)

    # TODO: Visualize features and gradients as circles and edges
    node_positions = {
        "x1": (0, 0.8), "x2": (0, 0.2),  # Input nodes
        "h1": (0.5, 0.6), "h2": (0.5, 0.4), "h3": (0.5, 0.2),  # Hidden layer nodes
        "y": (1, 0.4)  # Output node
    }
    edges = [
        ("x1", "h1"), ("x1", "h2"), ("x1", "h3"),
        ("x2", "h1"), ("x2", "h2"), ("x2", "h3"),
        ("h1", "y"), ("h2", "y"), ("h3", "y")
    ]

    # Draw nodes
    for node, (x, y) in node_positions.items():
        ax_gradient.scatter(x, y, s=1000, color='blue', alpha=0.8, zorder=2)
        ax_gradient.text(x, y, node, color='white', ha='center', va='center', fontsize=12, zorder=3)

    # Draw edges with thickness representing gradient magnitude
    edge_thickness = {
        "x1_h1": np.abs(mlp.W1[0, 0]).mean(),
        "x1_h2": np.abs(mlp.W1[0, 1]).mean(),
        "x1_h3": np.abs(mlp.W1[0, 2]).mean(),
        "x2_h1": np.abs(mlp.W1[1, 0]).mean(),
        "x2_h2": np.abs(mlp.W1[1, 1]).mean(),
        "x2_h3": np.abs(mlp.W1[1, 2]).mean(),
        "h1_y": np.abs(mlp.W2[0, 0]).mean(),
        "h2_y": np.abs(mlp.W2[1, 0]).mean(),
        "h3_y": np.abs(mlp.W2[2, 0]).mean()
    }

    for edge in edges:
        start, end = edge
        start_pos = node_positions[start]
        end_pos = node_positions[end]
        key = f"{start}_{end}".replace(" ", "_").lower()
        thickness = edge_thickness.get(key, 1) * 10  # Scale thickness
        ax_gradient.plot(
            [start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
            color='purple', linewidth=thickness, alpha=0.8, zorder=1
        )

    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis("off")
    ax_gradient.set_title(f"Gradients at Step {frame}", fontsize=10)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=5, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig, 
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), 
        frames=step_num // 10, 
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 2000
    visualize(activation, lr, step_num)