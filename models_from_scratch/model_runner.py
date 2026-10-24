import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from basic_neural_network import NeuralNetwork

# --------------------------------------------------
# Load images
# --------------------------------------------------

X, labels = load_digits(return_X_y=True)

# Keep only handwritten 3s and 8s
mask = (labels == 3) | (labels == 8)

X = X[mask]
labels = labels[mask]

# Binary labels:
# 3 -> 0
# 8 -> 1
y = (labels == 8).astype(np.float64)


# --------------------------------------------------
# Normalize pixel values
# original range: [0, 16]
# --------------------------------------------------

X = X.astype(np.float64) / 16.0


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=173,
    stratify=y
)


# --------------------------------------------------
# Convert to YOUR column-vector convention
#
# sklearn:   (samples, features)
# ours:      (features, samples)
# --------------------------------------------------

X_train = X_train.T
X_test = X_test.T

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


# --------------------------------------------------
# Build your neural network
# --------------------------------------------------

nn = NeuralNetwork()

nn.add_layer(64, 32, "relu")
nn.add_layer(32, 16, "relu")
nn.add_layer(16, 1, "sigmoid")


# --------------------------------------------------
# Train
# --------------------------------------------------

nn.train(
    X_train,
    y_train,
    epochs=100,
    learning_rate=0.05,
    batch_size=32
)


# --------------------------------------------------
# Test
# --------------------------------------------------

y_pred = nn.forward(X_test)

y_pred_class = (y_pred >= 0.5).astype(int)

accuracy = np.mean(y_pred_class == y_test)

print(f"Test accuracy: {accuracy:.4f}")
