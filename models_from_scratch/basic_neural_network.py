import numpy as np 

# NN implementation assuming ReLU hidden layers --> sigmoid output --> BCE Loss

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.outputs = []
        self.gradients = []
        self.rng = np.random.default_rng(173)

    def forward(self, X):
        self.outputs.clear()
        output = X
        for layer in self.layers:
            W, b, activation = layer
            Z = (W @ output) + b # Z = WX + b
            self.outputs.append((output, Z)) # (A^[l - 1], Z^[L])
            output = activation(Z) # a = g(z)
            
        return output

    def add_layer(self, input_size, output_size, activation):
        if activation == "relu":
            he_constant = np.sqrt(2/input_size)

            W = self.rng.normal(
                                loc=0.0, 
                                scale=1.0,
                                size=(output_size, input_size)) * he_constant
            b = np.zeros((output_size, 1))
            activation_func = self.relu

        elif activation == "sigmoid":
            xavier_constant = np.sqrt(2/(output_size + input_size))

            W = self.rng.normal(
                                loc=0.0,
                                scale=1.0,
                                size=(output_size, input_size)) * xavier_constant
            b = np.zeros((output_size, 1))
            activation_func = self.sigmoid
        else:
            raise ValueError("Invalid activation parameter")

        self.layers.append((W, b, activation_func))

    def backward(self, y, y_pred):
        self.gradients.clear()
        delta = y_pred - y

        for i in range(len(self.layers) - 1, -1, -1):
            W, b, activation = self.layers[i]
            a, z = self.outputs[i] 
            m = a.shape[1] # m is the batch size

            dw = (1/m) * (delta @ a.T) # dw = 1/m * (delta^[l] @ (a^[l - 1])^T)
            db = np.mean(delta, axis=1, keepdims=True) # db = 1/m * sum(col of delta)

            self.gradients.insert(0, (dw, db))

            # update delta^[l]
            if i > 0:
                a_next, z_next = self.outputs[i - 1]
                da = (W.T @ delta)
                delta = da * self.relu_prime(z_next)

    def train(self, X, y, epochs, learning_rate, batch_size=None):
        instances = X.shape[1]

        if batch_size is None:
            batch_size = instances

        for epoch in range(epochs):
            indices = self.rng.permutation(instances)

            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]

            start = 0
            epoch_loss = 0
            while start < instances:
                end = min(instances, start + batch_size)

                X_batch = X_shuffled[:, start: end]
                y_batch = y_shuffled[:, start:end]

                y_pred_batch = self.forward(X_batch)
                self.backward(y_batch, y_pred_batch)

                epoch_loss += np.sum(self.loss(y_pred_batch, y_batch))

                for layer_index in range(len(self.layers)):
                    W, b, activation  = self.layers[layer_index]
                    dW, db = self.gradients[layer_index]
                    W -= learning_rate * dW
                    b -= learning_rate * db
            
                start = end

            epoch_loss /= instances
            print(f"Epoch {epoch}: completed with loss {epoch_loss}")

    def loss(self, y_pred, y):
        epsilon = 1e-6
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def relu(self, X):
        return np.maximum(0, X)

    def relu_prime(self, X):
        return np.where(X > 0, 1, 0)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
