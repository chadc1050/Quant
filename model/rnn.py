import numpy as np

class RNN:

    def __init__(self, input_size, hidden_size = 64, seed = 42, learning_rate = 0.005, clip = 1):
        generator = np.random.default_rng(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_i_h = generator.uniform(0, 1, (hidden_size, input_size))
        self.w_h_o = generator.uniform(0, 1, (1, hidden_size))

        self.b_i_h = generator.uniform(0, 1, (hidden_size, 1))
        self.b_h_o = generator.uniform(0, 1, (1, 1))

        self.learning_rate = learning_rate

        self.clip = clip

    def forward_prop(self, inputs):
        """
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.
        - inputs is an array of one hot vectors with shape (input_size, 1).
        """

        h = np.zeros((self.hidden_size, 1))

        self.last_inputs = inputs
        self.last_hs = { 0: h }

        for i, x in enumerate(inputs):
            h = np.tanh(self.w_i_h @ x + self.b_i_h)
            self.last_hs[i + 1] = h

        y = self.w_h_o @ h + self.b_h_o

        return y, h

    def backward_prop(self, d_y):
        """
        Perform a backward pass of the RNN.
        - d_y (dL/dy) has shape (output_size, 1).
        - learn_rate is a float.
        """

        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby.
        d_w_h_o = np.zeros(self.w_h_o.shape)
        d_b_o = d_y

        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
        d_w_i_h = np.zeros(self.w_i_h.shape)
        d_b_h = np.zeros(self.b_i_h.shape)

        # Calculate dL/dh for the last h.
        # dL/dh = dL/dy * dy/dh
        d_h = self.w_h_o.T @ d_y

        # Backprop through time
        for t in reversed(range(n)):

            # An intermediate value: dL/dh * (1 - h^2)
            temp1 = (1 - self.last_hs[t + 1] ** 2)
            temp = temp1 * d_h

            d_b_h += temp
            d_w_i_h += temp @ self.last_inputs[t].T
            d_h = self.w_i_h @ temp

        # Clip to prevent exploding gradients.
        for d in [d_w_h_o, d_w_i_h, d_b_o, d_b_h]:
            np.clip(d, -self.clip, self.clip, out=d)

        self.w_i_h -= self.learning_rate * d_w_i_h
        self.w_h_o -= self.learning_rate * d_w_h_o
        self.b_i_h -= self.learning_rate * d_b_h
        self.b_h_o -= self.learning_rate * d_b_o