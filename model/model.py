import numpy as np
import matplotlib.pyplot as plt
from data import get_data


class Model:

    def __init__(self,
                 n_epochs = 20,
                 window_size = 20,
                 n_features = 3,
                 patience = 2,
                 learning_rate = 0.005,
                 hidden_dim = 75):

        self.n_epochs = n_epochs
        self.window_size = window_size
        self.T = window_size * n_features
        self.learning_rate = learning_rate
        self.patience = patience

        self.hidden_dim = hidden_dim
        self.output_dim = 1

        self.bptt_truncate = 5

        self.min_clip_value = -10
        self.max_clip_value = 10

        self.b_i_h = np.random.uniform(0, 1, (self.hidden_dim, self.T))
        self.w_i_h = np.random.uniform(0, 1, (self.hidden_dim, self.hidden_dim))
        self.w_h_o = np.random.uniform(0, 1, (self.output_dim, self.hidden_dim))


    def run(self):

        data = get_data()
        data.drop('observation_date', axis=1, inplace=True)

        X, Y = self.to_window(data)

        x_train, y_train = X[:8000], Y[:8000, [0]]

        best_loss = float('inf')
        loss_increase_count = 0

        for epoch in range(self.n_epochs):

            val_loss = 0.0

            for i in range(y_train.size):

                x, y = x_train[i].reshape(1, self.T).transpose(), y_train[i]

                layers = []
                prev_s = np.zeros((self.hidden_dim, 1))

                dU = np.zeros(self.b_i_h.shape)
                dV = np.zeros(self.w_h_o.shape)
                dW = np.zeros(self.w_i_h.shape)

                dU_t = np.zeros(self.b_i_h.shape)
                dW_t = np.zeros(self.w_i_h.shape)

                # forward pass
                for t in range(self.T):
                    new_i = np.zeros(x.shape)
                    new_i[t] = x[t]
                    mulu = np.dot(self.b_i_h, new_i)
                    mulw = np.dot(self.w_i_h, prev_s)
                    add = mulw + mulu
                    s = sigmoid(add)
                    mulv = np.dot(self.w_h_o, s)

                    layers.append({'s':s, 'prev_s':prev_s})
                    prev_s = s

                # Calculate loss (MSE)
                val_loss += (y - mulv) ** 2 / 2

                # derivative of pred
                dmulv = (mulv - y)

                # backward pass
                for t in range(self.T):
                    dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
                    dsv = np.dot(np.transpose(self.w_h_o), dmulv)

                    ds = dsv
                    dadd = add * (1 - add) * ds

                    dmulw = dadd * np.ones_like(mulw)

                    dprev_s = np.dot(np.transpose(self.w_i_h), dmulw)

                    for i in range(t - 1, max(-1, t - self.bptt_truncate - 1), -1):
                        ds = dsv + dprev_s
                        dadd = add * (1 - add) * ds

                        dmulw = dadd * np.ones_like(mulw)
                        dmulu = dadd * np.ones_like(mulu)

                        dW_i = np.dot(self.w_i_h, layers[t]['prev_s'])
                        dprev_s = np.dot(np.transpose(self.w_i_h), dmulw)

                        new_input = np.zeros(x.shape)
                        new_input[t] = x[t]
                        dU_i = np.dot(self.b_i_h, new_input)
                        dx = np.dot(np.transpose(self.b_i_h), dmulu)

                        dU_t += dU_i
                        dW_t += dW_i

                    dV += dV_t
                    dU += dU_t
                    dW += dW_t

                    if dU.max() > self.max_clip_value:
                        dU[dU > self.max_clip_value] = self.max_clip_value
                    if dV.max() > self.max_clip_value:
                        dV[dV > self.max_clip_value] = self.max_clip_value
                    if dW.max() > self.max_clip_value:
                        dW[dW > self.max_clip_value] = self.max_clip_value

                    if dU.min() < self.min_clip_value:
                        dU[dU < self.min_clip_value] = self.min_clip_value
                    if dV.min() < self.min_clip_value:
                        dV[dV < self.min_clip_value] = self.min_clip_value
                    if dW.min() < self.min_clip_value:
                        dW[dW < self.min_clip_value] = self.min_clip_value

                # update
                self.b_i_h -= self.learning_rate * dU
                self.w_i_h -= self.learning_rate * dW
                self.w_h_o -= self.learning_rate * dV

            print('Epoch: ', epoch + 1, 'Iterations: ', y_train.size, 'Loss: ', (val_loss / float(y_train.size))[0][0], ', Val Loss: ', val_loss[0][0])

            if val_loss[0][0] > best_loss:
                loss_increase_count += 1
                if loss_increase_count >= self.patience:
                    print('Early Stopping with patience: ', self.patience)
                    break
            else:
                loss_increase_count = 0

        # Validation
        x_val, y_val = X[8000:], Y[8000:, [0]]

        preds = []
        for i in range(y_val.shape[0]):
            x, y = x_val[i].reshape(1, self.T).transpose(), y_val[i]
            prev_s = np.zeros((self.hidden_dim, 1))
            # For each time step...
            for t in range(self.T):
                mulu = np.dot(self.b_i_h, x)
                mulw = np.dot(self.w_i_h, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(self.w_h_o, s)
                prev_s = s

            preds.append(mulv)

        preds = np.array(preds)

        plot_preds = preds[:, 0, 0]

        plt.plot(plot_preds, 'g')
        plt.plot(y_val, 'r')
        plt.show()

    def to_window(self, df):

        df_np = df.to_numpy()

        x = []
        y = []

        for i in range(len(df_np) - self.window_size):
            row = [df_np[i:i + self.window_size]]
            x.append(row)

            label = df_np[i + self.window_size]
            y.append(label)

        return np.array(x), np.array(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))