import random

from data import get_data
from rnn import RNN

import numpy as np

class Environment:
    def __init__(self, seed = 42):
        self.window_size = 20
        self.seed = seed

        random.seed(seed)

    def run(self):

        data = get_data()
        data.drop('observation_date', axis=1, inplace=True)

        items = self.to_window(data["index_value"])
        random.shuffle(items)

        rnn = RNN(input_size=self.window_size, seed=self.seed)

        for epoch in range(100):
            loss = 0.0

            for y, x in zip(*items):

                x = np.array(x)

                inputs = []
                for input in x:
                    inputs.append(input)

                # Forward Prop
                out, _ = rnn.forward_prop(x)

                # Calculate loss
                loss += (y - out[0][0]) ** 2 / 2

                # Backward prop
                d_l_d_y = out - y
                rnn.backward_prop(d_l_d_y)

            print('Epoch: ', epoch + 1, ' Value Loss: ', loss,' Loss: ', loss / len(items))

        # Validation



    def to_window(self, df):

        df_np = df.to_numpy()

        x = []
        y = []

        for i in range(len(df_np) - self.window_size):
            row = [df_np[i:i + self.window_size]]
            x.append(row)

            label = df_np[i + self.window_size]
            y.append(label)

        return list((x, y))
