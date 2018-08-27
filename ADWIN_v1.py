import numpy as np


class Adwin:
    def __init__(self, datastream, delta = 0.4):
        """
        The function implements the ADWIN_V1  algorithm, for Drift detection
        :param datastream: The datastream of Examples
        :param delta: The level of confidence to the detection made
        :return: W, a window of examples
        """
        self.data = datastream.tolist()
        self.delta_ = delta

    def detect_drift(self):
        drift_detected = False
        mean_w = 0
        # Initialize the Window
        height = len(self.data)
        rand = np.random.randint(1, 52)
        rand = 3

        W = self.data[0:rand]
        for xi in range(rand, len(self.data)):
            # df2 = pd.DataFrame([[xi + 1, "GISTEMP", "2012-10-27", datastream['Mean'][xi], rand + 1]], columns=["Unnamed: 0", "Source", "Date", "Mean", "index"])
            W.append(xi)
            # Splitting into 2 sets W0, W1
            len_w = len(W)
            for j in range(1, len_w):
                # print("J value", j)
                # print('wshape', W.shape)
                W0 = W[0:j]
                W1 = W[j:len_w + 1]

                n0 = len(W0)  # W0.shape[0] *W0.shape[1]
                n1 = len(W1)  # (W1.shape[0] )*( W1.shape[1])
                if n1 > 1:
                    # Compute the average
                    mean_W0_hat = np.mean(W0)
                    mean_W1_hat = np.mean(W1)

                    # Calculate epsilon
                    # print("n0", n0)
                    # print("n1", n1)
                    n = n0 + n1
                    m = 1 / (1 / n0 + 1 / n1)
                    sigmap = self.delta_ / n
                    epsilon = np.sqrt((1 / 2 * m) * (4 / sigmap))
                    # print('epsilon', epsilon)
                    diff = abs(mean_W0_hat - mean_W1_hat)
                    # print('diff', diff)
                    if diff < epsilon:
                        # print("ENTERED")
                        W.pop(0)
                        # print("Size of W", W.size)
                        drift_detected = False
                    else:
                        drift_detected = True
                        print("drift detected : ", drift_detected)
                        print(self.data[xi])
                        break
                        #  W.drop([W.shape[0] - 1])
            # if mean_w - np.mean(W) == 0
        return drift_detected

