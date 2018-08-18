import scipy.stats as stat
import sklearn.linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats
from Page_Hinkley import *



def load_data(filename):
    """ Load a file, given its name.
    filename-- name of the file we want to open.

    :return a list containing all rows
    """
    result = []
    dico={}
    line_i=[]
    with open(filename, 'r') as joint_prob_file:
        joint_prob_file.readline()
        for line in joint_prob_file.readlines():
            line_i = line.split()
            line_i = [float(string.strip()) for string in line_i]
            result.append(line_i)
    return result


# Preprocessing
def remove_outlier(input_data, label_type=None):
    """
    :param input_data: the datastream in which we remove outliers
    :param label_type: The type of label (Source) of the data (GCAG, GISTEMP)
    :return: indices, of the inliers values.
    """
    mean_data = np.mean(input_data['Mean'])  # Mean value of the data
    var_data = np.std(input_data['Mean'])  # Variance of the data
    output_data = [] # return the indices of the inliers values
    for j in range(len(input_data['Mean'])):
        if label_type == 'GCAG':
            if mean_data + 3*var_data > input_data['Mean'][2 * j] > mean_data - 3*var_data:
                output_data.append(2*j)
        else:
            if mean_data + 3*var_data > input_data['Mean'][2 * j + 1] > mean_data - 3*var_data:
                output_data.append(2*j + 1)
    return output_data


# remove_outlier 2
def remove_outlier2(input_data):
    """
    The function remove outliers to the input datastream.
    :param input_data: the datastream
    :return: the datastream, with remove outliers
    """
    mean_data = np.mean(input_data['Mean'])  # Mean value of the data
    var_data = np.std(input_data['Mean'])  # Variance of the data
    input_data_copy = input_data.copy()
    a = []
    for j in range(len(input_data_copy['Mean'])):
        if mean_data + 4*var_data < input_data_copy['Mean'][j] < mean_data - 4*var_data:
            # a.append(j)
            input_data_copy = input_data_copy.drop([j]) # Remove the corresponding rows
    # A = input_data_copy[a]
    return input_data_copy


def ADWIN(datastream, confidence_interval=None):
    """
    The function implements the ADWIN  algorithm, for Drift detection
    :param datastream: The datastream of Examples
    :param confidence_interval: The level of confidence to the detection made
    :return: W, a window of examples
    """
    if confidence_interval is None:
        confidence_interval = 0.4

    #drift_detected = False
    mean_w = 0
    # Initialize the Window
    height = datastream.shape[0]
    rand = np.random.randint(1, 52)
    rand = 20

    W = datastream[1:rand]
    for xi in datastream:
        # df2 = pd.DataFrame([[xi + 1, "GISTEMP", "2012-10-27", datastream['Mean'][xi], rand + 1]], columns=["Unnamed: 0", "Source", "Date", "Mean", "index"])
        W = np.append(W, xi)

        # Splitting into 2 sets W0, W1
        for j in range(1, W.shape[0]):
            print("J value", j)
            print('wshape', W.shape)
            W0 = W[0:j]
            W1 = W[j:W.shape[0] + 1]

            n0 = len(W0)  # W0.shape[0] *W0.shape[1]
            n1 = len(W1)  # (W1.shape[0] )*( W1.shape[1])
            if n1 > 1:
                # Compute the average
                mean_W0_hat = np.mean(W0)
                mean_W1_hat = np.mean(W1)

                # Calculate epsilon
                print("n0", n0)
                print("n1", n1)
                n = n0 + n1
                m = 1 / (1/n0 + 1/n1)
                sigmap = confidence_interval / n
                epsilon = np.sqrt((1/2*m) * (4/sigmap))
                print('epsilon', epsilon)
                diff = np.absolute(mean_W0_hat - mean_W1_hat)
                print('diff', diff)
                if diff < epsilon:
                    print("ENTERED")
                    W = np.delete(W, j)
                    print("Size of W", W.size)
                    drift_detected = False
                else:
                    drift_detected = True
                    break
                    #  W.drop([W.shape[0] - 1])
        #if mean_w - np.mean(W) == 0
    return drift_detected


def kolmogorov_smirnov(data, window_size=1000):
    """
    The function is the Kolmogorov smirnov test, that use the
    :param data: Column vector
    :param window_size: Size of the Scanning Window
    :return: True, False (True : Drift Present, False : Drift Absent)
    """
    # W0 = data[1:window_size]
    num = 0
    data_length = data.shape[0]

    for t in range(0, data_length, window_size):
        data_ = []
        # Splitting the data recursively in two using a sliding window
        sample1 = data[t:t+window_size]
        if t+window_size < data_length:
            sample2 = data[t+window_size:t + 2*window_size]
        else:
            # print("ca marche")
            sample2 = data[t:data_length]

        # --->Mean and std of the sample 1
        mean_samp1e2 = np.mean(sample1)  # Mean of the second sliding window.
        std_sample2 = np.std(sample1)  # Standard deviation

        # ---> Mean and std of the sample 2
        mean_samp1e1 = np.mean(sample2)
        std_sample1 = np.std(sample2)

        # Normalization of the value of the samples
        sample1 = [(item - mean_samp1e1) / std_sample1 for item in sample1]
        sample2 = [(item - mean_samp1e2) / std_sample2 for item in sample2]

        D_stat, p_value = stat.ks_2samp(sample1, sample2)

        print("Result P vlaue", p_value)
        if p_value < 0.05: # We reject the Null Hypothesis, so Drfit detected
            drift = True
            num = num + 1
            print("Drift detected between {} to {} and {} to {}".format(t, t+window_size-1, t+window_size,  t+2*window_size))
        else:
            drift = False
            print("t value..........{} and data length {}".format(t+window_size, t + 2*window_size))
    print("{} drifts detected".format(num))
    return drift


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_artificial_dataset(datastream, pause_=None, drift_type='virtual'):
    """
    Inject a drift on the dataset
    Generate an Artificial Dataset and output the result.
    :param datastream: The base data in which we inject Drift.
    :param pause_ : The time to pause the graphic
    :return: None
    """
    # Original data
    # ---Mean
    data_GCAG = data2[data2['Source'] == 'GCAG']
    data_GISTEMP = data2[data2['Source'] == 'GISTEMP']

    # --Date
    date_GCAG = data_GCAG['Date']
    date_GISTEMP = data_GISTEMP['Date']

    # for j in range(len(date_GCAG)):
    #     data_GCAG['Date'][2*j] = j
    #     data_GISTEMP['Date'][2*j + 1] = j

    # Remove outlier and add modify indexes
    # result = remove_outlier2(data2)
    # print("rmv_outlier 2", result.shape)

    # # print(data2)
    print("Type of data_gcag", type(data_GCAG))
    result_GCAG = remove_outlier(data_GCAG, 'GCAG')
    result_GISTEMP = remove_outlier(data_GISTEMP, 'GISTEMP')

    # # # GCAG data
    data_mean_GCAG = data_GCAG['Mean'][result_GCAG]
    data_mean_GCAG = data_mean_GCAG.tolist()  # Convert to list, to later convert to DataFrame
    data_mean_GCAG = pd.DataFrame(data_mean_GCAG)  #

    # Adding the last index to the mean column extracted
    # index = [j for j in range(data_mean_GCAG.shape[0])]
    # data_mean_GCAG['index'] = index
    # data_mean_GCAG = data_mean_GCAG.reindex(index)
    # print('data', data_mean_GCAG.shape[0])
    ##print(data_mean_GCAG)

    # # # GISTEMP data
    # data_mean_GISTEMP = data_GISTEMP['Mean'][result_GISTEMP]
    # data_date_GISTEMP = data_GISTEMP['Date'][result_GISTEMP]

    # # --Plotting the original data
    # plt.figure(1)
    # plt.plot(data_mean_GCAG)
    # plt.title("Original Data with outliers removes")
    # plt.ylabel("Mean temperature distribution")
    # plt.draw()
    # plt.pause(20)

    # # ---Plotting to see how it looks.
    # plt.figure(2)
    # plt.plot(data_mean_GCAG)
    # plt.title("After removing outliers... using label type")
    # plt.draw()
    # plt.pause(20)

    # Concatenating the actual datastream with the Sinus function
    x = np.linspace(-np.pi, np.pi, 644)

    # result.to_csv('test_df_csv.csv')
    data3 = pd.read_csv('test_df_csv.csv')
    # print(data3.shape)
    sin_template = np.sin(4 * np.pi * x)
    if drift_type == 'virtual':
        sin_template = np.ones((1000, 1))
    else:
        val = np.linspace(0, 10, 400)
        sin_template = sigmoid(val)
    y = -x + 1; min_y = np.min(y); max_y = np.max(y)
    y = [(item - min_y)/(max_y - min_y) for item in y ]
    print("Poumffffffffffffffffff",len(y))
    # print(sin_template)

    sin_template = list(sin_template)

    data_mean_GCAG = data_mean_GCAG.to_dict()
    data_repeat = list(data_mean_GCAG[0].values())
    data_mean_GCAG = list(data_mean_GCAG[0].values())

    data_mean_GCAG = data_mean_GCAG[1:300]

    data_mean_GCAG.append((data_mean_GCAG[len(data_mean_GCAG) - 1] + sin_template[0])/2)

    data_mean_GCAG.extend(sin_template)
    # data_mean_GCAG.extend(y)
    N = 200
    data_repeat = 0.17 + 0.3 * np.random.rand(N)
    data_mean_GCAG.append((data_mean_GCAG[len(data_mean_GCAG) - 1] + data_repeat[0])/2)
    data_mean_GCAG.extend(data_mean_GCAG[0:70])
    data_mean_GCAG.extend(data_repeat)
    data_mean_GCAG = pd.DataFrame(np.abs(data_mean_GCAG))

    # Test pour se rassurer que les valeurs sont comprises entre -1 et 1
    # print("=====================================================")
    # test = data_mean_GCAG > 1
    # # print(test[0])
    # q = [j for j in test[0] if j]
    # print(q)
    # print("==================================================")
    dat = data_mean_GCAG.copy()
    data_mean_GCAG = np.array(data_mean_GCAG)
    dat = np.array(dat)
    # print(np.append(dat[1:10], [2]))
    # print(dat[1:10])
    # rint(dat[0].append(data_mean_GCAG[0][0]))

    # data_mean_GCAG = data_mean_GCAG.extend(sin_template)
    # data_mean_GCAG = pd.DataFrame(data_mean_GCAG)

    # Plot the merge data
    if pause_ is None:
        plt.figure()
        plt.plot(data_mean_GCAG)
        plt.title('Gradual change')
        plt.draw()
        plt.show()
    else:
        plt.figure()
        plt.plot(data_mean_GCAG)
        plt.draw()
        plt.pause(pause_)
    # plt.pause(5)
    return data_mean_GCAG


if __name__ == '__main__':
    filename = 'data_set_test_weather/monthly_csv_temp.csv'
    # data = pd.DataFrame.from_csv(filename)
    data2 = pd.read_csv(filename)

    data_mean = generate_artificial_dataset(data2, pause_=20, drift_type='real')
    # # result.to_csv('test_df_csv.csv')
    # result_ADWIN = ADWIN(result)
    #
    # print(result)
    # plt.figure(2)
    # plt.plot(result['Mean'])
    # plt.show()

    # # Testing ADWIN
    print(data_mean.shape)
    # mean_w = ADWIN(data_mean)
    # print(mean_w)

    # # Testing Kolmogorov-Smirnov
    sample2 = data_mean[2639:3284, 0]
    # Sample1 = data_mean[1639:2639, 0]
    # sample1 = data_mean[3283:4921, 0]
    sample1 = data_mean[0:2283, 0]
    # plt.figure()
    # plt.plot(sample1)
    # plt.show()

    kolmogorov_smirnov(data_mean[:, 0])

    # # Testing the Page Hinkley statistic test
    # pg_hinkley = PageHinkley()
    # print("DATA MEAN pfppffffffffffffffff", data_mean[:,0])
    # print()
    # for data_element in data_mean[:, 0]:
    #     changed_detected = pg_hinkley.detect_drift(data_element)
    #     # print(changed_detected)
    #     if changed_detected:
    #         print("Changed detected using Page_hinkley at pt {}")







