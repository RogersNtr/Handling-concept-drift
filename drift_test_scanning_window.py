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


def kolmogorov_smirnov(data1, data2, window_size=None):
    """
    The function is the Kolmogorov smirnov test, that use the
    :param data: Column vector
    :param window_size: Size of the Scanning Window
    :return: True, False (True : Drift Present, False : Drift Absent)
    """
    # W0 = data[1:window_size]
    # for t in range(len(data)):
    #     W2 = data[window_size - t + 1:window_size]
    result = stat.ks_2samp(data1, data2)
    print("Result", result)
    # if result < 0.05: # We reject the Null Hypothesis, so Drfit detected
    #     drift = True
    # else:
    #     drift = False

    return False


def generate_artificial_dataset(datastream):
    """
    Inject a drift on the dataset
    Generate an Artificial Dataset and output the result.
    :param datastream: The base data in which we inject Drift.
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
    sin_template = np.ones((1000, 1))
    y = -x + 1; min_y = np.min(y); max_y = np.max(y)
    y = [(item - min_y)/(max_y - min_y) for item in y ]
    print("Poumffffffffffffffffff",len(y))
    # print(sin_template)

    sin_template = list(sin_template)

    data_mean_GCAG = data_mean_GCAG.to_dict()
    data_repeat = list(data_mean_GCAG[0].values())
    data_mean_GCAG = list(data_mean_GCAG[0].values())

    data_mean_GCAG.extend(sin_template)
    data_mean_GCAG.extend(y)
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
    # plt.figure()
    # plt.plot(data_mean_GCAG)
    # plt.draw()
    # plt.pause(5)
    return data_mean_GCAG

if __name__ == '__main__':
    filename = 'data_set_test_weather/monthly_csv_temp.csv'
    # data = pd.DataFrame.from_csv(filename)
    data2 = pd.read_csv(filename)

    data_mean = generate_artificial_dataset(data2)
    # # result.to_csv('test_df_csv.csv')
    # result_ADWIN = ADWIN(result)
    #
    # print(result)
    # plt.figure(2)
    # plt.plot(result['Mean'])
    # plt.show()

    # # Testing ADWIN
    print(data_mean.shape)
    mean_w = ADWIN(data_mean)
    print(mean_w)

    # # Testing Kolmogorov-Smirnov
    distrib1 = data_mean[2639:3284, 0]
    # distrib2 = data_mean[1639:2639, 0]
    distrib2 = data_mean[3283:4921, 0]
    kolmogorov_smirnov(distrib1, distrib2)

    # # Testing the Page Hinkley statistic test
    pg_hinkley = PageHinkley()
    print("DATA MEAN pfppffffffffffffffff", data_mean[:,0])
    print()
    for data_element in data_mean[:, 0]:
        changed_detected = pg_hinkley.detect_drift(data_element)
        # print(changed_detected)
        if changed_detected:
            print("Changed detected using Page_hinkley at pt {}")






