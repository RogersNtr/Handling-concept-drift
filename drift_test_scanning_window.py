import scipy
import sklearn.linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats


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
        if mean_data - 4*var_data < input_data_copy['Mean'][j] < mean_data + 4*var_data:
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
        confidence_interval = 2
    # Initialize the Window
    height = datastream.shape[0]
    rand = np.random.randint(1, 52)
    rand = 100

    W = datastream[1:rand]
    for xi in range(rand, datastream.shape[0]):
        df2 = pd.DataFrame([[xi + 1, "GISTEMP", "2012-10-27", datastream['Mean'][xi], rand + 1]], columns=["Unnamed: 0", "Source", "Date", "Mean", "index"])
        W = W.append(df2)

        # Splitting into 2 sets W0, W1
        W0 = W
        W1 = datastream[xi:height]

        # Compute the average
        mean_W0_hat = np.mean(W0)
        mean_W1_hat = np.mean(W1)

        # Calculate epsilon
        n0 = (W0.shape[0] )*( W0.shape[1])
        n1 = (W1.shape[0] )*( W1.shape[1])
        n = n0 + n1
        m = 1 / (1/n0 + 1/n1)
        sigmap = confidence_interval / n
        epsilon = np.sqrt((1/2*m) * (4/sigmap))
        print('epsilon', epsilon)
        diff = np.absolute(mean_W0_hat - mean_W1_hat)
        print(diff)
        while diff < epsilon:
            W.drop([W.shape[0] - 1])

    return W


def kolmogorov_smirnov(data1, data2, window_size):
    """
    The function is the Kolmogorov smirnov test, that use the
    :param data: Column vector
    :param window_size: Size of the Scanning Window
    :return: True, False (True : Drift Present, False : Drift Absent)
    """
    # W0 = data[1:window_size]
    # for t in range(len(data)):
    #     W2 = data[window_size - t + 1:window_size]
    result = stats.ks_2samp(data1, data2)
    if result == 0:
        drift = True
    else:
        drift = False

    return drift


if __name__ == '__main__':
    filename = 'data_set_test_weather/monthly_csv_temp.csv'
    # data = pd.DataFrame.from_csv(filename)
    data2 = pd.read_csv(filename)

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
    result = remove_outlier2(data2)
    print("rmv_outlier 2", result.shape)
    a = [j for j in range(result.shape[0])]
    result['index'] = a

    # Plotting the original data
    # print(result)
    plt.figure(1)
    plt.plot(result['Mean'])
    plt.title("Original Data with outliers removes")
    plt.ylabel("Mean temperature distribution")
    plt.draw()
    plt.pause(20)

    # # print(data2)
    result_GCAG = remove_outlier(data_GCAG, 'GCAG')
    result_GISTEMP = remove_outlier(data_GISTEMP, 'GISTEMP')

    # # # GCAG data
    data_mean_GCAG = data_GCAG['Mean'][result_GCAG]
    data_date_GCAG = data_GCAG['Date'][result_GCAG]
    print("rmv_outlier", data_mean_GCAG.shape)

    # # # GISTEMP data
    data_mean_GISTEMP = data_GISTEMP['Mean'][result_GISTEMP]
    data_date_GISTEMP = data_GISTEMP['Date'][result_GISTEMP]

    # # Plotting to see how it looks.
    plt.figure(2)
    plt.plot(data_mean_GCAG)
    plt.title("Test")
    plt.draw()
    plt.pause(20)
    # # plt.pause(10)
    # # plt.figure(2)
    # # plt.plot(data_date_GISTEMP, data_mean_GISTEMP)
    # # plt.show()
    #
    # x = np.linspace(-np.pi, np.pi, 644)
    #
    # # mean_data = stats.mean(x)  # Mean value of the data
    # # var_data = stats.variance(x)  # Variance of the data
    # # A = []
    # # for val in x:
    # #     if mean_data - 2*var_data < val < mean_data + 2*var_data:
    # #         A.append(val)
    # #
    # # print(len(A))
    #
    # result.to_csv('test_df_csv.csv')
    # data3 = pd.read_csv('test_df_csv.csv')
    #
    # sin_template = pd.DataFrame(x, columns=['Mean'])
    # date_add = ["2012-11-27" for j in range(sin_template.shape[0])]
    # origin_add = ["GISTEMP" for j in range(sin_template.shape[0])]
    # unnamed_add = [(data3.shape[0]) + j for j in range(sin_template.shape[0])]
    # index_add = [(data3.shape[0]) + j for j in range(sin_template.shape[0])]
    # h = np.vstack([[unnamed_add[j],  origin_add[j], date_add[j], sin_template['Mean'][j], index_add[j]] for j in range(sin_template.shape[0])])
    # h_pd = pd.DataFrame(h, columns=['Unnamed: 0', 'Source', 'Date','Mean','index'])
    # result.append(h_pd)
    #
    # #plt.plot(result)
    # # result.to_csv('test_df_csv.csv')
    # result_ADWIN = ADWIN(result)
    #
    # print(result)
    # plt.figure(2)
    # plt.plot(result['Mean'])
    # plt.show()

    # Testing ADWIN

    # sin_template = remove_outlier2(sin_template)
    #print(sin_template)

    # print(data3)

    # print('random', np.random.randint(1, 51))
    # plt.figure()
    # plt.plot(x, np.sin(4*np.pi*x))
    # plt.show()


