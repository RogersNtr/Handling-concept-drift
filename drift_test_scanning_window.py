import scipy
import sklearn.linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats


def load_data(filename):
    ''' Load a file, given its name.
    filename-- name of the file we want to open.

    :return dico
    '''
    result = []
    dico={}
    line_i=[]
    with open(filename, 'r') as joint_prob_file:
        # print(joint_prob_file.readlines())
        joint_prob_file.readline()
        #result = joint_prob_file.readlines()
        for line in joint_prob_file.readlines():
            line_i = line.split()
            line_i = [float(string.strip()) for string in line_i]
            result.append(line_i)
            # print(line)
            # line_i = line.split(',')
            # line_i = [string.strip() for string in line_i]  # remove white space and special character
            # dico[(line_i[0], line_i[1])] = float(line_i[2])
    return result


# Preprocessing
def remove_outlier(input_data, label_type):
    mean_data = stats.mean(input_data['Mean'])  # Mean value of the data
    var_data = stats.variance(input_data['Mean'])  # Variance of the data
    output_data = [] #input_data['Mean'][mean_data - 3*var_data < input_data['Mean'] < mean_data + 3*var_data]
    count = 0
    for j in range(len(input_data['Mean'])):
        if label_type == 'GCAG':
            if mean_data + 4*var_data > input_data['Mean'][2 * j] > mean_data - 4*var_data:
                # print("test", input_data['Mean'][2*j])
                count = count + 1
                # print(2*j)
                output_data.append(2*j)
        else:
            if mean_data + 4*var_data > input_data['Mean'][2 * j + 1] > mean_data - 4*var_data:
                # print("test", input_data['Mean'][2*j])
                count = count + 1
                # print(2*j)
                output_data.append(2*j + 1)
    print(count)
    print(len(input_data['Mean']))
    return output_data

# remove_outlier 2
def remove_outlier2(input_data, label_type=None):
    mean_data = stats.mean(input_data['Mean'])  # Mean value of the data
    var_data = stats.variance(input_data['Mean'])  # Variance of the data
    output_data = [] #input_data['Mean'][mean_data - 3*var_data < input_data['Mean'] < mean_data + 3*var_data]
    count = 0
    input_data_copy = input_data.copy()
    for j in range(len(input_data_copy['Mean'])):
        if mean_data + 4*var_data > input_data_copy['Mean'][j] > mean_data - 4*var_data:
            # print("test", input_data['Mean'][2*j])
            input_data_copy = input_data_copy.drop([input_data_copy['Mean'][j]])
            count = count + 1
            # print(2*j)
            output_data.append(2*j)
    print(count)
    print(len(input_data['Mean']))
    return input_data_copy

if __name__ == '__main__':
    filename = 'f_b_1.pow'
    # result = load_data(filename)
    # array_result = np.array(result)
    # print(array_result.shape)
    #
    # plt.figure()
    # plt.plot(array_result[1:500,3])
    # plt.show()
    filename = 'monthly_csv_temp.csv'
    data = pd.DataFrame.from_csv(filename)
    data2 = pd.read_csv(filename)
    # data_test = data.drop(['Date'], axis=1)
    A = data2['Mean']==0.48
    for j in range(len(data2['Mean'])):
        if data2['Mean'][j] ==0.48:
            print(j)


    data_GCAG = data2[data2['Source'] == 'GCAG']
    data_GISTEMP = data2[data2['Source'] == 'GISTEMP']
    print(type(data_GCAG['Date'][0]))
    date_GCAG = data_GCAG['Date']
    date_GISTEMP = data_GISTEMP['Date']
    # print(date_GCAG)
    # for j in range(len(date_GCAG)):
    #     data_GCAG['Date'][2*j] = j
    #     data_GISTEMP['Date'][2*j + 1] = j
    result = remove_outlier2(data2)
    # # print(data2)
    # result_GCAG = remove_outlier(data_GCAG, 'GCAG')
    # result_GISTEMP = remove_outlier(data_GISTEMP, 'GISTEMP')
    # # print(result)
    # # GCAG, data
    # data_mean_GCAG = data_GCAG['Mean'][result_GCAG]
    # data_date_GCAG = data_GCAG['Date'][result_GCAG]
    #
    # # GISTEMP data
    # data_mean_GISTEMP = data_GISTEMP['Mean'][result_GISTEMP]
    # data_date_GISTEMP = data_GISTEMP['Date'][result_GISTEMP]
    # print(data_mean)
    # Plotting to see how it looks.
    # plt.figure(1)
    # plt.plot(data_date_GCAG, data_mean_GCAG)
    # plt.show()
    # plt.pause(10)
    # plt.figure(2)
    # plt.plot(data_date_GISTEMP, data_mean_GISTEMP)
    # plt.show()
    x = np.linspace(-np.pi, np.pi, 1644)

    plt.plot(x, np.sin(4*np.pi*x))
    plt.show()


