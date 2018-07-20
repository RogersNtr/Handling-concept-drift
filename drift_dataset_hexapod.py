import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot  as plt
from datetime import datetime
from drift_test_scanning_window import kolmogorov_smirnov, ADWIN

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
        # result = joint_prob_file.readlines()
        for line in joint_prob_file.readlines():
            line_i = line.split()
            line_i = [float(string.strip()) for string in line_i]
            result.append(line_i)
            # print(line)
            # line_i = line.split(',')
            # line_i = [string.strip() for string in line_i]  # remove white space and special character
            # dico[(line_i[0], line_i[1])] = float(line_i[2])
    return result


def readfile_pow(file_ref):
    """
    method to read power log  data
    @input: filename .. full filename of the pow file
    @output: pow .. power log in format: time, voltage[V], current[A]
    """
    pow = np.empty((0, 3))  # )
    # open the file and parse the values
    with open(file_ref, 'r') as fp:
        for line in fp:
            q = line.split()

            # timestamp
            tt_i = (float(q[0]) + float(q[1]) / 1000000000)
            # voltage
            V_i = float(q[2])  # * 0.1
            # current
            I_i = (float(q[3]))  # * 10.0 / 1023.0) - 5

            # append data
            a = np.array([tt_i, V_i, I_i])  # ,dtype=DTYPE)
            pow = np.vstack([pow, a])
    return pow


def concatenate_data(data1, data2, window_size=1024):
    """
    Concatenate two data
    :param data1: data to concatenate
    :param data2: second data to concatenate
    :param window_size: 1024
    :return: the data already concatenate
    """
    shape_data1 = data1.shape[0]
    print(shape_data1)
    shape_data2 = data2.shape[0]
    print(shape_data2)
    max_length_data1_data2 = max(shape_data1, shape_data2)
    for i in range(0, max_length_data1_data2, window_size):
        if i+window_size < shape_data1:
            A = data1[i:i + window_size]
        else:
            A = data1[i:shape_data1]

        if i+window_size < shape_data2:
            B = data2[i:i + window_size]
        else:
            B = data2[i:shape_data2]
        # B = data2[i:i + window_size]

        # print(type(A))
        frames.append(A)
        if i-1+window_size < shape_data1:
            # print("i", i)
            smooth_A = [{'current': (data1['current'][i - 1 + window_size] + data2['current'][i]) / 2, 'label': -1}]
            smooth_A = pd.DataFrame(smooth_A)
            frames.append(smooth_A)
        else:
            # print("i>>", i)
            smooth_A = [{'current': (data1['current'][shape_data1 -1] + data2['current'][i]) / 2, 'label': -1}]
            smooth_A = pd.DataFrame(smooth_A)
            frames.append(smooth_A)
        # print(type(smooth_A))
        # dicts = [{'current':}, {}]

        frames.append(B)
    data_concatenate = pd.concat(frames, ignore_index=True)
    return data_concatenate


def norm_(x, min_, max_):
    return (x-min_)/(max_ - min_)


def normalize(datastream):
    """
    Normalize the data stream to [0,1]
    :param datastream: The data stream to normalize
    :return: The normalized data stream
    """
    min_ = np.min(datastream['current'])
    max_ = np.max(datastream['current'])
    for index, row in datastream.iterrows():
        datastream = datastream.replace(row['current'], norm_(row['current'], min_, max_))
    return datastream


def remove_outlier(datastream, normalized=None):
    """
    Remove outlier in the current  normalized datastream
    :param datastream: The datastream to clean
    :param normalized : A boolean, which indicate if the datastream is already normalized or not
    :return: The Clean Dataset with remove outlier (points outside [mean - 3*sigma, mean + 3*sigma]
    """
    mean_data = np.mean(datastream['current'])  # Mean value of the data
    var_data = np.std(datastream['current'])  # Variance of the data
    if normalized:
        input_data_copy = datastream.copy()
    elif normalized is None:
        input_data_copy = normalize(datastream)

    # for j in range(input_data_copy['current'].shape[0]):
    #     if mean_data + 3 * var_data < input_data_copy['current'][j] < mean_data - 3 * var_data:
    #         # a.append(j)
    #         input_data_copy = input_data_copy.drop([j])  # Remove the corresponding rows
    for index, row in input_data_copy.iterrows():
        if mean_data + 3 * var_data < row['current'] < mean_data - 3 * var_data:
            input_data_copy = input_data_copy.drop(index)  # Remove the corresponding rows
    return input_data_copy


if __name__ == '__main__':
    filepath_bf = 'Hexapod_dataset/black_flat/f_b_1.pow'  # Black_flat
    filepath_br = 'Hexapod_dataset/black_rough/t_b_1.pow'  # Black_rough
    filepath_cu = 'Hexapod_dataset/cubes/t_c_2.pow'  # Cube
    filepath_flat = 'Hexapod_dataset/flat/f_p_2.pow'  # Cube
    filepath_gf = 'Hexapod_dataset/grass_flat/f_g_1.pow'  # Grass_flat
    filepath_gr = 'Hexapod_dataset/grass_rough/t_g_1.pow'  # Grass_rough

    # result = load_data(filename)

    # Read the Pow files
    pow_bf = readfile_pow(filepath_bf)  # Time, Voltage , Current respectively
    pow_br = readfile_pow(filepath_br)  # Time, Voltage , Current respectively
    pow_cu = readfile_pow(filepath_cu)  # Time, Voltage , Current respectively
    pow_flat = readfile_pow(filepath_flat)  # Time, Voltage , Current respectively
    pow_gf = readfile_pow(filepath_gf)  # Time, Voltage , Current respectively
    pow_gr = readfile_pow(filepath_gr)  # Time, Voltage , Current respectively

    # Extract the current from each pow reading
    current_bf = pow_bf[:, 2]
    current_br = pow_br[:, 2]
    current_cu = pow_cu[:, 2]
    current_flat = pow_flat[:, 2]
    current_gf = pow_gf[:, 2]
    current_gr = pow_gr[:, 2]

    # Converting all the current data into a pandas dataframe
    current_bf_pd = pd.DataFrame(current_bf, columns=['current'])
    current_br_pd = pd.DataFrame(current_br, columns=['current'])
    current_cu_pd = pd.DataFrame(current_cu, columns=['current'])
    current_flat_pd = pd.DataFrame(current_flat, columns=['current'])
    current_gf_pd = pd.DataFrame(current_gf, columns=['current'])
    current_gr_pd = pd.DataFrame(current_gr, columns=['current'])

    # Adding the label to end of each current measurement
    # 0 : Black_flat, 1: Black_rough, 2 : Cubes, 3: flat, 4: grass-flat, 5: grass-rough
    shape_nber_bf = current_bf_pd.shape
    shape_nber_br = current_br_pd.shape
    shape_nber_cu = current_cu_pd.shape
    shape_nber_flat = current_flat_pd.shape
    shape_nber_gf = current_gf_pd.shape
    shape_nber_gr = current_gr_pd.shape
    zero_elt = np.zeros(shape_nber_bf, dtype=np.int8)
    one_elt_br = np.ones(shape_nber_br, dtype=np.int8)
    elt_cu = 2*np.ones(shape_nber_cu, dtype=np.int8)
    elt_flat = 3*np.ones(shape_nber_flat, dtype=np.int8)
    elt_gf = 4*np.ones(shape_nber_gf, dtype=np.int8)
    elt_gr = 5*np.ones(shape_nber_gr, dtype=np.int8)

    current_bf_pd['label'] = zero_elt
    current_br_pd['label'] = one_elt_br
    current_cu_pd['label'] = elt_cu
    current_flat_pd['label'] = elt_flat
    current_gf_pd['label'] = elt_gf
    current_gr_pd['label'] = elt_gr

    # Normalization of the current between [0, 1]
    window_size = 10
    A = current_bf_pd[1:window_size]
    B = current_br_pd[1:window_size]
    # for index, row in A.iterrows():
    #     print(row['current'])
    # print(A)
    frames = []

    # for i in range(0, 100, window_size):
    #     A = current_bf_pd[i:i+window_size]
    #     B = current_br_pd[i:i+window_size]
    #     # print(type(A))
    #     frames.append(A)
    #     smooth_A = [{'current':(current_bf_pd['current'][i-1+window_size] + current_br_pd['current'][i])/2, 'label':-1}]
    #     smooth_A = pd.DataFrame(smooth_A)
    #     # print(type(smooth_A))
    #     # dicts = [{'current':}, {}]
    #     frames.append(smooth_A)
    #     frames.append(B)
    # C = pd.concat(frames, ignore_index=True)


    # print(C)
    # current_bf_pd.set_index(['current', 'label'])
    # Normalize the data
    # current_bf_pd
    # D = normalize(C)
    # print(D)

    # # Removing Outliers
    start = datetime.now()
    # current_br_pd = remove_outlier(current_br_pd)
    current_bf_pd = remove_outlier(current_bf_pd)
    current_flat_pd = remove_outlier(current_flat_pd)
    # current_cu_pd = remove_outlier(current_cu_pd)
    # current_gf_pd = remove_outlier(current_gf_pd[1:200])
    current_gr_pd = remove_outlier(current_gr_pd
                                   )
    print(type(current_gr_pd))
    end = datetime.now() - start
    print("outlier start : {}, end : {} ".format(start, end))

    # # Concatenation
    start = datetime.now()
    CP1 = concatenate_data(current_gr_pd, current_flat_pd)  # Black Flat vs flat
    # CP1 = concatenate_data(current_gf_pd, CP1)  # Black Flat vs flat vs Grass Flat
    end = datetime.now() - start
    print("Concatenation start : {}, end : {} ".format(start, end))
    print(CP1.shape[0])
    # print(current_cu_pd.shape[0])
    plt.figure()
    plt.plot(CP1['current'][1:200])
    plt.show()

    print("\n\n\n")
    # # Detect Drift
    # start = datetime.now()
    # kolmogorov_smirnov(CP1, window_size=1024)
    # end = datetime.now() - start
    # print("Drift detection start : {}, end : {} ".format(start, end))

    print("\n\n\n")
    print(current_bf_pd.shape[0])
    print(current_flat_pd.shape[0])
    print(current_gf_pd.shape[0])
    print("rough")
    print(current_br_pd.shape[0])
    print(current_gr_pd.shape[0])

    # print(np.max(current_br_pd['current']))
    # print(current_cu_pd)
    # print(current_gf_pd)
    # print(current_gr_pd)
    # print(current_bf_pd)
    # plt.plot(result_pd['current'][1:500])
    # plt.plot(pow[:,0])
    # plt.show()
    # print(result_pd)
