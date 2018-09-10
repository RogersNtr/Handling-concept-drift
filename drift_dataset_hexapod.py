import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot  as plt
from datetime import datetime
import scipy.stats as stat
from PageHinkley import *
from Page_Hinkley import *
# from ADWIN_V1.adwin import *
from test_ADWIN.adwin import *
from plotting import *
from cumsum import *


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


def concatenate_data(data1, data2, window_size=200):
    """
    Concatenate two data
    :param data1: data to concatenate
    :param data2: second data to concatenate
    :param window_size: 1024 by default, it is the size of the scanning window.
    :return: the data already concatenate
    """
    frames = []
    print("\n\n concatenation start....")
    shape_data1 = data1.shape[0]
    # print("shape data 1", shape_data1)
    shape_data2 = data2.shape[0]
    # print("shape data 2", shape_data2)
    max_length_data1_data2 = max(shape_data1, shape_data2)
    # print("max length", max_length_data1_data2)
    if window_size <= max_length_data1_data2:
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
            if A.shape[0] !=0:
                frames.append(A)
            if i-1+window_size < shape_data1:
                # print("i", i)
                # print("data2__ Concatenate_data", data2['current'][i])
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

            if B.shape[0] != 0:
                frames.append(B)

    else:
        print("change the window size")
    # print("len frames", frames)
    if len(frames) !=0:
        data_concatenate = pd.concat(frames, ignore_index=True)
    else:
        return
    return data_concatenate


def kolmogorov_smirnov(data, window_size=100):
    """
    The function is the Kolmogorov smirnov test, that use the
    :param data: Column vector
    :param window_size: Size of the Scanning Window
    :return: True, False (True : Drift Present, False : Drift Absent)
    """
    # W0 = data[1:window_size]
    global drift_rejected
    num = 0
    num_iter = 0
    data_length = data.shape[0]
    print("datalength", data_length)
    for t in range(0, data_length, window_size):
        num_iter += 1
        data_ = []
        # print("t", t)
        # Splitting the data recursively in two using a sliding window

        sample1 = data[t:t+window_size]

        if t+window_size < data_length:
            if t+ 2*window_size < data_length:
                sample2 = data[t+window_size:t + 2*window_size]
            else:
                sample2 = data[t + window_size:data_length]
        else:
            # print("ca marche")
            break

        current_sample1 = sample1['current']
        current_sample2 = sample2['current']

        # # --->Mean and std of the sample 1
        mean_samp1e1 = current_sample1.mean()  # Mean of the second sliding window.
        std_sample1 = current_sample1.std()  # Standard deviation

        # #  ---> Mean and std of the sample 2
        mean_samp1e2 = current_sample2.mean()
        std_sample2 = current_sample2.std()

        # # Normalization of the value of the samples
        mean_samp1e1_pd = pd.DataFrame(mean_samp1e1*np.ones((sample1.shape[0], 1)))
        # mean_samp1e2_pd = pd.DataFrame(mean_samp1e2*np.ones((sample2.shape[0], 1)))
        mean_samp1e2_pd = mean_samp1e2*np.ones((sample2.shape[0], 1))

        sample1_sub = mean_samp1e1_pd.sub(current_sample1, axis=0)
        norm_s1 = - sample1_sub.div(std_sample1)

        # Using numpy in order to avoid the need for reindexing the indexes of the DataFrame,
        val_current2 = current_sample2.values
        val_current2 = val_current2.reshape(val_current2.shape[0], 1)
        sample2_sub = val_current2 - mean_samp1e2_pd
        sample2_sub = pd.DataFrame(sample2_sub)
        norm_s2 = - sample2_sub.div(std_sample2)

        # Transform to numpy(needed by the ks_2samp function
        val1 = norm_s1.values
        val2 = norm_s2.values
        D_stat, p_value = stat.ks_2samp(val1[:, 0], val2[:, 0])

        drift_rejected = 0   # The number of hypothesis rejected
        # print("Result P vlaue", p_value)
        if p_value < 0.05:  # D_stat > 0.04301p_value < 0.05:  # We reject the Null Hypothesis, so Drift detected
            drift = True
            num = num + 1
            if t + window_size < data_length:
                print("Drift detected between {} to {} and {} to {}".format(t, t+window_size-1, t+window_size,  t+2*window_size))
                # print("KS-test: drift at {}".format(t))
            else:
                print("Drift detected between {} to {} and {} to {}".format(t, t + 2*window_size - 1, t,
                                                                            data_length))
        else:
            drift = False
            drift_rejected +=1
            # print("t value..........{} and data length {}".format(t+window_size, t + 2*window_size))
    print("{} drifts detected using KS-test".format(num))
    print("{} iteration in ks test".format(num_iter))
    print("{} hypothesis rejected".format(drift_rejected))
    return drift


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


def run_drift_detection(data_size, confidence_level):
    return True


def get_actual_data(i):
    name_actual_data = "DP" + str(i)

    if name_actual_data == "DP1":
        now_data = DP1
        title_ = name_actual_data + ":{0, 3, 4}" + " (Black Flat vs Flat vs Grass Flat)"
    elif name_actual_data == "DP2":
        now_data = DP2
        title_ = name_actual_data + ":{1, 2, 5}" + " (Black Rough vs Wooden Cubes vs Grass Rough)"
    elif name_actual_data == "DP3":
        now_data = DP3
        title_ = name_actual_data + ":{2, 3}" + " (Wooden Cubes vs Flat)"
    elif name_actual_data == "DP4":
        now_data = DP4
        title_ = name_actual_data + ":{0, 5}" + " (Black Flat vs Grass Rough)"
    elif name_actual_data == "DP5":
        now_data = current_bf_pd
        title_ = name_actual_data + ":{0}" + " (Black Flat)"
    else:
        exit(5)

    return now_data, title_

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
    # print(type(current_bf))

    # Converting all the current data into a pandas dataframe
    current_bf_pd = pd.DataFrame(current_bf, columns=['current'])
    current_br_pd = pd.DataFrame(current_br, columns=['current'])
    current_cu_pd = pd.DataFrame(current_cu, columns=['current'])
    current_flat_pd = pd.DataFrame(current_flat, columns=['current'])
    current_gf_pd = pd.DataFrame(current_gf, columns=['current'])
    current_gr_pd = pd.DataFrame(current_gr, columns=['current'])

    # print("Shape bf", current_bf_pd.shape)
    # print("Shape br", current_br_pd.shape)
    # print("Shape cu", current_cu_pd.shape)
    # print("Shape flat", current_flat_pd.shape)
    # print("Shape gf", current_gf_pd.shape)
    # print("Shape gr", current_gr_pd.shape)

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
    # window_size = 200  # Window size on the data(only a subset)
    # A = current_bf_pd[1:window_size]
    # B = current_br_pd[1:window_size]
    # for index, row in A.iterrows():
    #     print(row['current'])
    # print(A)
    # frames = []

    # print(C)
    # current_bf_pd.set_index(['current', 'label'])
    # Normalize the data
    # current_bf_pd
    # D = normalize(C)
    # print(D)



    #################################################################
    #   Experiments for detecting drift
    #################################################################

    # # define  the size of the data, we want to consider and the window size for concatenation
    data_sizes = [200, 2000, 5000]; window_size = 200;rand_data = np.random.randint(0, 2) # rand for choosing a size

    choose_size = data_sizes[0]
    # # Removing Outliers and inner normalization
    print("removing outliers start....")
    start = datetime.now()
    current_br_pd = remove_outlier(current_br_pd[:choose_size])

    current_bf_pd = remove_outlier(current_bf_pd[:choose_size])
    current_flat_pd = remove_outlier(current_flat_pd[:choose_size])
    current_cu_pd = remove_outlier(current_cu_pd[:choose_size])
    current_gf_pd = remove_outlier(current_gf_pd[:choose_size])
    current_gr_pd = remove_outlier(current_gr_pd[:choose_size])
    end = datetime.now() - start
    print("outlier start : {}, end : {} ".format(start, end))

    # # Concatenation


    # Plot the current  for each terrain

    for i in range(7):
        if i == 0:
            actual_current = current_bf_pd
            title = "current on Black Flat(" + str(i) + ") terrain"
        elif i == 1:
            actual_current = current_br_pd
            title = "current on Black Rough(" + str(i) + ") terrain"
        elif i == 2:
            actual_current = current_cu_pd
            title = "current on Cubes(" + str(i) + ") terrain"
        elif i == 3:
            actual_current = current_flat_pd
            title = "current on Flat(" + str(i) + ") terrain"
        elif i == 4:
            actual_current = current_gf_pd
            title = "current on Grass Flat(" + str(i) + ") terrain"
        elif i == 5:
            actual_current = current_gr_pd
            title = "current on Grass Rough(" + str(i) + ") terrain"
        else:
            print("something wrong happen when ploting each current terrain")

        plt.figure()
        plt.xlabel("time steps")
        plt.ylabel("current")
        plt.plot(actual_current['current'])
        plt.title(title)
        plt.savefig("figure/" + "Terrain{}".format(i) + ".png")

    # # # DP1 : Flat i.e {0, 3, 4}
    start = datetime.now()
    flat = concatenate_data(current_bf_pd, current_flat_pd, window_size=window_size)  # Black Flat vs flat
    DP1 = concatenate_data(current_gf_pd, flat, window_size=200)  # Black Flat vs flat vs Grass Flat
    print("shapeDP1", DP1.shape)
    # print(DP1[400:460])

    # # # DP2 : {1, 2, 5}
    rough = concatenate_data(current_br_pd, current_cu_pd)  # Black Rough vs Wooden
    DP2 = concatenate_data(current_gr_pd, rough)
    print("shapeDP2", DP2.shape)
    end = datetime.now() - start

    # # # DP3 : {2, 3} Wooden vs flat
    DP3 = concatenate_data(current_cu_pd, current_flat_pd)
    print("shapeDP3", DP1.shape)

    # # # DP4 : {0, 5} Black Flat vs grass Rough
    DP4 = concatenate_data(current_bf_pd, current_gr_pd)
    print("shapeDP4", DP4.shape)
    print("Concatenation start : {}, end : {} ".format(start, end))

    # # # Run the drift detection over DPi, i =  1,...3
    start = datetime.now()
    delta_adwin = [0.001, 0.3, 2, 4]  # Values of delta for ADWIN_V1 (confidence value)
    min_len_win = [5, 10, 20, 32]
    adwin = Adwin(delta=1)
    delta_hinkley = [0.00005, 0.03, 0.6, 0.9]  # Different delta for the PH test (magintude of changes)
    lambda_hinkley = [5, 25, 50, 75, 100]  # Different lambda Threshold for the PH test
    ks_window_sizes = [5, 15, 50, 100]  # Different window size for
    adwin_min_clock = [5, 10, 20, 50, 100]
    # # # # # # #


    cusum1 = cumsumDM()

    ############################################
    #           ADWIN_V1 results                  #
    ############################################
    # print(DP1[0:200])
    print("######Size of the data {}".format(choose_size))
    for delta_i in delta_adwin:
        # adwin = Adwin(delta=delta_i, max_buckets=5, min_clock=5, min_length_window=5, min_length_sub_window=1)
        rand_nber = np.random.randint(0, 5)
        min_clock_val = adwin_min_clock[1]
        adwin = Adwin(delta=delta_i, max_buckets=5, min_clock=min_clock_val, min_length_window=5, min_length_sub_window=1)
        true_drift = 0
        false_drift = 0
        # actual_data = DP1
        print("#######################################################Result for delta = {} and min_clock = {}".format(delta_i, min_clock_val))
        for i in range(1, 6):
            actual_data, title_curve = get_actual_data(i)

            # print("ADWIN_V1, true positive : {}".format(true_drift))
            # # # # ------> ADWIN_V1
            # print("\n\n")
            print("\nADWIN start for {}.....".format(title_curve))
            index_drift = 0
            actual_data = actual_data['current']
            for dat in actual_data:
                index_drift += 1
                if adwin.set_input(dat):
                    print("ADWIN_V1: drift at {}".format(index_drift))
                    if i == 1:
                        if (150<= index_drift <=250) or (375<= index_drift <=460):
                            true_drift+=1
                        else:
                            false_drift+=1
                    elif i == 2:
                        if (150<= index_drift <=260) or (375<= index_drift <=420):
                            true_drift+=1
                        else:
                            false_drift+=1
                    elif i == 3:
                        if 150<= index_drift <=250:
                            true_drift+=1
                        else:
                            false_drift+=1
                    elif i == 4:
                        if 150 <= index_drift <=250:
                            true_drift+=1
                        else:
                            false_drift+=1
                    elif i == 5:
                        false_drift+=1
                elif i == 5:
                    true_drift+=1
            print("ADWIN_V1 : True Positive {}, False Positive {}".format(true_drift, false_drift))
            true_drift = 0
            false_drift = 0
                # else:
                #     print("pas de drift")

    ############################################
    #           Page-Hinkley Test results      #
    ############################################

    print("\n######-------------> Page-Hinley test <-----------#########\n")
    for lambda_i in lambda_hinkley:
        rand_delta_hinkley = np.random.randint(0, 4)  # A random number to choose a value of delta for the PH-test
        delta_test_ph = 0.01
        PH_ = Hinkley_test(delta=delta_test_ph, lambda_=lambda_i,
                           alpha=1 - 0.0001)  #1 - 0.0001
        # PH_2 = PH_test()
        true_drift = 0
        false_drift = 0
        print("#######################################################Result for lambda = {} and delta : {}".format(lambda_i, delta_test_ph))
        for i in range(1, 6):
            actual_data, title_curve = get_actual_data(i)
            print("PH-test start for {}.....".format(title_curve))
            index_drift = 0
            actual_data = actual_data['current']
            # PH_2 = PH_test(actual_data, delta_=delta_test_ph, lambda_=lambda_i, alpha_=1 - 0.0001)
            for dat1 in actual_data:
                index_drift += 1
                if PH_.set_data(dat1):
                    print("Page Hinkley: drift at {}".format(index_drift))
                    if i == 1:
                        if (150 <= index_drift <= 250) or (375 <= index_drift <= 460):
                            true_drift += 1
                        else:
                            false_drift += 1
                    elif i == 2:
                        if (150 <= index_drift <= 260) or (375 <= index_drift <= 420):
                            true_drift += 1
                        else:
                            false_drift += 1
                    elif i == 3:
                        if 150 <= index_drift <= 250:
                            true_drift += 1
                        else:
                            false_drift += 1
                    elif i == 4:
                        if 150 <= index_drift <= 250:
                            true_drift += 1
                        else:
                            false_drift += 1
                    elif i == 5:
                        false_drift += 1
                elif i == 5:
                    true_drift += 1
            print("PH-test : True Positive {}, False Positive {}".format(true_drift, false_drift))
            true_drift = 0
            false_drift = 0
                # else:
                #     print("pas de drift")

    ############################################
    #           K-S Test results               #
    ############################################
    print("\n######-------------> KS test <-----------#########\n")
    for win_i in ks_window_sizes:
        rand_delta_hinkley = np.random.randint(0, 4)  # A random number to choose a value of delta for the PH-test
        true_drift = 0
        false_drift = 0
        print("#######################################################Result for window_size = {}".format(win_i))
        for i in range(1, 6):
            actual_data, title_curve = get_actual_data(i)
            print("\nKS-test start for {}.....".format(title_curve))
            index_drift = 0
            kolmogorov_smirnov(actual_data, window_size=win_i)

    # for win in ks_window_sizes:
    #         # ploting the concatenate current
    #         # plotting the concatenations
    #         # plt.figure()  # plot in dust
    #         # plt.xlabel("time steps")
    #         # plt.ylabel("current")
    #         # plt.plot(actual_data['current'])
    #         # plt.title(title)
    #         # plt.savefig("figure/" + name_actual_data + ".png")
    #
    #         # # # # ------> KS test
    #         kolmogorov_smirnov(actual_data, window_size=100)
    #
    #
    #
    #         print("ADWIN_V1, true positive : {}".format(true_drift))
    #         # # # # ------> Page-Hinkley test
    #         # print("\n\n")
    #         print("\nPH test start....\n.")
    #         index_drift = 0
    #         for dat1 in actual_data:
    #             index_drift += 1
    #             if PH_.set_input(dat1):
    #                 print("Page Hinkley: drift at {}".format(index_drift))
    #              # else:
    #              #     print("pas de drift")

    end = datetime.now() - start
    print("All drifts detected in {}".format(end))


    # # # DP2 :
    # plt.title("Abrupt Drift")
    # plt.xlabel("black Flat")
    # plt.ylabel("black Rough")

    # plt.plot(CP1['current'][0:current_subset_bf.shape[0]])
    # plt.plot(CP1['current'])
    # plt.show()

    # print("\n\n\n")
    # # # Detect Drift
    # start = datetime.now()
    # kolmogorov_smirnov(DP1, window_size=200)
    # end = datetime.now() - start
    # print("Drift detection start : {}, end : {} ".format(start, end))

    # print("\n\n\n")
    # print(current_bf_pd.shape[0])
    # print(current_flat_pd.shape[0])
    # print(current_gf_pd.shape[0])
    # print("rough")
    # print(current_br_pd.shape[0])
    # print(current_gr_pd.shape[0])

    # # Testing the PageHinkley Algorithm
    # print(CP1["current"][300:350])
    # plt.plot(CP1["current"][319], '*')
    # plt.show()
    # DP1 = DP1["current"]
    # print(type(CP1))
    # PH_ = Hinkley_test()
    num=0
    A = [1, 1, 4, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print("A", A[:3])
    # for i in A:
    #     num+=1
    #     if PH_.set_input(i):
    #         print(num)
    #         print("Here is a drift")
    #     else:
    #         print("pas de drift")
    # page_hinkley = PH_test(CP1, delta_=0.005, lambda_=1, alpha_=1 - 0.0009)
    # page_hinkley.detect_drift()

    # adwin = AdwinAlgo(5)
    data_stream = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.7]

    # adwin = Adwin(delta=1)# , max_buckets=20, min_length_window=50, min_length_sub_window=20)

    # adwin = Adwin(delta=1, max_buckets=5)
    # adwin = Adwin(delta=0.1, max_buckets=5, min_clock=4, min_length_window=3, min_length_sub_window=2)
    # h=0
    # for i in DP1:
    #     h += 1
    #     if adwin.set_input(i):
    #         print("Here is a drift ", h)
    # for data in CP1:
    #     if adwin.update(data):
    #         print("Change has been detected in data: " + str(data))
    #         print(adwin.get_estimation())  # Prints the next value of the estimated form of data_stream
    #     else:
    #         print(adwin.get_estimation())
    #         print("Nodrfit Detected")
    # page_hinkley(CP1)

    # # Testing the drift detection on the dataset

    # 1. Flat vs Wodden


    # 2. Flat vs Rough

    # 3. Grass vs Black

    # 4. Grass vs Black vs Wodden
    # cusum1 = cumsumDM()
    # h=0
    # for i in data_stream:
    #     h += 1
    #     if cusum1.input(i):
    #         print("Here is a drift cusum ", h)

    # print(np.max(current_br_pd['current']))
    # print(current_cu_pd)
    # print(current_gf_pd)
    # print(current_gr_pd)
    # print(current_bf_pd)
    # plt.plot(result_pd['current'][1:500])
    # plt.plot(pow[:,0])
    # plt.show()
    # print(result_pd)

    # # testing pageHinkley algorithm