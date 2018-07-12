import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot  as plt

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
if __name__ == '__main__':
    filepath_bf = 'Hexapod_dataset/black_flat/f_b_1.pow'  # Black_flat
    filepath_br = 'Hexapod_dataset/black_rough/t_b_1.pow'  # Black_rough
    filepath_cu = 'Hexapod_dataset/cubes/t_c_2.pow'  # Cube
    filepath_gf = 'Hexapod_dataset/grass_flat/f_g_1.pow'  # Grass_flat
    filepath_gr = 'Hexapod_dataset/grass_rough/t_g_1.pow'  # Grass_rough

    # result = load_data(filename)

    # Read the Pow files
    pow_bf = readfile_pow(filepath_bf)  # Time, Voltage , Current respectively
    pow_br = readfile_pow(filepath_br)  # Time, Voltage , Current respectively
    pow_cu = readfile_pow(filepath_cu)  # Time, Voltage , Current respectively
    pow_gf = readfile_pow(filepath_gf)  # Time, Voltage , Current respectively
    pow_gr = readfile_pow(filepath_gr)  # Time, Voltage , Current respectively

    # Extract the current from each pow reading
    current_bf = pow_bf[:, 2]
    current_br = pow_br[:, 2]
    current_cu = pow_cu[:, 2]
    current_gf = pow_gf[:, 2]
    current_gr = pow_gr[:, 2]

    # Converting all the current data into a pandas dataframe
    current_bf_pd = pd.DataFrame(current_bf, columns=['current'])
    current_br_pd = pd.DataFrame(current_br, columns=['current'])
    current_cu_pd = pd.DataFrame(current_cu, columns=['current'])
    current_gf_pd = pd.DataFrame(current_gf, columns=['current'])
    current_gr_pd = pd.DataFrame(current_gr, columns=['current'])

    # Adding the label to end of each current measurement
    # 0 : Black_flat, 1: Black_rough, 2 : Cubes, 3: grass-flat, 4: grass-rough
    shape_nber_bf = current_bf_pd.shape
    shape_nber_br = current_br_pd.shape
    shape_nber_cu = current_cu_pd.shape
    shape_nber_gf = current_gf_pd.shape
    shape_nber_gr = current_gr_pd.shape
    zero_elt = np.zeros(shape_nber_bf, dtype=np.int8)
    one_elt_br = np.ones(shape_nber_br, dtype=np.int8)
    elt_cu = 2*np.ones(shape_nber_cu, dtype=np.int8)
    elt_gf = 3*np.ones(shape_nber_gf, dtype=np.int8)
    elt_gr = 4*np.ones(shape_nber_gr, dtype=np.int8)

    current_br_pd['label'] = one_elt_br
    current_cu_pd['label'] = elt_cu
    current_gf_pd['label'] = elt_gf
    current_gr_pd['label'] = elt_gr

    # Normalization of the current between [0, 1]
    current_bf_pd.set_index(['current', 'label'])

    print(np.max(current_br_pd['current']))
    # print(current_cu_pd)
    #print(current_gf_pd)
    #print(current_gr_pd)
    # print(current_bf_pd)
    # plt.plot(result_pd['current'][1:500])
    # plt.plot(pow[:,0])
    # plt.show()
    # print(result_pd)
