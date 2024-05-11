import pandas as pd
import numpy as np
import math


def remove_nan_strat_end(df, attr):
    index_start = 0
    index_end = 0
    if attr in list(df):
        for i in range(len(df) - 1):
            if np.isnan(df[attr][i]) == False:
                index_start = i
                break
        for j in range(len(df) - 1, 0, -1):
            if np.isnan(df[attr][j]) == False:
                index_end = j + 1
                break
        df = df[index_start:index_end]
        df = df.reset_index(drop=True)
        print("Preprocessing: Remove nan successfully!")
        return df
    else:
        print('Unable to remove: No Attibutes Found\n')
        return 0


# fill in nan values with "1"
# TODO: use average value to fill in missing CGMs 
def filling_CGM(testpath):
    filepath = "./data/processedcsv/ohio540.csv"
    a_test = pd.read_csv(filepath, usecols=['CGM'])
    a_test = remove_nan_strat_end(a_test, 'CGM')  
    M, N = a_test.shape           # M = 16175, N =1 
    AA = a_test['CGM'].fillna(1)
    return AA
    
    
    
    
    
    # Original code with normalization 
    # # AA[AA > 1] = 0 
    # AA = np.array(AA).reshape((-1, 1)) #ensure AA is only 1 col
    # a_test = np.array(a_test)
    # zero = np.zeros(M).reshape((-1, 1))
    # a_test = np.c_[zero, a_test] #concatenating zero horizontally to the left of a_test
    # AA = np.c_[zero, AA]

    # zero = np.zeros(N + 1).reshape((1, -1))
    # a_test = np.r_[zero, a_test]
    # zero = np.zeros(N).reshape((1, -1))
    # print("Shape of zero array:", zero.shape)
    # print("Shape of AA array:", AA.shape)
    # input()
    # AA = np.r_[zero, AA]
   




    # BB = np.zeros((M + 1, N + 1))
    # # print(a_test.shape, AA.shape, BB.shape)

    # for i in range(2, M + 1):
    #     if AA[i, 1] != 0:   # AA: i-row, 1st col
    #         BB[i, 2] = AA[i, 1] + BB[i - 1, 2]
    # for i in range(M, 1, -1):
    #     if AA[i, 1] != 0:
    #         BB[i, 1] = AA[i, 1] + BB[i + 1, 1]

    # for i in range(2, M + 1):
    #     if AA[i, 1] != 0:
    #         num = i % 288
    #         den = math.floor(i / 288.0)
    #         if BB[i, 2] < 4:
    #             a_test[i, 1] = 2 * a_test[i - 1, 1] - a_test[i - 2, 1]
    #         else:
    #             a_test[i, 1] = a_test[i, 2]
    #             AA = np.isnan(a_test) + 0
    #             AA = AA[:, 0:2]
    #             for j in range(2, M + 1):
    #                 if AA[j, 1] != 0:
    #                     BB[j, 2] = AA[j, 1] + BB[j - 1, 2]
    #             y1 = 2 * a_test[i - 1, 1] - a_test[i - 2, 1]
    #             s = np.mean(a_test[1:i, 1])
    #             t = den + 1
    #             ss = np.ones((2 * t - 1, 1)) * s
    #             for n in range(1, den + 1):
    #                 r = (n - 1) * 288 + num
    #                 if r == 0:
    #                     r = 1
    #                 if den >= 1:
    #                     ss[n + den + 1 - 1] = a_test[r, 1]
    #                 else:
    #                     ss[n + den + 1 - 1] = s
    #             y2 = np.nanmean(ss)
    #             a_test[i, 1] = y1 * pow(0.999, int(BB[i, 2])) + y2 - y2 * pow(0.999, int(BB[i, 2]))
    #         a_test[i, 1] = min(400, a_test[i, 1])
    #         a_test[i, 1] = max(40, a_test[i, 1])
    #     if AA[i - 1, 1] != 0 and AA[i, 1] == 0:
    #         if BB[i - 1, 2] >= 50:
    #             for k in range(1, 42 + 1):
    #                 yy1 = 2 * a_test[i - k + 1, 1] - a_test[i - k + 2, 1]
    #                 a_test[i - k, 1] = yy1 * pow(0.99, k) + a_test[i - k, 1] - a_test[i - k, 1] * pow(0.99, k)
    #         else:
    #             if BB[i - 1, 2] >= 12:
    #                 for k in range(1, int(BB[i - 1, 2]) + 1):
    #                     yy1 = 2 * a_test[i - k + 1, 1] - a_test[i - k + 2, 1]
    #                     a_test[i - k, 1] = yy1 * pow(0.99, k) + a_test[i - k, 1] - a_test[i - k, 1] * pow(0.99, k)
    #         a_test[i, 1] = min(400, a_test[i, 1])
    #         a_test[i, 1] = max(40, a_test[i, 1])
    # bg1 = a_test[1:, 1]
    # return bg1
