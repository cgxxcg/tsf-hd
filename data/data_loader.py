import os
import warnings
import copy

import data.missing_CGM_data_filling as missing_CGM
from torch.utils.data import Dataset, DataLoader



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features
from utils.tools import StandardScaler

warnings.filterwarnings("ignore")


class Dataset_ohio(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",  #single/univariate task
        data_path="ohio540.csv",
        target="CGM",
        scale=False,
        inverse=False,
        timeenc=0,  #time features
        freq="t",   
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()
        full_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(full_path)

        # border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        # border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1s = [0, 4 * 30 * 24 - self.seq_len, 5 * 30 * 24 - self.seq_len] #3 borders: split to two chunks based on row (time)
        border2s = [4 * 30 * 24, 5 * 30 * 24, 20 * 30 * 24]
        # print("border1s",border1s) #[0, 2784, 3504]
        # print("border2s", border2s) #[2880, 3600, 14400]

        border1 = border1s[self.set_type] #set_type = 0,1,2 train test val
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            #newly added: preprocess data
            df_data = missing_CGM.remove_nan_strat_end(df_data, 'CGM')
            df_data = df_data.fillna(0)
            bgorg = copy.deepcopy(df_data['CGM'])
            #fill in CGM
            df_data['CGM'] = missing_CGM.filling_CGM(df_data)
            
            
        #ohio: (single variable-S)
        elif self.features == "S":
            df_data = df_raw[[self.target]] #yes, select a single column (CGM)
            #newly added: preprocess data
            df_data = missing_CGM.remove_nan_strat_end(df_data, 'CGM')
            df_data = df_data.fillna(0)
            # bgorg = copy.deepcopy(df_data['CGM'])
            
            #fill in CGM with 1
            df_data['CGM'] = missing_CGM.filling_CGM(df_data) #type:Series
            

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values  #data = all CGM values
        
        if self.timeenc == 2:
            train_df_stamp = df_raw[["Time"]][border1s[0] : border2s[0]]  #[["date"]] 
            train_df_stamp["Time"] = pd.to_datetime(train_df_stamp.Time, format="%d-%b-%Y %H:%M:%S") #pd.to_datetime(train_df_stamp.date)
            train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc, freq=self.freq)
            date_scaler = sklearn_StandardScaler().fit(train_date_stamp)

            df_stamp = df_raw[["Time"]][border1:border2] #choose a segment from Time col
            df_stamp["Time"] = pd.to_datetime(df_stamp.Time, format="%d-%b-%Y %H:%M:%S") #change it to datetime: pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            data_stamp = date_scaler.transform(data_stamp)
        else:
            df_stamp = df_raw[["Time"]][border1:border2] #[0:2880]
            # print("dfstamp before\n", df_stamp.tail(5))
            df_stamp["Time"] = pd.to_datetime(df_stamp.Time, format="%d-%b-%Y %H:%M:%S") #pd.to_datetime(df_stamp.date)
           # print("dfstamp todatetime\n", df_stamp.tail(5))
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            


        self.data_x = data[border1:border2] #[0:2880]
        
        if self.inverse:
            self.data_y = df_data.values[border1:border2]  #[0:2880]
        else:
            self.data_y = data[border1:border2] #[0:2880]

        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="h",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        # border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1s = [0, 4 * 30 * 24 - self.seq_len, 5 * 30 * 24 - self.seq_len]
        border2s = [4 * 30 * 24, 5 * 30 * 24, 20 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values

        if self.timeenc == 2:
            train_df_stamp = df_raw[["date"]][border1s[0] : border2s[0]]
            train_df_stamp["date"] = pd.to_datetime(train_df_stamp.date)
            train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc)
            date_scaler = sklearn_StandardScaler().fit(train_date_stamp)

            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            data_stamp = date_scaler.transform(data_stamp)
        else:
            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # if self.set_type == 2: pdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="t",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        # border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1s = [0, 4 * 30 * 24 - self.seq_len, 5 * 30 * 24 - self.seq_len]
        border2s = [4 * 30 * 24, 5 * 30 * 24, 20 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.timeenc == 2:
            train_df_stamp = df_raw[["date"]][border1s[0] : border2s[0]]
            train_df_stamp["date"] = pd.to_datetime(train_df_stamp.date)
            train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc)
            date_scaler = sklearn_StandardScaler().fit(train_date_stamp)

            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            data_stamp = date_scaler.transform(data_stamp)
        else:
            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="h",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        # cols = list(df_raw.columns);

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.2)
        num_test = int(len(df_raw) * 0.75)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            

        df_stamp = df_raw[["date"]][border1:border2]
        
        
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["pred"]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin : r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin : r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


#newly added 
class ohio(Dataset):
    
    
    def __init__(self, csv_file, seq_len, pred_len, scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        
        # Parse the date column with the correct format string
        self.data['Time'] = pd.to_datetime(self.data['Time'], format="%d-%b-%Y %H:%M:%S")
        
        # Sort the dataframe by date
        self.data = self.data.sort_values(by='Time').reset_index(drop=True)
        
        # Scale the data if needed
        if self.scale:
            self.scaler = sklearn_StandardScaler()
            self.data['CGM'] = self.scaler.fit_transform(self.data['CGM'].values.reshape(-1, 1))
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        idx += self.seq_len  # Adjust index to account for sequence length
        
        # Extract sequence and target
        seq_x = self.data.iloc[idx - self.seq_len:idx]['CGM'].values.astype(np.float32)
        seq_y = self.data.iloc[idx:idx + self.pred_len]['CGM'].values.astype(np.float32)
        
        return seq_x, seq_y
    
    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            return data.flatten()

