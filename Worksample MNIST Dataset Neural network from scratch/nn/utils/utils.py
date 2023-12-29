import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class Utils:
    """
    Utils class for data manipulation / miscellaneous operations.

    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    @staticmethod
    def to_one_hot(y, n_classes):
        """
        Convert class indices to one-hot encoded vectors.
        :param y: numpy array of class indices.
        :param n_classes: Total number of classes.
        :return: one-hot encoded representation.
        """
        one_hot = np.zeros((y.size, n_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    @staticmethod
    def prepareData():
        # TODO: Refine???
        def read_idx(filename):
            with open(filename, 'rb') as f:
                magic_number = int.from_bytes(f.read(4), 'big')
                num_items = int.from_bytes(f.read(4), 'big')

                # number for image data
                if magic_number == 2051:
                    num_rows = int.from_bytes(f.read(4), 'big')
                    num_cols = int.from_bytes(f.read(4), 'big')
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    return data.reshape(num_items, num_rows * num_cols)

        def read_labels(filename):
            with open(filename, 'rb') as f:
                magic_number = int.from_bytes(f.read(4), 'big')
                num_items = int.from_bytes(f.read(4), 'big')

                # number for label data
                if magic_number == 2049:
                    return np.frombuffer(f.read(), dtype=np.uint8)

        # read train data
        train_images = read_idx('mnist_data/train-images.idx3-ubyte')
        train_labels = read_labels('mnist_data/train-labels.idx1-ubyte')

        # read test data
        test_images = read_idx('mnist_data/t10k-images.idx3-ubyte')
        test_labels = read_labels('mnist_data/t10k-labels.idx1-ubyte')

        # load train data into Dataframe
        df_train_images = pd.DataFrame(train_images)
        df_train_labels = pd.DataFrame(train_labels, columns=['Label'])

        # load test data into Dataframe
        df_test_images = pd.DataFrame(test_images)
        df_test_labels = pd.DataFrame(test_labels, columns=['Label'])

        # Concatenate the data frames horizontally
        df_train_combined = pd.concat([df_train_labels, df_train_images], axis=1)

        # df_train_combined['Label'] = df_train_combined['Label'].round().astype(int)

        df_test_combined = pd.concat([df_test_labels, df_test_images], axis=1)
        # df_test_combined['Label'] = df_test_combined['Label'].round().astype(int)

        X_train = df_train_combined.drop(columns=['Label'])
        y_train = df_train_combined['Label']

        # Load your test data
        X_test = df_test_combined.drop(columns=['Label'])
        y_test = df_test_combined['Label']

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_normalized = scaler_X.fit_transform(X_train)
        X_test_normalized = scaler_X.transform(X_test)

        # # Standardize the target variable
        # y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        # y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        return X_train_normalized, y_train, X_test_normalized, y_test
