import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def data_config_creator(circuit_config):

    return Data(**circuit_config)


class Data:
    def __init__(self, parameter_list, performance_list, arguments, order, sign,):

        self.arguments = dict(arguments)
        self.performance_list = list(performance_list)
        self.parameter_list = list(parameter_list)

        self.num_params = len(parameter_list)
        self.num_perf = len(performance_list)

        self.order = order
        self.sign = sign

        
class BasePytorchModelDataset(Dataset):
    def __init__(self, performance, parameters):
        self.parameters = np.array(parameters)
        self.performance = np.array(performance)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.performance[index], self.parameters[index]

    def getAll(self):
        return self.performance, self.parameters


class BaseDataset:
    def __init__(self, order, sign) -> None:
        self.order = order
        self.sign = np.array(sign)

    @staticmethod
    def transform_data(parameter, performance):
        """
        Preprocess data to be used in the model by scaling the data to be in the range of [-1, 1]
        """
        data = np.hstack((np.copy(parameter), np.copy(performance)))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        return scaled_data[:, :parameter.shape[1]], scaled_data[:, parameter.shape[1]:], scaler

    @staticmethod
    def inverse_transform(parameter, performance, scaler):
        """
        Inverse transform the data to the original scale
        """
        data = np.hstack((parameter, performance))
        data = scaler.inverse_transform(data)
        return data[:, :parameter.shape[1]], data[:, parameter.shape[1]:]


    def fit(self, parameter, performance):
        # make permutation of y according to order and sign

        fit_performance = np.copy(performance) * self.sign
        fit_performance = fit_performance[:, self.order]

        return parameter,fit_performance

    def inverse_fit(self, parameter,performance):
        reverse_order = [0 for _ in range(len(self.order))]
        for index in range(len(self.order)):
            reverse_order[self.order[index]] = index
        inverse_fit_performance = np.copy(performance)[:, reverse_order]
        inverse_fit_performance = inverse_fit_performance * self.sign
        return parameter,inverse_fit_performance

    def modify_data(self, train_parameter, train_performance, test_parameter, test_performance, train=True, extra_args=None):
        if train:
            return train_parameter, train_performance, {}
        else:
            return test_parameter, test_performance, {}