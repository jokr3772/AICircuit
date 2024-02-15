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