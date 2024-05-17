from Dataset.dataset import *
import os

from Model.models import ModelEvaluator
from utils.utils import load_circuit, load_train_config, load_visual_config, \
    save_result, saveDictToTxt, checkAlias, generate_train_config_for_single_pipeline, \
    update_train_config_given_model_type, check_comparison_value_diff
from utils.utils import get_margin_error, get_relative_margin_error
from Model.eval_model import *
from utils.visualutils import plot_multiple_loss_with_confidence_entrypoint, \
    plot_multiple_loss_with_confidence_comparison
from datetime import datetime


def generate_dataset_given_config(circuit_config):

    print("Return Dataset")
    return BaseDataset(circuit_config["order"], circuit_config["sign"])


def generate_circuit_given_config(circuit_name):
    config_path = os.path.join(os.path.join(os.getcwd(), "config"), "circuits")
    circuit_mapping = {
        "singlestageamplifier": os.path.join(config_path, "SingleStageAmplifier.yaml"),
        "cascode": os.path.join(config_path, "Cascode.yaml"),
        "lna": os.path.join(config_path, "LNA.yaml"),
        "mixer": os.path.join(config_path, "Mixer.yaml"),
        "twostage": os.path.join(config_path, "TwoStage.yaml"),
        "vco": os.path.join(config_path, "VCO.yaml"),
        "pa": os.path.join(config_path, "PA.yaml"),
        "vco_pa": os.path.join(config_path, "VCO_PA.yaml"),
        "receiver": os.path.join(config_path, "Receiver.yaml"),
        "transmitter-pa": os.path.join(config_path, "Transmitter-PA.yaml"),
        "transmitter-vco": os.path.join(config_path, "Transmitter-VCO.yaml"),
    }

    if circuit_name.lower() in circuit_mapping:
        circuit_definition_path = circuit_mapping[circuit_name.lower()]
    else:
        raise KeyError("The circuit you defined does not exist")

    circuit = load_circuit(circuit_definition_path)
    return circuit

def generate_model_given_config(model_config,num_params,num_perf):

    
    sklearn_model_mapping = {
        "RandomForestRegressor": RandomForest,
        "SupportVectorRegressor": SupportVector,
        "KNeighborsRegressor": KNeighbors,
    }

    dl_model_mapping = {
        "MultiLayerPerceptron": Model500GELU,
        "Transformer": Transformer
    }

    lookup_model_mapping = {
        "Lookup": None
    }

    if model_config["model"] in sklearn_model_mapping.keys():
        eval_model = sklearn_model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        copy_model_config.pop("extra_args", None)
        copy_model_config.pop("model", None)
        return eval_model(**copy_model_config), 0
    elif model_config["model"] in dl_model_mapping.keys():
        model_config['parameter_count'] = num_perf
        model_config['output_count'] = num_params
        eval_model = dl_model_mapping[model_config["model"]]
        copy_model_config = dict(model_config)
        copy_model_config.pop("extra_args", None)
        copy_model_config.pop("model", None)
        return eval_model(**copy_model_config), 1
    elif model_config["model"] in lookup_model_mapping.keys():
        return None, 2
    else:
        raise KeyError("The model you defined does not exist")


def generate_visual_given_result(result, train_config, visual_config, pipeline_save_name):
    folder_path = os.path.join(os.path.join(os.getcwd(), "out_plot"), pipeline_save_name)
    try:
        os.mkdir(folder_path)
    except:
        pass #if less than a minute passed
    result_dict = dict()

    if train_config["loss_per_epoch"]:
        loss_plot_result = plot_multiple_loss_with_confidence_entrypoint(train_config, visual_config, result, pipeline_save_name)
        result_dict.update(loss_plot_result)
    return result_dict


def generate_circuit_status(parameter, performance, path):

    circuit_dict = dict()
    circuit_dict["num_parameter"] = parameter.shape[1]
    circuit_dict["num_performance"] = performance.shape[1]
    circuit_dict["data_size"] = performance.shape[0]

    saveDictToTxt(circuit_dict, path)


def pipeline(configpath):

    train_config = load_train_config(configpath=configpath)
    visual_config = load_visual_config()


    if train_config["compare_dataset"] and train_config["compare_method"]:
        raise ValueError("You cannot compare dataset and method at the same time")

    if (train_config["compare_dataset"] or train_config["compare_method"]) and \
            (len(train_config["model_config"]) > 1 and len(train_config["dataset"]) > 1):
        raise ValueError("When you doing comparison testing, dataset and model can not be both greater than 1")

    for circuit in train_config['circuits']:
        print("Pipeline with {} circuit".format(circuit))
        pipeline_cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
        if train_config["compare_dataset"]:
            save_path = os.path.join(os.getcwd(), "out_plot", pipeline_cur_time + "-" + "compare-dataset-" + circuit)
        else:
            save_path = os.path.join(os.getcwd(), "out_plot", pipeline_cur_time + "-" + "compare-method-" + circuit)
        
        if train_config["compare_dataset"] or train_config["compare_method"]:
            print("Save comparison folder is {}".format(save_path))

        compare_loss_mean_list = []
        compare_loss_upper_bound_list = []
        compare_loss_lower_bound_list = []

        label = []

        epochs = None
        loss_per_epoch = None

        for model_template_config in train_config["model_config"]:
            print("Pipeline with {} model".format(model_template_config["model"]))
            for dataset_type_config in train_config["dataset"]:
                circuit_config = generate_circuit_given_config(circuit)

                dataset = generate_dataset_given_config(circuit_config)

                new_train_config = generate_train_config_for_single_pipeline(train_config, model_template_config, dataset_type_config)

                data_config = data_config_creator(circuit_config=circuit_config)

                model, model_type = generate_model_given_config(dict(model_template_config),num_params=data_config.num_params,
                                                                 num_perf=data_config.num_perf)
                update_train_config_given_model_type(model_type, new_train_config)
                new_train_config["subset_parameter_check"] = False
                new_train_config["model_type"] = model_type
                new_train_config["model_name"] = model_template_config["model"]

                loss_per_epoch = check_comparison_value_diff(new_train_config, loss_per_epoch, "loss_per_epoch")

                if new_train_config["loss_per_epoch"]:
                    epochs = check_comparison_value_diff(new_train_config, epochs, "epochs")
                    
                print("Load from saved data")
                parameter= np.load(os.path.join(data_config.arguments["out"], "x.npy"), allow_pickle=True)
                performance =np.load(os.path.join(data_config.arguments["out"], "y.npy"), allow_pickle=True)

                print("Check Alias Problem")
                checkAlias(parameter, performance)

                print("Generate Circuit Status")
                circuit_status_path = os.path.join(os.getcwd(), circuit_config["arguments"]["out"], "circuit_stats.txt")
                if not os.path.exists(circuit_status_path):
                    generate_circuit_status(parameter, performance, circuit_status_path)

                print("Pipeline Start")
                if new_train_config["metric"] == "absolute":
                    use_metric = get_margin_error
                else:
                    use_metric = get_relative_margin_error

                model_pipeline = ModelEvaluator(parameter, performance, dataset, metric=use_metric, simulator=data_config,
                                          train_config=new_train_config, model=model)

                cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
                pipeline_save_name = "{}-circuit-{}-method-{}".format(circuit, model_template_config["model"], cur_time)

                print("Pipeline save name is {}".format(pipeline_save_name))
                result = model_pipeline.eval()
                visual_result = generate_visual_given_result(result, new_train_config,
                                                             visual_config, pipeline_save_name)
                result.update(visual_result)
                save_result(result, pipeline_save_name, configpath)

                if new_train_config["compare_dataset"] or new_train_config["compare_method"]:
                    if new_train_config["loss_per_epoch"]:
                        compare_loss_mean_list.append(result["multi_train_loss"])
                        compare_loss_upper_bound_list.append(result["multi_train_loss_upper_bound"])
                        compare_loss_lower_bound_list.append(result["multi_train_loss_lower_bound"])
                if new_train_config["compare_dataset"]:
                    label.append(dataset_type_config["type"])
                if new_train_config["compare_method"]:
                    label.append(model_template_config["model"])

        if train_config["compare_dataset"] or train_config["compare_method"]:
            if loss_per_epoch:
                plot_multiple_loss_with_confidence_comparison(compare_loss_mean_list, compare_loss_upper_bound_list,
                                                              compare_loss_lower_bound_list, label, train_config["subset"],
                                                              save_path, visual_config, epochs)