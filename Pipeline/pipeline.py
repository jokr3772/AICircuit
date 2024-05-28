from Dataset.dataset import *
from Pipeline.modules import *
import os

from Model.model_evaluator import ModelEvaluator
from Utils.utils import load_train_config, load_visual_config, \
    save_result, checkAlias, generate_train_config_for_single_pipeline, \
    update_train_config_given_model_type, check_comparison_value_diff, load_data

from Model.models import *
from Utils.visualutils import plot_multiple_loss_with_confidence_comparison
from datetime import datetime


def pipeline(configpath):

    train_config = load_train_config(configpath=configpath)
    visual_config = load_visual_config()

    for circuit in train_config['circuits']:

        print("Pipeline with {} circuit".format(circuit))

        pipeline_cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
        
        if train_config["compare_method"]:
            save_path = os.path.join(os.getcwd(), "out_plot", "compare-method-" + circuit + "-" + pipeline_cur_time)
            print("Save comparison folder is {}".format(save_path))

        compare_loss_train = {"mean_list": [] , "upper_bound_list": [], "lower_bound_list": []}
        compare_loss_test = {"mean_list": [] , "upper_bound_list": [], "lower_bound_list": []}

        label = []

        epochs = None
        loss_per_epoch = None

        for model_template_config in train_config["model_config"]:

            print("Pipeline with {} model".format(model_template_config["model"]))

            circuit_config = generate_circuit_given_config(circuit)

            dataset = generate_dataset_given_config(circuit_config)

            new_train_config = generate_train_config_for_single_pipeline(train_config, model_template_config)

            data_config = data_config_creator(circuit_config=circuit_config)

            model, model_type = generate_model_given_config(dict(model_template_config),num_params=data_config.num_params,
                                                                num_perf=data_config.num_perf)
            update_train_config_given_model_type(model_type, new_train_config)

            new_train_config["model_type"] = model_type
            new_train_config["model_name"] = model_template_config["model"]

            loss_per_epoch = check_comparison_value_diff(new_train_config, loss_per_epoch, "loss_per_epoch")

            if new_train_config["loss_per_epoch"]:
                epochs = check_comparison_value_diff(new_train_config, epochs, "epochs")
                
            print("Load from saved data")
            parameter, performance = load_data(data_config, circuit)

            print("Check Alias Problem")
            checkAlias(parameter, performance)

            print("Generate Circuit Status")
            circuit_status_path = os.path.join(os.getcwd(), circuit_config["arguments"]["out"], "circuit_stats.txt")
            if not os.path.exists(circuit_status_path):
                generate_circuit_status(parameter, performance, circuit_status_path)

            print("Pipeline Start")
            print(new_train_config)
            model_pipeline = ModelEvaluator(parameter, performance, dataset, data_config=data_config,
                                        train_config=new_train_config, model=model)

            cur_time = str(datetime.now().strftime('%Y-%m-%d %H-%M'))
            pipeline_save_name = "{}-circuit-{}-method-{}".format(circuit, model_template_config["model"], cur_time)

            print("Pipeline save name is {}".format(pipeline_save_name))
            result = model_pipeline.eval()
            visual_result = generate_visual_given_result(result, new_train_config, visual_config, pipeline_save_name, circuit)
            result.update(visual_result)
            save_result(result, pipeline_save_name, configpath)

            if new_train_config["compare_method"]:
                label.append(model_template_config["model"])
                if new_train_config["loss_per_epoch"]:
                    compare_loss_train["mean_list"].append(result["multi_train_loss"])
                    compare_loss_train["upper_bound_list"].append(result["multi_train_loss_upper_bound"])
                    compare_loss_train["lower_bound_list"].append(result["multi_train_loss_lower_bound"])

                    compare_loss_test["mean_list"].append(result["multi_test_loss"])
                    compare_loss_test["upper_bound_list"].append(result["multi_test_loss_upper_bound"])
                    compare_loss_test["lower_bound_list"].append(result["multi_test_loss_lower_bound"])


        if train_config["compare_method"] and loss_per_epoch:
                plot_multiple_loss_with_confidence_comparison(compare_loss_train, label, train_config["subset"],
                                                              visual_config, epochs, circuit, 'Train', save_path)
                plot_multiple_loss_with_confidence_comparison(compare_loss_test, label, train_config["subset"],
                                                              visual_config, epochs, circuit, 'Test', save_path)