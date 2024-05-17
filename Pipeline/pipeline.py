from Dataset.dataset import *
import os

from Model.model_evaluator import ModelEvaluator
from Utils.utils import load_circuit, load_train_config, load_visual_config, \
    save_result, saveDictToTxt, checkAlias, generate_train_config_for_single_pipeline, \
    update_train_config_given_model_type, check_comparison_value_diff
from Utils.utils import get_margin_error, get_relative_margin_error
from Model.models import *
from Utils.visualutils import plot_multiple_loss_with_confidence_entrypoint, \
    plot_multiple_loss_with_confidence_comparison
from datetime import datetime




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