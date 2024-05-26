import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_multiple_loss_with_confidence_comparison(loss_mean, loss_upper_bound, loss_lower_bound,
                                                  labels, subsets, save_folder, visual_config, epochs, data_name, data_type):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    color = visual_config["color"]
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure()
    for percentage_index in range(len(subsets)):
        plt.clf()
        ax = fig.add_subplot()
        percentage_loss_mean_cross_comparison = [i[percentage_index] for i in loss_mean]
        percentage_loss_upper_bound_cross_comparison = [i[percentage_index] for i in loss_upper_bound]
        percentage_loss_lower_bound_cross_comparison = [i[percentage_index] for i in loss_lower_bound]

        for compared_item_index in range(len(percentage_loss_mean_cross_comparison)):
            ax.plot(np.arange(epochs), percentage_loss_mean_cross_comparison[compared_item_index], label=labels[compared_item_index],
                    color=color[compared_item_index])
            ax.fill_between(np.arange(epochs), percentage_loss_lower_bound_cross_comparison[compared_item_index],
                            percentage_loss_upper_bound_cross_comparison[compared_item_index], alpha=.3,
                            color=color[compared_item_index])

        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel(f'{data_type} Loss')
        plt.xlabel("Epochs")
        plt.title(f'{data_name}')

        image_save_path = os.path.join(save_folder, f'Subset-{subsets[percentage_index]}-{data_type}-loss.png')
        plt.savefig(image_save_path, dpi=250)


def plot_multiple_loss_with_confidence_entrypoint(train_config, visual_config, result, save_folder, circuit):
    
    plt.clf()

    multi_train_loss, multi_train_loss_lower_bounds, multi_train_loss_upper_bounds = generate_plot_loss_given_result(result["train_loss"],
                                                                                                                        train_config, visual_config, save_folder, "Train", circuit)
    multi_test_loss, multi_test_loss_lower_bounds, multi_test_loss_upper_bounds = generate_plot_loss_given_result(result["validation_loss"],
                                                                                                                     train_config, visual_config, save_folder, "Test", circuit)

    result_dict = dict()
    result_dict["multi_train_loss"] = multi_train_loss
    result_dict["multi_test_loss"] = multi_test_loss
    result_dict["multi_train_loss_lower_bound"] = multi_train_loss_lower_bounds
    result_dict["multi_test_loss_lower_bound"] = multi_test_loss_lower_bounds
    result_dict["multi_train_loss_upper_bound"] = multi_train_loss_upper_bounds
    result_dict["multi_test_loss_upper_bound"] = multi_test_loss_upper_bounds

    return result_dict


def generate_plot_loss_given_result(loss, train_config, visual_config, save_folder, data_type, circuit):

    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []

    save_path = os.path.join(save_folder, data_type + "-loss.png")

    for percentage_loss_index in range(len(loss)):
        temp_loss = [loss[percentage_loss_index][i][0] for i
                     in range(len(loss[percentage_loss_index]))]
        temp_loss_mean = np.average(temp_loss, axis=0)
        temp_loss_std = stats.sem(temp_loss, axis=0) if len(temp_loss) > 1 else [np.nan for i in range(len(temp_loss[0]))]

        multi_loss.append(temp_loss_mean)
        multi_loss_lower_bounds.append(temp_loss_mean - temp_loss_std)
        multi_loss_upper_bounds.append(temp_loss_mean + temp_loss_std)

    plot_loss(multi_loss, multi_loss_upper_bounds, multi_loss_lower_bounds, visual_config, train_config, save_path, circuit, data_type)

    return multi_loss, multi_loss_lower_bounds, multi_loss_upper_bounds


def plot_loss(multi_loss_mean, multi_loss_upper_bounds, multi_loss_lower_bounds, visual_config, train_config, save_name, data_name, data_type):

    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    fig = plt.figure()
    ax = fig.add_subplot()
    num_subset = len(multi_loss_mean)
    color = visual_config["color"][:num_subset]
    subset = train_config["subset"]
    epochs = train_config["epochs"]

    for i in range(len(multi_loss_mean)):
        
        if data_type == 'Train':
            temp_label = "{:3.1f}% data".format(subset[i] * 100)
        else:
            temp_label = "{:3.1f}% data".format((1-subset[i]) * 100)

        ax.plot(np.arange(epochs), multi_loss_mean[i], label=temp_label, color=color[i], linewidth=3)
        ax.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3, color=color[i])

        ax.set_xlim([0, epochs + 1])
        ax.set_ylim([0, None])

        if len(subset) > 1:
            ax.legend()

        plt.title(f'{data_name}')
        plt.ylabel(f'{data_type} Loss')
        plt.xlabel("Epochs")
        plt.savefig(save_name, dpi=250)