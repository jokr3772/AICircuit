import os

def create_folder(path, folder_name):
    if os.path.exists(path):
        print("{} FOLDER EXISTS ALREADY".format(folder_name))
    else:
        os.mkdir(path)
        print("{} FOLDER DOES NOT EXIST, CREATING NOW".format(folder_name))

def setup():
    '''
    RUN THIS FUNCTION TO SETUP REQUIRED FOLDER FOR RUNNING PIPELINE
    '''

    pwd = os.getcwd()


    train_result_folder = os.path.join(pwd, "result_out")
    train_plot_folder = os.path.join(pwd, "out_plot")
    test_data_out_folder = os.path.join(pwd, "data")

    nmos_data_out_path = os.path.join(test_data_out_folder, "SingleStageAmplifier")
    cascode_data_out_path = os.path.join(test_data_out_folder, "Cascode")
    LNA_data_out_path = os.path.join(test_data_out_folder, "LNA")
    mixer_data_out_path = os.path.join(test_data_out_folder, "Mixer")
    two_stage_data_out_path = os.path.join(test_data_out_folder, "TwoStage")
    VCO_data_out_path = os.path.join(test_data_out_folder, "VCO")
    pa_data_out_path = os.path.join(test_data_out_folder, "PA")

    create_folder(train_result_folder, "RESULTS")
    create_folder(train_plot_folder, "PLOTS")
    create_folder(test_data_out_folder, "DATA")
    create_folder(nmos_data_out_path, "SINGLE STAGE AMPLIFIER DATA")
    create_folder(cascode_data_out_path, "CASCODE DATA")
    create_folder(LNA_data_out_path, "LNA DATA")
    create_folder(mixer_data_out_path, "MIXER DATA")
    create_folder(two_stage_data_out_path, "TWO STAGE DATA")
    create_folder(VCO_data_out_path, "VCO DATA")
    create_folder(pa_data_out_path, "PA DATA")

if __name__ == '__main__':
    setup()