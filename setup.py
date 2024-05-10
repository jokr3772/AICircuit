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

    result_folder = os.path.join(pwd, "result_out")
    create_folder(result_folder, "RESULTS")

    plot_folder = os.path.join(pwd, "out_plot")
    create_folder(plot_folder, "PLOTS")

    data_folder = os.path.join(pwd, "data")
    create_folder(data_folder, "DATA")

    print("\n")

    circuits = ["SingleStageAmplifier", "Cascode", "LNA", "Mixer", "TwoStage", "VCO", "PA", "VCO_PA", "Receiver"] 
    models = ["MultiLayerPerceptron", "RandomForest", "SupportVectorRegressor", "Transformer"]

    for circuit in circuits:
        p = os.path.join(data_folder, circuit)
        create_folder(p, circuit.upper() + " DATA")

        for model in models:
            pm = os.path.join(data_folder, circuit, model)
            create_folder(pm, circuit.upper() + " DATA " + model.upper() + " MODEL")
        
        print("\n")
    

if __name__ == '__main__':
    setup()