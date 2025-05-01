from omnifold import PET
import h5py as h5
import argparse 
import numpy as np
import gzip
import pickle
import os
import yaml
import horovod.tensorflow.keras as hvd
hvd.init()

def expit(x):
    return 1. / (1. + np.exp(-x))

def reweight(events,model,batch_size=None):
    f = expit(model.predict(events,batch_size=batch_size))
    weights = f / (1. - f)  # this is the crux of the reweight, approximates likelihood ratio
    weights = np.nan_to_num(weights[:,0],posinf=1)
    return weights

def parse_arguments():
    parser = argparse.ArgumentParser(description="Make predictions using trained PET model")
    parser.add_argument("--data_dir", type=str, default="/pscratch/sd/a/aelabd/omnifold_examples/ryans_way/OmniLearn/data/", help="Folder containing input files")
    parser.add_argument("--save_dir", type=str, default="/pscratch/sd/a/aelabd/omnifold_examples/ryans_way/unbinned_unfolding/unbinned_unfolding_paper_code/output", help="Folder containing input files")
    parser.add_argument("--run_id", type=int, default=None, help="Unique identifier for training run; used i.f.f. start_N is provided and not 0")
    args = parser.parse_args()
    return args

def main():

    flags = parse_arguments()
    run_dir = os.path.join(flags.save_dir, f"run_{flags.run_id}")
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
    num_data = config['num_data']
    data_dir = flags.data_dir

    synthetic_file_path = data_dir + "test_pythia.h5"
    synthetic  =  h5.File(synthetic_file_path, 'r')
    synthetic_gen_parts = synthetic['gen'][:num_data]
    model = PET(synthetic_gen_parts.shape[2], 
                num_part=synthetic_gen_parts.shape[1], 
                num_heads = config['num_heads'], 
                num_transformer = config['num_layers'], 
                local = True, 
                projection_dim = config['p_dim'], 
                K = 10)
    model.load_weights(os.path.join(run_dir, "model_weights", f"OmniFold_{config['model_name']}_iter{config['num_iterations']-1}_step2.weights.h5"))
    weights = reweight(synthetic_gen_parts, model)

    data_to_save = {
        "PET_weights": weights
    }

    output_dir = os.path.join(run_dir, "predictions")
    if (hvd.rank()==0):
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/test_pythia_final_reweight.pickle.gz'
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(data_to_save, f)


if __name__ == '__main__':
    main()