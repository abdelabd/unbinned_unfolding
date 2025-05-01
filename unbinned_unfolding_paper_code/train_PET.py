
import h5py as h5
import argparse 
import os 
import numpy as np
import time
import yaml

import horovod.tensorflow.keras as hvd
hvd.init()
print("Number of visible GPUs {}".format(hvd.size()))
print("Each GPU ID: {}".format(hvd.rank()))

# local 
from omnifold import DataLoader, MultiFold, PET

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a PET model using Pythia and Herwig data.")
    parser.add_argument("--data_dir", type=str, default="/pscratch/sd/a/aelabd/omnifold_examples/ryans_way/OmniLearn/data/", help="Folder containing input files")
    parser.add_argument("--save_dir", type=str, default="/pscratch/sd/a/aelabd/omnifold_examples/ryans_way/unbinned_unfolding/unbinned_unfolding_paper_code/output/", help="Folder to store trained model weights")
    parser.add_argument("--run_id", type=int, default=None, help="Unique identifier for training run; used i.f.f. start_N is provided and not 0")
    parser.add_argument("--start_N", type=int, default=0, help="Number of iteration to start with")
    parser.add_argument("--num_data", type=int, default=-1, help="Number of data to train with")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations to use during training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of iteration to start with")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of transformer heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers per transformer head")
    parser.add_argument("--p_dim", type=int, default=128, help="Transformer projection dimension")
    parser.add_argument("--model-name", type=str, default="")
    args = parser.parse_args()
    return args

def main():
    flags = parse_arguments()

    # Create/retrieve run_id
    if flags.run_id is None: # If this is a new run, create a unique run id
        prev_runs = [int(i.replace("run_", "")) for i in os.listdir(flags.save_dir)]
        if len(prev_runs)==0:
            curr_run = 0
        else: curr_run = max(prev_runs)+1
    else: # If we're continuing from a previous run, retrieve that run_id
        curr_run = flags.run_id

    # Create/retrieve output directories
    run_dir = os.path.join(flags.save_dir + f"run_{curr_run}")
    model_weights_dir = os.path.join(run_dir, "model_weights")
    hist_weights_dir = os.path.join(run_dir, "hist_weights")
    if flags.run_id is None: # If new run, create directories
        if (hvd.rank())==0:
            os.makedirs(run_dir, exist_ok = False)
            os.makedirs(model_weights_dir, exist_ok = False)
            os.makedirs(hist_weights_dir, exist_ok = False)
    else: # Otherwise, assert that they already exist
        assert(os.path.isdir(run_dir))
        assert(os.path.isdir(model_weights_dir))
        assert(os.path.isdir(hist_weights_dir))

    # Create/retrieve config file
    if flags.run_id is None:
        config = {}
        for k,v in flags.__dict__.items():
            if k in ["data_dir", "save_dir", "run_id", "start_N"]: continue
            config[k] = v
        if (hvd.rank())==0:
            with open(os.path.join(run_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False)
    else:
        with open(os.path.join(run_dir, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
    
    # Load data
    synthetic_file_path = flags.data_dir + "train_pythia.h5"
    nature_file_path = flags.data_dir + "train_herwig.h5"
    synthetic  =  h5.File(synthetic_file_path, 'r')
    nature = h5.File(nature_file_path, 'r')
    synthetic_pass_reco = (synthetic['reco_jets'][:config['num_data'],0]>150)
    nature_pass_reco = (nature['reco_jets'][:config['num_data'],0]>150)

    synthetic_gen_parts = synthetic['gen'][:config['num_data']]
    synthetic_reco_parts = synthetic['reco'][:config['num_data']]
    nature_reco_parts = nature['reco'][:config['num_data']]
    if (hvd.rank()==0):
        print(f"synthetic_gen_parts.shape: {synthetic_gen_parts.shape}")
        print(f"synthetic_reco_parts.shape: {synthetic_reco_parts.shape}")
        print(f"nature_reco_parts.shape: {nature_reco_parts.shape}")
        print(f"nature_pass_reco.shape: {nature_pass_reco.shape}")

    synthetic_parts_dataloader = DataLoader(reco = synthetic_reco_parts,
                                            gen = synthetic_gen_parts,
                                            pass_reco = synthetic_pass_reco,
                                            normalize = True,
                                            rank=hvd.rank(),
                                            size=hvd.size(),)
    nature_parts_dataloader = DataLoader(reco = nature_reco_parts,
                                        pass_reco = nature_pass_reco,
                                        normalize = True,
                                        rank=hvd.rank(),
                                        size=hvd.size(),)

    # Create model
    synthetic_PET_model = PET(synthetic_gen_parts.shape[2], num_part=synthetic_gen_parts.shape[1], num_heads = config['num_heads'], num_transformer = config['num_layers'], local = True, projection_dim = config['p_dim'], K = 10)
    nature_PET_model = PET(synthetic_gen_parts.shape[2], num_part=synthetic_gen_parts.shape[1], num_heads = config['num_heads'], num_transformer = config['num_layers'], local = True, projection_dim = config['p_dim'], K = 10)
    omnifold_PET = MultiFold(
        config['model_name'],
        model_reco = nature_PET_model,
        model_gen = synthetic_PET_model,
        data = nature_parts_dataloader,
        mc = synthetic_parts_dataloader,
        start = flags.start_N,
        niter = config['num_iterations'],
        weights_folder = model_weights_dir,
        verbose=True,
        batch_size = 256,
        early_stop=3,
        rank=hvd.rank(),
        size=hvd.size(),
        epochs = config['num_epochs']
    )

    # Unfold
    tic = time.time()
    omnifold_PET.Unfold()
    toc = time.time()
    dur = toc - tic
    if (hvd.rank())==0:
        print(f"Unfolding: {dur//60} minutes, {dur%60} seconds")

if __name__ == '__main__':
    main()