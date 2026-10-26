"""Training entry point for vision / multimodal ESN, optional goal conditioning."""
import argparse
from pathlib import Path
import numpy as np
import cupy as cp
import parameters as p
from data_utils import load_split
from esn_test import build_model, save_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--neurons',type=int,nargs='+',default=p.NEURON_COUNTS)
    ap.add_argument('--runs',type=int,default=p.RUNS)
    args=ap.parse_args()
    train=load_split('train', data_path=p.DATA_PATH)
    print(f'Experiment={p.experiment_tag()}, training sequences={len(train)}, frame range={min(r["length"] for r in train)}..{max(r["length"] for r in train)}')
    seeds=np.random.SeedSequence(p.RANDOM_SEED).spawn(args.runs)
    for neurons in args.neurons:
        for run,seq in enumerate(seeds,1):
            seed=int(seq.generate_state(1,dtype=np.uint32)[0])
            model=build_model(train,neurons,seed=seed)
            target=p.MODEL_DIR / f'N_{neurons}' / f'run_{run:02d}' / 'model.npz'
            save_model(target,model)
            print('Saved:',target)
            del model
            cp.get_default_memory_pool().free_all_blocks()

if __name__=='__main__': main()