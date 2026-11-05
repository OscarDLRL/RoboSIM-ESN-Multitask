"""Train the ESN."""

import argparse, sys
import numpy as np
import cupy as cp
from model_loader import load_model_package


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="baseline")
    ap.add_argument("--neurons",type=int,nargs="+")
    ap.add_argument("--runs",type=int)
    a=ap.parse_args()

    p,esn=load_model_package(a.model)
    sys.modules["parameters"],sys.modules["esn"]=p,esn
    from data_utils import load_split

    neurons=a.neurons or p.NEURON_COUNTS
    runs=a.runs or p.RUNS

    splits=getattr(p,"TRAIN_SPLITS",("train",))

    train=[]
    for split in splits:
        rows=load_split(split,data_path=p.DATA_PATH)
        print(f"{split.upper()}={len(rows)}")
        train.extend(rows)

    print(
        f"\nMODEL={a.model}"
        f"\nARCH={esn.ARCHITECTURE}"
        f"\nDATA={p.DATA_PATH}"
        f"\nRESULTS={p.RESULTS_DIR}"
        f"\nTRAIN_SPLITS={splits}"
    )

    print(
        f"TOTAL TRAIN={len(train)} | "
        f"frames={min(x['length'] for x in train)}.."
        f"{max(x['length'] for x in train)}\n"
    )

    seeds=np.random.SeedSequence(p.RANDOM_SEED).spawn(runs)

    for n in neurons:
        for run,seq in enumerate(seeds,1):
            seed=int(seq.generate_state(1,dtype=np.uint32)[0])

            model=esn.build_model(
                train,
                n,
                seed=seed,
            )

            path=p.MODEL_DIR/f"N_{n}"/f"run_{run:02d}"/"model.npz"
            esn.save_model(path,model)

            print("Saved:",path)

            del model
            cp.get_default_memory_pool().free_all_blocks()


if __name__=="__main__":
    main()