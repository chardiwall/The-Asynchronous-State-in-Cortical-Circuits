"""One-off: tonic's N-MNIST download -> one compact .npz per split (x, y, p, t_us, offsets,
labels), so the experiment code reads plain numpy and needs no tonic at run time.

    python src/lsm/prepare_nmnist.py <tonic_save_to_dir> <out_dir>      (needs tonic)
"""
import os
import sys

import numpy as np
import tonic

save_to, out_dir = (os.path.expanduser(a) for a in sys.argv[1:3])
os.makedirs(out_dir, exist_ok=True)
for split, train in (("train", True), ("test", False)):
    dataset = tonic.datasets.NMNIST(save_to=save_to, train=train)
    events, labels = zip(*(dataset[i] for i in range(len(dataset))))
    offsets = np.concatenate([[0], np.cumsum([len(e) for e in events])])
    np.savez(f"{out_dir}/{split}.npz",
             x=np.concatenate([e["x"] for e in events]).astype(np.uint8),
             y=np.concatenate([e["y"] for e in events]).astype(np.uint8),
             p=np.concatenate([e["p"] for e in events]).astype(np.uint8),
             t_us=np.concatenate([e["t"] for e in events]).astype(np.int32),
             offsets=offsets, labels=np.asarray(labels, dtype=np.int16))
    print(split, len(labels), offsets[-1], "events", flush=True)
