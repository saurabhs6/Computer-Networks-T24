"""Builds cifar10.npz from the PNG copy of CIFAR-10 at https://github.com/YoongiKim/CIFAR-10-images
Usage:  git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images cifar_png
        python cifar_prepare.py cifar_png ../data/cifar10.npz"""
import os, sys
import numpy as np
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
root, out = sys.argv[1], sys.argv[2]
arrays = {}
for split in ("train", "test"):
    X, Y = [], []
    for i, c in enumerate(CLASSES):
        d = os.path.join(root, split, c)
        for f in sorted(os.listdir(d)):
            im = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), np.uint8)
            assert im.shape == (32, 32, 3)
            X.append(im); Y.append(i)
    arrays[split + "_x"] = np.stack(X); arrays[split + "_y"] = np.array(Y, np.uint8)
    print(split, arrays[split + "_x"].shape)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
np.savez_compressed(out, **arrays)
