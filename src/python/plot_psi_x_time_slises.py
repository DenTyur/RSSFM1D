import numpy as np
from matplotlib import pyplot as plt
import time
import os
import gc

basedir = os.path.abspath(os.getcwd())
src_dir = os.path.abspath(os.path.join(basedir, ".."))

x = np.load(src_dir + "/arrays_saved/x.npy")
t = np.load(src_dir + "/arrays_saved/time_evol/t.npy")


if not os.path.exists(src_dir + "/imgs/time_evol/psi_x"):
    os.makedirs(src_dir + "/imgs/time_evol/psi_x")

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), layout="constrained")

for i in range(len(t)):
    ts = time.time()
    psi = np.load(src_dir + f"/arrays_saved/time_evol/psi_x/psi_t_{i}.npy")
    axs.set(
        title=f"step={i} of {len(t)}; t = {t[i]:.{5}f} a.u.",
    )
    axs.plot(x, np.abs(psi) ** 2, color="blue")
    fig.savefig(src_dir + f"/imgs/time_evol/psi_x/psi_t_{i}.png")
    axs.clear()
    gc.collect()
    print(f"step {i} of {len(t)}; time of step = {(time.time()-ts):.{5}f}")
