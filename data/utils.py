import pandas as pd
import os


def load_dir(dir) -> pd.DataFrame:
    local_files = os.listdir(dir)
    ra, dec = [], []
    for i in range(len(local_files)):
        if ".fits" in local_files[i]:
            t_ra, t_dec = float(local_files[i].split("_")[0]), float(local_files[i].split("_")[1].split(".fits")[0])
            ra.append(t_ra)
            dec.append(t_dec)
    return pd.DataFrame(list(zip(ra, dec)), columns=["ra", "dec"])
