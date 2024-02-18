
from pathlib import Path

import numpy as np
import pandas as pd

from utilsLoaders import read_trc, read_mot
from utils import trc_arm_angles


def brooke_trc_feats(xyz, markers):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)
    mean_sa = (rsa + lsa) / 2
    mean_ea = (rea + lea) / 2
    max_mean_sa = np.max(mean_sa)

    min_sa = np.vstack([rsa, lsa]).min(0)
    max_min_sa = min_sa.max()

    # rea -= rea.min()
    # lea -= lea.min()
    max_ea = np.vstack([rea, lea]).max(0)
    max_max_ea = np.max(max_ea)
    window = max_ea < 30
    if np.any(window):
        min_sa_at_ea_break = np.max(min_sa[window]) # TODO magic number
    else:
        min_sa_at_ea_break = np.min(min_sa)
    max_ea_at_max_min_sa = max_ea[np.argmax(min_sa)]

    max_sa_ea_ratio = np.max(mean_sa / (mean_ea+90))

    return {
            'brooke_max_mean_sa': float(max_mean_sa),
            # 'brooke_max_max_ea': float(max_max_ea),
            'brooke_max_min_sa': float(max_min_sa),
            'brooke_max_ea_at_max_min_sa': float(max_ea_at_max_min_sa),
            # 'brooke_min_sa_at_ea_break': float(min_sa_at_ea_break),
            'brooke_max_sa_ea_ratio': float(max_sa_ea_ratio),
           }


def brooke_sto_feats(df):
    # TODO max shoulder moment (SDU to fix shoulder model)

    # return None
    return {}


def feats_brooke(trc_fpath, sto_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    trc_feats = brooke_trc_feats(xyz, markers)

    df = read_mot(sto_fpath)
    sto_feats = brooke_sto_feats(df)

    feats = trc_feats.copy()
    feats.update(sto_feats)
    return feats


if __name__ == '__main__':
    feats = feats_brooke(snakemake.input['trc'], snakemake.input['sto'])

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)

    # import sys

    # print(sys.argv)
    # feats = feats_brooke(sys.argv[1], sys.argv[2])



