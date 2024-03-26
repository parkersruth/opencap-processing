
from pathlib import Path

import numpy as np
import pandas as pd

from utilsLoaders import read_trc
from utils import trc_arm_angles



def curls_trc_feats(xyz, markers):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)
    max_rea = np.max(rea)
    max_lea = np.max(lea)
    mean_ea = (rea + lea) / 2

    min_max_ea = min(max_rea, max_lea)
    max_mean_ea = np.max(mean_ea)

    return {
            'curls_min_max_ea': float(min_max_ea),
            'curls_max_mean_ea': float(max_mean_ea),
           }


def feats_curls(trc_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    feats = curls_trc_feats(xyz, markers)

    return feats


if __name__ == '__main__':
    feats = feats_curls(snakemake.input['trc'])
    # feats['sid'] = snakemake.wildcards['sid']
    # feats['trial'] = snakemake.wildcards['trial']

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)



