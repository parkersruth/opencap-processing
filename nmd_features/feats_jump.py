

from pathlib import Path

import numpy as np
import pandas as pd

from utilsLoaders import read_trc
from utils import center_of_mass


def jump_trc_feats(com_xyz, fps):
    max_com_vel = np.max(np.diff(com_xyz[:,1])) * fps

    return {
            'jump_max_com_vel': float(max_com_vel),
           }


def feats_jump(trc_fpath, mot_fpath, model_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    com_xyz = center_of_mass(model_fpath, mot_fpath)
    feats = jump_trc_feats(com_xyz, fps)

    return feats


if __name__ == '__main__':
    feats = feats_jump(snakemake.input['trc'],
                       snakemake.input['mot'],
                       snakemake.input['model'])
    # feats['sid'] = snakemake.wildcards['sid']
    # feats['trial'] = snakemake.wildcards['trial']

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


