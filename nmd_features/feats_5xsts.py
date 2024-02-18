
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

from numpy.linalg import norm

from utilsLoaders import read_trc, read_mot
from utils import angle_between_all


def sts_trc_feats(xyz, markers, fps):
    c7 = xyz[:,np.argmax(markers=='C7_study'),:]
    # mh = xyz[:,np.argmax(markers=='midHip'),1]
    rh = xyz[:,np.argmax(markers=='RHJC_study'),:]
    lh = xyz[:,np.argmax(markers=='lHJC_study'),:]
    mh = (rh + lh) / 2

    h = mh[:,1]
    h -= h.min()
    h /= h.max()
    locs, _ = ss.find_peaks(h, height=0.75, prominence=0.5)

    la = locs[0] - np.argmax(h[locs[0]::-1] < 0.25)
    if la == locs[0]:
        la = np.argmin(h[:locs[0]])
    lb = locs[-1] + np.argmax(h[locs[-1]:] < 0.25)
    if lb == locs[-1]:
        lb = locs[-1] + np.argmin(h[locs[-1]:])

    if len(locs) > 1:
        tdiffs = np.diff(locs) / fps
        sts_time = np.median(tdiffs)
    else:
        sts_time = (lb - la)/fps

    # sts_time = locs.ptp() / (len(locs)-1) / fps
    sts_speed = 1 / sts_time

    # sts_time_5 = sts_time * 5
    sts_time_5 = (lb - la)/fps / len(locs) * 5

    # gravity vector
    grav = np.zeros_like(c7)
    grav[:,1] = -1

    trunk_angle = angle_between_all(mh-c7, grav) * 180 / np.pi

    # smooth trunk angle with 0.5s hann window
    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)
    trunk_angle = ss.convolve(trunk_angle, win, mode='same')

    lean_ptps = []
    lean_maxs = []
    lean_avels = []
    if len(locs) > 1:
        for i in range(len(locs)-1):
            la, lb = locs[i], locs[i+1]
            seg = trunk_angle[la:lb]
            lean_ptps.append(seg.ptp())
            lean_maxs.append(seg.max())
            lean_avels.append(np.max(np.diff(seg))*fps)
    else:
        seg = trunk_angle[la:lb]
        lean_ptps.append(seg.ptp())
        lean_maxs.append(seg.max())
        lean_avels.append(np.max(np.diff(seg))*fps)

    lean_ptp = np.median(lean_ptps)
    lean_max = np.median(lean_maxs)
    lean_avel = np.median(lean_avels)

    return {
            '5xsts_time_1': float(sts_time),
            '5xsts_time_5': float(sts_time_5),
            # '5xsts_num': float(len(locs)),
            '5xsts_speed': float(sts_speed),
            '5xsts_lean_ptp': float(lean_ptp),
            '5xsts_lean_max': float(lean_max),
            # '5xsts_lean_avel': float(lean_avel),
           }


def sts_mot_feats(df):
    return {}


def feats_5xsts(trc_fpath, mot_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    trc_feats = sts_trc_feats(xyz, markers, fps)

    df = read_mot(mot_fpath)
    mot_feats = sts_mot_feats(df)

    feats = trc_feats.copy()
    feats.update(mot_feats)
    return feats


if __name__ == '__main__':
    feats = feats_5xsts(snakemake.input['trc'], snakemake.input['mot'])

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)

