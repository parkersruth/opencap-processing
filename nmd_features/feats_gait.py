

from pathlib import Path

import numpy as np
import pandas as pd

from numpy.linalg import norm

from utilsLoaders import read_trc, read_mot
from utils import center_of_mass


def gait_trc_feats(xyz, markers, fps, com, trial_clean):
    # com = xyz[:,np.argmax(markers == 'midHip'),:] # TODO uses actual CoM
    com -= com[-1,:]

    com_dist = norm(com[:,[0,2]], axis=-1)
    # last_3m = np.argmax(com_dist < 4)
    last_3m = np.argmax((1 < com_dist) & (com_dist < 4))
    time_3m = (xyz.shape[0] - last_3m)/fps
    time_10m = time_3m * 10 / 3
    speed = 3 / time_3m

    com_bob = com[last_3m:,1].ptp()

    com_xz = com[last_3m:,[0,2]].copy()
    direction = com_xz[0,:] - com_xz[-1,:]
    direction /= norm(direction)
    com_xz -= np.outer(com_xz @ direction, direction)
    com_sway = norm(com_xz, axis=1).ptp()

    # TODO scale bob and sway by height

    # TODO joint impedance? See Cavallo 2022

    # return time_10m, speed, com_bob, com_sway, last_3m
    return {
            f'{trial_clean}_time_10m': float(time_10m),
            f'{trial_clean}_speed': float(speed),
            f'{trial_clean}_com_bob': float(com_bob),
            f'{trial_clean}_com_sway': float(com_sway),
           }


def gait_mot_feats(df, trial_clean):

    rha = df.hip_adduction_r.to_numpy()
    lha = df.hip_adduction_l.to_numpy()
    rka = df.knee_angle_r.to_numpy()
    lka = df.knee_angle_l.to_numpy()

    ptp_r_hip_add = rha.ptp()
    ptp_l_hip_add = lha.ptp()
    mean_ptp_hip_add = (ptp_r_hip_add + ptp_l_hip_add) / 2
    
    max_rka = rka.max()
    max_lka = lka.max()
    mean_max_ka = (max_rka + max_lka) / 2

    # TODO spatiotemporal features

    # stride length (normalized by height?)
    # foot drop: max dorsiflexion angle during swing
    # hip circumduction (secondary to foot drop)
    # peak knee flexion angle during swing and stance

    # ankle plantar flexion moments - dynamic simulation
    # peak saggital moments and power at all three lower extremity joints

    # return mean_ptp_hip_add, mean_max_ka

    return {
            f'{trial_clean}_mean_ptp_hip_add': float(mean_ptp_hip_add),
            f'{trial_clean}_mean_max_ka': float(mean_max_ka),
           }


def feats_gait(trc_fpath, mot_fpath, model_fpath, trial_clean='gait'):
    fps, markers, xyz = read_trc(trc_fpath)
    com_xyz = center_of_mass(model_fpath, mot_fpath)
    trc_feats = gait_trc_feats(xyz, markers, fps, com_xyz, trial_clean)

    df = read_mot(mot_fpath)
    mot_feats = gait_mot_feats(df, trial_clean)

    feats = trc_feats.copy()
    feats.update(mot_feats)
    return feats


if __name__ == '__main__':
    feats = feats_gait(snakemake.input['trc'],
                       snakemake.input['mot'],
                       snakemake.input['model'],
                       )
    # feats['sid'] = snakemake.wildcards['sid']
    # feats['trial'] = snakemake.wildcards['trial']

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)




