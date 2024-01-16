

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

from numpy.linalg import norm

from utilsLoaders import read_trc, read_mot
from utils import center_of_mass, center_of_mass_vel


def gait_trc_feats(xyz, markers, fps, com, comv, trial_clean):
    # com = xyz[:,np.argmax(markers == 'midHip'),:] # TODO uses actual CoM
    com -= com[-2,:]
    
    # compute direction of travel
    com_xz = com[:,[0,2]].copy()
    direction = com_xz[0,:] - com_xz[-1,:]
    direction /= norm(direction)
    
    # compute lateral sway
    comv_xz = comv[:,[0,2]].copy()
    comv_xz -= np.outer(comv_xz @ direction, direction)
    com_sway = comv_xz[:,0]
    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)
    com_sway = ss.convolve(com_sway, win, mode='same')
    
    # compute usable kinematics zone
    com_dist = norm(com[:,[0,2]], axis=-1)
    zone_start = np.argmax(com_dist < 4)
    zone_stop = np.argmax(com_dist < 1)
    zone = np.arange(xyz.shape[0])
    zone = (zone >= zone_start) & (zone < zone_stop)
    
    # compute metrics
    time_3m = (zone_stop-zone_start)/fps
    time_10m = time_3m * 10 / 3
    speed = 3 / time_3m
    com_sway = com_sway[zone].std()
    
    # TODO joint impedance? See Cavallo 2022
    
    # return time_10m, speed, com_bob, com_sway, last_3m
    return {
            f'{trial_clean}_time_10m': float(time_10m),
            f'{trial_clean}_speed': float(speed),
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
    com = center_of_mass(model_fpath, mot_fpath)
    comv = center_of_mass_vel(model_fpath, mot_fpath)
    trc_feats = gait_trc_feats(xyz, markers, fps, com, comv, trial_clean)

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




