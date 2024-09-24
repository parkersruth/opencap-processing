
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

from numpy.linalg import norm

from utilsLoaders import read_trc, read_mot
from utils import center_of_mass, center_of_mass_vel
from utils import segment_gait_cycles
from utils import angle_between_all


def gait_trc_feats(xyz, markers, fps, com, comv, trial_clean):
    half_cycles, full_cycles, h = segment_gait_cycles(xyz, markers, fps)

    # compute usable kinematics zone
    com_dist = norm((com - com[-2,:])[:,[0,2]], axis=-1)
    zone_start = np.argmax(com_dist < 4)
    zone_stop = np.argmax(com_dist < 1)
    zone = np.arange(xyz.shape[0])
    zone = (zone >= zone_start) & (zone < zone_stop)

    # LP filter kernel
    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)

    # transform to basis aligned with walk direction
    direc = com[-1,:] - com[0,:]
    direc /= norm(direc)
    pos_z = np.array([0.0, 1.0, 0.0])
    agrav = pos_z - (pos_z @ direc) / (direc @ direc) * direc
    agrav /= norm(agrav)
    perp = np.cross(agrav, direc)
    perp /= norm(perp)
    new_basis = np.stack([perp, agrav, direc])
    P = np.linalg.inv(new_basis)
    xyz = np.einsum('...ij,jk->...ik', xyz, P)

    # compute lateral sway
    com_sway = comv[:,0]
    com_sway = ss.convolve(com_sway, win, mode='same')

    # compute lateral lean
    c7 = xyz[:,np.argmax(markers=='r_C7'),:]
    rh = xyz[:,np.argmax(markers=='RHJC_study'),:].copy()
    lh = xyz[:,np.argmax(markers=='LHJC_study'),:].copy()
    midhip = (rh + lh) / 2
    trunk = c7 - midhip
    trunk_tilt = np.arctan2(trunk[:,0], trunk[:,1]) * 180/np.pi
    trunk_tilt = ss.convolve(trunk_tilt, win, mode='same')

    # compute metrics
    time_3m = (zone_stop-zone_start)/fps
    time_10m = time_3m * 10 / 3 # TODO magic numbers
    speed = 3 / time_3m # TODO magic numbers
    com_sway = com_sway[zone].std()
    trunk_lean = np.mean(np.abs(trunk_tilt[zone]))
    trunk_lean_asym = np.abs(np.mean(trunk_tilt[zone]))

    stride_time = np.diff(full_cycles, 1).mean() / fps

    ra = xyz[:,np.argmax(markers=='r_ankle_study'),:].copy()
    la = xyz[:,np.argmax(markers=='L_ankle_study'),:].copy()

    stride_lens = []
    ankle_elevs = []
    for cyc in full_cycles:
        if h[cyc[0]] > 0:
            lenny = norm(np.diff(la[cyc],0))
        else:
            lenny = norm(np.diff(ra[cyc],0))
        stride_lens.append(lenny)

        # find ankle elevation at mid-swing
        assert len(cyc) == 2
        ia, ib = cyc[0], cyc[1]
        ms = ia + np.argmin(np.abs(la[ia:ib,2]-ra[ia:ib,2]))
        ankle_elevs.append(np.abs(la[ms]-ra[ms]))
    stride_len = np.median(stride_lens)
    ankle_elev = np.median(ankle_elevs)


#     # get arm vectors
#     rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:].copy()
#     ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:].copy()
#     re = xyz[:,np.argmax(markers=='r_lelbow_study'),:].copy()
#     le = xyz[:,np.argmax(markers=='L_lelbow_study'),:].copy()
#     rw = xyz[:,np.argmax(markers=='r_lwrist_study'),:].copy()
#     lw = xyz[:,np.argmax(markers=='L_lwrist_study'),:].copy()
#     ls[:,0] *= -1
#     le[:,0] *= -1
#     lw[:,0] *= -1

#     # compute normalized segment vectors
#     rua = (re - rs)
#     lua = (le - ls)
#     rla = (rw - rs)
#     lla = (lw - ls)
#     rua /= norm(rua, axis=-1, keepdims=True)
#     rla /= norm(rla, axis=-1, keepdims=True)
#     lua /= norm(lua, axis=-1, keepdims=True)
#     lla /= norm(lla, axis=-1, keepdims=True)

#     # sort arm segments by contralateral and ipsilateral patterns
#     W = int(np.mean([lb - la for (la, lb) in full_cycles])) # resample width
#     r_ipsi = []
#     r_contra = []
#     l_ipsi = []
#     l_contra = []
#     for k, (loca, locb) in enumerate(full_cycles):
#         if h[loca] < 0:
#             r_ipsi.append(ss.resample(rla[loca:locb], W))
#             l_contra.append(ss.resample(lla[loca:locb], W))
#         else:
#             r_contra.append(ss.resample(rla[loca:locb], W))
#             l_ipsi.append(ss.resample(lla[loca:locb], W))

#     # join and concatenate to make mean arm profile for R and L
#     r_mean = np.concatenate([np.array(r_ipsi).mean(0), np.array(r_contra).mean(0)])
#     l_mean = np.concatenate([np.array(l_ipsi).mean(0), np.array(l_contra).mean(0)])

#     # compute asymmetry between R and L profiles
#     # aba = angle_between_all(r_mean, l_mean) * 180 / np.pi
#     # arm_asym = np.mean(aba)
#     arm_asym = np.sqrt(np.mean((r_mean - l_mean)**2))


    return {
            # f'{trial_clean}_time_10m': float(time_10m),
            f'{trial_clean}_speed': float(speed),
            f'{trial_clean}_com_sway': float(com_sway),
            f'{trial_clean}_stride_time': float(stride_time),
            f'{trial_clean}_stride_len': float(stride_len),
            f'{trial_clean}_trunk_lean': float(trunk_lean),
            # f'{trial_clean}_trunk_lean_asym': float(trunk_lean_asym),
            f'{trial_clean}_ankle_elev': float(ankle_elev),
            # f'{trial_clean}_arm_asym': float(arm_asym),
           }


def gait_mot_feats(df, trial_clean):

    rha = df.hip_adduction_r.to_numpy()
    lha = df.hip_adduction_l.to_numpy()
    rka = df.knee_angle_r.to_numpy()
    lka = df.knee_angle_l.to_numpy()

    rha = ss.medfilt(rha, 11)
    lha = ss.medfilt(lha, 11)
    rka = ss.medfilt(rka, 11)
    lka = ss.medfilt(lka, 11)

    # raa = df.ankle_angle_r.to_numpy()
    # laa = df.ankle_angle_l.to_numpy()
    # raa = ss.medfilt(raa, 5)
    # laa = ss.medfilt(laa, 5)

    # max_raa = raa.max()
    # max_laa = laa.max()
    # mean_max_aa = (max_raa + max_laa) / 2

    ptp_r_hip_add = rha.ptp()
    ptp_l_hip_add = lha.ptp()
    mean_ptp_hip_add = (ptp_r_hip_add + ptp_l_hip_add) / 2

    # TODO separately analyze abduction vs. adduction
    # cycle phase-specific metrics could be more informative

    max_rka = rka.max()
    max_lka = lka.max()
    mean_max_ka = (max_rka + max_lka) / 2

    # TODO max vals are noisy -- use median over segmented gait cycles instead

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
            # f'{trial_clean}_mean_max_aa': float(mean_max_ka),
           }


def feats_gait(trc_fpath, mot_fpath, model_fpath, trial_clean='gait'):
    fps, markers, xyz = read_trc(trc_fpath)

    com = center_of_mass(model_fpath, mot_fpath)
    comv = center_of_mass_vel(model_fpath, mot_fpath)
    # rh = xyz[:,np.argmax(markers=='RHJC_study'),:].copy()
    # lh = xyz[:,np.argmax(markers=='LHJC_study'),:].copy()
    # com = (rh + lh) / 2
    # comv = np.diff(com, axis=1, prepend=0)

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





