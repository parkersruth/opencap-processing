

from pathlib import Path

import numpy as np
import pandas as pd

from utilsLoaders import read_trc, read_mot
from utils import center_of_mass, angle_between_all


def toe_stand_trc_feats(xyz, markers, fps, com):
    # TODO find a participant who has low CoM y disp, see if they have x CoM disp

    start_win = int(fps*1)
    com_start = com[:start_win,:].mean(0)
    com -= com_start

    # com_height = com_start[1]
    # com /= com_height

    rc = xyz[:,np.argmax(markers=='r_calc_study'),:]
    lc = xyz[:,np.argmax(markers=='L_calc_study'),:]
    rc -= rc[:start_win,:].mean(0)
    lc -= lc[:start_win,:].mean(0)

    dt = 1/fps

    int_com_elev = np.sum(np.clip(com[:,1], 0, None)) * dt
    int_com_fwd = np.sum(com[:,2]) * dt
    int_mean_heel_elev = np.sum(np.clip((rc[:,1] + lc[:,1])/2, 0, None)) * dt

    # Trunk lean
    c7 = xyz[:,np.argmax(markers=='r_C7'),:]
    trunk = c7 - com.copy()
    grav = np.zeros_like(trunk)
    grav[:,1] = -1
    trunk_lean = angle_between_all(-grav, trunk) * 180/np.pi
    int_trunk_lean = np.sum(trunk_lean) * dt

    # or swap CoM vertical with peak knee flexion?

    # pick a steady state period
    # pick a 0.5 sec or so window where heels are high and mostly still

    # something about center of mass variance in 3D (teasing out balance)

    # kinematic trifecta
    # - com height
    # - knee flexion angle
    # - plantar flexion angle
    
    # muscle-driven simulation trifecta
    # gastroc-soleus ratio 
    # peak moment during steady state
    # integrated angle/height

    # TODO integral of plantar flexion torque

    # return int_com_elev, int_com_fwd, int_mean_heel_elev
    return {
            'toe_stand_int_com_elev': float(int_com_elev),
            # 'toe_stand_int_com_fwd': float(int_com_fwd),
            'toe_stand_int_mean_heel_elev': float(int_mean_heel_elev),
            'toe_stand_int_trunk_lean': float(int_trunk_lean),
           }


def toe_stand_mot_feats(df):
    raa = df['ankle_angle_r'].values
    laa = df['ankle_angle_l'].values

    dt = df.time[1] - df.time[0]
    int_raa = np.sum(raa) * dt
    int_laa = np.sum(laa) * dt
    mean_int_aa = (int_raa + int_laa) / 2

    # TODO integral of plantar flexion torque

    return {
            'toe_stand_mean_int_aa': float(mean_int_aa),
           }


def feats_toe_stand(trc_fpath, mot_fpath, model_fpath):
    fps, markers, xyz = read_trc(trc_fpath)

    com_xyz = center_of_mass(model_fpath, mot_fpath)

    # rh = xyz[:,np.argmax(markers=='RHJC_study'),:].copy()
    # lh = xyz[:,np.argmax(markers=='LHJC_study'),:].copy()
    # com_xyz = (rh + lh) / 2

    trc_feats = toe_stand_trc_feats(xyz, markers, fps, com_xyz)

    df = read_mot(mot_fpath)
    mot_feats = toe_stand_mot_feats(df)

    feats = trc_feats.copy()
    feats.update(mot_feats)
    return feats


if __name__ == '__main__':
    feats = feats_toe_stand(snakemake.input['trc'],
                            snakemake.input['mot'],
                            snakemake.input['model'])
    # feats['sid'] = snakemake.wildcards['sid']
    # feats['trial'] = snakemake.wildcards['trial']

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)






