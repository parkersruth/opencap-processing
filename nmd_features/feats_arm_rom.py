
from pathlib import Path

import numpy as np
import pandas as pd

from numpy.linalg import norm
from scipy.spatial import ConvexHull

from utilsLoaders import read_trc, read_mot
from utils import trc_arm_angles

from scipy.spatial.transform import Rotation as R


# def workspace_r(xyz, markers):
def workspace_right(rs, ls, rw, arm_len):
    # center on right shoulder
    rw = rw - rs
    ls = ls - rs

    # remove torso yaw and roll
    roty = np.arctan(ls[:,2] / ls[:,0])
    rotz = np.arctan(ls[:,1] / norm(ls[:,[0,2]], axis=1))
    for i in range(rw.shape[0]):
        rot = R.from_euler('yz', [roty[i], rotz[i]])
        rw[i,:] = rot.apply(rw[i,:])

    # clip contralateral crossover
    rw[:,0] = rw[:,0].clip(0, None)

    # normalize by arm length
    # arm_len = np.max(norm(rw, axis=1)) # TODO better definition?
    rw /= arm_len

    # compute reachable workspace metrics
    ch = ConvexHull(rw)
    rewo_area = ch.area / (3 * np.pi) # hemisphere area
    rewo_sphere = ch.volume / (1/3 * np.pi) # hemisphere volume
    rewo_cube = np.product(rw.ptp(axis=0)) / 4 # cuboid volume

    return rw*arm_len, rewo_area, rewo_sphere, rewo_cube


def workspace(xyz, markers, xyz_neu):
    # get shoulder and wrist marker trajectories
    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    rw = xyz[:,np.argmax(markers=='r_mwrist_study'),:]
    lw = xyz[:,np.argmax(markers=='L_mwrist_study'),:]

    rsn = xyz_neu[:,np.argmax(markers=='r_shoulder_study'),:]
    lsn = xyz_neu[:,np.argmax(markers=='L_shoulder_study'),:]
    rwn = xyz_neu[:,np.argmax(markers=='r_mwrist_study'),:]
    lwn = xyz_neu[:,np.argmax(markers=='L_mwrist_study'),:]

    arm_len_r = np.mean(norm(rwn - rsn, axis=1))
    arm_len_l = np.mean(norm(lwn - lsn, axis=1))

    # compute right arm reachable workspace
    _, rw_area_r, rw_sphere_r, rw_cube_r = workspace_right(rs, ls, rw, arm_len_r)

    # sagittal flip
    ls[:,0] *= -1
    rs[:,0] *= -1
    lw[:,0] *= -1
    _, rw_area_l, rw_sphere_l, rw_cube_l = workspace_right(ls, rs, lw, arm_len_l)

    # sum right and left scores
    rw_area = rw_area_r + rw_area_l
    rw_sphere = rw_sphere_r + rw_sphere_l
    rw_cube = rw_cube_r + rw_cube_l

    return rw_area, rw_sphere, rw_cube


def arm_rom_trc_feats(xyz, markers, xyz_neu):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)

    mean_sa = (rsa + lsa) / 2
    mean_ea = (rea + lea) / 2
    max_mean_sa = np.max(mean_sa)
    mean_ea_at_max_mean_sa = mean_ea[np.argmax(mean_sa)]

    min_sa = np.vstack([rsa, lsa]).min(0)
    max_min_sa = min_sa.max()

    max_ea = np.vstack([rea, lea]).max(0)
    max_ea_at_max_min_sa = max_ea[np.argmax(min_sa)]

    rw_area, rw_sphere, rw_cube = workspace(xyz, markers, xyz_neu)

    return {
            'arm_rom_max_mean_sa': float(max_mean_sa),
            'arm_rom_max_min_sa': float(max_min_sa),
            'arm_rom_max_ea_at_max_min_sa': float(max_ea_at_max_min_sa),
            # 'arm_rom_mean_ea_at_max_mean_sa': float(mean_ea_at_max_mean_sa),
            'arm_rom_rw_area': float(rw_area),
            # 'arm_rom_rw_sphere': float(rw_sphere),
            # 'arm_rom_rw_cube': float(rw_cube),
           }


def arm_rom_sto_feats(df):
    # TODO max shoulder moment 

    # return None
    return {}


def feats_arm_rom(trc_fpath, sto_fpath, neu_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    _, _, xyz_neu = read_trc(neu_fpath)
    trc_feats = arm_rom_trc_feats(xyz, markers, xyz_neu)

    df = read_mot(sto_fpath)
    sto_feats = arm_rom_sto_feats(df)

    feats = trc_feats.copy()
    feats.update(sto_feats)
    return feats


if __name__ == '__main__':
    feats = feats_arm_rom(snakemake.input['trc'],
                          snakemake.input['sto'],
                          snakemake.input['neu'])
    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


