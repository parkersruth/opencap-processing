
import numpy as np
from numpy.linalg import norm

from utilsKinematics import OpenSimModelWrapper

# from https://stackoverflow.com/a/13849249
def angle_between(v1, v2):
    # gets the angle between two vectors
    v1_u = v1 / norm(v1)
    v2_u = v2 / norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_all(s1, s2):
    # gets the angles between two vectors over time
    assert s1.shape == s2.shape
    out = np.empty(s1.shape[0])
    for i in range(s1.shape[0]):
        out[i] = angle_between(s1[i,:], s2[i,:])
    return out

def trc_arm_angles(xyz, markers):
    # shoulder, elbow, and wrist markers
    # rs = xyz[:,np.argmax(markers=='RShoulder'),:]
    # ls = xyz[:,np.argmax(markers=='LShoulder'),:]
    # re = xyz[:,np.argmax(markers=='RElbow'),:]
    # le = xyz[:,np.argmax(markers=='LElbow'),:]
    # rw = xyz[:,np.argmax(markers=='RWrist'),:]
    # lw = xyz[:,np.argmax(markers=='LWrist'),:]

    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    re = xyz[:,np.argmax(markers=='r_melbow_study'),:]
    le = xyz[:,np.argmax(markers=='L_melbow_study'),:]
    rw = xyz[:,np.argmax(markers=='r_mwrist_study'),:]
    lw = xyz[:,np.argmax(markers=='L_mwrist_study'),:]

    # gravity vector
    grav = np.zeros_like(rs)
    grav[:,1] = -1

    # shoulder and elbow angles
    rsa = angle_between_all(re-rs, grav) * 180 / np.pi
    rea = angle_between_all(rw-re, re-rs) * 180 / np.pi
    lsa = angle_between_all(le-ls, grav) * 180 / np.pi
    lea = angle_between_all(lw-le, le-ls) * 180 / np.pi

    return rsa, rea, lsa, lea


def center_of_mass(modelPath, motionPath):
    model = OpenSimModelWrapper(modelPath, motionPath)
    df_com = model.get_center_of_mass_values()
    return df_com[['z', 'y', 'x']].values


def center_of_mass_vel(modelPath, motionPath):
    model = OpenSimModelWrapper(modelPath, motionPath)
    df_com = model.get_center_of_mass_speeds()
    return df_com[['z', 'y', 'x']].values




