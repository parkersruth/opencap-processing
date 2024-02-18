
from pathlib import Path

import pandas as pd

def agg(fpaths):
    datadir = Path('./datadir/')
    oc_feats_dir = datadir / 'opencap_features'

    df_part = pd.read_excel(datadir / 'participant_info.xlsx')
    df_trial = pd.read_excel(datadir / 'trial_info.xlsx')
    # df_session = pd.read_excel(datadir / 'session_info.xlsx')

    dfs = []
    for fpath in fpaths:
        pid, sid, trial = Path(fpath).stem.split('__')
        trial_clean = Path(fpath).parent.stem
        df = pd.read_csv(fpath, header=None, names=['feature', 'value'])
        df['pid'] = pid
        df['sid'] = sid
        df['trial'] = trial
        df['trial_clean'] = trial_clean
        dfs.append(df)

    df_feat = pd.concat(dfs)#.groupby(['pid', 'trial_clean'])

    # add datetime object and date string from trial info
    df_date = df_trial[['sid', 'pid', 'trial', 'created_at']].copy()
    df_date['dt'] = pd.to_datetime(df_date['created_at'])

    # create date strings (MDF conference in different timezone)
    df_date['date'] = ''
    is_mdf = df_date.pid.str.startswith('mdf_')
    df_date.loc[~is_mdf,'date'] = df_date.loc[~is_mdf,'dt'].dt.tz_convert('America/Los_Angeles').dt.strftime('%Y-%m-%d')
    df_date.loc[is_mdf,'date'] = df_date.loc[is_mdf,'dt'].dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    df_date = df_date[['sid', 'trial', 'date', 'created_at']]

    # merge date fields with features
    df_feat = df_date.merge(df_feat, on=['sid', 'trial'], how='right')

    # merge with participant info
    df_feat = df_feat.merge(df_part[['pid', 'date', 'type']], how='left')

    # sort rows
    df_feat.sort_values(['pid', 'date', 'trial_clean', 'feature'], inplace=True)
    df_feat.reset_index(inplace=True)

    # reorder columns
    first_cols = ['pid', 'date', 'trial_clean']
    last_cols = ['sid', 'trial']
    cols = df_feat.columns
    cols = first_cols + [c for c in cols if c not in first_cols + last_cols] + last_cols
    df_feat = df_feat[cols]

    return df_feat


if __name__ == '__main__':
    df_feat = agg(snakemake.input)
    df_feat.to_pickle(snakemake.output[0])

    oc_feats = list(df_feat.feature.unique())
    df_feat_list = pd.DataFrame({'feature': oc_feats})
    df_feat_list.to_csv(snakemake.output[1], index=False, header=False)


