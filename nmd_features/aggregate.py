
from pathlib import Path

import pandas as pd

def agg(fpaths):
    datadir = Path('./datadir/')
    oc_feats_dir = datadir / 'opencap_features'

    df_part = pd.read_excel(datadir / 'participant_info.xlsx')
    df_session = pd.read_excel(datadir / 'session_info.xlsx')

    dfs = []
    for fpath in fpaths:
        pid, sid, trial = Path(fpath).stem.split('__')
        trial_clean = Path(fpath).parent.stem
        df = pd.read_csv(fpath, header=None, names=['feature', 'value'])
        df['pid'] = pid
        df['sid'] = sid
        df['trial'] = trial_clean
        df['trial_clean'] = trial_clean
        dfs.append(df)

    df_feat = pd.concat(dfs)#.groupby(['pid', 'trial_clean'])

    # add datetime object and date string from session info
    df_date = df_session[['sid', 'created_at']].copy()
    df_date['date'] = pd.to_datetime(df_date['created_at'])
    df_date['date'] = df_date.date.dt.tz_convert('America/Los_Angeles')
    # TODO handle MDF time zone differently?

    df_date['date'] = df_date.date.dt.strftime('%Y-%m-%d')
    df_feat = df_date[['sid', 'date']].merge(df_feat, on='sid', how='right')

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


