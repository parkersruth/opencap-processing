
from pathlib import Path

import pandas as pd

from feats_tug import feats_tug


if __name__ == '__main__':
    feats = feats_tug(snakemake.input['trc'],
                      snakemake.input['mot'],
                      'tug_line'
                      )

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


