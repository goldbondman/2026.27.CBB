from __future__ import annotations
import json
import pandas as pd

URLS={
 'spreads':'https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/canonical/spreads/fg/spreads_fg_all.csv',
 'totals':'https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/canonical/totals/fg/totals_fg_all.csv',
}
for k,u in URLS.items():
    try:
        d=pd.read_csv(u,nrows=5)
        print(k, json.dumps({'columns':list(d.columns),'rows':d.astype(str).to_dict('records')},indent=2))
    except Exception as e:
        print(k,'ERROR',repr(e))
