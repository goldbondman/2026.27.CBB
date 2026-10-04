from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import gate_batch_15 as base

OUT=Path('results/residual_ml_legal_check'); OUT.mkdir(parents=True,exist_ok=True)

def main():
    g=base.load_games(); C=base.continuity(); tp=base.run_frozen(g,C,True); pace=base.pace_predictions(g); games=base.make_games(tp,pace)
    feats=['pred_margin','pred_total','n_avg']
    rows=[]
    for sy in [2023,2024,2025]:
        tr=games[(games.season<sy)&(games.season>=2022)].copy(); te=games[games.season==sy].copy()
        med=tr[feats].median(); Xtr=tr[feats].fillna(med); Xte=te[feats].fillna(med); y=tr.margin_resid
        models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),'elastic':make_pipeline(StandardScaler(),ElasticNet(alpha=.02,l1_ratio=.2,max_iter=10000))}
        base_mae=float(te.margin_resid.abs().mean())
        for name,m in models.items():
            m.fit(Xtr,y); corr=m.predict(Xte); new=np.abs(te.margin_resid-corr)
            rows.append({'season':sy,'model':name,'base_mae':base_mae,'new_mae':float(new.mean()),'delta':float(new.mean()-base_mae),'n':len(te)})
    import pandas as pd
    d=pd.DataFrame(rows); out={'features':feats,'rows':rows}
    for name in ['ridge','elastic']:
        q=d[d.model==name]; out[name]={'mean_delta':float(q.delta.mean()),'season_wins':int((q.delta<0).sum())}
    print('LEGAL_ML_RESULTS'); print(json.dumps(out,indent=2)); (OUT/'results.json').write_text(json.dumps(out,indent=2)); d.to_csv(OUT/'results.csv',index=False)
if __name__=='__main__': main()
