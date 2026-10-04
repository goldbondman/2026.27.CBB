from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gate_batch_15 as base
import gate_batch_20 as b20
OUT=Path('results/market_edge_validation'); OUT.mkdir(parents=True,exist_ok=True)

def stats(q,edge_col,resid_col):
    if len(q)==0:return {'n':0}
    e=q[edge_col].to_numpy(float); r=q[resid_col].to_numpy(float)
    hit=(np.sign(e)*r>0); push=np.isclose(r,0)
    return {'n':int(len(q)),'hit_rate':float(hit.mean()),'push_rate':float(push.mean()),'mean_abs_edge':float(np.abs(e).mean())}

def main():
    g=base.load_games();C=base.continuity();tp=base.run_frozen(g,C,True);pace=base.pace_predictions(g);games=base.make_games(tp,pace)
    tc=b20.total_calibration(games); scores,spc,ttc,_,_=b20.market_data();j=b20.match_market(games,scores,spc,ttc)
    # rolling legal total intercept per season, not one future-informed shift
    shift={int(r.season):float(r.learned_global) for r in tc.itertuples(index=False)}
    j['season']=pd.to_numeric(j.season,errors='coerce').astype('Int64')
    spread=j.dropna(subset=['spread_line']).copy(); spread['mkt_margin']=-spread.spread_line;spread['edge']=spread.model_home_margin-spread.mkt_margin;spread['resid']=spread.actual_home_margin-spread.mkt_margin
    total=j.dropna(subset=['total_line']).copy();total['model_corr']=total.apply(lambda r:r.pred_total+shift.get(int(r.season),-1.65),axis=1);total['edge']=total.model_corr-total.total_line;total['resid']=total.actual_total-total.total_line
    rows=[]
    for market,d in [('spread',spread),('total',total)]:
        for sy in sorted(int(x) for x in d.season.dropna().unique()):
            z=d[d.season==sy]
            for lo in [0.25,2,4,6,8,12]:
                q=z[z.edge.abs()>=lo]
                s=stats(q,'edge','resid');s.update({'market':market,'season':sy,'threshold':lo});rows.append(s)
            for stage,mask in [('early',z.n_avg<5),('mature',z.n_avg>=5)]:
                for lo in [4,8,12]:
                    q=z[mask & (z.edge.abs()>=lo)];s=stats(q,'edge','resid');s.update({'market':market,'season':sy,'threshold':lo,'stage':stage});rows.append(s)
    df=pd.DataFrame(rows);df.to_csv(OUT/'thresholds.csv',index=False)
    core=df[df.stage.isna() if 'stage' in df else True]
    summary={}
    for market in ['spread','total']:
        summary[market]={}
        for lo in [4,6,8,12]:
            q=df[(df.market==market)&(df.threshold==lo)&(df.get('stage',pd.Series(index=df.index,dtype=object)).isna())]
            valid=q[q.n>=30]
            summary[market][str(lo)]={'seasons':valid[['season','n','hit_rate']].to_dict('records'),'wins_over_50':int((valid.hit_rate>.5).sum()),'weighted_hit':float(np.average(valid.hit_rate,weights=valid.n)) if len(valid) else None}
    result={'coverage':{'spread':len(spread),'total':len(total)},'summary':summary,'decision_spread':'PASS' if summary['spread']['8']['wins_over_50']>=2 and (summary['spread']['8']['weighted_hit'] or 0)>.53 else 'HOLD','decision_total':'PASS' if summary['total']['8']['wins_over_50']>=2 and (summary['total']['8']['weighted_hit'] or 0)>.53 else 'HOLD'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print('MARKET_EDGE_VALIDATION');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
