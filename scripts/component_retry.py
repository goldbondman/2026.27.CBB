from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import gate_batch_15 as base
import gate_batch_20 as b20

OUT=Path('results/component_retry'); OUT.mkdir(parents=True,exist_ok=True)

def mae(x): return float(np.mean(np.abs(np.asarray(x,float))))

def component_rows(g):
    # corpus-native columns verified in G20: fgm/fga/tpm/to/orb/drb/fta
    need=['fgm','fga','tpm','to','orb','drb','fta']
    miss=[c for c in need if c not in g.columns]
    if miss: return None,{'missing':miss,'columns':list(g.columns)}
    z=g.copy()
    for c in need: z[c]=pd.to_numeric(z[c],errors='coerce')
    z['efg']=(z.fgm+.5*z.tpm)/z.fga.replace(0,np.nan)
    z['tovr']=z['to']/z.poss.replace(0,np.nan)
    z['ftr']=z.fta/z.fga.replace(0,np.nan)
    od=z[['game_id','ix','drb']].copy(); od['ix']=1-od.ix; od=od.rename(columns={'drb':'opp_drb'})
    z=z.merge(od,on=['game_id','ix'],how='left',validate='one_to_one')
    z['orbp']=z.orb/(z.orb+z.opp_drb).replace(0,np.nan)
    comps=['efg','tovr','orbp','ftr']; defaults={'efg':.50,'tovr':.20,'orbp':.30,'ftr':.30}
    off={}; allow={}; cnt={}; rows=[]; season=None
    for date,day in z.sort_values(['date','game_id','team']).groupby('date',sort=True):
        sy=int(day.season.iloc[0])
        if season!=sy: off={};allow={};cnt={};season=sy
        day_rows=list(day.itertuples(index=False)); lookup={(r.game_id,r.team):r for r in day_rows}
        for r in day_rows:
            a,b=r.team,r.opp
            rec={'season':sy,'date':date,'game_id':r.game_id,'team':a,'opp':b,'prior_games':cnt.get(a,0)}
            for c in comps:
                rec[f'off_{c}']=off.get((a,c),defaults[c])
                rec[f'defallow_{c}']=allow.get((b,c),defaults[c])
                rec[f'match_{c}']=rec[f'off_{c}']+rec[f'defallow_{c}']-defaults[c]
            rows.append(rec)
        # all updates after full date to preserve same-date isolation
        for r in day_rows:
            a,b=r.team,r.opp;n=cnt.get(a,0);opp=lookup.get((r.game_id,b))
            for c in comps:
                v=getattr(r,c)
                if pd.notna(v): off[(a,c)]=(off.get((a,c),0)*n+v)/(n+1)
                if opp is not None:
                    ov=getattr(opp,c)
                    if pd.notna(ov): allow[(a,c)]=(allow.get((a,c),0)*n+ov)/(n+1)
            cnt[a]=n+1
    return pd.DataFrame(rows),{'rows':len(rows),'components':comps}

def game_features(games,cr):
    fs=['match_efg','match_tovr','match_orbp','match_ftr']; out=[]
    by={gid:q.set_index('team') for gid,q in cr.groupby('game_id')}
    for r in games.itertuples(index=False):
        q=by.get(r.game_id)
        if q is None or r.team0 not in q.index or r.team1 not in q.index: continue
        a,b=q.loc[r.team0],q.loc[r.team1]
        rec={'game_id':r.game_id,'season':r.season,'date':r.date,'margin_resid':r.margin_resid,'total_resid':r.total_resid,'pred_margin':r.pred_margin,'pred_total':r.pred_total,'n_avg':r.n_avg}
        for f in fs:
            rec[f+'_diff']=float(a[f]-b[f]);rec[f+'_avg']=float((a[f]+b[f])/2)
        out.append(rec)
    return pd.DataFrame(out)

def main():
    g=base.load_games(); C=base.continuity(); tp=base.run_frozen(g,C,True); pace=base.pace_predictions(g); games=base.make_games(tp,pace)
    cr,meta=component_rows(g)
    if cr is None:
        result={'compiler':'HOLD','meta':meta}; print('COMPONENT_RETRY');print(json.dumps(result,indent=2));return
    cg=game_features(games,cr)
    vars=[c for c in cg.columns if c.endswith('_diff') or c.endswith('_avg')]
    cohorts=[]
    for v in vars:
        z=cg.dropna(subset=[v]).copy();z['bucket']=pd.qcut(z[v],5,labels=['Q1','Q2','Q3','Q4','Q5'],duplicates='drop')
        for (sy,b),q in z.groupby(['season','bucket'],observed=True):
            cohorts.append({'feature':v,'season':int(sy),'bucket':str(b),'n':len(q),'margin_bias':float(q.margin_resid.mean()),'margin_mae':mae(q.margin_resid),'total_bias':float(q.total_resid.mean()),'total_mae':mae(q.total_resid)})
    perf=[]
    for sy in [2023,2024,2025]:
        tr=cg[(cg.season>=2022)&(cg.season<sy)].dropna(subset=vars);te=cg[cg.season==sy].dropna(subset=vars)
        model=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(tr[vars],tr.margin_resid)
        corr=model.predict(te[vars]); perf.append({'season':sy,'n':len(te),'base_mae':mae(te.margin_resid),'new_mae':mae(te.margin_resid-corr),'delta':mae(te.margin_resid-corr)-mae(te.margin_resid)})
    p=pd.DataFrame(perf); co=pd.DataFrame(cohorts)
    result={'compiler':'PASS','component_rows':len(cr),'game_rows':len(cg),'perf':perf,'mean_delta':float(p.delta.mean()),'wins':int((p.delta<0).sum()),'max_abs_margin_bias':float(co.margin_bias.abs().max()),'max_abs_total_bias':float(co.total_bias.abs().max()),'incremental_decision':'PASS' if float(p.delta.mean())<=-.05 and int((p.delta<0).sum())>=2 else 'REJECTED'}
    co.to_csv(OUT/'component_cohorts.csv',index=False);p.to_csv(OUT/'component_perf.csv',index=False);(OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print('COMPONENT_RETRY');print(json.dumps(result,indent=2))
if __name__=='__main__': main()
