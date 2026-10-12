from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import prediction_first_alt_factors15 as o
import prediction_first_alt_factors15_v2 as r

OUT=Path('results/prediction_first_alt_factors15_v3'); OUT.mkdir(parents=True,exist_ok=True)

def sfold(q,features):
    avail=[]; coverage={}
    for c in features:
        if c in q:
            cv=float(q[c].notna().mean()); coverage[c]=cv
            if cv>=.50: avail.append(c)
        else: coverage[c]=0.0
    if not avail:return pd.DataFrame(),avail,coverage
    rows=[]
    for sy in [2024,2025]:
        tr=q[q.season.lt(sy)].dropna(subset=['resid']+avail); te=q[q.season.eq(sy)].dropna(subset=['resid']+avail)
        if len(tr)<300 or len(te)<200: continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[avail],tr.resid); pred=m.predict(te[avail])
        b=mean_absolute_error(te.actual_margin,te.base_margin); c=mean_absolute_error(te.actual_margin,te.base_margin+pred)
        rows.append({'season':sy,'n':len(te),'base_mae':b,'challenger_mae':c,'gain':b-c})
    return pd.DataFrame(rows),avail,coverage

def mg(x):return float(x.gain.mean()) if len(x) else np.nan

def main():
    a,q=r.build(); G=[]
    def A(i,s,d,res,n):G.append({'gate':f'T34.{i:02d}','status':s,'decision':d,'result':res,'next_action':n})
    # 1 TS vs eFG persistence
    rows=[]
    for sy in [2023,2024,2025]:
        z=a[a.season.eq(sy)]; e=z[z.game_no.le(5)].groupby('team_id')[['ts','efg']].mean(); l=z[z.game_no.gt(5)].groupby('team_id')[['ts','efg']].mean(); x=e.join(l,lsuffix='_e',rsuffix='_l').dropna()
        rows.append({'season':sy,'n':len(x),'ts_rho':o.corr(x.ts_e,x.ts_l),'efg_rho':o.corr(x.efg_e,x.efg_l)})
    A(1,'GREEN','PASS — TS% VS eFG% PERSISTENCE MAPPED',str(rows),'Compare direct v14 residual lift')
    # helper
    def run(fs):return sfold(q,fs)
    # 2 shooting
    efg,ae,ce=run(['d_pre_efg']); ts,at,ct=run(['d_pre_ts']); both,ab,cb=run(['d_pre_efg','d_pre_ts']); vals=[x for x in [mg(efg),mg(ts),mg(both)] if np.isfinite(x)]
    A(2,'GREEN' if vals and max(vals)>0 else 'REJECT','PASS — SHOOTING FORMULATION HAS RESIDUAL SIGNAL' if vals and max(vals)>0 else 'REJECT — SHOOTING FORMULATION DOES NOT ADD',f'efg={efg.to_dict("records")}; ts={ts.to_dict("records")}; both={both.to_dict("records")}; coverage={cb}','Retain simplest repeatable formulation')
    # 3 rebounds
    orb,_,co=run(['d_pre_orbr']); od,av,cod=run(['d_pre_orbr','d_pre_drbr']); inc=mg(od)-mg(orb) if len(od) and len(orb) else np.nan
    st='GREEN' if np.isfinite(inc) and inc>0 else ('HOLD' if cod.get('d_pre_drbr',0)<.5 else 'REJECT')
    A(3,st,'PASS — DRB ADDS BEYOND ORB' if st=='GREEN' else ('HOLD — DRB COVERAGE INSUFFICIENT' if st=='HOLD' else 'REJECT — DRB DOES NOT ADD'),f'orb={orb.to_dict("records")}; orb_drb={od.to_dict("records")}; coverage={cod}; incremental={inc}','Repair/retain DRB only if it survives nested OOS test')
    # 4 turnovers
    sep,_,cs=run(['h_pre_tovr','a_pre_tovr']); dif,_,cd=run(['d_pre_tovr']); vals=[x for x in [mg(sep),mg(dif)] if np.isfinite(x)]
    A(4,'GREEN' if vals and max(vals)>0 else 'REJECT','PASS — TURNOVER REPRESENTATION ADDS' if vals and max(vals)>0 else 'REJECT — TURNOVER REPRESENTATION DOES NOT ADD',f'separate={sep.to_dict("records")}; differential={dif.to_dict("records")}; coverage={cs|cd}','Prefer differential unless separate sides materially win')
    # 5 FTR after TS
    ts0,_,_=run(['d_pre_ts']); tsf,_,cf=run(['d_pre_ts','d_pre_ftr']); inc5=mg(tsf)-mg(ts0) if len(tsf) and len(ts0) else np.nan
    A(5,'GREEN' if np.isfinite(inc5) and inc5>0 else 'REJECT','PASS — FTR ADDS AFTER TS' if np.isfinite(inc5) and inc5>0 else 'REJECT — FTR REDUNDANT AFTER TS',f'ts={ts0.to_dict("records")}; ts_ftr={tsf.to_dict("records")}; coverage={cf}; incremental={inc5}','Drop FTR from mean model if redundant')
    # 6 weights with available factors
    fac=['d_pre_ts','d_pre_tovr','d_pre_orbr','d_pre_drbr','d_pre_ftr']; fw,av,cf=run(fac)
    A(6,'GREEN' if len(fw) and mg(fw)>0 else ('HOLD' if len(av)<4 else 'REJECT'),'PASS — EMPIRICAL FACTOR WEIGHTS IMPROVE V14' if len(fw) and mg(fw)>0 else ('HOLD — FACTOR FAMILY INCOMPLETE' if len(av)<4 else 'REJECT — WEIGHTS DO NOT ADD'),f'available={av}; coverage={cf}; folds={fw.to_dict("records")}; mean_gain={mg(fw)}','Audit coefficient stability only for usable factors')
    # 7 coefficients
    coeff=[]
    if len(av):
      for sy in [2024,2025]:
        tr=q[q.season.lt(sy)].dropna(subset=['resid']+av)
        if len(tr)<300:continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[av],tr.resid); coeff.append({'season':sy,**{f:float(v) for f,v in zip(av,m[-1].coef_)}})
    stable=0; signs={}
    if len(coeff)>=2:
      cdf=pd.DataFrame(coeff)
      for f in av: signs[f]=bool(np.sign(cdf[f]).nunique()==1); stable+=int(signs[f])
    else:cdf=pd.DataFrame(coeff)
    A(7,'GREEN' if len(coeff)>=2 and stable>=max(2,len(av)//2) else 'HOLD','PASS — FACTOR WEIGHT SIGNS MOSTLY STABLE' if len(coeff)>=2 and stable>=max(2,len(av)//2) else 'HOLD — FACTOR WEIGHTS NOT STABLE/COMPLETE',f'coefficients={coeff}; sign_stability={signs}','Interpret only stable coefficient families')
    # 8 combined factors
    fc,avc,cc=run(['d_pre_ts','d_pre_tovr','d_pre_orbr','d_pre_drbr','d_pre_ftr','d_pre_three_rate','d_pre_poss'])
    A(8,'GREEN' if len(fc) and mg(fc)>0 else 'REJECT','PASS — COMBINED ALTERNATIVE FACTORS ADD' if len(fc) and mg(fc)>0 else 'REJECT — COMBINED ALTERNATIVE FACTORS DO NOT ADD',f'available={avc}; folds={fc.to_dict("records")}; mean_gain={mg(fc)}','Run ablations on any positive family next')
    # 9-13 form
    l5,_,_=run(['d_l5_margin']); l7,_,_=run(['d_l7_margin']); a5,_,_=run(['d_adj_l5_margin']); a7,_,_=run(['d_adj_l7_margin']); r5,_,_=run(['d_rec_l5_margin']); r7,_,_=run(['d_rec_l7_margin'])
    A(9,'GREEN' if len(l5) and mg(l5)>0 else 'REJECT','PASS — RAW L5 ADDS' if len(l5) and mg(l5)>0 else 'REJECT — RAW L5 DOES NOT ADD',f'{l5.to_dict("records")}; gain={mg(l5)}','Compare adjusted form')
    A(10,'GREEN' if len(l7) and mg(l7)>0 else 'REJECT','PASS — RAW L7 ADDS' if len(l7) and mg(l7)>0 else 'REJECT — RAW L7 DOES NOT ADD',f'{l7.to_dict("records")}; gain={mg(l7)}','Compare adjusted form')
    A(11,'GREEN' if len(a5) and mg(a5)>max(0,mg(l5)) else 'HOLD','PASS — ADJUSTED L5 BEATS RAW' if len(a5) and mg(a5)>mg(l5) else 'HOLD — ADJUSTED L5 DOES NOT BEAT RAW',f'raw={mg(l5)}; adjusted={mg(a5)}; folds={a5.to_dict("records")}','Keep only if incremental')
    A(12,'GREEN' if len(a7) and mg(a7)>max(0,mg(l7)) else 'HOLD','PASS — ADJUSTED L7 BEATS RAW' if len(a7) and mg(a7)>mg(l7) else 'HOLD — ADJUSTED L7 DOES NOT BEAT RAW',f'raw={mg(l7)}; adjusted={mg(a7)}; folds={a7.to_dict("records")}','Keep only if incremental')
    rawbest=max([x for x in [mg(l5),mg(l7)] if np.isfinite(x)] or [-999]); recbest=max([x for x in [mg(r5),mg(r7)] if np.isfinite(x)] or [-999])
    A(13,'GREEN' if recbest>max(0,rawbest) else 'HOLD','PASS — RECURSIVE OPPONENT FORM ADDS' if recbest>max(0,rawbest) else 'HOLD — RECURSIVE FORM DOES NOT BEAT RAW',f'raw5={mg(l5)}; raw7={mg(l7)}; rec5={mg(r5)}; rec7={mg(r7)}','Stop at this recursion depth unless clear gain')
    # 14 side residual decomposition
    z=q.copy(); z['eh']=z.actual_home-z.base_home; z['ea']=z.actual_away-z.base_away; rows=[]
    for sy in [2023,2024,2025]:
      x=z[z.season.eq(sy)]; rows.append({'season':sy,'n':len(x),'home_resid':float(x.eh.mean()),'away_resid':float(x.ea.mean()),'margin_resid':float((x.eh-x.ea).mean())})
    A(14,'GREEN','PASS — HOME/AWAY SCORE VALUE DECOMPOSED',str(rows),'Next: explain side residuals through venue-linked factor mechanisms')
    # 15 matchup interactions robustly using available columns
    candidates={
      'three_alignment': q.h_pre_three_rate*q.a_pre_efg-q.a_pre_three_rate*q.h_pre_efg,
      'rebound_alignment': q.h_pre_orbr*(1-q.a_pre_drbr)-q.a_pre_orbr*(1-q.h_pre_drbr) if q[['h_pre_drbr','a_pre_drbr']].notna().mean().min()>.5 else pd.Series(np.nan,index=q.index),
      'tov_stress': -(q.h_pre_tovr*q.a_pre_tovr),
      'ftr_stress': -(q.h_pre_ftr*q.a_pre_ftr)}
    singles={}
    for k,v in candidates.items(): q['mx_'+k]=v; x,_,_=run(['mx_'+k]); singles[k]=mg(x)
    use=['mx_'+k for k,v in singles.items() if np.isfinite(v)]; allx,_,_=run(use) if use else (pd.DataFrame(),[],{})
    best=max([v for v in singles.values() if np.isfinite(v)] or [-999]); cg=mg(allx)
    A(15,'GREEN' if best>0 or (np.isfinite(cg) and cg>0) else 'REJECT','PASS — INITIAL MATCHUP RESIDUAL SCREEN FINDS SIGNAL' if best>0 or (np.isfinite(cg) and cg>0) else 'REJECT — INITIAL MATCHUP INTERACTIONS ADD NO SIGNAL',f'single_gains={singles}; combined_features={use}; combined={allx.to_dict("records")}; combined_gain={cg}','Preregister only positive interactions for mechanistic replication')
    pd.DataFrame(G).to_csv(OUT/'t34_alt_factors_15_gates.csv',index=False); cdf.to_csv(OUT/'t34_factor_coefficients.csv',index=False)
    miss={c:float(q[c].notna().mean()) for c in q.columns if c.startswith(('d_pre_','d_l','d_adj','d_rec'))}; pd.DataFrame([miss]).to_csv(OUT/'t34_feature_coverage.csv',index=False)
    json.dump({'gates':G,'joined_games':len(q),'green':sum(x['status']=='GREEN' for x in G),'hold':sum(x['status']=='HOLD' for x in G),'reject':sum(x['status']=='REJECT' for x in G),'production_reopen':False,'outcomes_2026_used':False},open(OUT/'summary.json','w'),indent=2)
    print(pd.DataFrame(G).to_string(index=False))
if __name__=='__main__':main()
