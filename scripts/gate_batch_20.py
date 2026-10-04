from __future__ import annotations

import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import gate_batch_15 as base

OUT=Path('results/gate_batch_20'); OUT.mkdir(parents=True,exist_ok=True)
MKT='https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/'

def mae(x): return float(np.mean(np.abs(np.asarray(x,float))))
def normname(x): return re.sub(r'[^a-z0-9]','',str(x).lower())

def total_calibration(games):
    rows=[]
    for sy in [2023,2024,2025]:
        tr=games[(games.season>=2022)&(games.season<sy)].copy(); te=games[games.season==sy].copy()
        global_mu=float(tr.total_resid.mean())
        # stage means are learned only on prior seasons
        tr['stage']=pd.cut(tr.n_avg,[-1,1,4,9,1e9],labels=['0-1','2-4','5-9','10+'])
        te['stage']=pd.cut(te.n_avg,[-1,1,4,9,1e9],labels=['0-1','2-4','5-9','10+'])
        sm=tr.groupby('stage',observed=True).total_resid.mean().to_dict()
        tr['mb']=pd.cut(tr.pred_margin.abs(),[-.001,3,7,12,1e9],labels=['0-3','3-7','7-12','12+'])
        te['mb']=pd.cut(te.pred_margin.abs(),[-.001,3,7,12,1e9],labels=['0-3','3-7','7-12','12+'])
        mm=tr.groupby('mb',observed=True).total_resid.mean().to_dict()
        base_err=te.total_resid.values
        e_global=base_err-global_mu
        e_stage=base_err-te.stage.map(sm).astype(float).fillna(global_mu).values
        e_margin=base_err-te.mb.map(mm).astype(float).fillna(global_mu).values
        rows.append({'season':sy,'n':len(te),'learned_global':global_mu,
                     'base_mae':mae(base_err),'global_mae':mae(e_global),'stage_mae':mae(e_stage),'marginband_mae':mae(e_margin),
                     'base_bias':float(base_err.mean()),'global_bias':float(e_global.mean()),'stage_bias':float(e_stage.mean()),'marginband_bias':float(e_margin.mean())})
    return pd.DataFrame(rows)

def pick(df,opts):
    return next((c for c in opts if c in df.columns),None)

def component_rows(g):
    fgm=pick(g,['fgm','field_goals_made']); fga=pick(g,['fga','field_goals_attempted'])
    tpm=pick(g,['three_point_field_goals_made','three_pointers_made','three_points_made','fg3m','threePointFieldGoalsMade'])
    tov=pick(g,['total_turnovers','turnovers','totalTurnovers']); orb=pick(g,['orb','offensive_rebounds','offensiveRebounds'])
    drb=pick(g,['drb','defensive_rebounds','defensiveRebounds']); fta=pick(g,['fta','free_throws_attempted','freeThrowsAttempted'])
    needed=[fgm,fga,tpm,tov,orb,drb,fta]
    if any(c is None for c in needed):
        return None, {'missing':needed,'columns':list(g.columns)}
    z=g.copy()
    for c in needed: z[c]=pd.to_numeric(z[c],errors='coerce')
    z['efg']=(z[fgm]+.5*z[tpm])/z[fga].replace(0,np.nan)
    z['tovr']=z[tov]/z.poss.replace(0,np.nan)
    z['ftr']=z[fta]/z[fga].replace(0,np.nan)
    # opponent DRB for ORB%
    od=z[['game_id','ix',drb]].copy(); od['ix']=1-od.ix; od=od.rename(columns={drb:'opp_drb'})
    z=z.merge(od,on=['game_id','ix'],how='left',validate='one_to_one')
    z['orbp']=z[orb]/(z[orb]+z.opp_drb).replace(0,np.nan)
    comps=['efg','tovr','orbp','ftr']
    # leakage-safe current-season component offense/defense states; same-date isolation
    off={}; deff={}; cnt={}; rows=[]; prev=None
    global_defaults={'efg':.50,'tovr':.20,'orbp':.30,'ftr':.30}
    for date,day in z.sort_values(['date','game_id','team']).groupby('date',sort=True):
        sy=int(day.season.iloc[0])
        if prev!=sy: off={};deff={};cnt={};prev=sy
        drows=[]
        for r in day.itertuples(index=False):
            a=r.team;b=r.opp;n=cnt.get(a,0); nb=cnt.get(b,0)
            rec={'season':sy,'date':date,'game_id':r.game_id,'team':a,'opp':b,'prior_games':n}
            for c in comps:
                rec[f'off_{c}']=off.get((a,c),global_defaults[c])
                rec[f'oppdef_{c}']=deff.get((b,c),global_defaults[c])
                rec[f'match_{c}']=rec[f'off_{c}']+rec[f'oppdef_{c}']-global_defaults[c]
                rec[f'actual_{c}']=getattr(r,c)
            rows.append(rec); drows.append(r)
        # update after all same-date predictions
        bygame={}
        for r in drows: bygame[(r.game_id,r.team)]=r
        for r in drows:
            a=r.team;b=r.opp;n=cnt.get(a,0)
            opp=bygame.get((r.game_id,b))
            for c in comps:
                v=getattr(r,c)
                if pd.notna(v): off[(a,c)]=(off.get((a,c),0)*n+v)/(n+1)
                if opp is not None:
                    ov=getattr(opp,c)
                    if pd.notna(ov): deff[(a,c)]=(deff.get((a,c),0)*n+ov)/(n+1)
            cnt[a]=n+1
    return pd.DataFrame(rows), {'columns':needed}

def component_games(games,cr):
    if cr is None: return None
    feats=['match_efg','match_tovr','match_orbp','match_ftr']
    x=cr[['game_id','team']+feats].copy()
    out=[]
    for r in games.itertuples(index=False):
        q=x[x.game_id==r.game_id]
        a=q[q.team==r.team0];b=q[q.team==r.team1]
        if len(a)!=1 or len(b)!=1: continue
        rec={'game_id':r.game_id,'season':r.season,'date':r.date,'margin_resid':r.margin_resid,'total_resid':r.total_resid,'pred_margin':r.pred_margin,'pred_total':r.pred_total,'n_avg':r.n_avg}
        for f in feats:
            rec[f+'_diff']=float(a.iloc[0][f]-b.iloc[0][f]); rec[f+'_avg']=float((a.iloc[0][f]+b.iloc[0][f])/2)
        out.append(rec)
    return pd.DataFrame(out)

def component_diagnostics(cg):
    if cg is None or cg.empty: return pd.DataFrame(),pd.DataFrame()
    vars=[c for c in cg.columns if c.endswith('_diff') or c.endswith('_avg')]
    rows=[]
    for v in vars:
        z=cg.dropna(subset=[v]).copy(); z['bucket']=pd.qcut(z[v],5,labels=['Q1','Q2','Q3','Q4','Q5'],duplicates='drop')
        for (sy,b),q in z.groupby(['season','bucket'],observed=True):
            rows.append({'feature':v,'season':int(sy),'bucket':str(b),'n':len(q),'margin_bias':float(q.margin_resid.mean()),'margin_mae':float(q.margin_resid.abs().mean()),'total_bias':float(q.total_resid.mean()),'total_mae':float(q.total_resid.abs().mean())})
    d=pd.DataFrame(rows)
    # legal incremental ridge from components, expanding seasons
    perf=[]
    for sy in [2023,2024,2025]:
        tr=cg[(cg.season>=2022)&(cg.season<sy)].dropna(subset=vars); te=cg[cg.season==sy].dropna(subset=vars)
        if len(tr)<500 or len(te)<200: continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(tr[vars],tr.margin_resid)
        corr=m.predict(te[vars]); perf.append({'season':sy,'n':len(te),'base_mae':mae(te.margin_resid),'new_mae':mae(te.margin_resid-corr),'delta':mae(te.margin_resid-corr)-mae(te.margin_resid)})
    return d,pd.DataFrame(perf)

def nonlinear_ml(games):
    feats=['pred_margin','pred_total','n_avg']; rows=[]; pred_store={}
    for sy in [2023,2024,2025]:
        tr=games[(games.season>=2022)&(games.season<sy)].copy();te=games[games.season==sy].copy(); y=tr.margin_resid.values
        models={
          'spline':make_pipeline(SplineTransformer(n_knots=5,degree=3),Ridge(alpha=100.0)),
          'hgb':HistGradientBoostingRegressor(max_iter=120,max_depth=3,learning_rate=.04,l2_regularization=5,random_state=7),
          'rf':RandomForestRegressor(n_estimators=160,max_depth=6,min_samples_leaf=40,n_jobs=-1,random_state=7),
        }
        base_mae=mae(te.margin_resid)
        for name,m in models.items():
            m.fit(tr[feats],y); p=m.predict(te[feats]); new=mae(te.margin_resid-p)
            rows.append({'season':sy,'model':name,'n':len(te),'base_mae':base_mae,'new_mae':new,'delta':new-base_mae})
            pred_store[(sy,name)]=p
        ens=(pred_store[(sy,'spline')]+pred_store[(sy,'hgb')]+pred_store[(sy,'rf')])/3
        rows.append({'season':sy,'model':'ensemble3','n':len(te),'base_mae':base_mae,'new_mae':mae(te.margin_resid-ens),'delta':mae(te.margin_resid-ens)-base_mae})
    return pd.DataFrame(rows)

def market_data():
    scores=pd.read_csv(MKT+'canonicalized/scores/fg/games_all_canonical.csv')
    sp=pd.read_csv(MKT+'odds/canonical/spreads/fg/spreads_fg_all.csv')
    tt=pd.read_csv(MKT+'odds/canonical/totals/fg/totals_fg_all.csv')
    for d in [scores,sp,tt]:
        if 'date' in d: d['date']=pd.to_datetime(d['date'],errors='coerce').dt.date
        if 'game_date' in d: d['game_date']=pd.to_datetime(d['game_date'],errors='coerce').dt.date
    for d in [sp,tt]:
        d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='coerce'); d['commence_time']=pd.to_datetime(d.commence_time,utc=True,errors='coerce')
        d.dropna(subset=['timestamp','commence_time'],inplace=True); d=d[d.timestamp<=d.commence_time]
    # latest available pregame timestamp for event, then median across books
    def consensus(d,val):
        z=d[d.timestamp<=d.commence_time].copy(); mx=z.groupby('event_id').timestamp.transform('max'); z=z[z.timestamp==mx]
        a=z.groupby('event_id').agg(game_date=('game_date','first'),home=('home_team_canonical','first'),away=('away_team_canonical','first'),commence=('commence_time','first'),snapshot=('timestamp','max'),books=('bookmaker','nunique'),line=(val,'median'),line_sd=(val,'std')).reset_index()
        return a
    return scores,consensus(sp,'spread'),consensus(tt,'total'),sp,tt

def match_market(games,scores,spc,ttc):
    s=scores.copy(); s['game_date']=pd.to_datetime(s.date,errors='coerce').dt.date
    # unique score-pair/date key; high precision, sacrificing ambiguous matches
    s['lo']=s[['home_score','away_score']].min(axis=1);s['hi']=s[['home_score','away_score']].max(axis=1)
    dup=s.groupby(['game_date','lo','hi']).size(); good=set(dup[dup==1].index)
    gm=games.copy(); gm['game_date']=pd.to_datetime(gm.date).dt.date;gm['lo']=gm[['actual0','actual1']].min(axis=1);gm['hi']=gm[['actual0','actual1']].max(axis=1)
    gm['key']=list(zip(gm.game_date,gm.lo,gm.hi));gm=gm[gm.key.isin(good)].copy()
    s['key']=list(zip(s.game_date,s.lo,s.hi)); sm=s[['key','home_canonical','away_canonical','home_score','away_score']]
    j=gm.merge(sm,on='key',how='inner')
    # orient model to home by final-score identity
    home_is_0=(j.actual0==j.home_score)&(j.actual1==j.away_score)
    home_is_1=(j.actual1==j.home_score)&(j.actual0==j.away_score)
    j=j[home_is_0|home_is_1].copy(); h0=(j.actual0==j.home_score)&(j.actual1==j.away_score)
    j['model_home_margin']=np.where(h0,j.pred_margin,-j.pred_margin);j['actual_home_margin']=j.home_score-j.away_score
    # attach consensus by canonical names/date
    spc=spc.rename(columns={'home':'home_canonical','away':'away_canonical','line':'spread_line','books':'spread_books','line_sd':'spread_sd','snapshot':'spread_snapshot'})
    ttc=ttc.rename(columns={'home':'home_canonical','away':'away_canonical','line':'total_line','books':'total_books','line_sd':'total_sd','snapshot':'total_snapshot'})
    j=j.merge(spc[['game_date','home_canonical','away_canonical','spread_line','spread_books','spread_sd','spread_snapshot']],on=['game_date','home_canonical','away_canonical'],how='left')
    j=j.merge(ttc[['game_date','home_canonical','away_canonical','total_line','total_books','total_sd','total_snapshot']],on=['game_date','home_canonical','away_canonical'],how='left')
    return j

def market_tests(j,total_shift):
    out={}; spread=j.dropna(subset=['spread_line']).copy(); total=j.dropna(subset=['total_line']).copy()
    out['coverage']={'matched_games':len(j),'spread_games':len(spread),'total_games':len(total),'spread_book_median':float(spread.spread_books.median()) if len(spread) else np.nan,'total_book_median':float(total.total_books.median()) if len(total) else np.nan}
    if len(spread):
        spread['mkt_margin']=-spread.spread_line; spread['edge']=spread.model_home_margin-spread.mkt_margin; spread['ats_resid']=spread.actual_home_margin-spread.mkt_margin
        spread=spread[spread.edge.abs()>.25].copy(); spread['hit']=(np.sign(spread.edge)*spread.ats_resid>0).astype(int);spread['push']=(spread.ats_resid.abs()<1e-9)
        spread['bucket']=pd.cut(spread.edge.abs(),[.25,2,4,6,8,12,1e9],labels=['.25-2','2-4','4-6','6-8','8-12','12+'])
        out['spread_buckets']=spread.groupby('bucket',observed=True).agg(n=('hit','size'),hit_rate=('hit','mean'),mean_edge=('edge',lambda x:float(np.mean(np.abs(x)))),push_rate=('push','mean')).reset_index().to_dict('records')
    if len(total):
        total['model_total_corr']=total.pred_total+total_shift; total['edge']=total.model_total_corr-total.total_line; total['ou_resid']=total.actual_total-total.total_line
        total=total[total.edge.abs()>.25].copy(); total['hit']=(np.sign(total.edge)*total.ou_resid>0).astype(int); total['push']=(total.ou_resid.abs()<1e-9)
        total['bucket']=pd.cut(total.edge.abs(),[.25,2,4,6,8,12,1e9],labels=['.25-2','2-4','4-6','6-8','8-12','12+'])
        out['total_buckets']=total.groupby('bucket',observed=True).agg(n=('hit','size'),hit_rate=('hit','mean'),mean_edge=('edge',lambda x:float(np.mean(np.abs(x)))),push_rate=('push','mean')).reset_index().to_dict('records')
    return out

def main():
    g=base.load_games(); C=base.continuity(); tp=base.run_frozen(g,C,True); pace=base.pace_predictions(g); games=base.make_games(tp,pace)
    tc=total_calibration(games)
    cr,cmeta=component_rows(g); cg=component_games(games,cr); cohorts,cperf=component_diagnostics(cg)
    ml=nonlinear_ml(games)
    scores,spc,ttc,spraw,ttraw=market_data(); joined=match_market(games,scores,spc,ttc)
    shift=float(tc.learned_global.iloc[-1])
    mt=market_tests(joined,shift)
    # summaries and governed decisions
    td={c:float(tc[c].mean()) for c in ['base_mae','global_mae','stage_mae','marginband_mae','base_bias','global_bias','stage_bias','marginband_bias']}
    best_total=min(['global_mae','stage_mae','marginband_mae'],key=lambda c:td[c]); global_gain=td['base_mae']-td['global_mae']
    comp_delta=float(cperf.delta.mean()) if len(cperf) else np.nan; comp_wins=int((cperf.delta<0).sum()) if len(cperf) else 0
    ml_sum={m:{'mean_delta':float(q.delta.mean()),'wins':int((q.delta<0).sum())} for m,q in ml.groupby('model')}
    def monotonic(rows):
        if not rows or len(rows)<3:return False
        r=pd.DataFrame(rows); return float(r.tail(2).hit_rate.mean())>float(r.head(2).hit_rate.mean()) and float(r.tail(2).hit_rate.mean())>.50
    spmono=monotonic(mt.get('spread_buckets')); ttmono=monotonic(mt.get('total_buckets'))
    decisions={
      'G01_total_global_intercept':{'decision':'PASS' if global_gain>.05 else 'REJECTED','evidence':{'mean_base_mae':td['base_mae'],'mean_global_mae':td['global_mae'],'gain':global_gain,'mean_post_bias':td['global_bias']}},
      'G02_total_stage_correction':{'decision':'PASS' if td['stage_mae']+0.02<td['global_mae'] else 'REJECTED','evidence':{'global':td['global_mae'],'stage':td['stage_mae']}},
      'G03_total_marginband_correction':{'decision':'PASS' if td['marginband_mae']+0.02<td['global_mae'] else 'REJECTED','evidence':{'global':td['global_mae'],'marginband':td['marginband_mae']}},
      'G04_total_mean_solution':{'decision':'PASS','evidence':{'best':best_total,'table':tc.to_dict('records'),'rule':'promote simplest correction within 0.02 MAE of best'}},
      'G05_component_compiler':{'decision':'PASS' if cr is not None and len(cr)>40000 else 'HOLD','evidence':{'rows':0 if cr is None else len(cr),'meta':cmeta}},
      'G06_component_extreme_cohorts':{'decision':'PASS' if len(cohorts)>0 else 'HOLD','evidence':{'rows':len(cohorts),'max_abs_margin_bias':float(cohorts.margin_bias.abs().max()) if len(cohorts) else np.nan}},
      'G07_component_replication':{'decision':'PASS' if len(cperf)==3 else 'HOLD','evidence':cperf.to_dict('records')},
      'G08_T08_01_component_incremental':{'decision':'PASS' if comp_delta<=-.05 and comp_wins>=2 else 'REJECTED','evidence':{'mean_delta':comp_delta,'wins':comp_wins}},
      'G09_T08_02_resistance':{'decision':'HOLD' if comp_delta<=-.05 and comp_wins>=2 else 'REJECTED','evidence':{'reason':'only earned if base component states materially pass'}},
      'G10_T08_03_matchup_integration':{'decision':'HOLD' if comp_delta<=-.05 and comp_wins>=2 else 'REJECTED','evidence':{'reason':'only integrate promoted component interactions'}},
      'G11_T10_03_spline':{'decision':'PASS' if ml_sum['spline']['mean_delta']<=-.05 and ml_sum['spline']['wins']>=2 else 'REJECTED','evidence':ml_sum['spline']},
      'G12_T10_04_hgb':{'decision':'PASS' if ml_sum['hgb']['mean_delta']<=-.05 and ml_sum['hgb']['wins']>=2 else 'REJECTED','evidence':ml_sum['hgb']},
      'G13_T10_04_rf':{'decision':'PASS' if ml_sum['rf']['mean_delta']<=-.05 and ml_sum['rf']['wins']>=2 else 'REJECTED','evidence':ml_sum['rf']},
      'G14_T10_05_ensemble':{'decision':'PASS' if ml_sum['ensemble3']['mean_delta']<=-.05 and ml_sum['ensemble3']['wins']>=2 else 'REJECTED','evidence':ml_sum['ensemble3']},
      'G15_T11_01_line_corpus':{'decision':'PASS','evidence':{'spread_rows':len(spraw),'total_rows':len(ttraw),'spread_events':int(spraw.event_id.nunique()),'total_events':int(ttraw.event_id.nunique())}},
      'G16_T11_01_timestamp_book_coverage':{'decision':'PASS' if mt['coverage']['spread_book_median']>=3 and mt['coverage']['total_book_median']>=3 else 'HOLD','evidence':mt['coverage']},
      'G17_T11_01_model_market_join':{'decision':'PASS' if mt['coverage']['spread_games']>=1500 and mt['coverage']['total_games']>=1500 else 'HOLD','evidence':mt['coverage']},
      'G18_T11_02_no_vig_prices':{'decision':'HOLD','evidence':{'reason':'public canonical corpus preserves line/book/timestamp but not quoted odds price/juice; exact no-vig probabilities and ROI cannot be certified'}},
      'G19_T11_03_04_line_edge_informativeness':{'decision':'PASS' if (spmono or ttmono) else 'HOLD','evidence':{'spread_monotonic':spmono,'total_monotonic':ttmono,'spread_buckets':mt.get('spread_buckets',[]),'total_buckets':mt.get('total_buckets',[])}},
      'G20_T12_expression_readiness':{'decision':'HOLD','evidence':{'reason':'line disagreement can be tested, but contract choice/EV requires quoted prices; keep side/total expression selection open until price corpus exists'}}
    }
    summary={'total_calibration':tc.to_dict('records'),'total_summary':td,'component_perf':cperf.to_dict('records'),'nonlinear_ml':ml.to_dict('records'),'market':mt,'decisions':decisions}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    tc.to_csv(OUT/'total_calibration.csv',index=False); cohorts.to_csv(OUT/'component_cohorts.csv',index=False); cperf.to_csv(OUT/'component_perf.csv',index=False); ml.to_csv(OUT/'nonlinear_ml.csv',index=False); joined.to_csv(OUT/'market_join_sample.csv',index=False)
    print('GATE20_RESULTS'); print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
