from __future__ import annotations

import json, re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT=Path('results/prediction_first_venue_alias_final'); OUT.mkdir(parents=True,exist_ok=True)
G15=Path('/tmp/g15')
NCAA='https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/ncaa_mbb_schedule_master.parquet'
ESPN='https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{}.parquet'
EVAL=[2023,2024,2025]


def nn(x): return re.sub(r'[^a-z0-9]','',str(x).lower()) if pd.notna(x) else ''

def sim(a,b): return SequenceMatcher(None,a,b).ratio()

def ece(p,y,bins=10):
    d=pd.DataFrame({'p':p,'y':y}).dropna();
    if d.empty:return np.nan
    d['b']=np.minimum((d.p*bins).astype(int),bins-1)
    return float(sum(len(z)/len(d)*abs(z.p.mean()-z.y.mean()) for _,z in d.groupby('b')))

def logloss(p,y):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))

# Frozen prediction object + NCAA home/away orientation
g=pd.read_parquet(G15/'game_predictions.parquet'); g['game_id']=g.game_id.astype(str); g['date']=pd.to_datetime(g.date).dt.normalize()
s=pd.read_parquet(NCAA); gid='game_id' if 'game_id' in s.columns else 'contest_id'; s=s.rename(columns={gid:'game_id'}); s['game_id']=s.game_id.astype(str); s['hk']=s.home.map(nn); s['ak']=s.away.map(nn)
b=g.merge(s[['game_id','home','away','hk','ak']].drop_duplicates('game_id'),on='game_id',how='left'); b['t0']=b.team0.map(nn); b['t1']=b.team1.map(nn); b['h0']=b.t0.eq(b.hk); b['h1']=b.t1.eq(b.hk); b=b[b.h0^b.h1].copy();
b['pred_home']=np.where(b.h0,b.pred0,b.pred1); b['pred_away']=np.where(b.h0,b.pred1,b.pred0); b['actual_home']=np.where(b.h0,b.actual0,b.actual1); b['actual_away']=np.where(b.h0,b.actual1,b.actual0); b['pred_margin']=b.pred_home-b.pred_away; b['actual_margin']=b.actual_home-b.actual_away; b['base_resid']=b.actual_margin-b.pred_margin

# ESPN venue/neutral schedule. Location fields are school labels rather than mascot display names.
xs=[]
for sy in range(2022,2026):
    d=pd.read_parquet(ESPN.format(sy)); d['file_sy']=sy; xs.append(d)
e=pd.concat(xs,ignore_index=True); e['edate']=pd.to_datetime(e.game_date).dt.normalize(); e['eh']=e.home_location.map(nn); e['ea']=e.away_location.map(nn)
bydate=defaultdict(list)
for r in e.itertuples(index=False): bydate[r.edate].append(r)

# Outcome-blind matching tiers. All use only date + two team strings.
rows=[]
for r in b.itertuples(index=False):
    cand=[]
    for day_offset in [0,-1,1]:
        dd=r.date+pd.Timedelta(days=day_offset)
        for c in bydate.get(dd,[]):
            sh=sim(r.hk,c.eh); sa=sim(r.ak,c.ea); joint=(sh+sa)/2
            cand.append((joint,min(sh,sa),max(sh,sa),-abs(day_offset),sh,sa,c))
    cand.sort(key=lambda x:(x[0],x[1],x[2],x[3]),reverse=True)
    chosen=None; method='unmatched'; top=runner=np.nan
    if cand:
        top=cand[0][0]; runner=cand[1][0] if len(cand)>1 else 0.0; margin=top-runner; j,mn,mx,_,sh,sa,c=cand[0]
        if sh==1.0 and sa==1.0 and c.edate==r.date: chosen=c; method='exact'
        elif sh==1.0 and sa==1.0: chosen=c; method='exact_pm1'
        elif j>=0.88 and margin>=0.03: chosen=c; method='fuzzy_strict'
        elif j>=0.72 and mn>=0.55 and margin>=0.06: chosen=c; method='fuzzy_balanced'
        elif j>=0.62 and mx>=0.90 and margin>=0.10: chosen=c; method='fuzzy_anchor'
    rec={'game_id':r.game_id,'match_method':method,'match_score':top,'runner_score':runner}
    if chosen is not None:
        rec.update({'neutral':bool(chosen.neutral_site) if pd.notna(chosen.neutral_site) else False,'venue_id':chosen.venue_id,'venue_full_name':chosen.venue_full_name,'venue_city':chosen.venue_address_city,'venue_state':chosen.venue_address_state,'espn_game_id':chosen.game_id})
    else: rec.update({'neutral':np.nan,'venue_id':np.nan,'venue_full_name':np.nan,'venue_city':np.nan,'venue_state':np.nan,'espn_game_id':np.nan})
    rows.append(rec)
m=pd.DataFrame(rows); ctx=b.merge(m,on='game_id',how='left',validate='one_to_one'); known=ctx[ctx.neutral.notna()].copy(); known['neutral']=known.neutral.astype(bool)
method_counts=m.match_method.value_counts().to_dict(); coverage=float(m.neutral.notna().mean())

# OOS venue candidates. Static 2.7 and learned pooled are both zeroed on neutral games.
season_rows=[]; fold_rows=[]
for sy in EVAL:
    tr=known[(known.season>=2022)&(known.season<sy)].copy(); te=known[known.season==sy].copy(); trn=tr[~tr.neutral]; pooled=float(trn.base_resid.mean()); hs=trn.groupby('hk').base_resid.agg(['mean','count'])
    te['hca_none']=0.0; te['hca_static27']=np.where(te.neutral,0.0,2.7); te['hca_pooled']=np.where(te.neutral,0.0,pooled)
    for k in [10,20]:
        vals=[]
        for rr in te.itertuples(index=False):
            if rr.neutral: vals.append(0.0)
            elif rr.hk in hs.index:
                n=float(hs.loc[rr.hk,'count']); mm=float(hs.loc[rr.hk,'mean']); w=n/(n+k); vals.append(pooled+w*(mm-pooled))
            else: vals.append(pooled)
        te[f'hca_team_k{k}']=vals
    for v in ['none','static27','pooled','team_k10','team_k20']:
        hc='hca_'+v; pc='pm_'+v; te[pc]=te.pred_margin+te[hc]
        # prior-season calibration for exact candidate
        tt=tr.copy()
        if v=='none': tt[hc]=0.0
        elif v=='static27': tt[hc]=np.where(tt.neutral,0.0,2.7)
        elif v=='pooled': tt[hc]=np.where(tt.neutral,0.0,pooled)
        else:
            k=int(v.split('k')[1]); vv=[]
            for rr in tt.itertuples(index=False):
                if rr.neutral: vv.append(0.0)
                elif rr.hk in hs.index:
                    n=float(hs.loc[rr.hk,'count']); mm=float(hs.loc[rr.hk,'mean']); w=n/(n+k); vv.append(pooled+w*(mm-pooled))
                else: vv.append(pooled)
            tt[hc]=vv
        tt[pc]=tt.pred_margin+tt[hc]; res=tt.actual_margin-tt[pc]; mu=float(res.mean()); sd=float(res.std(ddof=1)); p=norm.cdf((te[pc]+mu)/sd); y=(te.actual_margin>0).astype(int); rr=te.actual_margin-te[pc]
        season_rows.append({'season':sy,'variant':v,'n':len(te),'margin_mae':float(abs(rr).mean()),'margin_bias':float(rr.mean()),'brier':float(np.mean((p-y)**2)),'logloss':logloss(p,y),'ece':ece(p,y),'pooled_hca':pooled})
    fold_rows.append(te)
season=pd.DataFrame(season_rows); summary=season.groupby('variant').agg(margin_mae=('margin_mae','mean'),brier=('brier','mean'),logloss=('logloss','mean'),ece=('ece','mean')).sort_values(['margin_mae','brier'])
base=summary.loc['none']; none_s=season[season.variant=='none'].set_index('season').margin_mae; complexity={'static27':1,'pooled':2,'team_k20':3,'team_k10':3}; passers=[]
for v in complexity:
    ss=season[season.variant==v].set_index('season').margin_mae; gain=float(base.margin_mae-summary.loc[v,'margin_mae']); wins=int((ss<none_s).sum())
    if gain>=.05 and wins>=2: passers.append({'variant':v,'gain':gain,'wins':wins,'mae':float(summary.loc[v,'margin_mae']),'complexity':complexity[v]})
if passers:
    best_mae=min(x['mae'] for x in passers); near=[x for x in passers if x['mae']<=best_mae+.02]; selected=min(near,key=lambda x:x['complexity'])['variant']
else:selected='none'

folds=pd.concat(fold_rows,ignore_index=True)
# Neutral calibration under selected model. Acceptance preregistered here: |weighted bias| <= 1.0 and weighted Brier <= overall Brier + .02.
neutral_rows=[]
for sy in EVAL:
    te=folds[(folds.season==sy)&folds.neutral].copy(); allte=folds[folds.season==sy].copy(); tr=folds[folds.season<sy].copy(); pc='pm_'+selected
    if te.empty: continue
    res=tr.actual_margin-tr[pc]; mu=float(res.mean()); sd=float(res.std(ddof=1)); pn=norm.cdf((te[pc]+mu)/sd); yn=(te.actual_margin>0).astype(int); pa=norm.cdf((allte[pc]+mu)/sd); ya=(allte.actual_margin>0).astype(int); rr=te.actual_margin-te[pc]
    neutral_rows.append({'season':sy,'n':len(te),'margin_mae':float(abs(rr).mean()),'bias':float(rr.mean()),'brier':float(np.mean((pn-yn)**2)),'overall_brier':float(np.mean((pa-ya)**2)),'ece':ece(pn,yn)})
neutral=pd.DataFrame(neutral_rows); wbias=float(np.average(neutral.bias,weights=neutral.n)) if len(neutral) else np.nan; wbrier=float(np.average(neutral.brier,weights=neutral.n)) if len(neutral) else np.nan; wall=float(np.average(neutral.overall_brier,weights=neutral.n)) if len(neutral) else np.nan; neutral_pass=bool(len(neutral)==3 and abs(wbias)<=1.0 and wbrier<=wall+.02)

# Partial pooling incremental vs selected simpler location model.
best_team=min(['team_k10','team_k20'],key=lambda v:summary.loc[v,'margin_mae']); sel_s=season[season.variant==selected].set_index('season').margin_mae; bt_s=season[season.variant==best_team].set_index('season').margin_mae; team_gain=float(summary.loc[selected,'margin_mae']-summary.loc[best_team,'margin_mae']); team_wins=int((bt_s<sel_s).sum()); team_pass=bool(team_gain>=.03 and team_wins>=2)

result={'coverage':coverage,'matched':int(m.neutral.notna().sum()),'total':len(m),'neutral_games':int(known.neutral.sum()),'methods':method_counts,'location_summary':summary.reset_index().to_dict(orient='records'),'selected':selected,'passers':passers,'neutral_weighted_bias':wbias,'neutral_weighted_brier':wbrier,'overall_weighted_brier_for_neutral_seasons':wall,'neutral_pass':neutral_pass,'best_team':best_team,'team_gain_vs_selected':team_gain,'team_wins_vs_selected':team_wins,'team_pass':team_pass}
(OUT/'summary.json').write_text(json.dumps(result,indent=2,default=str)); m.to_csv(OUT/'identity_match_ledger.csv',index=False); season.to_csv(OUT/'location_season_metrics.csv',index=False); summary.reset_index().to_csv(OUT/'location_model_summary.csv',index=False); neutral.to_csv(OUT/'neutral_audit.csv',index=False); print(json.dumps(result,indent=2,default=str))
