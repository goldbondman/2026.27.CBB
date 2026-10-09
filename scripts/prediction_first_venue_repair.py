from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

OUT = Path('results/prediction_first_venue_repair')
OUT.mkdir(parents=True, exist_ok=True)
G15 = Path('/tmp/g15')
G10 = Path('/tmp/g10')
NCAA_RAW = 'https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/'
ESPN_REL = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/'
EVAL = [2023, 2024, 2025]
TRAIN_MIN = 2022


def norm_name(x):
    if pd.isna(x): return ''
    s = re.sub(r'[^a-z0-9]', '', str(x).lower())
    replacements = {
        'saint':'st', 'university':'', 'the':'',
    }
    # conservative text normalization only; no learned outcome aliases
    return s


def pick(df, cols):
    for c in cols:
        if c in df.columns: return c
    raise KeyError(f'none of {cols} in {list(df.columns)}')


def ece(p,y,bins=10):
    d=pd.DataFrame({'p':p,'y':y}).dropna()
    if d.empty:return np.nan
    d['b']=np.minimum((d.p*bins).astype(int),bins-1)
    return float(sum(len(z)/len(d)*abs(z.p.mean()-z.y.mean()) for _,z in d.groupby('b')))


def logloss(p,y):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def load_base():
    games=pd.read_parquet(G15/'game_predictions.parquet'); pace=pd.read_parquet(G10/'pace_predictions.parquet')
    games['game_id']=games.game_id.astype(str); pace['game_id']=pace.game_id.astype(str)
    s=pd.read_parquet(NCAA_RAW+'mbb/ncaa_mbb_schedule_master.parquet')
    gid=pick(s,['game_id','contest_id']); s=s.rename(columns={gid:'game_id'}); s['game_id']=s.game_id.astype(str)
    s['date']=pd.to_datetime(s['game_date'],errors='coerce').dt.normalize()
    s['home_key']=s.home.map(norm_name); s['away_key']=s.away.map(norm_name)
    b=games.merge(s[['game_id','date','home','away','home_key','away_key']],on='game_id',how='left',validate='one_to_one')
    b['team0_key']=b.team0.map(norm_name); b['team1_key']=b.team1.map(norm_name)
    b['home_is_0']=b.team0_key.eq(b.home_key); b['home_is_1']=b.team1_key.eq(b.home_key); b['orientation_ok']=b.home_is_0^b.home_is_1
    b=b[b.orientation_ok].copy()
    b['pred_home']=np.where(b.home_is_0,b.pred0,b.pred1); b['pred_away']=np.where(b.home_is_0,b.pred1,b.pred0)
    b['actual_home']=np.where(b.home_is_0,b.actual0,b.actual1); b['actual_away']=np.where(b.home_is_0,b.actual1,b.actual0)
    b['pred_home_margin']=b.pred_home-b.pred_away; b['actual_home_margin']=b.actual_home-b.actual_away; b['home_margin_resid']=b.actual_home_margin-b.pred_home_margin
    b['pred_total_ha']=b.pred_home+b.pred_away; b['actual_total_ha']=b.actual_home+b.actual_away
    return b,pace


def load_espn():
    xs=[]
    for sy in range(2022,2026):
        u=ESPN_REL+f'mbb_schedule_{sy}.parquet'; d=pd.read_parquet(u)
        d['season_key']=sy; xs.append(d)
    e=pd.concat(xs,ignore_index=True)
    e['espn_date']=pd.to_datetime(e['game_date'],errors='coerce').dt.normalize()
    hcol='home_display_name' if 'home_display_name' in e else 'home_name'; acol='away_display_name' if 'away_display_name' in e else 'away_name'
    e['ehome']=e[hcol].map(norm_name); e['eaway']=e[acol].map(norm_name)
    keep=['game_id','season_key','espn_date','ehome','eaway','neutral_site','venue_id','venue_full_name','venue_address_city','venue_address_state','home_venue_id']
    for c in keep:
        if c not in e: e[c]=np.nan
    return e[keep].copy()


def match_espn(base,e):
    # outcome-blind identity matching: date and team names only
    exact={}
    for r in e.itertuples(index=False): exact.setdefault((r.espn_date,r.ehome,r.eaway),[]).append(r)
    bydate=defaultdict(list)
    for r in e.itertuples(index=False): bydate[r.espn_date].append(r)
    rows=[]; stats=defaultdict(int)
    for r in base.itertuples(index=False):
        found=None; method=None; candidates=[]
        for dd in [r.date, r.date-pd.Timedelta(days=1), r.date+pd.Timedelta(days=1)]:
            z=exact.get((dd,r.home_key,r.away_key),[])
            if len(z)==1: found=z[0]; method='exact' if dd==r.date else 'exact_pm1'; break
        if found is None:
            candidates=bydate.get(r.date,[])
            scored=[]
            for c in candidates:
                sh=SequenceMatcher(None,r.home_key,c.ehome).ratio(); sa=SequenceMatcher(None,r.away_key,c.eaway).ratio(); sc=(sh+sa)/2
                if sc>=0.78: scored.append((sc,c))
            scored.sort(key=lambda x:x[0],reverse=True)
            if scored and scored[0][0]>=0.88 and (len(scored)==1 or scored[0][0]-scored[1][0]>=0.06): found=scored[0][1]; method='fuzzy'
        rec={'game_id':r.game_id,'match_method':method or 'unmatched'}
        if found is not None:
            rec.update({'neutral':bool(found.neutral_site) if pd.notna(found.neutral_site) else False,'venue_id':found.venue_id,'venue_full_name':found.venue_full_name,'venue_city':found.venue_address_city,'venue_state':found.venue_address_state,'espn_game_id':found.game_id}); stats[method]+=1
        else:
            rec.update({'neutral':np.nan,'venue_id':np.nan,'venue_full_name':np.nan,'venue_city':np.nan,'venue_state':np.nan,'espn_game_id':np.nan}); stats['unmatched']+=1
        rows.append(rec)
    m=pd.DataFrame(rows); z=base.merge(m,on='game_id',how='left',validate='one_to_one')
    diag={'n':len(z),'matched':int(z.neutral.notna().sum()),'coverage':float(z.neutral.notna().mean()),'neutral_games':int(z.neutral.fillna(False).sum()),'venue_name_coverage':float(z.venue_full_name.notna().mean()),'venue_city_coverage':float(z.venue_city.notna().mean()),'methods':dict(stats)}
    return z,diag


def score_metrics(z,pc,hc):
    resid=z.actual_home_margin-z[pc]; h=z[hc]
    ph=z.pred_home+h/2; pa=z.pred_away-h/2
    return {'n':len(z),'margin_mae':float(abs(resid).mean()),'margin_bias':float(resid.mean()),'team_score_mae':float(np.mean(np.r_[abs(z.actual_home-ph),abs(z.actual_away-pa)])),'total_mae':float(abs(z.actual_total_ha-(ph+pa)).mean())}


def location_eval(ctx):
    summaries=[]; foldframes=[]
    c=ctx.dropna(subset=['neutral']).copy(); c['neutral']=c.neutral.astype(bool)
    for sy in EVAL:
        tr=c[(c.season>=TRAIN_MIN)&(c.season<sy)]; te=c[c.season==sy].copy(); tr_non=tr[~tr.neutral]
        pooled=float(tr_non.home_margin_resid.mean()); hs=tr_non.groupby('home_key').home_margin_resid.agg(['mean','count'])
        te['hca_none']=0.0; te['hca_static27']=np.where(te.neutral,0.0,2.7); te['hca_pooled']=np.where(te.neutral,0.0,pooled)
        for k in [5,10,20]:
            vals=[]
            for r in te.itertuples(index=False):
                if r.neutral: vals.append(0.0)
                elif r.home_key in hs.index:
                    n=float(hs.loc[r.home_key,'count']); mm=float(hs.loc[r.home_key,'mean']); w=n/(n+k); vals.append(pooled+w*(mm-pooled))
                else: vals.append(pooled)
            te[f'hca_team_k{k}']=vals
        for v in ['none','static27','pooled','team_k5','team_k10','team_k20']:
            hc=f'hca_{v}'; pc=f'pred_margin_{v}'; te[pc]=te.pred_home_margin+te[hc]
            # Probability layer is re-estimated using prior-season residuals for this exact location model.
            trc=tr.copy()
            if v=='none': trc[hc]=0.0
            elif v=='static27': trc[hc]=np.where(trc.neutral,0.0,2.7)
            elif v=='pooled': trc[hc]=np.where(trc.neutral,0.0,pooled)
            else:
                k=int(v.split('k')[1]); vv=[]
                for r in trc.itertuples(index=False):
                    if r.neutral: vv.append(0.0)
                    elif r.home_key in hs.index:
                        n=float(hs.loc[r.home_key,'count']); mm=float(hs.loc[r.home_key,'mean']); w=n/(n+k); vv.append(pooled+w*(mm-pooled))
                    else: vv.append(pooled)
                trc[hc]=vv
            trc[pc]=trc.pred_home_margin+trc[hc]; rr=trc.actual_home_margin-trc[pc]; mu=float(rr.mean()); sd=float(rr.std(ddof=1)); p=norm.cdf((te[pc]+mu)/sd); y=(te.actual_home_margin>0).astype(int)
            sm=score_metrics(te,pc,hc); sm.update({'season':sy,'variant':v,'pooled_train':pooled,'brier':float(np.mean((p-y)**2)),'logloss':logloss(p,y),'ece':ece(p,y)}); summaries.append(sm)
        foldframes.append(te)
    s=pd.DataFrame(summaries); p=s.groupby('variant').agg(margin_mae=('margin_mae','mean'),team_score_mae=('team_score_mae','mean'),brier=('brier','mean'),logloss=('logloss','mean'),ece=('ece','mean')).sort_values(['margin_mae','brier'])
    none=p.loc['none']; season_none=s[s.variant=='none'].set_index('season').margin_mae
    passers=[]
    complexity={'static27':1,'pooled':2,'team_k20':3,'team_k10':3,'team_k5':3}
    for v in ['static27','pooled','team_k5','team_k10','team_k20']:
        ss=s[s.variant==v].set_index('season').margin_mae; gain=float(none.margin_mae-p.loc[v,'margin_mae']); wins=int((ss<season_none).sum())
        if gain>=.05 and wins>=2: passers.append((v,gain,wins,float(p.loc[v,'margin_mae'])))
    if passers:
        best=min(passers,key=lambda x:x[3]); best_mae=best[3]; near=[x for x in passers if x[3]<=best_mae+.02]; selected=min(near,key=lambda x:complexity[x[0]])[0]
    else:selected='none'
    sel=s[s.variant==selected].set_index('season').margin_mae; pooled=s[s.variant=='pooled'].set_index('season').margin_mae
    best_team=min(['team_k5','team_k10','team_k20'],key=lambda v:p.loc[v,'margin_mae']); bt=s[s.variant==best_team].set_index('season').margin_mae
    dec={'selected':selected,'passing_variants':passers,'static_gain_vs_none':float(none.margin_mae-p.loc['static27','margin_mae']),'static_wins':int((s[s.variant=='static27'].set_index('season').margin_mae<season_none).sum()),'pooled_gain_vs_none':float(none.margin_mae-p.loc['pooled','margin_mae']),'pooled_wins':int((pooled<season_none).sum()),'best_team':best_team,'team_gain_vs_selected':float(p.loc[selected,'margin_mae']-p.loc[best_team,'margin_mae']),'team_wins_vs_selected':int((bt<sel).sum())}
    return pd.concat(foldframes,ignore_index=True),s,p,dec


def neutral_audit(folds,selected):
    rows=[]
    for sy in EVAL:
        te=folds[(folds.season==sy)&(folds.neutral.astype(bool))].copy(); tr=folds[folds.season<sy].copy()
        if te.empty: continue
        hc=f'hca_{selected}'; pc=f'pred_margin_{selected}'; rr=tr.actual_home_margin-tr[pc]; mu=float(rr.mean()); sd=float(rr.std(ddof=1)); p=norm.cdf((te[pc]+mu)/sd); y=(te.actual_home_margin>0).astype(int); resid=te.actual_home_margin-te[pc]
        rows.append({'season':sy,'n':len(te),'margin_mae':float(abs(resid).mean()),'margin_bias':float(resid.mean()),'brier':float(np.mean((p-y)**2)),'logloss':logloss(p,y),'ece':ece(p,y)})
    return pd.DataFrame(rows)


def schedule_features(ctx):
    x=ctx[['game_id','season','date','home_key','away_key']].drop_duplicates().sort_values(['date','game_id']); hist=defaultdict(deque); rows=[]
    for r in x.itertuples(index=False):
        dt=pd.Timestamp(r.date); rec={'game_id':r.game_id}
        for side,key in [('home',r.home_key),('away',r.away_key)]:
            q=hist[key]; prior=[d for d in q if d<dt]; raw=(dt-prior[-1]).days if prior else np.nan; rec[f'{side}_rest_raw']=raw; rec[f'{side}_rest']=7.0 if pd.isna(raw) else float(np.clip(raw,0,7)); rec[f'{side}_g7']=sum(1 for d in prior if 0<(dt-d).days<=7); rec[f'{side}_g14']=sum(1 for d in prior if 0<(dt-d).days<=14)
        hist[r.home_key].append(dt); hist[r.away_key].append(dt); rows.append(rec)
    f=pd.DataFrame(rows); f['rest_diff']=f.home_rest-f.away_rest; f['g7_diff']=f.home_g7-f.away_g7; f['g14_diff']=f.home_g14-f.away_g14; f['stress']=((f[['home_rest','away_rest']].min(axis=1)<=1)|(f[['home_g7','away_g7']].max(axis=1)>=3)).astype(int); return f


def context_tests(ctx,folds,selected,pace):
    sf=schedule_features(ctx); x=ctx.merge(sf,on='game_id',how='inner',validate='one_to_one'); pc=f'pred_margin_{selected}'; hc=f'hca_{selected}'
    # recreate selected location prediction columns on full ctx from fold output
    key=folds[['game_id',pc,hc]].drop_duplicates('game_id'); x=x.merge(key,on='game_id',how='inner',validate='one_to_one'); x['resid']=x.actual_home_margin-x[pc]
    features=['rest_diff','g7_diff','g14_diff']; mrows=[]; urows=[]
    for sy in EVAL:
        tr=x[(x.season>=TRAIN_MIN)&(x.season<sy)].dropna(subset=features+['resid']); te=x[x.season==sy].dropna(subset=features+['resid']).copy(); lm=LinearRegression().fit(tr[features],tr.resid); corr=lm.predict(te[features]); base=float(abs(te.resid).mean()); new=float(abs(te.resid-corr).mean()); mrows.append({'season':sy,'n':len(te),'base_mae':base,'rest_mae':new,'gain':base-new,'coef_rest':lm.coef_[0],'coef_g7':lm.coef_[1],'coef_g14':lm.coef_[2]})
        # simple global sigma then stress multiplier, avoiding re-optimizing stage buckets here
        sd=float(tr.resid.std(ddof=1)); std=tr.resid/sd; a=std[tr.stress==1]; b=std[tr.stress==0]; raw=float(a.std(ddof=1)/b.std(ddof=1)) if len(a)>20 and len(b)>50 else 1.0; w=len(a)/(len(a)+50); mult=1+w*(raw-1); bs=np.full(len(te),sd); cs=bs*np.where(te.stress.values==1,mult,1.0)
        def nll(r,s): return float(np.mean(.5*np.log(2*np.pi*s*s)+.5*(r/s)**2))
        urows.append({'season':sy,'n':len(te),'base_nll':nll(te.resid.values,bs),'stress_nll':nll(te.resid.values,cs),'gain':nll(te.resid.values,bs)-nll(te.resid.values,cs),'multiplier':mult,'stress_n':int(te.stress.sum())})
    # pace
    pg=pace[['game_id','pace_pred','pace_actual']].drop_duplicates('game_id').merge(sf,on='game_id',how='inner'); pg=pg.merge(ctx[['game_id','season']].drop_duplicates(),on='game_id',how='inner'); pg['resid']=pg.pace_actual-pg.pace_pred; prows=[]
    for sy in EVAL:
        tr=pg[(pg.season>=TRAIN_MIN)&(pg.season<sy)].dropna(subset=features+['resid']); te=pg[pg.season==sy].dropna(subset=features+['resid']).copy(); lm=LinearRegression().fit(tr[features],tr.resid); pr=te.pace_pred+lm.predict(te[features]); b=float(abs(te.pace_actual-te.pace_pred).mean()); n=float(abs(te.pace_actual-pr).mean()); prows.append({'season':sy,'n':len(te),'base_pace_mae':b,'rest_pace_mae':n,'gain':b-n,'coef_rest':lm.coef_[0],'coef_g7':lm.coef_[1],'coef_g14':lm.coef_[2]})
    return sf,pd.DataFrame(mrows),pd.DataFrame(prows),pd.DataFrame(urows)


def main():
    base,pace=load_base(); espn=load_espn(); ctx,diag=match_espn(base,espn); print('MATCH',json.dumps(diag,indent=2,default=str))
    if diag['coverage']<0.90: raise RuntimeError(f'ESPN neutral join coverage too low: {diag}')
    folds,season,summary,dec=location_eval(ctx); neutral=neutral_audit(folds,dec['selected']); sf,rest,pace_res,unc=context_tests(ctx,folds,dec['selected'],pace)
    # source audit: venue names/cities are available, but no lat/lon/timezone/altitude in this schedule table.
    travel={'venue_name_coverage':diag['venue_name_coverage'],'venue_city_coverage':diag['venue_city_coverage'],'coordinate_fields_present':False,'decision':'HOLD','reason':'ESPN schedule repairs venue identity/city/state but does not supply latitude/longitude/timezone/altitude; a static audited coordinate table is still required for distance/time-zone/altitude features.'}
    out={'match':diag,'location_decision':dec,'neutral_rows':neutral.to_dict(orient='records'),'rest_mean_gain':float(rest.gain.mean()),'rest_mean_wins':int((rest.gain>0).sum()),'pace_gain':float(pace_res.gain.mean()),'pace_wins':int((pace_res.gain>0).sum()),'unc_nll_gain':float(unc.gain.mean()),'unc_wins':int((unc.gain>0).sum()),'travel':travel}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2,default=str)); season.to_csv(OUT/'location_season_metrics.csv',index=False); summary.reset_index().to_csv(OUT/'location_model_summary.csv',index=False); neutral.to_csv(OUT/'neutral_site_audit.csv',index=False); rest.to_csv(OUT/'rest_margin_metrics.csv',index=False); pace_res.to_csv(OUT/'rest_pace_metrics.csv',index=False); unc.to_csv(OUT/'schedule_uncertainty_metrics.csv',index=False); ctx[['game_id','season','neutral','venue_id','venue_full_name','venue_city','venue_state','match_method']].to_csv(OUT/'venue_context_join.csv',index=False)
    print(json.dumps(out,indent=2,default=str))

if __name__=='__main__': main()
