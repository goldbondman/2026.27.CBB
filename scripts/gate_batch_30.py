from __future__ import annotations

import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

import gate_batch_15 as base

OUT=Path('results/gate_batch_30'); OUT.mkdir(parents=True,exist_ok=True)
SPREAD_URL='https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/canonical/spreads/fg/spreads_fg_all.csv'
TOTAL_URL='https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/canonical/totals/fg/totals_fg_all.csv'
YEARS=[2022,2023,2024,2025]
ALIASES={
 'connecticut':'uconn','uconn':'uconn','miamifl':'miamifl','miamiflorida':'miamifl',
 'northcarolinastate':'ncstate','ncstate':'ncstate','saintmarys':'stmarys','stmarys':'stmarys',
 'saintpeters':'stpeters','stpeters':'stpeters','mississippi':'olemiss','olemiss':'olemiss',
 'brighamyoung':'byu','byu':'byu','southerncalifornia':'usc','usc':'usc',
 'centralflorida':'ucf','ucf':'ucf','texaschristian':'tcu','tcu':'tcu',
 'southernmethodist':'smu','smu':'smu','virginiacommonwealth':'vcu','vcu':'vcu',
 'nevada':'nevada','unlv':'unlv','massachusetts':'umass','umass':'umass',
}

def normname(x):
    s=re.sub(r'[^a-z0-9]','',str(x).lower())
    for tail in ['wildcats','bulldogs','tigers','eagles','hawks','aggies','bears','panthers','cougars','cardinals','spartans','terrapins','mountaineers','cavaliers','paladins','bison','jayhawks','bluejays','razorbacks','crimson','huskies','volunteers','hoosiers','boilermakers','cowboys','cowgirls','utes','ducks','beavers','bruins','trojans','rebels','rams','flyers','zags']:
        if s.endswith(tail) and len(s)>len(tail)+3: s=s[:-len(tail)]
    return ALIASES.get(s,s)

def name_match(a,b):
    a,b=normname(a),normname(b)
    return a==b or (len(a)>=5 and a in b) or (len(b)>=5 and b in a)

def load_model_games():
    g=base.load_games(); C=base.continuity(); tp=base.run_frozen(g,C,True); pace=base.pace_predictions(g); games=base.make_games(tp,pace)
    games['date']=pd.to_datetime(games.date).dt.date
    return games

def load_market(url,line_col):
    d=pd.read_csv(url)
    d=d[d.season.isin(YEARS)].copy()
    d['game_date']=pd.to_datetime(d.game_date).dt.date
    d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='coerce')
    d['commence_time']=pd.to_datetime(d.commence_time,utc=True,errors='coerce')
    d['lead_min']=(d.commence_time-d.timestamp).dt.total_seconds()/60
    d[line_col]=pd.to_numeric(d[line_col],errors='coerce')
    d=d.dropna(subset=['event_id','game_date','home_team_canonical','away_team_canonical',line_col])
    return d

def attach_model(events,games,line_col):
    rows=[]
    for r in events.itertuples(index=False):
        q=games[(games.season==int(r.season)) & (games.date==r.game_date)]
        hit=None; home0=None
        for x in q.itertuples(index=False):
            if name_match(r.home_team_canonical,x.team0) and name_match(r.away_team_canonical,x.team1): hit=x; home0=True; break
            if name_match(r.home_team_canonical,x.team1) and name_match(r.away_team_canonical,x.team0): hit=x; home0=False; break
        if hit is None: continue
        d=r._asdict()
        d.update({
            'game_id':hit.game_id,
            'team0':hit.team0,'team1':hit.team1,
            'pred_home_margin':float(hit.pred_margin if home0 else -hit.pred_margin),
            'actual_home_margin':float(hit.actual_margin if home0 else -hit.actual_margin),
            'pred_total':float(hit.pred_total),'actual_total':float(hit.actual_total),
            'n_avg':float(hit.n_avg),
        })
        rows.append(d)
    return pd.DataFrame(rows)

def consensus(m,line_col):
    agg=m.groupby(['season','event_id','game_date','game_id','home_team_canonical','away_team_canonical','is_march_madness'],as_index=False).agg(
        line=(line_col,'median'),line_mean=(line_col,'mean'),line_sd=(line_col,'std'),books=('bookmaker','nunique'),
        lead_min=('lead_min','median'),pred_home_margin=('pred_home_margin','first'),actual_home_margin=('actual_home_margin','first'),
        pred_total=('pred_total','first'),actual_total=('actual_total','first'),n_avg=('n_avg','first'))
    agg['line_sd']=agg.line_sd.fillna(0)
    return agg

def perf_binary(y):
    y=np.asarray(y,float)
    if len(y)==0:return {'n':0,'hit':None,'roi110':None,'profit110':0}
    hit=float((y>0).mean()); push=float((y==0).mean())
    roi=(hit*(100/110) - (1-hit-push))
    return {'n':int(len(y)),'hit':hit,'push':push,'roi110':float(roi),'profit110':float(roi*len(y))}

def ci_hit(y,seed=7,B=2000):
    y=np.asarray(y,float); y=y[y!=0]
    if len(y)<5:return {'n':int(len(y)),'lo':None,'hi':None}
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(B): vals.append(np.mean(rng.choice(y>0,size=len(y),replace=True)))
    return {'n':int(len(y)),'lo':float(np.quantile(vals,.025)),'hi':float(np.quantile(vals,.975))}

def spread_tests(s):
    z=s.copy(); z['edge_pts']=z.pred_home_margin + z.line
    z['pick_home']=z.edge_pts>=0
    z['ats_margin']=np.where(z.pick_home,z.actual_home_margin+z.line,-z.actual_home_margin-z.line)
    z['edge_abs']=z.edge_pts.abs()
    out={}
    for t in [0,2,4,6,8,10]: out[str(t)]=perf_binary(z.loc[z.edge_abs>=t,'ats_margin'])
    out['season']={str(int(y)):perf_binary(q.loc[q.edge_abs>=6,'ats_margin']) for y,q in z.groupby('season')}
    out['march']={str(bool(k)):perf_binary(q.loc[q.edge_abs>=6,'ats_margin']) for k,q in z.groupby('is_march_madness')}
    z['maturity']=pd.cut(z.n_avg,[-1,4.999,9.999,19.999,1e9],labels=['<5','5-9','10-19','20+'])
    out['maturity']={str(k):perf_binary(q.loc[q.edge_abs>=6,'ats_margin']) for k,q in z.groupby('maturity',observed=True)}
    z['market_fav']=np.where(z.line<0,'home',np.where(z.line>0,'away','pick'))
    z['pick_side']=np.where(z.pick_home,'home','away')
    z['pick_type']=np.where(z.pick_side==z.market_fav,'favorite',np.where(z.market_fav=='pick','pickem','dog'))
    out['pick_type']={str(k):perf_binary(q.loc[q.edge_abs>=6,'ats_margin']) for k,q in z.groupby('pick_type')}
    z['line_bucket']=pd.cut(z.line.abs(),[-.1,3,7,12,1e9],labels=['0-3','3.5-7','7.5-12','12.5+'])
    out['line_bucket']={str(k):perf_binary(q.loc[q.edge_abs>=6,'ats_margin']) for k,q in z.groupby('line_bucket',observed=True)}
    out['dispersion']={
        'low':perf_binary(z.loc[(z.edge_abs>=6)&(z.line_sd<=z.line_sd.median()),'ats_margin']),
        'high':perf_binary(z.loc[(z.edge_abs>=6)&(z.line_sd>z.line_sd.median()),'ats_margin'])}
    out['bootstrap_hit_ci_6']=ci_hit(z.loc[z.edge_abs>=6,'ats_margin'])
    # standard-vig probability proxy, trained only on prior seasons
    evrows=[]
    for sy in [2023,2024,2025]:
        tr=z[z.season<sy]; te=z[z.season==sy].copy()
        resid=tr.actual_home_margin-tr.pred_home_margin
        mu=float(resid.mean()); sd=float(resid.std(ddof=1))
        ph=1-norm.cdf((-te.line-te.pred_home_margin-mu)/sd)
        p=np.where(te.pick_home,ph,1-ph)
        te['p_cover']=p; te['ev110']=p*(100/110)-(1-p)
        evrows.append(te)
    ev=pd.concat(evrows,ignore_index=True)
    out['ev_proxy']={str(t):perf_binary(ev.loc[ev.ev110>=t,'ats_margin']) for t in [0,.02,.05,.08,.10]}
    out['ev_proxy_season']={str(int(y)):perf_binary(q.loc[q.ev110>=.05,'ats_margin']) for y,q in ev.groupby('season')}
    return z,ev,out

def total_tests(t):
    z=t.copy(); z['edge_pts']=z.pred_total-z.line; z['pick_over']=z.edge_pts>=0; z['edge_abs']=z.edge_pts.abs()
    z['ou_margin']=np.where(z.pick_over,z.actual_total-z.line,z.line-z.actual_total)
    out={}
    for th in [0,4,6,8,10,12]: out[str(th)]=perf_binary(z.loc[z.edge_abs>=th,'ou_margin'])
    out['season']={str(int(y)):perf_binary(q.loc[q.edge_abs>=8,'ou_margin']) for y,q in z.groupby('season')}
    out['march']={str(bool(k)):perf_binary(q.loc[q.edge_abs>=8,'ou_margin']) for k,q in z.groupby('is_march_madness')}
    z['maturity']=pd.cut(z.n_avg,[-1,4.999,9.999,19.999,1e9],labels=['<5','5-9','10-19','20+'])
    out['maturity']={str(k):perf_binary(q.loc[q.edge_abs>=8,'ou_margin']) for k,q in z.groupby('maturity',observed=True)}
    out['direction']={str(k):perf_binary(q.loc[q.edge_abs>=8,'ou_margin']) for k,q in z.groupby(z.pick_over.map({True:'over',False:'under'}))}
    out['dispersion']={'low':perf_binary(z.loc[(z.edge_abs>=8)&(z.line_sd<=z.line_sd.median()),'ou_margin']),'high':perf_binary(z.loc[(z.edge_abs>=8)&(z.line_sd>z.line_sd.median()),'ou_margin'])}
    out['bootstrap_hit_ci_8']=ci_hit(z.loc[z.edge_abs>=8,'ou_margin'])
    evrows=[]
    for sy in [2023,2024,2025]:
        tr=z[z.season<sy]; te=z[z.season==sy].copy(); resid=tr.actual_total-tr.pred_total; mu=float(resid.mean()); sd=float(resid.std(ddof=1))
        po=1-norm.cdf((te.line-te.pred_total-mu)/sd); p=np.where(te.pick_over,po,1-po)
        te['p_cover']=p; te['ev110']=p*(100/110)-(1-p); evrows.append(te)
    ev=pd.concat(evrows,ignore_index=True)
    out['ev_proxy']={str(x):perf_binary(ev.loc[ev.ev110>=x,'ou_margin']) for x in [0,.02,.05,.08,.10]}
    out['ev_proxy_season']={str(int(y)):perf_binary(q.loc[q.ev110>=.05,'ou_margin']) for y,q in ev.groupby('season')}
    return z,ev,out

def main():
    games=load_model_games(); sp=load_market(SPREAD_URL,'spread'); to=load_market(TOTAL_URL,'total')
    sm=attach_model(sp,games,'spread'); tm=attach_model(to,games,'total')
    sc=consensus(sm,'spread'); tc=consensus(tm,'total')
    sz,sev,st=spread_tests(sc); tz,tev,tt=total_tests(tc)
    source={
      'spread_rows':int(len(sp)),'total_rows':int(len(to)),'spread_events':int(sp.event_id.nunique()),'total_events':int(to.event_id.nunique()),
      'spread_books':int(sp.bookmaker.nunique()),'total_books':int(to.bookmaker.nunique()),
      'spread_match_events':int(sc.event_id.nunique()),'total_match_events':int(tc.event_id.nunique()),
      'spread_season_events':{str(int(k)):int(v) for k,v in sc.groupby('season').size().items()},
      'total_season_events':{str(int(k)):int(v) for k,v in tc.groupby('season').size().items()},
      'lead_min':{'spread_median':float(sp.lead_min.median()),'spread_p95':float(sp.lead_min.quantile(.95)),'total_median':float(to.lead_min.median()),'total_p95':float(to.lead_min.quantile(.95)),
                  'spread_nonpregame':int((sp.lead_min<0).sum()),'total_nonpregame':int((to.lead_min<0).sum())},
      'books_per_event':{'spread_median':float(sc.books.median()),'total_median':float(tc.books.median())},
      'line_dispersion':{'spread_median_sd':float(sc.line_sd.median()),'total_median_sd':float(tc.line_sd.median())},
    }
    # leave-one-season summaries at fixed point-edge policies
    loo={'spread':{},'total':{}}
    for sy in YEARS:
        loo['spread'][str(sy)]=perf_binary(sz.loc[(sz.season==sy)&(sz.edge_abs>=6),'ats_margin'])
        loo['total'][str(sy)]=perf_binary(tz.loc[(tz.season==sy)&(tz.edge_abs>=8),'ou_margin'])
    summary={'source':source,'spread':st,'total':tt,'leave_one_season':loo}
    sz.to_csv(OUT/'spread_games.csv',index=False); sev.to_csv(OUT/'spread_ev_proxy.csv',index=False); tz.to_csv(OUT/'total_games.csv',index=False); tev.to_csv(OUT/'total_ev_proxy.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('GATE_BATCH_30'); print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
