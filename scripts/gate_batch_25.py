from __future__ import annotations

import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

import gate_batch_15 as base

OUT=Path('results/gate_batch_25'); OUT.mkdir(parents=True,exist_ok=True)
SRC='https://raw.githubusercontent.com/bsneaks/march-madness-betting/70dfdfa27832111ac2d01f1b3889b5ed1046a3e9/data/'
YEARS=[2023,2024,2025]
THRESH=[0,.02,.05,.08,.10]
ALIASES={'uconn':'connecticut','connecticut':'connecticut','miamifl':'miamifl','miamiflorida':'miamifl','ncstate':'northcarolinastate','stjohnsny':'stjohns','saintmarys':'stmarys','stpeters':'saintpeters','saintpeters':'saintpeters','olemiss':'mississippi'}

def normname(x):
    s=re.sub(r'[^a-z0-9]','',str(x).lower())
    return ALIASES.get(s,s)

def name_match(a,b):
    a,b=normname(a),normname(b)
    return a==b or (len(a)>=5 and a in b) or (len(b)>=5 and b in a)

def implied(o):
    o=float(o)
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def profit_mult(o):
    o=float(o)
    return o/100 if o>0 else 100/abs(o)

def ret(outcome,o):
    return 0.0 if outcome==0 else (profit_mult(o) if outcome>0 else -1.0)

def load_model():
    g=base.load_games(); C=base.continuity(); tp=base.run_frozen(g,C,True); pace=base.pace_predictions(g); games=base.make_games(tp,pace)
    dist={}
    for sy in YEARS:
        tr=games[(games.season>=2022)&(games.season<sy)]
        dist[sy]={'mm':float(tr.margin_resid.mean()),'ms':float(tr.margin_resid.std(ddof=1)),'tm':float(tr.total_resid.mean()),'ts':float(tr.total_resid.std(ddof=1))}
    return games,dist

def load_prices():
    xs=[]
    for sy in YEARS:
        z=pd.read_csv(SRC+f'historical_odds_{sy}.csv'); z['year']=sy; xs.append(z)
    odds=pd.concat(xs,ignore_index=True)
    res=pd.read_csv(SRC+'tournament_results.csv'); res=res[res.year.isin(YEARS)].copy()
    return odds,res

def attach_results(odds,res):
    rows=[]
    for r in odds.itertuples(index=False):
        q=res[res.year==r.year]
        hit=None
        for x in q.itertuples(index=False):
            if (name_match(r.home_team,x.team1) and name_match(r.away_team,x.team2)) or (name_match(r.home_team,x.team2) and name_match(r.away_team,x.team1)):
                hit=x; break
        if hit is None: continue
        if name_match(r.home_team,hit.team1): hs,as_=hit.score1,hit.score2
        else: hs,as_=hit.score2,hit.score1
        d=r._asdict(); d.update({'home_score':float(hs),'away_score':float(as_),'round_num':int(hit.round_num)}); rows.append(d)
    return pd.DataFrame(rows)

def attach_model(odds,games):
    rows=[]
    for r in odds.itertuples(index=False):
        q=games[games.season==r.year]
        q=q[((q.actual0==r.home_score)&(q.actual1==r.away_score))|((q.actual1==r.home_score)&(q.actual0==r.away_score))]
        if len(q)>1:
            qq=[]
            for x in q.itertuples(index=False):
                if (name_match(r.home_team,x.team0) and name_match(r.away_team,x.team1)) or (name_match(r.home_team,x.team1) and name_match(r.away_team,x.team0)): qq.append(x)
            if len(qq)==1: x=qq[0]
            else: continue
        elif len(q)==1: x=next(q.itertuples(index=False))
        else: continue
        h0=(x.actual0==r.home_score and x.actual1==r.away_score)
        d=r._asdict(); d.update({'game_id':x.game_id,'model_home_margin':float(x.pred_margin if h0 else -x.pred_margin),'model_total':float(x.pred_total),'actual_home_margin':float(r.home_score-r.away_score),'actual_total':float(r.home_score+r.away_score),'n_avg':float(x.n_avg),'home_is_team0':bool(h0)}); rows.append(d)
    return pd.DataFrame(rows)

def book_prefixes(cols):
    out=set()
    suffixes=['_home_ml','_away_ml','_home_spread','_away_spread','_over','_under']
    for c in cols:
        for s in suffixes:
            if c.endswith(s): out.add(c[:-len(s)])
    return sorted(x for x in out if x and x not in {'home','away'})

def outcome(v):
    return 1 if v>1e-9 else (-1 if v<-1e-9 else 0)

def contracts(matched,dist):
    rows=[]; books=book_prefixes(matched.columns)
    for r in matched.itertuples(index=False):
        d=dist[int(r.year)]; mm=d['mm'] if r.home_is_team0 else -d['mm']; mu_m=r.model_home_margin+mm; ms=d['ms']; mu_t=r.model_total+d['tm']; ts=d['ts']
        for b in books:
            # moneyline
            hm=getattr(r,b+'_home_ml',np.nan); am=getattr(r,b+'_away_ml',np.nan)
            if pd.notna(hm) and pd.notna(am) and hm!=0 and am!=0:
                ph=float(norm.cdf(mu_m/ms)); vals=[('home',ph,hm,outcome(r.actual_home_margin)),('away',1-ph,am,outcome(-r.actual_home_margin))]
                ov=implied(hm)+implied(am)
                for side,p,o,y in vals:
                    rows.append({'year':r.year,'game_id':r.game_id,'market':'ml','book':b,'side':side,'line':np.nan,'odds':float(o),'p_model':p,'imp':implied(o),'overround':ov,'ev':p*profit_mult(o)-(1-p),'result':y,'ret':ret(y,o),'n_avg':r.n_avg})
            # spread
            hl=getattr(r,b+'_home_spread',np.nan); hp=getattr(r,b+'_home_spread_price',np.nan); al=getattr(r,b+'_away_spread',np.nan); ap=getattr(r,b+'_away_spread_price',np.nan)
            if all(pd.notna(x) for x in [hl,hp,al,ap]) and hp!=0 and ap!=0:
                ph=float(1-norm.cdf((-float(hl)-mu_m)/ms)); pa=float(norm.cdf((float(al)-mu_m)/ms)); ov=implied(hp)+implied(ap)
                vals=[('home',ph,hp,float(hl),outcome(r.actual_home_margin+float(hl))),('away',pa,ap,float(al),outcome(-r.actual_home_margin+float(al)))]
                for side,p,o,line,y in vals: rows.append({'year':r.year,'game_id':r.game_id,'market':'spread','book':b,'side':side,'line':line,'odds':float(o),'p_model':p,'imp':implied(o),'overround':ov,'ev':p*profit_mult(o)-(1-p),'result':y,'ret':ret(y,o),'n_avg':r.n_avg})
            # total
            ol=getattr(r,b+'_over',np.nan); op=getattr(r,b+'_over_price',np.nan); ul=getattr(r,b+'_under',np.nan); up=getattr(r,b+'_under_price',np.nan)
            if all(pd.notna(x) for x in [ol,op,ul,up]) and op!=0 and up!=0:
                po=float(1-norm.cdf((float(ol)-mu_t)/ts)); pu=float(norm.cdf((float(ul)-mu_t)/ts)); ov=implied(op)+implied(up)
                vals=[('over',po,op,float(ol),outcome(r.actual_total-float(ol))),('under',pu,up,float(ul),outcome(float(ul)-r.actual_total))]
                for side,p,o,line,y in vals: rows.append({'year':r.year,'game_id':r.game_id,'market':'total','book':b,'side':side,'line':line,'odds':float(o),'p_model':p,'imp':implied(o),'overround':ov,'ev':p*profit_mult(o)-(1-p),'result':y,'ret':ret(y,o),'n_avg':r.n_avg})
    return pd.DataFrame(rows)

def best_by_game(c,market=None,book=None,shrink=1.0):
    z=c.copy()
    if market: z=z[z.market==market]
    if book: z=z[z.book==book]
    if shrink!=1:
        z=z.copy(); z['p_adj']=.5+shrink*(z.p_model-.5); z['ev_adj']=z.p_adj*z.odds.map(profit_mult)-(1-z.p_adj); key='ev_adj'
    else: key='ev'
    if z.empty:return z
    return z.sort_values(key).groupby('game_id',as_index=False).tail(1).copy().rename(columns={key:'decision_ev'})

def perf(z,threshold=.05):
    if z.empty:return {'n':0,'roi':None,'hit':None,'profit':0}
    q=z[z.decision_ev>=threshold].copy();
    if q.empty:return {'n':0,'roi':None,'hit':None,'profit':0}
    return {'n':int(len(q)),'roi':float(q['ret'].mean()),'hit':float((q.result>0).mean()),'profit':float(q['ret'].sum()),'avg_ev':float(q.decision_ev.mean())}

def season_perf(z,t=.05):
    return {str(int(y)):perf(q,t) for y,q in z.groupby('year')}

def max_drawdown(returns):
    eq=1+np.cumsum(np.asarray(returns,float)); peak=np.maximum.accumulate(eq); return float(np.max((peak-eq)/np.maximum(peak,1e-9))) if len(eq) else 0.0

def kelly_path(z,threshold=.05,fraction=.25,cap=.02,daily_cap=None):
    q=z[z.decision_ev>=threshold].copy().sort_values(['year','game_id']); bank=1.0; peak=1.0; mdd=0; n=0
    for _,r in q.iterrows():
        b=profit_mult(r.odds); p=r.p_model; f=max(0,(b*p-(1-p))/b)*fraction; f=min(f,cap)
        if daily_cap is not None: f=min(f,daily_cap)
        bank*=1+f*r['ret']; peak=max(peak,bank); mdd=max(mdd,(peak-bank)/peak); n+=1
    return {'n':n,'ending_bankroll':float(bank),'return':float(bank-1),'max_drawdown':float(mdd)}

def main():
    games,dist=load_model(); odds,res=load_prices(); ar=attach_results(odds,res); m=attach_model(ar,games); c=contracts(m,dist)
    ar.to_csv(OUT/'odds_with_results.csv',index=False); m.to_csv(OUT/'matched_games.csv',index=False); c.to_csv(OUT/'contracts.csv',index=False)
    summary={'source':{'odds_rows':len(odds),'results_attached':len(ar),'model_matched':len(m),'years':m.groupby('year').size().to_dict(),'books':len(book_prefixes(odds.columns))},'price_completeness':{},'overround':{},'thresholds':{},'season':{},'shopping':{},'expression':{},'maturity':{},'shrink':{},'staking':{},'portfolio':{}}
    for mk in ['spread','total','ml']:
        z=c[c.market==mk]; summary['price_completeness'][mk]={'contracts':int(len(z)),'games':int(z.game_id.nunique()),'books':int(z.book.nunique())}; summary['overround'][mk]={'median':float(z.overround.median()),'mean':float(z.overround.mean()),'p05':float(z.overround.quantile(.05)),'p95':float(z.overround.quantile(.95))}
        best=best_by_game(c,mk)
        summary['thresholds'][mk]={str(t):perf(best,t) for t in THRESH}; summary['season'][mk]=season_perf(best,.05)
        dk=best_by_game(c,mk,'draftkings'); summary['shopping'][mk]={'best':perf(best,.05),'draftkings':perf(dk,.05),'mean_best_ev':float(best.decision_ev.mean()) if len(best) else None,'mean_dk_ev':float(dk.decision_ev.mean()) if len(dk) else None}
    spread=best_by_game(c,'spread'); ml=best_by_game(c,'ml'); total=best_by_game(c,'total')
    side=pd.concat([spread.assign(expr='spread'),ml.assign(expr='ml')]).sort_values('decision_ev').groupby('game_id').tail(1); allm=pd.concat([spread.assign(expr='spread'),ml.assign(expr='ml'),total.assign(expr='total')]).sort_values('decision_ev').groupby('game_id').tail(1)
    summary['expression']={'spread_only':perf(spread,.05),'ml_only':perf(ml,.05),'total_only':perf(total,.05),'best_side':perf(side,.05),'best_any':perf(allm,.05),'best_side_mix':side[side.decision_ev>=.05].expr.value_counts().to_dict(),'best_any_mix':allm[allm.decision_ev>=.05].expr.value_counts().to_dict()}
    summary['maturity']={'min':float(m.n_avg.min()) if len(m) else None,'median':float(m.n_avg.median()) if len(m) else None,'share_ge5':float((m.n_avg>=5).mean()) if len(m) else None}
    for mk in ['spread','total','ml']:
        raw=best_by_game(c,mk); shr=best_by_game(c,mk,shrink=.85); summary['shrink'][mk]={'raw':perf(raw,.05),'shrink85':perf(shr,.05)}
    summary['staking']={'quarter_kelly':kelly_path(allm,.05,.25,.02),'half_kelly':kelly_path(allm,.05,.5,.02),'quarter_kelly_cap1':kelly_path(allm,.05,.25,.01)}
    # one selected market per game already removes same-game stacking; proxy daily cap by tighter per-bet fraction
    summary['portfolio']={'one_market_per_game':True,'uncapped2pct':kelly_path(allm,.05,.25,.02),'conservative1pct':kelly_path(allm,.05,.25,.01)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); print('GATE_BATCH_25'); print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
