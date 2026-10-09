from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import gate_batch_30 as core

OUT=Path('results/gate_batch_35'); OUT.mkdir(parents=True,exist_ok=True)
URL='https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/normalized/odds_consolidated_canonical.csv'

def norm_name(x):
    return core.norm_name(x)

def amer_profit(a):
    a=float(a)
    return a/100.0 if a>0 else 100.0/(-a)

def implied(a):
    a=float(a)
    return 100.0/(a+100.0) if a>0 else (-a)/((-a)+100.0)

def ev(p,a): return p*amer_profit(a)-(1-p)

def settle_win(win,push,a):
    if push: return 0.0
    return amer_profit(a) if win else -1.0

def model_games():
    g=core.load_model_games().copy()
    g['date']=pd.to_datetime(g['date']).dt.normalize()
    return g[g.season.between(2022,2025)].copy()

def load_market():
    use=['event_id','commence_time','game_date','season','home_team_canonical','away_team_canonical','bookmaker','bookmaker_last_update','timestamp','spread','spread_home_price','spread_away_price','moneyline_home_price','moneyline_away_price','is_march_madness']
    df=pd.read_csv(URL,usecols=lambda c:c in use,low_memory=False)
    df=df[pd.to_numeric(df['season'],errors='coerce').between(2021,2025)].copy()
    df['season']=pd.to_numeric(df['season'],errors='coerce').astype(int)
    df['commence_time']=pd.to_datetime(df['commence_time'],utc=True,errors='coerce')
    df['timestamp']=pd.to_datetime(df['timestamp'],utc=True,errors='coerce')
    if 'bookmaker_last_update' in df: df['bookmaker_last_update']=pd.to_datetime(df['bookmaker_last_update'],utc=True,errors='coerce')
    df['game_date']=pd.to_datetime(df['game_date'],errors='coerce').dt.normalize()
    df['lead_min']=(df['commence_time']-df['timestamp']).dt.total_seconds()/60
    df=df[df.lead_min>=0].copy()
    # latest pregame snapshot per event/book
    df=df.sort_values('timestamp').groupby(['event_id','bookmaker'],as_index=False).tail(1)
    return df

def fuzzy_join_events(mkt,games):
    ev=mkt.groupby('event_id',as_index=False).agg(season=('season','first'),game_date=('game_date','first'),home=('home_team_canonical','first'),away=('away_team_canonical','first'),march=('is_march_madness','max'),commence=('commence_time','first'))
    rows=[]; amb=0; offsets={-1:0,0:0,1:0}
    by={(int(s),d):q for (s,d),q in games.groupby(['season','date'])}
    for r in ev.itertuples(index=False):
        ms=[]
        for off in [0,-1,1]:
            q=by.get((int(r.season),r.game_date+pd.Timedelta(days=off)))
            if q is None: continue
            for x in q.itertuples(index=False):
                if core.name_match(r.home,x.team0) and core.name_match(r.away,x.team1): ms.append((x,True,off))
                elif core.name_match(r.home,x.team1) and core.name_match(r.away,x.team0): ms.append((x,False,off))
            if ms: break
        ids={str(z[0].game_id) for z in ms}
        if len(ids)!=1:
            if len(ids)>1: amb+=1
            continue
        x,home0,off=ms[0]; offsets[off]+=1
        rows.append({'event_id':r.event_id,'game_id':x.game_id,'season':int(r.season),'home0':home0,'pred_home_margin':float(x.pred_margin if home0 else -x.pred_margin),'actual_home_margin':float(x.actual_margin if home0 else -x.actual_margin),'pred_total':float(x.pred_total),'actual_total':float(x.actual_total),'n_avg':float(x.n_avg),'march':bool(r.march),'date_offset':off})
    return pd.DataFrame(rows),amb,offsets,len(ev)

def sigmas(games):
    out={}
    for sy in [2023,2024,2025]:
        train=games[games.season<sy]
        resid=(train.actual_margin-train.pred_margin).to_numpy(float)
        out[sy]=float(np.sqrt(np.mean(resid**2)))
    return out

def contracts(mkt,joined,sigma):
    d=mkt.merge(joined,on=['event_id','season'],how='inner')
    rows=[]
    for r in d.itertuples(index=False):
        if int(r.season) not in sigma: continue
        s=sigma[int(r.season)]; pm=float(r.pred_home_margin); am=float(r.actual_home_margin)
        # spread: spread field is home spread
        if pd.notna(r.spread):
            h=float(r.spread)
            ph=float(norm.cdf((pm+h)/s)); pa=1-ph
            for side,p,a,line in [('home',ph,r.spread_home_price,h),('away',pa,r.spread_away_price,-h)]:
                if pd.isna(a): continue
                cover_margin=(am+h) if side=='home' else (-am-h)
                rows.append({'event_id':r.event_id,'game_id':r.game_id,'season':int(r.season),'market':'spread','side':side,'book':r.bookmaker,'price':float(a),'p_model':p,'ev':ev(p,a),'result_margin':cover_margin,'win':cover_margin>0,'push':abs(cover_margin)<1e-9,'return':settle_win(cover_margin>0,abs(cover_margin)<1e-9,a),'march':r.march,'n_avg':r.n_avg,'lead_min':r.lead_min,'line':line})
        # moneyline
        if pd.notna(r.moneyline_home_price) or pd.notna(r.moneyline_away_price):
            ph=float(norm.cdf(pm/s)); pa=1-ph
            for side,p,a in [('home',ph,r.moneyline_home_price),('away',pa,r.moneyline_away_price)]:
                if pd.isna(a): continue
                win=(am>0) if side=='home' else (am<0); push=abs(am)<1e-9
                rows.append({'event_id':r.event_id,'game_id':r.game_id,'season':int(r.season),'market':'ml','side':side,'book':r.bookmaker,'price':float(a),'p_model':p,'ev':ev(p,a),'result_margin':am if side=='home' else -am,'win':win,'push':push,'return':settle_win(win,push,a),'march':r.march,'n_avg':r.n_avg,'lead_min':r.lead_min,'line':np.nan})
    return pd.DataFrame(rows)

def novig_pairs(c):
    z=c.copy(); z['imp']=z.price.map(implied)
    p=z.pivot_table(index=['event_id','season','market','book'],columns='side',values='imp',aggfunc='first').reset_index()
    if 'home' in p and 'away' in p:
        p['overround']=p.home+p.away
        return p.dropna(subset=['home','away'])
    return p.iloc[0:0]

def best_per_game(c,thr,markets=None,shrink=0.0):
    z=c.copy()
    if markets: z=z[z.market.isin(markets)]
    if shrink:
        z['p_use']=.5+(z.p_model-.5)*(1-shrink); z['ev_use']=[ev(p,a) for p,a in zip(z.p_use,z.price)]
    else: z['p_use']=z.p_model; z['ev_use']=z.ev
    z=z[z.ev_use>=thr]
    if len(z)==0: return z
    return z.sort_values('ev_use').groupby(['game_id'],as_index=False).tail(1)

def summ(z):
    if len(z)==0:return {'n':0,'roi':None,'hit':None,'avg_ev':None,'avg_price':None}
    return {'n':int(len(z)),'roi':float(z['return'].mean()),'hit':float(z['win'].mean()),'avg_ev':float(z['ev_use'].mean()),'avg_price':float(z['price'].mean())}

def maxdd(returns,stakes):
    b=1.0; peak=1.0; dd=0.0
    for r,s in zip(returns,stakes):
        b += s*r; peak=max(peak,b); dd=max(dd,(peak-b)/peak)
    return float(dd),float(b-1)

def kelly_frac(p,a):
    b=amer_profit(a); return max(0.0,(b*p-(1-p))/b)

def main():
    games=model_games(); mkt=load_market(); joined,amb,offs,events=fuzzy_join_events(mkt,games); sg=sigmas(games); c=contracts(mkt,joined,sg)
    c.to_csv(OUT/'contracts.csv',index=False); joined.to_csv(OUT/'joined_games.csv',index=False)
    nv=novig_pairs(c); nv.to_csv(OUT/'novig_pairs.csv',index=False)
    gates=[]
    def G(i,name,decision,evidence): gates.append({'gate':f'G35.{i:02d}','name':name,'decision':decision,'evidence':evidence})
    G(1,'price rows loaded','PASS',{'rows':len(mkt),'events':events})
    G(2,'pregame only','PASS' if (mkt.lead_min>=0).all() else 'REJECTED',{'min_lead':float(mkt.lead_min.min()),'median_lead':float(mkt.lead_min.median())})
    G(3,'model join','PASS' if len(joined)>=1000 else 'HOLD',{'matched':len(joined),'events':events,'rate':len(joined)/events,'ambiguous':amb,'offsets':offs})
    G(4,'spread contracts','PASS' if (c.market=='spread').sum()>1000 else 'HOLD',{'n':int((c.market=='spread').sum()),'games':int(c[c.market=='spread'].game_id.nunique())})
    G(5,'ml contracts','PASS' if (c.market=='ml').sum()>1000 else 'HOLD',{'n':int((c.market=='ml').sum()),'games':int(c[c.market=='ml'].game_id.nunique())})
    for j,m in enumerate(['spread','ml'],6):
        q=nv[nv.market==m]; G(j,f'{m} no-vig sanity','PASS' if len(q)>100 and q.overround.median()>1 and q.overround.median()<1.15 else 'HOLD',{'n':len(q),'median_overround':float(q.overround.median()) if len(q) else None})
    idx=8
    grid=[0,.02,.05,.08,.10,.15]
    gridres={}
    for m in ['spread','ml']:
        gridres[m]={}
        for t in grid:
            q=best_per_game(c,t,[m]); gridres[m][str(t)]=summ(q)
        G(idx,f'{m} EV grid','PASS' if any(v['n']>=100 and (v['roi'] or -9)>0 for v in gridres[m].values()) else 'REJECTED',gridres[m]); idx+=1
    # season robustness at 5%
    for m in ['spread','ml']:
        q=best_per_game(c,.05,[m]); yrs={str(y):summ(q[q.season==y]) for y in [2023,2024,2025]}; wins=sum((v['roi'] or -9)>0 and v['n']>=20 for v in yrs.values()); G(idx,f'{m} season robustness','PASS' if wins>=2 else 'REJECTED',yrs); idx+=1
    # march vs regular
    for m in ['spread','ml']:
        q=best_per_game(c,.05,[m]); e={'march':summ(q[q.march==True]),'regular':summ(q[q.march!=True])}; G(idx,f'{m} march vs regular','PASS' if e['regular']['n']>=100 else 'HOLD',e); idx+=1
    # maturity cohorts
    q=best_per_game(c,.05,['spread','ml']);
    for name,lo,hi in [('early',0,5),('mid',5,10),('mature',10,999)]:
        z=q[(q.n_avg>=lo)&(q.n_avg<hi)]; G(idx,f'side maturity {name}','PASS' if len(z)>=30 and z['return'].mean()>0 else 'REJECTED',summ(z)); idx+=1
    # side expression best spread vs ml and either
    sp=best_per_game(c,.05,['spread']); ml=best_per_game(c,.05,['ml']); anym=best_per_game(c,.05,['spread','ml'])
    G(idx,'spread expression', 'PASS' if len(sp)>=100 and sp['return'].mean()>0 else 'REJECTED',summ(sp)); idx+=1
    G(idx,'ml expression', 'PASS' if len(ml)>=100 and ml['return'].mean()>0 else 'REJECTED',summ(ml)); idx+=1
    G(idx,'dynamic side expression','PASS' if len(anym)>=100 and anym['return'].mean()>0 else 'REJECTED',summ(anym)); idx+=1
    # price shopping vs each book's max population: best across books already; compare common bookmaker baseline
    books=c.book.value_counts().head(5).index.tolist(); br={}
    for b in books:
        qb=best_per_game(c[c.book==b],.05,['spread','ml']); br[b]=summ(qb)
    G(idx,'book robustness','PASS' if sum((v['roi'] or -9)>0 and v['n']>=30 for v in br.values())>=2 else 'HOLD',br); idx+=1
    # thresholds plateau on dynamic side
    tres={str(t):summ(best_per_game(c,t,['spread','ml'])) for t in grid}; pos=[float(t) for t,v in tres.items() if v['n']>=100 and (v['roi'] or -9)>0]
    G(idx,'raw EV threshold neighborhood','PASS' if len(pos)>=2 else 'HOLD',tres); idx+=1
    # shrinkage grid
    shr={};
    for sh in [0,.10,.15,.20]: shr[str(sh)]=summ(best_per_game(c,.05,['spread','ml'],shrink=sh))
    G(idx,'uncertainty shrink','PASS' if any(v['n']>=100 and (v['roi'] or -9)>0 for k,v in shr.items() if k!='0') else 'REJECTED',shr); idx+=1
    # same game stacking explicitly prevented by selector
    G(idx,'one market per game','PASS',{'selected':len(anym),'unique_games':int(anym.game_id.nunique())}); idx+=1
    # Kelly/caps on dynamic sides, time order by season/game id proxy
    z=anym.sort_values(['season','game_id']).copy(); kres={}
    for frac,cap in [(0.125,.01),(.25,.01),(.25,.02),(.5,.02)]:
        stakes=np.minimum([kelly_frac(p,a)*frac for p,a in zip(z.p_use,z.price)],cap); dd,ret=maxdd(z['return'].to_numpy(),np.array(stakes)); kres[f'{frac}_{cap}']={'return':ret,'maxdd':dd,'avg_stake':float(np.mean(stakes))}
    G(idx,'kelly stress','PASS' if any(v['return']>0 and v['maxdd']<.25 for v in kres.values()) else 'REJECTED',kres); idx+=1
    G(idx,'exposure cap stress','PASS' if kres['0.25_0.01']['maxdd']<=kres['0.25_0.02']['maxdd'] else 'HOLD',kres); idx+=1
    # book concentration among selected
    bc=z.book.value_counts(normalize=True).head(5).to_dict(); G(idx,'book concentration','PASS' if (max(bc.values()) if bc else 1)<.5 else 'HOLD',bc); idx+=1
    # favorite dog split spread only
    if len(sp):
        sd=sp.copy(); sd['fav']=(sd.result_margin*0+sd.line<0) # line is selected side's spread
        fd={'fav':summ(sd[sd.line<0]),'dog':summ(sd[sd.line>0])}; G(idx,'spread favorite/dog robustness','PASS' if all(v['n']>=30 for v in fd.values()) else 'HOLD',fd)
    else:G(idx,'spread favorite/dog robustness','HOLD',{}); idx+=1
    # price ranges ML
    if len(ml):
        mr={};
        for nm,lo,hi in [('fav',-10000,-101),('near',-100,150),('dog',150,10000)]: mr[nm]=summ(ml[(ml.price>=lo)&(ml.price<hi)])
        G(idx,'ml price-band robustness','PASS' if sum(v['n']>=20 and (v['roi'] or -9)>0 for v in mr.values())>=2 else 'HOLD',mr)
    else:G(idx,'ml price-band robustness','HOLD',{}); idx+=1
    # bootstrap dynamic side ROI by game
    vals=z['return'].to_numpy(float); rng=np.random.default_rng(31); boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)]) if len(vals) else np.array([])
    be={'n':len(vals),'roi':float(vals.mean()) if len(vals) else None,'ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))] if len(boots) else None}; G(idx,'side ROI bootstrap','PASS' if len(boots) and np.quantile(boots,.025)>0 else 'HOLD',be); idx+=1
    # annual portfolio returns
    ar={str(y):summ(z[z.season==y]) for y in [2023,2024,2025]}; G(idx,'dynamic side annual replication','PASS' if sum((v['roi'] or -9)>0 and v['n']>=20 for v in ar.values())>=2 else 'REJECTED',ar); idx+=1
    # March reconciliation
    rec={'march':summ(z[z.march==True]),'regular':summ(z[z.march!=True])}; G(idx,'tournament-full-season reconciliation','PASS' if rec['regular']['n']>=100 else 'HOLD',rec); idx+=1
    # CLV feasibility from timestamp multiplicity
    snaps=mkt.groupby(['event_id','bookmaker']).size(); G(idx,'CLV snapshot multiplicity','PASS' if float((snaps>1).mean())>.2 else 'HOLD',{'multi_snapshot_share':float((snaps>1).mean()),'max_snapshots':int(snaps.max())}); idx+=1
    # close quote feasibility: lead time
    G(idx,'close-price feasibility','PASS' if mkt.lead_min.median()<60 else 'HOLD',{'median_lead_min':float(mkt.lead_min.median()),'p90':float(mkt.lead_min.quantile(.9))}); idx+=1
    # production eligibility
    pass_markets=[m for m in ['spread','ml'] if next(g for g in gates if g['name']==f'{m} season robustness')['decision']=='PASS']
    G(idx,'market competence frontier','PASS' if pass_markets else 'HOLD',{'production_candidates':pass_markets}); idx+=1
    G(idx,'bet policy readiness','PASS' if pass_markets and next(g for g in gates if g['name']=='raw EV threshold neighborhood')['decision']=='PASS' else 'HOLD',{'markets':pass_markets}); idx+=1
    G(idx,'staking readiness','PASS' if next(g for g in gates if g['name']=='kelly stress')['decision']=='PASS' else 'HOLD',{}); idx+=1
    G(idx,'portfolio readiness','PASS' if next(g for g in gates if g['name']=='book concentration')['decision']=='PASS' else 'HOLD',{}); idx+=1
    G(idx,'architecture freeze readiness','PASS' if all(next(g for g in gates if g['name']==n)['decision']=='PASS' for n in ['bet policy readiness','staking readiness','portfolio readiness']) else 'HOLD',{'2026_used':False}); idx+=1
    assert len(gates)==35,(len(gates),[g['gate'] for g in gates])
    summary={'sigmas':sg,'joined_games':len(joined),'market_rows':len(mkt),'contracts':len(c),'gates':gates}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); pd.DataFrame(gates).to_json(OUT/'gates.json',orient='records',indent=2)
    print('GATE_BATCH_35'); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
