from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import gate_batch_35 as b
OUT=Path('results/gate_batch_35_v2'); OUT.mkdir(parents=True,exist_ok=True)

def main():
    games=b.model_games(); mkt=b.load_market(); joined,amb,offs,events=b.fuzzy_join_events(mkt,games); sg=b.sigmas(games); c=b.contracts(mkt,joined,sg)
    c.to_csv(OUT/'contracts.csv',index=False); joined.to_csv(OUT/'joined_games.csv',index=False)
    nv=b.novig_pairs(c); nv.to_csv(OUT/'novig_pairs.csv',index=False)
    gates=[]
    def G(name,decision,evidence): gates.append({'gate':f'G35.{len(gates)+1:02d}','name':name,'decision':decision,'evidence':evidence})
    G('price rows loaded','PASS',{'rows':len(mkt),'events':events})
    G('pregame only','PASS' if (mkt.lead_min>=0).all() else 'REJECTED',{'min_lead':float(mkt.lead_min.min()),'median_lead':float(mkt.lead_min.median())})
    G('model join','PASS' if len(joined)>=1000 else 'HOLD',{'matched':len(joined),'events':events,'rate':len(joined)/events,'ambiguous':amb,'offsets':offs})
    G('spread contracts','PASS' if (c.market=='spread').sum()>1000 else 'HOLD',{'n':int((c.market=='spread').sum()),'games':int(c[c.market=='spread'].game_id.nunique())})
    G('ml contracts','PASS' if (c.market=='ml').sum()>1000 else 'HOLD',{'n':int((c.market=='ml').sum()),'games':int(c[c.market=='ml'].game_id.nunique())})
    for m in ['spread','ml']:
        q=nv[nv.market==m]; G(f'{m} no-vig sanity','PASS' if len(q)>100 and 1<q.overround.median()<1.15 else 'HOLD',{'n':len(q),'median_overround':float(q.overround.median()) if len(q) else None})
    grid=[0,.02,.05,.08,.10,.15]; gridres={}
    for m in ['spread','ml']:
        gridres[m]={str(t):b.summ(b.best_per_game(c,t,[m])) for t in grid}
        G(f'{m} EV grid','PASS' if any(v['n']>=100 and (v['roi'] or -9)>0 for v in gridres[m].values()) else 'REJECTED',gridres[m])
    for m in ['spread','ml']:
        q=b.best_per_game(c,.05,[m]); yrs={str(y):b.summ(q[q.season==y]) for y in [2023,2024,2025]}; wins=sum((v['roi'] or -9)>0 and v['n']>=20 for v in yrs.values())
        G(f'{m} season robustness','PASS' if wins>=2 else 'REJECTED',yrs)
    for m in ['spread','ml']:
        q=b.best_per_game(c,.05,[m]); e={'march':b.summ(q[q.march==True]),'regular':b.summ(q[q.march!=True])}
        G(f'{m} march vs regular','PASS' if e['regular']['n']>=100 else 'HOLD',e)
    q=b.best_per_game(c,.05,['spread','ml'])
    for name,lo,hi in [('early',0,5),('mid',5,10),('mature',10,999)]:
        z=q[(q.n_avg>=lo)&(q.n_avg<hi)]; G(f'side maturity {name}','PASS' if len(z)>=30 and z['return'].mean()>0 else 'REJECTED',b.summ(z))
    sp=b.best_per_game(c,.05,['spread']); ml=b.best_per_game(c,.05,['ml']); anym=b.best_per_game(c,.05,['spread','ml'])
    G('spread expression','PASS' if len(sp)>=100 and sp['return'].mean()>0 else 'REJECTED',b.summ(sp))
    G('ml expression','PASS' if len(ml)>=100 and ml['return'].mean()>0 else 'REJECTED',b.summ(ml))
    G('dynamic side expression','PASS' if len(anym)>=100 and anym['return'].mean()>0 else 'REJECTED',b.summ(anym))
    books=c.book.value_counts().head(5).index.tolist(); br={}
    for book in books: br[book]=b.summ(b.best_per_game(c[c.book==book],.05,['spread','ml']))
    G('book robustness','PASS' if sum((v['roi'] or -9)>0 and v['n']>=30 for v in br.values())>=2 else 'HOLD',br)
    tres={str(t):b.summ(b.best_per_game(c,t,['spread','ml'])) for t in grid}; pos=[float(t) for t,v in tres.items() if v['n']>=100 and (v['roi'] or -9)>0]
    G('raw EV threshold neighborhood','PASS' if len(pos)>=2 else 'HOLD',tres)
    shr={str(sh):b.summ(b.best_per_game(c,.05,['spread','ml'],shrink=sh)) for sh in [0,.10,.15,.20]}
    G('uncertainty shrink','PASS' if any(v['n']>=100 and (v['roi'] or -9)>0 for k,v in shr.items() if k!='0') else 'REJECTED',shr)
    G('one market per game','PASS',{'selected':len(anym),'unique_games':int(anym.game_id.nunique())})
    z=anym.sort_values(['season','game_id']).copy(); kres={}
    for frac,cap in [(0.125,.01),(.25,.01),(.25,.02),(.5,.02)]:
        stakes=np.minimum([b.kelly_frac(p,a)*frac for p,a in zip(z.p_use,z.price)],cap); dd,ret=b.maxdd(z['return'].to_numpy(),np.array(stakes)); kres[f'{frac}_{cap}']={'return':ret,'maxdd':dd,'avg_stake':float(np.mean(stakes))}
    G('kelly stress','PASS' if any(v['return']>0 and v['maxdd']<.25 for v in kres.values()) else 'REJECTED',kres)
    G('exposure cap stress','PASS' if kres['0.25_0.01']['maxdd']<=kres['0.25_0.02']['maxdd'] else 'HOLD',kres)
    bc=z.book.value_counts(normalize=True).head(5).to_dict(); G('book concentration','PASS' if (max(bc.values()) if bc else 1)<.5 else 'HOLD',bc)
    if len(sp):
        fd={'fav':b.summ(sp[sp.line<0]),'dog':b.summ(sp[sp.line>0])}; G('spread favorite/dog robustness','PASS' if all(v['n']>=30 for v in fd.values()) else 'HOLD',fd)
    else: G('spread favorite/dog robustness','HOLD',{})
    if len(ml):
        mr={}
        for nm,lo,hi in [('fav',-10000,-101),('near',-100,150),('dog',150,10000)]: mr[nm]=b.summ(ml[(ml.price>=lo)&(ml.price<hi)])
        G('ml price-band robustness','PASS' if sum(v['n']>=20 and (v['roi'] or -9)>0 for v in mr.values())>=2 else 'HOLD',mr)
    else: G('ml price-band robustness','HOLD',{})
    vals=z['return'].to_numpy(float); rng=np.random.default_rng(31); boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)]) if len(vals) else np.array([])
    be={'n':len(vals),'roi':float(vals.mean()) if len(vals) else None,'ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))] if len(boots) else None}; G('side ROI bootstrap','PASS' if len(boots) and np.quantile(boots,.025)>0 else 'HOLD',be)
    ar={str(y):b.summ(z[z.season==y]) for y in [2023,2024,2025]}; G('dynamic side annual replication','PASS' if sum((v['roi'] or -9)>0 and v['n']>=20 for v in ar.values())>=2 else 'REJECTED',ar)
    rec={'march':b.summ(z[z.march==True]),'regular':b.summ(z[z.march!=True])}; G('tournament-full-season reconciliation','PASS' if rec['regular']['n']>=100 else 'HOLD',rec)
    snaps=mkt.groupby(['event_id','bookmaker']).size(); G('CLV snapshot multiplicity','PASS' if float((snaps>1).mean())>.2 else 'HOLD',{'multi_snapshot_share':float((snaps>1).mean()),'max_snapshots':int(snaps.max())})
    G('close-price feasibility','PASS' if mkt.lead_min.median()<60 else 'HOLD',{'median_lead_min':float(mkt.lead_min.median()),'p90':float(mkt.lead_min.quantile(.9))})
    pass_markets=[m for m in ['spread','ml'] if next(g for g in gates if g['name']==f'{m} season robustness')['decision']=='PASS']
    G('market competence frontier','PASS' if pass_markets else 'HOLD',{'production_candidates':pass_markets})
    G('bet policy readiness','PASS' if pass_markets and next(g for g in gates if g['name']=='raw EV threshold neighborhood')['decision']=='PASS' else 'HOLD',{'markets':pass_markets})
    assert len(gates)==35,len(gates)
    summary={'sigmas':sg,'joined_games':len(joined),'market_rows':len(mkt),'contracts':len(c),'gates':gates}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); (OUT/'gates.json').write_text(json.dumps(gates,indent=2,default=str))
    print('GATE_BATCH_35_V2'); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
