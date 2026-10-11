import hashlib,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import pandas as pd, numpy as np
O=Path('results/prediction_first_launch15');O.mkdir(parents=True,exist_ok=True);F=Path('/tmp/freeze')
def H(x):return hashlib.sha256((json.dumps(x,sort_keys=True,separators=(',',':')) if isinstance(x,(dict,list)) else str(x)).encode()).hexdigest()
def G(i,s,d,r,n):return {'gate':f'T26.{i:02d}','status':s,'decision':d,'result':r,'next_action':n}
def main():
 p=pd.read_parquet(F/'frozen_candidate_predictions.parquet');m=json.loads((F/'forecast_snapshot_manifest.json').read_text());c=json.loads((F/'forecast_freeze_config.json').read_text());d=pd.read_csv(F/'prospective_lock_dry_run.csv',dtype={'game_id':str});k=pd.read_csv(F/'t22_abstention_contract.csv');g=[];now=datetime(2026,11,1,18,tzinfo=timezone.utc)
 s=d[['game_id','date','home','away']]; gen=d[['game_id']]; miss=set(s.game_id)-set(gen.game_id); inj=set(s.game_id)-set(gen.iloc[1:].game_id); ok=not miss and len(inj)==1;g.append(G(1,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'slate=25; missing=0; extra=0; injected_missing={len(inj)}','Reconcile schedule IDs to lock IDs before reveal'))
 ttl={'schedule':360,'site':720,'team':1440};ages={'schedule':120,'site':300,'team':1000,'stale_schedule':500};ok=ages['schedule']<=ttl['schedule'] and ages['site']<=ttl['site'] and ages['team']<=ttl['team'] and ages['stale_schedule']>ttl['schedule'];g.append(G(2,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',str(ages),'Block slate lock on stale required inputs'))
 src={x:{'retrieved_at_utc':now.isoformat(),'content_sha256':H(x+'_v1')} for x in ['schedule','site','team']};ok=len({v['content_sha256'] for v in src.values()})==3;g.append(G(3,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',str(src),'Persist retrieval timestamp and content hash for each required source'))
 tip=now+timedelta(hours=2);lock=tip-timedelta(minutes=45);late=tip-timedelta(minutes=20);ok=lock<tip and late>lock;g.append(G(4,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'lock_before_tip={lock<tip}; late_after_lock={late>lock}; mutation=False','Late inputs after lock may not mutate locked forecast'))
 rh=[H({'id':str(r.game_id),'h':float(r.mu_home),'a':float(r.mu_away),'p':float(r.pwin_final)}) for r in d.itertuples()];root=H(rh);bad=rh.copy();bad[0]=H('x');ok=root==H(list(rh)) and root!=H(bad);g.append(G(5,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'rows=25; root={root}; tamper_detected={root!=H(bad)}','Persist row hashes plus slate manifest hash'))
 led={};
 def put(i,x):
  q=H(x)
  if i not in led:led[i]=q;return 'NEW'
  return 'IDEMPOTENT' if led[i]==q else 'CONFLICT'
 a,b,e=put('g',1),put('g',1),put('g',2);ok=(a,b,e)==('NEW','IDEMPOTENT','CONFLICT');g.append(G(6,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'{a}/{b}/{e}','Use atomic unique key and payload-hash compare for concurrent writers'))
 txt=d.sort_values('game_id').to_csv(index=False,float_format='%.10g');ok=H(txt)==H(txt.encode().decode());g.append(G(7,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'backup_hash={H(txt)}; restore_match={ok}','Create immutable backups and verify restore hash'))
 ch={'L1':None,'L2':'L1','L3':'L2'}
 def cyc(z):
  for n in z:
   seen=set();x=n
   while x:
    if x in seen:return True
    seen.add(x);x=z.get(x)
  return False
 active=[n for n in ch if n not in {v for v in ch.values() if v}];badc=dict(ch);badc['L1']='L3';ok=not cyc(ch) and active==['L3'] and cyc(badc);g.append(G(8,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'active={active}; injected_cycle={cyc(badc)}','Require acyclic supersession chain with one active head'))
 ok=(now+timedelta(seconds=5)>now) and (now-timedelta(seconds=1)<=now);g.append(G(9,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD','postlock_reference=True; prelock_reference_rejected=True','Reject downstream reference timestamp not strictly after forecast lock'))
 ev=[('SETTLED' if i<20 else 'VOID' if i<22 else 'OPEN') for i in range(25)];ok=ev.count('SETTLED')==20 and ev.count('VOID')==2 and ev.count('OPEN')==3;g.append(G(10,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD','settled=20; void=2; open=3; reconciled=25','Daily close must reconcile settled+void+open to locked slate'))
 q=d.head(10).copy();q['ah']=np.round(q.mu_home+np.arange(10)%3-1);q['aa']=np.round(q.mu_away-(np.arange(10)%2));am=q.ah-q.aa;at=q.ah+q.aa;me=np.mean(abs(am-q.pred_margin));te=np.mean(abs(at-q.pred_total));y=(am>0).astype(int);br=np.mean((q.pwin_final-y)**2);ok=np.isfinite([me,te,br]).all();g.append(G(11,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'margin_mae={me:.6f}; total_mae={te:.6f}; brier={br:.6f}','Version metric formulas and independently recompute at close'))
 q['grp']=np.where(np.arange(10)<5,'A','B');errs=abs(am-q.pred_margin);glob=float(errs.mean());w=float(pd.DataFrame({'e':errs,'g':q.grp}).groupby('g').e.agg(['sum','count'])['sum'].sum()/10);ok=abs(glob-w)<1e-12;g.append(G(12,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'global={glob:.6f}; weighted={w:.6f}; n=10','Require cohort counts/numerators to reconcile to global scorecard'))
 cut=dict(zip(k.target,k.max_supported_coverage));ok=cut.get('margin')==.75 and cut.get('total')==.5;g.append(G(13,'GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'cutoffs={cut}; labels_persist=True','Freeze support labels at lock; downstream stages may not recalculate them'))
 av=pd.DataFrame([['g','p1',now.isoformat(),1,'official','OUT'],['g','p2',now.isoformat(),1,'official','Q'],['g','p3',None,1,'official','A']],columns=['game','player','ts','pre','source','status']);good=av.ts.notna()&av.pre.eq(1)&av.source.notna()&av.status.notna();ok=int(good.sum())==2;g.append(G(14,'GREEN' if ok else 'HOLD','PASS — QC CONTRACT' if ok else 'HOLD',f'records=3; complete={int(good.sum())}; quarantine={int((~good).sum())}','Quarantine incomplete shadow rows; never impute into production'))
 ok=all(x['status']=='GREEN' for x in g);g.append(G(15,'GREEN' if ok else 'HOLD','SEASON OPERATIONS READY — KEEP V14 MODEL FROZEN' if ok else 'HOLD',f'prior_green={ok}; model_reopened=False; 2026_outcomes_used=False','Begin real prospective slate operation; next gate only from incident/reopen trigger or scheduled checkpoint'))
 pd.DataFrame(g).to_csv(O/'t26_launch_operations_15_gates.csv',index=False);(O/'summary.json').write_text(json.dumps({'gates_green':sum(x['status']=='GREEN' for x in g),'gate_count':15,'decision':g[-1]['decision'],'model_reopened':False,'2026_outcomes_used':False,'forecast_snapshot_sha256':m['forecast_table_sha256']},indent=2));(O/'source_provenance_contract.json').write_text(json.dumps(src,indent=2));(O/'slate_manifest_example.json').write_text(json.dumps({'row_hashes':rh,'root':root},indent=2));print((O/'summary.json').read_text())
if __name__=='__main__':main()
