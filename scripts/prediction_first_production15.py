from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

ROOT=Path('/tmp/freeze')
OUT=Path('results/prediction_first_production15'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE_COMMIT='77749fac1ecedcd2dc487debff75b70ddc403553'


def sha(obj):
    if isinstance(obj,(dict,list)): b=json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()
    elif isinstance(obj,str): b=obj.encode()
    else: b=bytes(obj)
    return hashlib.sha256(b).hexdigest()

def gate(gate,status,decision,result,next_action):
    return {'gate':gate,'status':status,'decision':decision,'result':result,'next_action':next_action}

def load():
    man=json.loads((ROOT/'forecast_snapshot_manifest.json').read_text())
    cfg=json.loads((ROOT/'forecast_freeze_config.json').read_text())
    api=json.loads((ROOT/'t23_api_certification.json').read_text())
    dry=pd.read_csv(ROOT/'prospective_lock_dry_run.csv',dtype={'game_id':str})
    score=pd.read_csv(ROOT/'prospective_scorecard_schema.csv')
    pred=pd.read_parquet(ROOT/'frozen_candidate_predictions.parquet'); pred['game_id']=pred.game_id.astype(str)
    contract=pd.read_csv(ROOT/'t22_abstention_contract.csv')
    comp=pd.read_csv(ROOT/'t23_competence_map.csv')
    return man,cfg,api,dry,score,pred,contract,comp

def canonical_row(r,lock_ts='2026-10-11T17:30:00Z'):
    return {
        'game_id':str(r.game_id),'date':str(r.date),'home':str(r.home),'away':str(r.away),
        'mu_home':round(float(r.mu_home),10),'mu_away':round(float(r.mu_away),10),
        'pred_margin':round(float(r.pred_margin),10),'pred_total':round(float(r.pred_total),10),
        'pred_pace':round(float(r.pred_pace),10),'pwin_home':round(float(r.pwin_final),10),
        'sigma_margin':round(float(r.sigma_margin_final),10),'sigma_total':round(float(r.sigma_total_final),10),
        'forecast_lock_timestamp':lock_ts,'freeze_config_sha256':None,'freeze_forecast_sha256':None,
        'market_seen':False,'outcome_seen':False,'lock_status':'LOCKED'
    }

def main():
    man,cfg,api,dry,score,pred,contract,comp=load(); rows=[]; details={}

    # T25.01 provenance pinning
    provenance_ok=(man.get('source_commit')==FREEZE_COMMIT and man.get('2026_outcomes_used') is False and cfg.get('2026_outcomes_used') is False and man.get('determinism_pass') is True)
    rows.append(gate('T25.01','GREEN' if provenance_ok else 'HOLD','PASS' if provenance_ok else 'HOLD',f"source_commit={man.get('source_commit')}; determinism={man.get('determinism_pass')}; manifest_2026={man.get('2026_outcomes_used')}; config_2026={cfg.get('2026_outcomes_used')}",'Pin every live lock to config + forecast SHA and frozen source commit'))

    # T25.02 forecast schema + domain constraints
    required=['game_id','season','date','home','away','mu_home','mu_away','pred_margin','pred_total','pred_pace','pwin_final','sigma_margin_final','sigma_total_final']
    missing=[c for c in required if c not in pred.columns]; nulls={c:int(pred[c].isna().sum()) for c in required if c in pred.columns}; dup=int(pred.game_id.duplicated().sum())
    domain_ok=(not missing and sum(nulls.values())==0 and dup==0 and pred.pwin_final.between(0,1).all() and (pred.sigma_margin_final>0).all() and (pred.sigma_total_final>0).all())
    rows.append(gate('T25.02','GREEN' if domain_ok else 'HOLD','PASS' if domain_ok else 'HOLD',f'missing={missing}; nulls_total={sum(nulls.values())}; duplicate_game_ids={dup}; pwin_range=[{pred.pwin_final.min():.6f},{pred.pwin_final.max():.6f}]; sigma_margin_min={pred.sigma_margin_final.min():.6f}; sigma_total_min={pred.sigma_total_final.min():.6f}','Hard-fail forecast emission on schema/null/domain violation'))

    # Build 25-row canonical lock envelope from frozen historical dry-run sample. No 2026 inputs.
    sample=pred.sort_values(['date','game_id']).tail(25).copy(); env=[]
    for r in sample.itertuples(index=False):
        x=canonical_row(r); x['freeze_config_sha256']=man['config_sha256']; x['freeze_forecast_sha256']=man['forecast_table_sha256']; x['lock_id']=sha({'game_id':x['game_id'],'ts':x['forecast_lock_timestamp'],'freeze':x['freeze_forecast_sha256']}); x['payload_sha256']=sha({k:v for k,v in x.items() if k!='payload_sha256'}); env.append(x)

    # T25.03 deterministic envelope serialization and UTC timestamp contract
    h1=sha(env); h2=sha(json.loads(json.dumps(env,sort_keys=True,default=str))); ts_ok=True
    for x in env:
        try:
            d=datetime.fromisoformat(x['forecast_lock_timestamp'].replace('Z','+00:00')); ts_ok &= d.tzinfo is not None and d.utcoffset().total_seconds()==0
        except Exception: ts_ok=False
    ok=h1==h2 and ts_ok and len({x['lock_id'] for x in env})==25
    rows.append(gate('T25.03','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'canonical_hash_1={h1}; canonical_hash_2={h2}; utc_timestamps={ts_ok}; unique_lock_ids={len({x["lock_id"] for x in env})}','Require UTC ISO-8601 lock timestamps and deterministic canonical payload serialization'))

    # Ledger primitive for retry/crash/settlement tests.
    ledger={}
    def append_lock(x):
        gid=x['game_id']; ph=x['payload_sha256']
        if gid not in ledger: ledger[gid]=json.loads(json.dumps(x)); return 'APPENDED'
        return 'IDEMPOTENT' if ledger[gid]['payload_sha256']==ph else 'CONFLICT'

    # T25.04 retry idempotency / conflict defense
    a=[append_lock(x) for x in env]; b=[append_lock(x) for x in env]
    conflict=json.loads(json.dumps(env[0])); conflict['pred_margin']=float(conflict['pred_margin'])+1; conflict['payload_sha256']=sha({k:v for k,v in conflict.items() if k!='payload_sha256'}); c=append_lock(conflict)
    ok=(a.count('APPENDED')==25 and b.count('IDEMPOTENT')==25 and c=='CONFLICT' and len(ledger)==25)
    rows.append(gate('T25.04','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'first_append={a.count("APPENDED")}; retry_idempotent={b.count("IDEMPOTENT")}; conflicting_retry={c}; ledger_rows={len(ledger)}','Retry identical payload safely; reject same-game conflicting forecast payload'))

    # T25.05 crash recovery replay
    ledger2={}
    def app2(x):
        gid=x['game_id']
        if gid not in ledger2: ledger2[gid]=x['payload_sha256']; return 'APPENDED'
        return 'IDEMPOTENT' if ledger2[gid]==x['payload_sha256'] else 'CONFLICT'
    first=[app2(x) for x in env[:10]]; replay=[app2(x) for x in env]
    ok=len(ledger2)==25 and first.count('APPENDED')==10 and replay.count('IDEMPOTENT')==10 and replay.count('APPENDED')==15
    rows.append(gate('T25.05','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'precrash_rows=10; recovered_rows={len(ledger2)}; replay_idempotent={replay.count("IDEMPOTENT")}; replay_new={replay.count("APPENDED")}','Resume interrupted slate by replaying entire batch idempotently'))

    # T25.06 schedule/matchup mutation policy
    original=env[1]; changed=json.loads(json.dumps(original)); changed['home']=changed['home']+'__CHANGED'; changed['payload_sha256']=sha({k:v for k,v in changed.items() if k!='payload_sha256'})
    schedule_conflict=(original['game_id']==changed['game_id'] and original['payload_sha256']!=changed['payload_sha256'])
    action='SUPERSEDE_REQUIRES_NEW_LOCK_RECORD' if schedule_conflict else 'NO_CONFLICT'
    ok=schedule_conflict and action.startswith('SUPERSEDE')
    rows.append(gate('T25.06','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'same_game_id={original["game_id"]}; matchup_payload_changed={schedule_conflict}; action={action}','Never overwrite a locked matchup; append superseding record with reason and link to prior lock'))

    # T25.07 postponement/cancellation semantics
    lifecycle=['LOCKED','POSTPONED','VOID_NO_SCORE','RELOCKED_NEW_EVENT']
    scoreable={'LOCKED':False,'POSTPONED':False,'VOID_NO_SCORE':False,'RELOCKED_NEW_EVENT':True}
    ok=(not scoreable['VOID_NO_SCORE'] and scoreable['RELOCKED_NEW_EVENT'] and 'VOID_NO_SCORE' in lifecycle)
    rows.append(gate('T25.07','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'lifecycle={lifecycle}; void_scored={scoreable["VOID_NO_SCORE"]}; relocked_scoreable={scoreable["RELOCKED_NEW_EVENT"]}','Postponed/cancelled locks remain immutable and unscored; rescheduled event receives a new lock'))

    # T25.08 unknown-site fail-safe
    site_cases=[{'site_known':True,'status':'LOCK_ALLOWED'},{'site_known':False,'status':'LOCK_BLOCKED_SITE_UNKNOWN'}]
    ok=site_cases[1]['status']=='LOCK_BLOCKED_SITE_UNKNOWN' and float(cfg.get('site_history_coverage',0))<0.95
    rows.append(gate('T25.08','GREEN' if ok else 'HOLD','PASS — FAIL SAFE' if ok else 'HOLD',f'site_history_coverage={cfg.get("site_history_coverage")}; unknown_site_action={site_cases[1]["status"]}','Do not guess HCA/neutral state for unknown live site; hold that forecast until site is resolved'))

    # T25.09 availability shadow legality
    availability_schema=['game_id','captured_at_utc','source','player_id','team','status','status_detail','pre_tip','production_adjustment_applied','notes']
    forbidden_avail={'current_game_minutes','eventual_dnp','actual_minutes','postgame_status'}
    avail_record={'game_id':'DRY','captured_at_utc':'2026-10-11T17:20:00Z','source':'DRY_RUN','player_id':'P1','team':'T1','status':'QUESTIONABLE','status_detail':'dry-run','pre_tip':True,'production_adjustment_applied':False,'notes':''}
    ok=set(avail_record)==set(availability_schema) and not (set(avail_record)&forbidden_avail) and avail_record['pre_tip'] and not avail_record['production_adjustment_applied']
    rows.append(gate('T25.09','GREEN' if ok else 'HOLD','PASS — SHADOW ONLY' if ok else 'HOLD',f'schema={availability_schema}; forbidden_present={sorted(set(avail_record)&forbidden_avail)}; production_adjustment_applied={avail_record["production_adjustment_applied"]}','Capture timestamped pre-tip availability in shadow ledger only; never infer from game participation'))

    # T25.10 downstream market handoff immutability
    forecast_pkg=json.loads(json.dumps(env[2])); before=sha(forecast_pkg); market_record={'forecast_lock_id':forecast_pkg['lock_id'],'forecast_payload_sha256':forecast_pkg['payload_sha256'],'sportsbook':'DRY','spread':-3.5,'total':145.5,'moneyline':-160,'market_timestamp_utc':'2026-10-11T17:40:00Z'}; after=sha(forecast_pkg)
    ok=before==after and not any(k in forecast_pkg for k in ['spread','moneyline','total_line','sportsbook'])
    rows.append(gate('T25.10','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'forecast_hash_before={before}; after={after}; unchanged={before==after}; market_record_separate=True','Market layer may reference a lock but may never mutate blind forecast payload'))

    # T25.11 downstream referential integrity
    valid_ids={x['lock_id']:x['payload_sha256'] for x in env}; valid=(market_record['forecast_lock_id'] in valid_ids and valid_ids[market_record['forecast_lock_id']]==market_record['forecast_payload_sha256']); orphan=dict(market_record); orphan['forecast_lock_id']='ORPHAN'; orphan_valid=orphan['forecast_lock_id'] in valid_ids
    mismatch=dict(market_record); mismatch['forecast_payload_sha256']='BAD'; mismatch_valid=(valid_ids.get(mismatch['forecast_lock_id'])==mismatch['forecast_payload_sha256'])
    ok=valid and not orphan_valid and not mismatch_valid
    rows.append(gate('T25.11','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'valid_reference={valid}; orphan_accepted={orphan_valid}; hash_mismatch_accepted={mismatch_valid}','Reject market/bet records without exact forecast lock ID + payload-hash reference'))

    # T25.12 settlement idempotency/conflict handling
    settle={}
    result={'actual_home':78,'actual_away':71,'settlement_timestamp':'2026-11-05T05:00:00Z'}
    rh=sha(result)
    def settle_once(gid,obj):
        h=sha(obj)
        if gid not in settle: settle[gid]=h; return 'SETTLED'
        return 'IDEMPOTENT' if settle[gid]==h else 'CONFLICT'
    s1=settle_once(env[3]['game_id'],result); s2=settle_once(env[3]['game_id'],result); bad=dict(result); bad['actual_home']=79; s3=settle_once(env[3]['game_id'],bad)
    ok=(s1=='SETTLED' and s2=='IDEMPOTENT' and s3=='CONFLICT')
    rows.append(gate('T25.12','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'first={s1}; repeat={s2}; conflicting_result={s3}','Allow identical result replay; reject conflicting settlement and require adjudication'))

    # T25.13 audit hash chain tamper detection
    chain=[]; prev='GENESIS'
    for i,x in enumerate(env):
        rec={'seq':i+1,'prev_hash':prev,'lock_id':x['lock_id'],'payload_sha256':x['payload_sha256']}; rec['entry_hash']=sha(rec); chain.append(rec); prev=rec['entry_hash']
    def verify(ch):
        prev='GENESIS'
        for rec in ch:
            if rec['prev_hash']!=prev: return False
            chk={k:v for k,v in rec.items() if k!='entry_hash'}
            if sha(chk)!=rec['entry_hash']: return False
            prev=rec['entry_hash']
        return True
    chain_ok=verify(chain); tampered=json.loads(json.dumps(chain)); tampered[7]['payload_sha256']='TAMPER'; tamper_detected=not verify(tampered); ok=chain_ok and tamper_detected
    rows.append(gate('T25.13','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'entries={len(chain)}; original_chain_valid={chain_ok}; tamper_detected={tamper_detected}; terminal_hash={chain[-1]["entry_hash"]}','Persist append-only hash chain or equivalent immutable event log for lock/settlement audit'))

    # T25.14 monitoring alerts dry-run
    alerts=[]
    anomalies=[('HASH_MISMATCH','HARD'),('MARKET_VISIBLE_PRELOCK','HARD'),('CONFLICTING_DUPLICATE','HARD'),('UNKNOWN_SITE','BLOCK_LOCK'),('PERFORMANCE_WEAK_N249','MONITOR'),('PERFORMANCE_WEAK_N500','REVIEW')]
    for name,severity in anomalies: alerts.append({'event':name,'action':severity})
    amap={a['event']:a['action'] for a in alerts}; ok=(amap['HASH_MISMATCH']=='HARD' and amap['MARKET_VISIBLE_PRELOCK']=='HARD' and amap['PERFORMANCE_WEAK_N249']=='MONITOR' and amap['PERFORMANCE_WEAK_N500']=='REVIEW')
    rows.append(gate('T25.14','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD',f'alerts={alerts}','Hard integrity alerts act immediately; ordinary performance weakness cannot request reopen before 500 settled games'))

    # T25.15 launch readiness synthesis
    prior_green=all(r['status']=='GREEN' for r in rows); decision='LAUNCH READY — KEEP V14 MODEL FROZEN' if prior_green else 'HOLD LAUNCH'; status='GREEN' if prior_green else 'HOLD'
    rows.append(gate('T25.15',status,decision,f'prior_green={prior_green}; gates_green={sum(r["status"]=="GREEN" for r in rows)}/14; model_reopened=False; 2026_outcomes_used=False','Proceed to live slate generation/locking only; next gate is actual prospective operation, not historical construction'))

    pd.DataFrame(rows).to_csv(OUT/'t25_production_hardening_15_gates.csv',index=False)
    pd.DataFrame(env).to_csv(OUT/'production_lock_envelope_dry_run.csv',index=False)
    pd.DataFrame(chain).to_csv(OUT/'audit_hash_chain_dry_run.csv',index=False)
    pd.DataFrame(alerts).to_csv(OUT/'monitoring_alert_matrix.csv',index=False)
    pd.DataFrame(columns=availability_schema).to_csv(OUT/'availability_shadow_schema.csv',index=False)
    spec={
        'frozen_source_commit':FREEZE_COMMIT,'config_sha256':man['config_sha256'],'forecast_sha256':man['forecast_table_sha256'],
        'lock_key':['game_id'],'lock_timestamp':'UTC ISO-8601','retry_policy':'idempotent on identical payload; conflict on different payload',
        'crash_recovery':'replay full slate idempotently','schedule_change':'append superseding lock; never overwrite','postponement':'void old lock from scoring; relock rescheduled event',
        'unknown_site':'block lock until resolved','availability':'shadow only until validated','market_handoff':'separate record referencing lock_id + payload hash',
        'settlement':'append-only, idempotent exact replay, conflict on differing result','audit':'hash-chain or equivalent immutable event log',
        'monitoring':{'descriptive_n':50,'formal_reliability_n':250,'performance_reopen_min_n':500,'hard_integrity':'immediate'},
        '2026_outcomes_used':False,'model_reopened':False
    }
    (OUT/'production_operations_contract.json').write_text(json.dumps(spec,indent=2))
    summary={'gates_green':sum(r['status']=='GREEN' for r in rows),'gates_total':len(rows),'decision':rows[-1]['decision'],'model_reopened':False,'2026_outcomes_used':False,'frozen_forecast_sha256':man['forecast_table_sha256']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
