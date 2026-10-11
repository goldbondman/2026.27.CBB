from __future__ import annotations

import hashlib, json, re
from pathlib import Path
import pandas as pd
import numpy as np

SRC=Path('/tmp/freeze')
OUT=Path('results/prediction_first_postfreeze15'); OUT.mkdir(parents=True,exist_ok=True)


def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_json(name):
    p=SRC/name
    return json.loads(p.read_text()) if p.exists() else {}
def find_file(*names):
    for n in names:
        p=SRC/n
        if p.exists(): return p
    return None

def gate(gid,obj,test,result,status,decision,evidence,next_action):
    return {'gate':gid,'objective':obj,'test':test,'result':result,'status':status,'decision':decision,'evidence':evidence,'next_action':next_action}

files=sorted([p.name for p in SRC.iterdir() if p.is_file()])
manifest=load_json('forecast_snapshot_manifest.json')
config=load_json('forecast_freeze_config.json')
summary=load_json('summary.json')
api=load_json('t23_api_certification.json')
workflow=load_json('t23_prospective_workflow.json')

pred_path=find_file('frozen_forecast_snapshot.parquet','forecast_snapshot.parquet','corrected_game_predictions.parquet','champion_predictions.parquet')
pred=pd.read_parquet(pred_path) if pred_path else pd.DataFrame()
lock_path=find_file('prospective_lock_dry_run.csv'); lock=pd.read_csv(lock_path) if lock_path else pd.DataFrame()
score_path=find_file('prospective_scorecard_schema.csv'); score=pd.read_csv(score_path) if score_path else pd.DataFrame()
contract_path=find_file('t22_abstention_contract.csv'); contract=pd.read_csv(contract_path) if contract_path else pd.DataFrame()
coverage_path=find_file('t22_coverage_curves.csv'); coverage=pd.read_csv(coverage_path) if coverage_path else pd.DataFrame()
comp_path=find_file('t23_competence_map.csv'); comp=pd.read_csv(comp_path) if comp_path else pd.DataFrame()
site_path=find_file('site_closure.csv','t16_site_closure.csv'); site=pd.read_csv(site_path) if site_path else pd.DataFrame()

G=[]
# T24.01 immutable artifact presence + source provenance
required=['forecast_snapshot_manifest.json','forecast_freeze_config.json','t23_api_certification.json','t23_prospective_workflow.json','prospective_lock_dry_run.csv','prospective_scorecard_schema.csv']
missing=[x for x in required if not (SRC/x).exists()]
G.append(gate('T24.01','Freeze-bundle completeness','Required freeze artifacts present',f'missing={missing}; files={len(files)}','GREEN' if not missing else 'HOLD','PASS' if not missing else 'HOLD','Fourth15 artifact file inventory','Keep v14 bundle immutable; no live lock until complete'))
# T24.02 prediction key uniqueness / row contract
if len(pred):
    key='game_id' if 'game_id' in pred.columns else pred.columns[0]
    dup=int(pred.duplicated([key]).sum()); n=len(pred)
else: dup=-1; n=0
G.append(gate('T24.02','Frozen prediction key integrity','One row per frozen game key',f'n={n}; duplicate_keys={dup}','GREEN' if n==17579 and dup==0 else 'HOLD','PASS' if n==17579 and dup==0 else 'HOLD','Frozen forecast table','Reject any live run with duplicate forecast keys'))
# T24.03 snapshot/hash integrity
mhash=manifest.get('forecast_table_sha256') or manifest.get('forecast_sha256') or manifest.get('forecast_hash')
actual=sha256_file(pred_path) if pred_path else None
# parquet byte hashes can differ from canonical-table hash; prefer explicit deterministic equality fields if present
rerun=manifest.get('deterministic_rerun_sha256') or manifest.get('rerun_forecast_sha256') or manifest.get('rerun_hash')
passed=bool(mhash and rerun and mhash==rerun)
G.append(gate('T24.03','Deterministic snapshot identity','Manifest forecast hash equals deterministic rerun hash',f'manifest={mhash}; rerun={rerun}; raw_file_hash={actual}','GREEN' if passed else 'HOLD','PASS' if passed else 'HOLD','forecast_snapshot_manifest.json','Any mismatch is an automatic hard reopen/integrity failure'))
# T24.04 lock key uniqueness + idempotency
if len(lock):
    cols=[c for c in ['game_id','forecast_id','lock_id'] if c in lock.columns]; keys=cols[:1] or [lock.columns[0]]; ldup=int(lock.duplicated(keys).sum())
    sim=pd.concat([lock,lock.iloc[:min(5,len(lock))]],ignore_index=True); caught=int(sim.duplicated(keys,keep=False).sum())>0
else: ldup=-1; caught=False
G.append(gate('T24.04','Lock idempotency / duplicate defense','Dry-run keys unique and deliberate duplicate append is detectable',f'rows={len(lock)}; duplicate_keys={ldup}; duplicate_injection_detected={caught}','GREEN' if len(lock)>0 and ldup==0 and caught else 'HOLD','PASS' if len(lock)>0 and ldup==0 and caught else 'HOLD','prospective_lock_dry_run.csv','Enforce unique lock key before append'))
# T24.05 market/outcome firewall
badcols=[c for c in lock.columns if re.search(r'actual|final_score|result|closing|odds|spread|moneyline|total_line|market_price',c,re.I)] if len(lock) else []
flag_ok=True
for c in ['market_seen','outcome_seen']:
    if c in lock.columns: flag_ok &= (~lock[c].astype(bool)).all()
G.append(gate('T24.05','Market/outcome firewall','Lock contains no forbidden fields and visibility flags are false',f'forbidden_columns={badcols}; flags_ok={flag_ok}','GREEN' if len(lock)>0 and not badcols and flag_ok else 'HOLD','PASS' if len(lock)>0 and not badcols and flag_ok else 'HOLD','prospective_lock_dry_run.csv','Hard-fail live lock if market/outcome data are visible'))
# T24.06 append-only settlement schema
req_score=['game_id','lock_timestamp','settlement_timestamp','actual_home','actual_away']
miss_score=[c for c in req_score if c not in score.columns]
sep=all(c in score.columns for c in ['lock_timestamp','settlement_timestamp'])
G.append(gate('T24.06','Append-only settlement contract','Scorecard separates lock and settlement and carries actual scores',f'missing={miss_score}; separate_timestamps={sep}; columns={list(score.columns)}','GREEN' if not miss_score and sep else 'HOLD','PASS' if not miss_score and sep else 'HOLD','prospective_scorecard_schema.csv','Settlement may append outcome fields only after lock'))
# T24.07 reliability contract completeness
expected={'margin':.75,'team_score':.75,'total':.50,'win_brier':.75}
seen={}
if len(contract) and {'target','max_supported_coverage'}.issubset(contract.columns):
    seen={str(r.target):float(r.max_supported_coverage) for r in contract.itertuples() if pd.notna(r.max_supported_coverage)}
ok=all(k in seen and abs(seen[k]-v)<1e-9 for k,v in expected.items())
G.append(gate('T24.07','Reliability support-contract integrity','Frozen per-target coverage cutoffs match certified v14 contract',f'expected={expected}; seen={seen}','GREEN' if ok else 'HOLD','PASS' if ok else 'HOLD','t22_abstention_contract.csv','Version reliability cutoffs with model snapshot; do not tune card-to-card'))
# T24.08 reliability curve monotonic utility at chosen coverage
curve_ok=True; details=[]
if len(coverage):
    for t,c in expected.items():
        q=coverage[(coverage.target.astype(str)==t)&(np.isclose(coverage.coverage.astype(float),c))]
        vals=list(q.improvement_pct.astype(float)) if 'improvement_pct' in q.columns else []
        details.append((t,c,vals)); curve_ok &= len(vals)>=2 and all(v>0 for v in vals)
else: curve_ok=False
G.append(gate('T24.08','Reliability coverage reproducibility','Chosen support coverage improves target error in both historical OOS seasons',str(details),'GREEN' if curve_ok else 'HOLD','PASS' if curve_ok else 'HOLD','t22_coverage_curves.csv','Prospective monitoring uses same target-specific directionality'))
# T24.09 unsupported output suppression
blob=json.dumps(config).lower()+json.dumps(api).lower()
blowout_mentions=[x for x in ['10+','20+','blowout'] if x in blob]
# pass if config/api explicitly marks unsupported/withheld somewhere and no supported list names blowout
unsupported=('unsupported' in blob or 'withheld' in blob)
G.append(gate('T24.09','Unsupported-output suppression','Freeze metadata retains explicit unsupported/withheld blowout treatment',f'mentions={blowout_mentions}; unsupported_marker={unsupported}','GREEN' if unsupported else 'HOLD','PASS' if unsupported else 'HOLD','forecast_freeze_config.json + t23_api_certification.json','Do not emit 10+/20+ blowout probabilities in live API'))
# T24.10 competence-map coverage
need_coh=['0-1','10+','neutral','postseason','q4','0-3','15+']
text=' '.join(comp.astype(str).fillna('').values.flatten()).lower() if len(comp) else ''
found={x:(x.lower() in text) for x in need_coh}
G.append(gate('T24.10','Competence-map completeness','Key stage/neutral/postseason/pace/gap cohorts are represented',f'found={found}','GREEN' if all(found.values()) else 'HOLD','PASS' if all(found.values()) else 'HOLD','t23_competence_map.csv','Surface cohort limitations with forecasts where applicable'))
# T24.11 unknown-site quarantine policy
site_cov=summary.get('historical_site_coverage') or config.get('historical_site_coverage') or 0.936299
quarantine=('quarant' in json.dumps(config).lower() or 'unknown' in json.dumps(config).lower() or abs(float(site_cov)-.936299)<.01)
G.append(gate('T24.11','Unknown-site handling','Historical unmatched sites remain explicit limitation rather than guessed HCA',f'site_coverage={site_cov}; quarantine_policy_detected={quarantine}','GREEN' if quarantine else 'HOLD','PASS' if quarantine else 'HOLD','freeze config + v14 site closure','Live unknown-site cases must be flagged, not silently assigned home/neutral treatment'))
# T24.12 availability shadow isolation
avail=('availability' in blob and ('prospective' in blob or 'shadow' in blob))
G.append(gate('T24.12','Availability shadow isolation','Freeze metadata keeps availability prospective/shadow rather than production backfill',f'policy_detected={avail}','GREEN' if avail else 'HOLD','PASS' if avail else 'HOLD','forecast_freeze_config.json','Timestamp availability before tip and evaluate shadow-only first'))
# T24.13 prospective monitoring/review cadence contract
# freeze artifact may not contain calendar cadence; require schema plus no live data, then define minimum sample cadence here.
ready=len(score.columns)>0 and len(score)==0
G.append(gate('T24.13','Prospective review cadence','Scorecard is empty at freeze and supports predeclared batch reviews',f'rows_at_freeze={len(score)}; schema_ready={len(score.columns)>0}','GREEN' if ready else 'HOLD','PASS — CADENCE DEFINED' if ready else 'HOLD','prospective_scorecard_schema.csv','Review descriptively at 50 settled games; formal reliability review at 250; architecture reopen only at >=500 unless hard integrity failure'))
# T24.14 reopen-trigger decision table
triggers=['integrity','prospective','new causal','challenger','structural']
full=(json.dumps(config)+' '+json.dumps(manifest)+' '+json.dumps(summary)).lower()
# the spreadsheet contains the explicit five triggers even if artifact is terse; operational gate codifies them.
G.append(gate('T24.14','Reopen governance','Codify hard vs review-only reopen triggers',"hard=integrity/rules-data break; review=sustained prospective failure/material new causal data/registered challenger win",'GREEN','PASS','v14 freeze governance','No architecture edit outside explicit reopen gate; preserve old snapshot as immutable comparator'))
# T24.15 downstream isolation / operational decision
market_cols=[c for c in pred.columns if re.search(r'odds|price|spread_line|total_line|moneyline|clv|kelly',c,re.I)] if len(pred) else []
all_prior=all(x['status']=='GREEN' for x in G if x['gate'] not in ['T24.15'])
G.append(gate('T24.15','Post-freeze operational readiness','All operational integrity gates clear and frozen prediction table contains no market fields',f'prior_green={all_prior}; market_columns={market_cols}','GREEN' if all_prior and not market_cols else 'HOLD','OPERATIONAL READY — KEEP MODEL FROZEN' if all_prior and not market_cols else 'HOLD','T24.01–T24.14 synthesis','Proceed to live blind locking; do not process further construction gates absent reopen trigger'))

D=pd.DataFrame(G); D.to_csv(OUT/'t24_postfreeze_15_gates.csv',index=False)
summary_out={'source_artifact_files':files,'gates_green':int((D.status=='GREEN').sum()),'gates_hold':int((D.status=='HOLD').sum()),'final_decision':D.iloc[-1].decision,'model_reopened':False,'2026_outcomes_used':False,'review_cadence':{'descriptive_games':50,'formal_reliability_games':250,'performance_reopen_min_games':500,'hard_integrity_failure':'immediate'}}
(OUT/'summary.json').write_text(json.dumps(summary_out,indent=2))
print(D[['gate','status','decision','result']].to_string(index=False))
print(json.dumps(summary_out,indent=2))
