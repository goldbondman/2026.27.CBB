#!/usr/bin/env python3
"""One-off pregame 2026-27 NCAA basketball source extraction.
Does not touch production data or credentials. Prints bounded, row-level NDJSON to job logs
and saves validation summaries in artifacts for cross-source reconciliation.
"""
import csv, hashlib, io, json, os, pathlib, sys, time
from datetime import datetime, timezone
import requests
import pyarrow.parquet as pq
import pandas as pd

OUT = pathlib.Path("source_staging_2027")
OUT.mkdir(exist_ok=True)
URLS = {
 "ESPN": "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/team_box/parquet/team_box_2027.parquet",
 "NCAA": "https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/team_box/parquet/ncaa_mbb_team_box_2027.parquet",
}
def report(kind, obj):
 print("NCAAB2027_"+kind+" "+json.dumps(obj,sort_keys=True,separators=(",",":"),default=str),flush=True)
def value(row, key, default=None):
 v=row.get(key,default)
 if v is None or pd.isna(v): return default
 if isinstance(v,(int,float,bool,str)): return v
 return str(v)
def integer(row,key,default=None):
 v=value(row,key)
 try: return int(v)
 except: return default
def get_data(name,url):
 r=requests.get(url,timeout=45,headers={"User-Agent":"ncaab-research-source-QA-2026-27"});r.raise_for_status()
 if len(r.content)<100 or r.content[:4]!=b"PAR1" or r.content[-4:]!=b"PAR1": raise RuntimeError(name+" corrupt Parquet header/footer")
 b=r.content
 t=pq.read_table(io.BytesIO(b))
 data=t.to_pandas()
 report("SOURCE",{"source":name,"url":url,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"rows":len(data),"cols":len(data.columns),"columns":list(data.columns),"as_of_utc":datetime.now(timezone.utc).isoformat()})
 data.to_parquet(OUT/(name.lower()+"_raw.parquet"))
 return data

es=get_data("ESPN",URLS["ESPN"])
nc=get_data("NCAA",URLS["NCAA"])
pairs={}
for rec in es.to_dict("records"):
 game=integer(rec,"game_id")
 team=integer(rec,"team_id")
 opp=integer(rec,"opponent_team_id")
 if game is None or team is None or opp is None: continue
 ha=str(value(rec,"team_home_away","")).lower()
 p={"game_id":game,"season":"2026-27","game_datetime":str(value(rec,"game_date_time",value(rec,"game_date",""))),
 "team_id":team,"team":str(value(rec,"team_display_name","")),"opponent_id":opp,
 "opponent":str(value(rec,"opponent_team_display_name","")),"home_away":ha,
 "points":integer(rec,"team_score"),"opp_points":integer(rec,"opponent_team_score"),
 "fgm":integer(rec,"field_goals_made"),"fga":integer(rec,"field_goals_attempted"),
 "three_pm":integer(rec,"three_point_field_goals_made"),"three_pa":integer(rec,"three_point_field_goals_attempted"),
 "ftm":integer(rec,"free_throws_made"),"fta":integer(rec,"free_throws_attempted"),
 "orb":integer(rec,"offensive_rebounds"),"drb":integer(rec,"defensive_rebounds"),
 "tov":integer(rec,"total_turnovers"),"ast":integer(rec,"assists"),
 "stl":integer(rec,"steals"),"blk":integer(rec,"blocks"),"pf":integer(rec,"fouls")}
 pairs.setdefault(game,[]).append(p)
valid=[];bad=[]
for game,ls in sorted(pairs.items()):
 reasons=[]
 if len(ls)!=2:reasons.append("PAIR_NOT_TWO")
 if len(ls)==2:
  a,b=ls
  if len({a["team_id"],b["team_id"]})!=2:reasons.append("DUP_TEAM")
  if {a["home_away"],b["home_away"]}!={"home","away"}:reasons.append("HOME_AWAY")
  if a["opponent_id"]!=b["team_id"] or b["opponent_id"]!=a["team_id"]:reasons.append("OPP_ID")
  if a["points"]!=b["opp_points"] or b["points"]!=a["opp_points"]:reasons.append("SCORE_MIRROR")
  for t in ls:
   keys=["points","fgm","fga","three_pm","three_pa","ftm","fta","orb","drb","tov"]
   if any(t[k] is None for k in keys):reasons.append("MISSING_BOX")
   elif not (t["fgm"]<=t["fga"] and t["three_pm"]<=t["fgm"] and t["three_pa"]<=t["fga"] and t["ftm"]<=t["fta"]):reasons.append("BOX_IMPOSSIBLE")
   elif 2*t["fgm"]+t["three_pm"]+t["ftm"]!=t["points"]:reasons.append("POINT_ARITHMETIC")
 if reasons:bad.append({"game":game,"reasons":sorted(set(reasons))})
 else:valid.extend(ls)
for row in valid:
 report("TEAM",row)
report("PAIR_SUMMARY",{"espn_source_rows":len(es),"espn_game_ids":len(pairs),"valid_pairs":len(valid)//2,"valid_team_rows":len(valid),"quarantined_game_ids":len(bad),"quarantined_examples":bad[:30],"ncaa_source_rows":len(nc),"ncaa_schema":list(nc.columns)})
with open(OUT/"valid_espn_2027.csv","w",newline="") as f:
 if valid:
  w=csv.DictWriter(f,fieldnames=list(valid[0]));w.writeheader();w.writerows(valid)
with open(OUT/"qa.json","w") as f:json.dump({"valid_games":len(valid)//2,"invalid_games":bad,"source_espn_rows":len(es),"source_ncaa_rows":len(nc)},f,indent=2)

# Second-source team-score and identity reconciliation. Keep ESPN and NCAA namespaces distinct.
ndf=nc.to_dict("records")
eidx={(v["game_id"],v["team_id"]):v for v in valid}
reconciled=[];unmatched=[];score_conflict=[];nc_seen=set()
for rec in ndf:
 gid=integer(rec,"espn_game_id");tid=integer(rec,"team_espn_team_id")
 if gid is None or tid is None:continue
 k=(gid,tid)
 if k in nc_seen:
  unmatched.append({"game":gid,"team":tid,"why":"NCAA_DUPLICATE"});continue
 nc_seen.add(k)
 row=eidx.get(k)
 pts=integer(rec,"pts");opp=integer(rec,"d_pts")
 if row is None:
  unmatched.append({"game":gid,"team":tid,"why":"NOT_IN_ESPN_VALIDATED"});continue
 if row["points"]!=pts or row["opp_points"]!=opp:
  score_conflict.append({"game":gid,"team":tid,"espn_pts":row["points"],"ncaa_pts":pts,"espn_opp":row["opp_points"],"ncaa_opp":opp});continue
 reconciled.append({"game_id":gid,"team_id":tid,"pts":pts,"opp_pts":opp,"home_espn_team_id":integer(rec,"home_espn_team_id"),"away_espn_team_id":integer(rec,"away_espn_team_id"),"ncaa_contest_id":integer(rec,"contest_id"),"ncaa_team_id":integer(rec,"team_ncaa_team_id")})
for row in reconciled:report("NCAA_MATCH",row)
groups={}
for row in reconciled: groups.setdefault(row["game_id"],set()).add(row["team_id"])
full={gid for gid,teams in groups.items() if len(teams)==2}
report("CROSS_SOURCE",{"ncaa_rows":len(nc),"ncaa_with_valid_espn_match":len(reconciled),"cross_source_two_team_games":len(full),"cross_source_two_team_rows":2*len(full),"unmatched_ncaa":len(unmatched),"score_conflicts":len(score_conflict),"unmatched_examples":unmatched[:15],"conflict_examples":score_conflict[:15],"verified_game_ids":sorted(full)})
with open(OUT/"cross_source_match.csv","w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(reconciled[0]) if reconciled else ["game_id"]);w.writeheader();w.writerows(reconciled)

report("NCAA_DEBUG",{"sample_rows":nc.head(3)[["contest_id","espn_game_id","team_espn_team_id","home","away","team","pts","d_pts","team_ncaa_team_id"]].astype(str).to_dict("records"),"nonnull_espn_game_id":int(nc["espn_game_id"].notna().sum()),"nonnull_espn_team_id":int(nc["team_espn_team_id"].notna().sum()),"contest_id_dtype":str(nc["contest_id"].dtype),"espn_game_id_dtype":str(nc["espn_game_id"].dtype)})

# Pair NCAA records by native contest ID, then match ESPN via *both* ESPN team IDs and score identity.
# NCAA does not publish usable espn_game_id in this source version.
ngroups={}
for record in ndf:
 k=str(value(record,"contest_id",""))
 ngroups.setdefault(k,[]).append(record)
npairs=[];nbad=[]
for contest,records in ngroups.items():
 if len(records)!=2:
  nbad.append({"contest_id":contest,"reason":"NOT_TWO_ROWS"});continue
 home=str(value(records[0],"home",""));away=str(value(records[0],"away",""))
 nh=next((q for q in records if str(value(q,"team",""))==home),None)
 na=next((q for q in records if str(value(q,"team",""))==away),None)
 if nh is None or na is None:
  nbad.append({"contest_id":contest,"reason":"MISSING_HOME_AWAY_MATCH"});continue
 hp=integer(nh,"pts");ap=integer(na,"pts")
 if hp is None or ap is None or integer(nh,"d_pts")!=ap or integer(na,"d_pts")!=hp:
  nbad.append({"contest_id":contest,"reason":"SCORE_MIRROR"});continue
 h=integer(nh,"team_espn_team_id");a=integer(na,"team_espn_team_id")
 q={"contest_id":contest,"home":home,"away":away,"home_espn_id":h,"away_espn_id":a,"home_points":hp,"away_points":ap}
 npairs.append(q);report("NCAA_PAIR",q)
report("NCAA_PAIR_SUMMARY",{"games":len(npairs),"both_espn_ids":sum(1 for t in npairs if t["home_espn_id"] is not None and t["away_espn_id"] is not None),"missing_ids":sum(1 for t in npairs if t["home_espn_id"] is None or t["away_espn_id"] is None),"quarantine":nbad[:15],"total_ncaacontest_ids":len(ngroups)})

# Independent ESPN scoreboard status check for already completed source game dates.
# The date set comes from imported box rows; this does not request upcoming future Nov6 games.
status_index={}
dates=sorted({str(d)[:10].replace("-","") for d in es["game_date"].tolist() if str(d)[:4]=="2026"})
for day in dates:
 try:
  endpoint="https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
  resp=requests.get(endpoint,params={"dates":day,"limit":500,"groups":50},timeout=25,headers={"User-Agent":"NCAABResearch2026QA/1.0"})
  resp.raise_for_status()
  events=resp.json().get("events",[])
  report("SCOREBOARD_BATCH",{"date":day,"events":len(events),"status_code":resp.status_code})
  for event in events:
   gid=int(event["id"]);comp=event.get("competitions",[])
   if not comp:continue
   first=comp[0];participants=first.get("competitors",[]);observed={}
   for t in participants:
    try:
     tid=int(t.get("team",{}).get("id"))
     observed[tid]={"points":int(t.get("score")),"homeAway":t.get("homeAway")}
    except Exception:pass
   stat=first.get("status",event.get("status",{}));period=stat.get("period")
   completed=stat.get("type",{}).get("completed")
   venue=first.get("venue",{})
   status_index[gid]={"game_id":gid,"date":day,"completed":completed,"period":period,"neutral_site":first.get("neutralSite"),"venue_id":venue.get("id"),"scoreboard":observed}
 except Exception as exc: report("SCOREBOARD_ERROR",{"date":day,"error":str(exc)[:200]})
official_valid=0;official_conflict=0
for gid,pair in pairs.items():
 s=status_index.get(gid)
 if s is None:continue
 good=(s["completed"] is True and len(pair)==2 and
       all(t["team_id"] in s["scoreboard"] and
           t["points"]==s["scoreboard"][t["team_id"]]["points"] and
           t["home_away"]==s["scoreboard"][t["team_id"]]["homeAway"] for t in pair))
 if good:official_valid+=1
 else:official_conflict+=1
 report("OFFICIAL_STATUS",{"game_id":gid,"score_valid":good,"completed":s["completed"],"period":s["period"],"neutral_site":s["neutral_site"],"venue_id":s["venue_id"]})
report("OFFICIAL_SUMMARY",{"dates":dates,"events_found":len(status_index),"paired_game_ids":len(pairs),"official_espn_final_score_verified":official_valid,"official_conflict_or_incomplete":official_conflict,"no_scoreboard_event":len(pairs)-official_valid-official_conflict})
