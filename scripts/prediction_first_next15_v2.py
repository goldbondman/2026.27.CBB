from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson, binom, multivariate_normal

src = Path('scripts/prediction_first_next15.py').read_text()
src = src.replace("['turnovers','total_turnovers','totalTurnovers','tov']", "['to','turnovers','total_turnovers','totalTurnovers','tov']")
src = src.replace("['three_point_field_goals_attempted','three_point_attempts','three_point_field_goal_attempts','threepa','3pa']", "['tpa','three_point_field_goals_attempted','three_point_attempts','three_point_field_goal_attempts','threepa','3pa']")
src = src.replace("['three_point_field_goals_made','three_point_made','three_point_field_goal_makes','threepm','3pm']", "['tpm','three_point_field_goals_made','three_point_made','three_point_field_goal_makes','threepm','3pm']")
src = src.replace("vals=q.sort_values('season').corr.values", "vals=q.sort_values('season')['corr'].values")
g={'__name__':'__main__'}
exec(compile(src, 'prediction_first_next15_v2_runtime.py', 'exec'), g)

OUT=Path('results/prediction_first_next15')
f=pd.read_parquet(OUT/'component_features.parquet')

def pnll(y,mu):
    y=np.rint(np.asarray(y,float)).astype(int); mu=np.clip(np.asarray(mu,float),1e-6,None)
    return float(-np.mean(poisson.logpmf(y,mu)))
def bnll(k,n,p):
    k=np.rint(np.asarray(k,float)).astype(int); n=np.maximum(np.rint(np.asarray(n,float)).astype(int),0); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); ok=n>=k
    return float(-np.mean(binom.logpmf(k[ok],n[ok],p[ok])))
rows=[]
for sy in [2023,2024,2025]:
    tr=f[(f.season>=2022)&(f.season<sy)]; te=f[f.season==sy]
    gt=tr.tov.sum()/tr.gposs.sum(); a=pnll(te.tov,gt*te.pred_gposs); b=pnll(te.tov,te.pred_tov_rate*te.pred_gposs); rows.append([sy,'TOV_mean',a,b,a-b])
    gf=tr.fta.sum()/tr.gposs.sum(); a=pnll(te.fta,gf*te.pred_gposs); b=pnll(te.fta,te.pred_fta_rate*te.pred_gposs); rows.append([sy,'FTA_mean',a,b,a-b])
    go=tr.orb.sum()/tr['miss'].sum(); a=bnll(te.orb,te['miss'],np.repeat(go,len(te))); b=bnll(te.orb,te['miss'],te.pred_orb_rate); rows.append([sy,'ORB_rate',a,b,a-b])
    ga=tr.threepa.sum()/tr.fga.sum(); a=bnll(te.threepa,te.fga,np.repeat(ga,len(te))); b=bnll(te.threepa,te.fga,te.pred_threepa_rate); rows.append([sy,'3PA_rate',a,b,a-b])
    gp=tr.threepm.sum()/tr.threepa.sum(); a=bnll(te.threepm,te.threepa,np.repeat(gp,len(te))); b=bnll(te.threepm,te.threepa,te.pred_threep_pct); rows.append([sy,'3PM_pct',a,b,a-b])
pd.DataFrame(rows,columns=['season','component','global_nll','teamopp_nll','gain']).to_csv(OUT/'event_mean_vs_global.csv',index=False)

def rawres(df):
    return pd.DataFrame({'tov':df.tov-df.pred_tov_rate*df.pred_gposs,'fta':df.fta-df.pred_fta_rate*df.pred_gposs,'orb':df.orb-df.pred_orb_rate*df['miss'],'threepa':df.threepa-df.pred_threepa_rate*df.fga,'threepm':df.threepm-df.pred_threep_pct*df.threepa})
deps=[]
for sy in [2023,2024,2025]:
    tr=f[(f.season>=2022)&(f.season<sy)]; te=f[f.season==sy]; rt=rawres(tr); re=rawres(te); sd=rt.std(ddof=1).replace(0,np.nan); ztr=(rt/sd).dropna(); zte=(re/sd).dropna(); cov=ztr.cov().to_numpy()+np.eye(5)*1e-6
    ind=-multivariate_normal(mean=np.zeros(5),cov=np.eye(5)).logpdf(zte.to_numpy()).mean(); corr=-multivariate_normal(mean=np.zeros(5),cov=cov,allow_singular=True).logpdf(zte.to_numpy()).mean(); deps.append([sy,ind,corr,ind-corr])
pd.DataFrame(deps,columns=['season','independent_nll','correlated_nll','gain']).to_csv(OUT/'dependence_joint_nll.csv',index=False)
