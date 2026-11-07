import pandas as pd,numpy as np,json,hashlib,pathlib
root=pathlib.Path('/tmp/cbb/parallel_hypothesis_lab')
a=pd.read_csv(root/'favorite_game_level.csv')
b=pd.read_csv(root/'factors_game_level.csv')
c=pd.read_csv(root/'pace_game_level.csv')
assert all(df.game_id.is_unique for df in [a,b,c])
x=a.merge(b,on=['game_id','game_date','home_id','away_id'],suffixes=('_fav','_fac'),validate='one_to_one')
x=x.merge(c,on=['game_id','game_date','home_id','away_id'],validate='one_to_one')
assert len(x)==597,len(x)
assert (x.actual_margin_fav==x.actual_margin_fac).all()
assert (x.actual_total_fav==x.actual_total_fac).all()
assert (x.actual_total_fav==x.actual_total).all()
margin={'FAV_original':x.ORIGINAL_margin,'FAV_weakdef':x.WEAK_DEF_ONLY_margin,
 'FAV_offdef':x.OFF_DEF_JOINT_margin,'FACT_F0':x.F0_BASE_margin,
 'FACT_fouls':x.OPP_FOULS_margin}
total={'FAV_original':x.ORIGINAL_total,'FAV_weakdef':x.WEAK_DEF_ONLY_total,
 'PACE_original':x.EQUAL_MEAN_total,'PACE_adj':x.OPP_ADJUSTED_total,
 'FACT_F0':x.F0_BASE_total,'FACT_TOV':x.FOUR_TOV_total}
def summarize(models,ycol,target):
    targetvals=x[ycol].to_numpy()
    pp={k:np.asarray(z,dtype=float) for k,z in models.items()}
    out={'individual_mae':{k:float(np.mean(abs(v-targetvals))) for k,v in pp.items()},
         'pairwise_error_corr':pd.DataFrame({k:v-targetvals for k,v in pp.items()}).corr().round(4).to_dict()}
    pairs={'weakdef_factor_base':('FAV_weakdef','FACT_F0')} if target=='margin' else {
           'pace_factor_base':('PACE_adj','FACT_F0'),
           'weakdef_factor_base':('FAV_weakdef','FACT_F0')}
    combos=[]
    for label,(left,right) in pairs.items():
        pred=(pp[left]+pp[right])/2
        combos.append(dict(pair=label,mean_absolute_error=float(np.mean(abs(pred-targetvals))),
            left_mae=out['individual_mae'][left],right_mae=out['individual_mae'][right],
            error_corr=float(np.corrcoef(pp[left]-targetvals,pp[right]-targetvals)[0,1]),
            gain_vs_best_posthoc=float(min(out['individual_mae'][left],out['individual_mae'][right])-np.mean(abs(pred-targetvals)))))
    out['equal_blend']=combos
    return out
result={'classification':'REUSED_HISTORICAL_2024_25_RESEARCH_ONLY','matched_games':len(x),
        'margin':summarize(margin,'actual_margin_fav','margin'),
        'total':summarize(total,'actual_total_fav','total'),
        'source_csv_hashes':{key:hashlib.sha256((root/(key+'_game_level.csv')).read_bytes()).hexdigest() for key in ['favorite','factors','pace']},
        'limitations':'Ensemble comparisons exploratory; same structural baseline/inputs, post-hoc best individual model is not valid future-selection rule.'}
(root/'complementarity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'n':result['matched_games'],'margin':{'models':result['margin']['individual_mae'],'blend':result['margin']['equal_blend']},'total':{'models':result['total']['individual_mae'],'blend':result['total']['equal_blend']}},separators=(',',':')))

