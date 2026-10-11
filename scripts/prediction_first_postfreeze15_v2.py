from pathlib import Path

p=Path(__file__).with_name('prediction_first_postfreeze15.py')
s=p.read_text()
s=s.replace("pred_path=find_file('frozen_forecast_snapshot.parquet','forecast_snapshot.parquet','corrected_game_predictions.parquet','champion_predictions.parquet')", "pred_path=find_file('frozen_candidate_predictions.parquet','frozen_forecast_snapshot.parquet','forecast_snapshot.parquet','corrected_game_predictions.parquet','champion_predictions.parquet')")
s=s.replace("req_score=['game_id','lock_timestamp','settlement_timestamp','actual_home','actual_away']", "req_score=['game_id','forecast_lock_timestamp','settlement_timestamp','actual_home','actual_away']")
s=s.replace("sep=all(c in score.columns for c in ['lock_timestamp','settlement_timestamp'])", "sep=all(c in score.columns for c in ['forecast_lock_timestamp','settlement_timestamp'])")
exec(compile(s,str(p),'exec'))
