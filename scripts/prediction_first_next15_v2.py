from pathlib import Path

src = Path('scripts/prediction_first_next15.py').read_text()
src = src.replace("['turnovers','total_turnovers','totalTurnovers','tov']", "['to','turnovers','total_turnovers','totalTurnovers','tov']")
src = src.replace("['three_point_field_goals_attempted','three_point_attempts','three_point_field_goal_attempts','threepa','3pa']", "['tpa','three_point_field_goals_attempted','three_point_attempts','three_point_field_goal_attempts','threepa','3pa']")
src = src.replace("['three_point_field_goals_made','three_point_made','three_point_field_goal_makes','threepm','3pm']", "['tpm','three_point_field_goals_made','three_point_made','three_point_field_goal_makes','threepm','3pm']")
exec(compile(src, 'prediction_first_next15_v2_runtime.py', 'exec'), {'__name__':'__main__'})
