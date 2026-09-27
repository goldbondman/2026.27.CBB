import pandas as pd
from cbb.canonical.boxscore import compile_team_games


def test_turnovers_are_not_double_counted():
    row = pd.DataFrame([{"fga": 60, "orb": 10, "total_turnovers": 12, "team_turnovers": 3, "fta": 20, "points": 75}])
    result = compile_team_games(row).iloc[0]
    assert result.tov == 12
    assert result.possessions == 70.8

