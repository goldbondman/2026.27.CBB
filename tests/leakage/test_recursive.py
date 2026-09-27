import pandas as pd
from cbb.ratings.recursive import RecursiveOD


def rows():
    return pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "game_id": "a",
                "team_id": "A",
                "opponent_id": "B",
                "ortg": 110.0,
                "possessions": 70,
            },
            {
                "date": "2025-01-01",
                "game_id": "a",
                "team_id": "B",
                "opponent_id": "A",
                "ortg": 90.0,
                "possessions": 70,
            },
            {
                "date": "2025-01-01",
                "game_id": "b",
                "team_id": "A",
                "opponent_id": "C",
                "ortg": 105.0,
                "possessions": 70,
            },
            {
                "date": "2025-01-02",
                "game_id": "c",
                "team_id": "A",
                "opponent_id": "B",
                "ortg": 100.0,
                "possessions": 70,
            },
        ]
    )


def test_current_and_same_day_mutation_cannot_change_prediction():
    base = RecursiveOD().run(rows())
    changed = rows()
    changed.loc[0, "ortg"] = 999
    other = RecursiveOD().run(changed)
    assert base.loc[:2, "prediction"].tolist() == other.loc[:2, "prediction"].tolist()


def test_past_changes_future_and_future_does_not_change_past():
    base = RecursiveOD().run(rows())
    past = rows()
    past.loc[0, "ortg"] = 130
    assert base.iloc[3].prediction != RecursiveOD().run(past).iloc[3].prediction
    future = rows()
    future.loc[3, "ortg"] = 999
    assert (
        base.iloc[:3].prediction.tolist() == RecursiveOD().run(future).iloc[:3].prediction.tolist()
    )
