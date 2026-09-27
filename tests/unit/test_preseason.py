from cbb.ratings.recursive import RecursiveOD


def test_preseason_uses_only_explicit_prior_inputs():
    model = RecursiveOD()
    model.initialize_season({"A": (10.0, -5.0)}, {"A": 0.375}, {"A": 0.0})
    assert model.offense["A"] == 12.0
    assert model.defense["A"] == -6.0

