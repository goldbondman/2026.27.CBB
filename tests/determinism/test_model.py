from cbb.ratings.recursive import RecursiveOD
from tests.leakage.test_recursive import rows


def test_reverse_input_is_deterministic():
    a = RecursiveOD().run(rows())
    b = RecursiveOD().run(rows().iloc[::-1])
    assert a.to_dict("records") == b.to_dict("records")

