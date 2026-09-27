import pytest
from cbb.governance.sealed import SealedHoldoutAccessError, assert_unsealed


def test_2026_is_sealed():
    with pytest.raises(SealedHoldoutAccessError, match="SEALED_HOLDOUT_ACCESS_ERROR"):
        assert_unsealed([2025, 2026])

