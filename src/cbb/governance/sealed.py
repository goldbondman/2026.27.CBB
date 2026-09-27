"""Sealed-holdout controls shared by all development entry points."""

SEALED_SEASON = 2026


class SealedHoldoutAccessError(RuntimeError):
    """Raised when ordinary development code attempts to inspect the holdout."""


def assert_unsealed(seasons: object) -> None:
    values = {int(value) for value in seasons}
    if SEALED_SEASON in values:
        raise SealedHoldoutAccessError("SEALED_HOLDOUT_ACCESS_ERROR: season 2026 is protected")

