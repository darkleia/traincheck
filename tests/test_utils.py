"""Tests for shared parsing helpers in traincheck.utils."""

from traincheck.utils import parse_gdr_level


def test_parse_gdr_level_accepts_named_levels():
    for name in ("LOC", "PIX", "PXB", "PHB", "SYS"):
        assert parse_gdr_level(name) == name


def test_parse_gdr_level_is_case_insensitive_and_normalizes_to_upper():
    assert parse_gdr_level("pxb") == "PXB"


def test_parse_gdr_level_accepts_numeric_values():
    assert parse_gdr_level("5") == 5
    assert parse_gdr_level("0") == 0


def test_parse_gdr_level_does_not_renormalize_numeric_meaning():
    """The numeric meaning of a given level shifted across NCCL versions
    (SYS was 4 before 2.4.7, 5 after) - parse_gdr_level must return
    exactly what was given, not remap it to a canonical scheme.
    """
    assert parse_gdr_level("4") == 4
    assert parse_gdr_level("5") == 5


def test_parse_gdr_level_none_stays_none():
    assert parse_gdr_level(None) is None


def test_parse_gdr_level_garbage_returns_none():
    assert parse_gdr_level("not-a-level") is None
