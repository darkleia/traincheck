"""Tests for shared parsing helpers in traincheck.utils."""

from traincheck.utils import parse_gdr_level, parse_pinned_version


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


def test_parse_pinned_version_reads_an_exact_pin():
    assert parse_pinned_version("==1.13.0") == (1, 13, 0)


def test_parse_pinned_version_reads_a_bare_lockfile_version():
    # uv.lock/poetry.lock/Pipfile.lock store an already-resolved version
    # with no "==" prefix at all.
    assert parse_pinned_version("0.18.5") == (0, 18, 5)


def test_parse_pinned_version_strips_a_local_version_segment():
    assert parse_pinned_version("2.3.0+cu121") == (2, 3, 0)


def test_parse_pinned_version_refuses_to_guess_inside_a_range():
    assert parse_pinned_version(">=1.13.0,<1.14.0") is None
    assert parse_pinned_version("~=1.13.0") is None
    assert parse_pinned_version(">1.0") is None


def test_parse_pinned_version_none_and_empty_stay_none():
    assert parse_pinned_version(None) is None
    assert parse_pinned_version("") is None
