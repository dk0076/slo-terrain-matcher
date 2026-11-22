"""Tests for the pleasantness score.

The score is a judgement call, not a measurement, so these tests pin the
*orderings* that matter rather than exact numbers. If a tweak makes a
four-lane arterial rank above a named park path, that is a regression no
matter what the absolute values are.
"""

from __future__ import annotations

import pytest

from conftest import make_edge
from pleasant import BARRED, BASE, _parse_maxspeed_mph, score_edge, score_network


def _edge(**kwargs):
    """An edge with pleasantness-relevant tags."""
    tags = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key.endswith("_tag")
    }
    edge = make_edge(**kwargs)
    for key, value in tags.items():
        object.__setattr__(edge, key, value) if False else setattr(edge, key, value)
    return edge


def _score(**kwargs) -> float:
    return score_edge(_edge(**kwargs))[0]


# ---------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------


def test_a_named_path_beats_an_arterial():
    """The whole point: steer off main roads and onto paths."""
    path = _score(highway="path", name="Bob Jones Trail", foot_tag="designated")
    arterial = _score(highway="primary", name="Main Street", lanes_tag="4")
    assert path > arterial
    assert path > 0.8 and arterial < 0.3


def test_the_road_hierarchy_is_ordered():
    """Quieter classes of road should score higher, in order."""
    primary = _score(highway="primary")
    secondary = _score(highway="secondary")
    tertiary = _score(highway="tertiary")
    residential = _score(highway="residential")
    footway = _score(highway="footway")
    assert primary < secondary < tertiary < residential < footway


def test_an_unremarkable_residential_street_is_near_neutral():
    assert 0.45 < _score(highway="residential") < 0.65


def test_naming_a_path_helps_but_naming_a_road_does_not():
    """A named path suggests people go there; every road has a name."""
    named_path = _score(highway="path", name="Creek Trail")
    unnamed_path = _score(highway="path", name="")
    named_road = _score(highway="residential", name="Elm Street")
    unnamed_road = _score(highway="residential", name="")
    assert named_path > unnamed_path
    assert named_road == unnamed_road


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------


def test_being_barred_overrides_every_bonus():
    """A way you may not run on is excluded, not merely discounted.

    A named park path tagged "no pedestrians" collects enough bonuses to
    outrank an arterial unless the restriction overrides them outright.
    """
    barred = _score(highway="path", name="Lovely Trail", foot_tag="designated")
    assert barred > 0.8  # before the restriction
    assert _score(highway="path", name="Lovely Trail", foot_tag="no") == BARRED
    assert _score(highway="residential", access_tag="private") == BARRED
    assert _score(highway="path", foot_tag="no") < _score(highway="primary")


def test_customer_only_access_is_capped_but_not_barred():
    assert _score(highway="service", access_tag="customers") <= 0.25


def test_a_sidewalk_helps_and_its_absence_on_a_busy_road_hurts():
    with_walk = _score(highway="secondary", sidewalk_tag="both")
    plain = _score(highway="secondary")
    without = _score(highway="secondary", sidewalk_tag="no")
    assert with_walk > plain > without


def test_fast_traffic_is_penalised():
    assert _score(highway="tertiary", maxspeed_tag="55 mph") < _score(
        highway="tertiary", maxspeed_tag="25 mph"
    )


def test_many_lanes_are_penalised():
    assert _score(highway="secondary", lanes_tag="6") < _score(
        highway="secondary", lanes_tag="2"
    )


def test_a_driveway_is_worse_than_a_service_road():
    assert _score(highway="service", service_tag="driveway") < _score(
        highway="service"
    )


def test_greenspace_helps():
    edge = _edge(highway="footway")
    assert score_edge(edge, in_greenspace=True)[0] > score_edge(edge)[0]


def test_steps_are_penalised():
    assert _score(highway="steps") < _score(highway="footway")


# ---------------------------------------------------------------------------
# Bounds and robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"highway": "path", "name": "Trail", "foot_tag": "designated", "lit_tag": "yes"},
        {"highway": "trunk", "lanes_tag": "8", "maxspeed_tag": "70 mph",
         "access_tag": "private", "foot_tag": "no"},
        {"highway": ""},
        {"highway": "unheard_of_value"},
    ],
)
def test_scores_stay_in_range(kwargs):
    assert 0.0 <= _score(**kwargs) <= 1.0


def test_unknown_tags_do_not_crash_or_swing_the_score():
    """OSM contains anything; an unrecognised value must be inert."""
    assert _score(highway="residential", lanes_tag="lots") == _score(
        highway="residential"
    )
    assert _score(highway="residential", maxspeed_tag="fast") == _score(
        highway="residential"
    )


def test_an_untagged_way_sits_at_the_baseline():
    assert _score(highway="") == pytest.approx(BASE)


def test_reasons_explain_the_score():
    """The score is a judgement, so it has to be able to justify itself."""
    score, reasons = score_edge(
        _edge(highway="primary", lanes_tag="6", maxspeed_tag="50 mph")
    )
    labels = " ".join(r.label for r in reasons)
    assert "primary" in labels and "lanes" in labels and "mph" in labels
    # The reasons must add up to the score, after clamping to [0, 1].
    expected = max(0.0, min(1.0, BASE + sum(r.delta for r in reasons)))
    assert score == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# maxspeed parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30 mph", 30.0),
        ("30mph", 30.0),
        ("50", 31.07),  # bare values are km/h
        ("100", 62.14),
        ("walk", None),
        ("", None),
    ],
)
def test_maxspeed_parsing(raw, expected):
    result = _parse_maxspeed_mph(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, abs=0.05)


# ---------------------------------------------------------------------------
# Whole network
# ---------------------------------------------------------------------------


def test_score_network_covers_every_edge():
    edges = [
        make_edge(edge_id=0, highway="footway"),
        make_edge(edge_id=1, highway="primary"),
        make_edge(edge_id=2, highway="residential"),
    ]
    scores = score_network(edges, None)
    assert set(scores) == {0, 1, 2}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert scores[0] > scores[2] > scores[1]


def test_score_network_without_greenspace_is_fine():
    """The park fetch is optional and may fail; scoring must still work."""
    assert score_network([make_edge(edge_id=0)], None)
