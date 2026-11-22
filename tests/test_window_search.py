"""Tests for the sliding-window search, deduplication and scoring."""

from __future__ import annotations

import pytest

import geo
from analyze import (
    build_corridors,
    build_edges,
    dedupe_windows,
    rolling_extreme_grades,
    materialise_match,
    score_match,
    search_profile,
)
from conftest import make_corridor, make_edge, make_profile, make_target

MILE = geo.METRES_PER_MILE


# ---------------------------------------------------------------------------
# Window search
# ---------------------------------------------------------------------------


def test_finds_a_window_that_matches_exactly():
    """A 0.4 mi ramp at 5.5% is found by a 0.2-0.5 mi, 4-7% target."""
    profile = make_profile([(0.4 * MILE, 5.5)])
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))

    hits = search_profile(profile, target, min_window_m=100.0, step_samples=1)

    assert hits, "the ramp should match its own specification"
    for _, _, grade in hits:
        assert 4.0 <= grade <= 7.0


def test_rejects_a_grade_outside_the_band():
    """A 2% ramp must not satisfy a 4-7% target at any window length."""
    profile = make_profile([(0.6 * MILE, 2.0)])
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))

    assert search_profile(profile, target, 100.0, 1) == []


def test_rejects_a_segment_that_is_too_short():
    """A 0.1 mi ramp cannot satisfy a 0.2 mi minimum."""
    profile = make_profile([(0.1 * MILE, 5.5)])
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))

    assert search_profile(profile, target, 100.0, 1) == []


def test_window_lengths_stay_inside_the_target_band():
    profile = make_profile([(1.5 * MILE, 5.0)])
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))

    hits = search_profile(profile, target, 100.0, 1)

    assert hits
    for start, end, _ in hits:
        length = profile.dist_m[end] - profile.dist_m[start]
        assert 0.2 * MILE - 20.0 <= length <= 0.5 * MILE + 1e-6


def test_minimum_window_floor_overrides_a_shorter_target():
    """``min_window_m`` wins even when the target permits something shorter.

    Grade over a very short window is DEM noise, so the floor is a hard rule.
    """
    profile = make_profile([(400.0, 6.0)])
    target = make_target(length_mi=(0.01, 0.5), grade_pct=(4.0, 7.0))

    hits = search_profile(profile, target, min_window_m=200.0, step_samples=1)

    assert hits
    for start, end, _ in hits:
        assert profile.dist_m[end] - profile.dist_m[start] >= 200.0


def test_finds_the_steep_section_inside_a_longer_road():
    """A 0.3 mi pitch between two flat kilometres is located, not averaged away."""
    flat = 1000.0
    pitch = 0.3 * MILE
    profile = make_profile([(flat, 0.0), (pitch, 6.0), (flat, 0.0)])
    target = make_target(length_mi=(0.25, 0.35), grade_pct=(5.0, 7.0))

    hits = dedupe_windows(
        search_profile(profile, target, 100.0, 1), target
    )

    assert len(hits) == 1
    start, end, grade = hits[0]
    assert grade == pytest.approx(6.0, abs=0.3)
    assert profile.dist_m[start] == pytest.approx(flat, abs=40.0)
    assert profile.dist_m[end] == pytest.approx(flat + pitch, abs=40.0)


def test_descent_target_matches_only_descents():
    profile = make_profile([(0.5 * MILE, -5.1), (0.5 * MILE, 5.1)])
    target = make_target(
        key="descent",
        length_mi=(0.3, 0.6),
        grade_pct=(-6.5, -4.0),
        ideal_grade_pct=-5.1,
    )

    hits = search_profile(profile, target, 100.0, 1)

    assert hits
    for start, end, grade in hits:
        assert grade < 0
        # Each match is predominantly in the descending half. A little of the
        # following climb is allowed by the reversal tolerance, which is what
        # keeps genuinely undulating roads from being disqualified.
        midpoint = (profile.dist_m[start] + profile.dist_m[end]) / 2.0
        assert midpoint < 0.5 * MILE


def test_flat_target_accepts_either_direction():
    """A ``-1 to +1%`` target matches gentle up and gentle down alike."""
    target = make_target(
        key="flat", length_mi=(1.0, 2.0), grade_pct=(-1.0, 1.0), ideal_grade_pct=0.0
    )
    for grade in (-0.4, 0.0, 0.4):
        profile = make_profile([(1.5 * MILE, grade)])
        assert search_profile(profile, target, 100.0, 1), f"{grade}% should match"


def test_long_continuous_target_needs_a_long_profile():
    """A 2-3 mi target cannot be satisfied by a 1 mi road."""
    profile = make_profile([(1.0 * MILE, 1.5)])
    target = make_target(key="bridge", length_mi=(2.0, 3.2), grade_pct=(1.0, 2.0))

    assert search_profile(profile, target, 100.0, 1) == []


def test_step_samples_reduces_the_candidate_count_without_losing_the_feature():
    profile = make_profile([(1.0 * MILE, 5.5)])
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))

    fine = search_profile(profile, target, 100.0, 1)
    coarse = search_profile(profile, target, 100.0, 4)

    assert len(coarse) < len(fine)
    assert coarse, "coarsening must not lose the feature entirely"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_dedupe_returns_non_overlapping_windows():
    profile = make_profile([(2.0 * MILE, 5.5)])
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))

    kept = dedupe_windows(search_profile(profile, target, 100.0, 1), target)

    assert kept
    for (s1, e1, _), (s2, e2, _) in zip(kept[:-1], kept[1:]):
        assert e1 <= s2, "deduped windows must not overlap"


def test_dedupe_prefers_the_grade_closest_to_ideal():
    """Among overlapping candidates the one nearest the ideal grade wins."""
    target = make_target(
        length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0), ideal_grade_pct=5.5
    )
    candidates = [
        (0, 30, 4.1),  # in band, far from ideal
        (5, 35, 5.5),  # in band, exactly ideal, overlaps the first
        (10, 40, 6.9),  # in band, far from ideal, overlaps both
    ]

    kept = dedupe_windows(candidates, target)

    assert len(kept) == 1
    assert kept[0][2] == pytest.approx(5.5)


def test_dedupe_breaks_grade_ties_by_length():
    target = make_target(ideal_grade_pct=5.5)
    candidates = [(0, 20, 5.5), (0, 40, 5.5)]

    kept = dedupe_windows(candidates, target)

    assert kept == [(0, 40, 5.5)]


def test_dedupe_keeps_genuinely_separate_features():
    target = make_target(ideal_grade_pct=5.5)
    candidates = [(0, 20, 5.5), (100, 120, 5.4)]

    assert len(dedupe_windows(candidates, target)) == 2


def test_dedupe_of_nothing_is_nothing():
    assert dedupe_windows([], make_target()) == []


# ---------------------------------------------------------------------------
# Match construction
# ---------------------------------------------------------------------------


def _materialise(profile, target, start, end, grade, **kwargs):
    """Helper that fills in the boilerplate arguments."""
    extremes = rolling_extreme_grades(profile, 100.0)
    return materialise_match(
        profile=profile,
        target=target,
        start_index=start,
        end_index=end,
        mean_grade=grade,
        extremes=extremes,
        min_window_m=100.0,
        waterway_edges=kwargs.pop("waterway_edges", frozenset()),
        reference_point=kwargs.pop("reference_point", (35.0, 0.0)),
    )


def test_match_reports_length_grade_and_gain():
    profile = make_profile([(0.4 * MILE, 5.0)])
    target = make_target()
    end = profile.size - 1

    match = _materialise(profile, target, 0, end, 5.0)

    assert match.length_mi == pytest.approx(0.4, abs=0.02)
    assert match.mean_grade_pct == pytest.approx(5.0)
    expected_gain = 0.4 * MILE * 0.05
    assert match.gain_m == pytest.approx(expected_gain, rel=0.02)
    assert match.loss_m == pytest.approx(0.0)


def test_match_max_grade_is_signed_by_the_direction_of_travel():
    """On a descent, "max grade" must be the steepest *downhill* figure."""
    profile = make_profile([(200.0, -2.0), (300.0, -8.0), (200.0, -2.0)])
    target = make_target(grade_pct=(-9.0, -1.0), ideal_grade_pct=-5.0)
    end = profile.size - 1

    match = _materialise(profile, target, 0, end, -5.0)

    assert match.steepest_grade_pct == pytest.approx(-8.0, abs=0.5)
    assert match.steepest_grade_pct < 0


def test_match_max_grade_on_a_climb_is_positive():
    profile = make_profile([(200.0, 2.0), (300.0, 9.0), (200.0, 2.0)])
    target = make_target(grade_pct=(1.0, 10.0), ideal_grade_pct=5.0)

    match = _materialise(profile, target, 0, profile.size - 1, 5.0)

    assert match.steepest_grade_pct == pytest.approx(9.0, abs=0.5)


def test_match_surface_is_length_weighted():
    """A mostly-paved corridor with a dirt stretch comes back unpaved."""
    edges = [
        make_edge(edge_id=0, start_node=1, end_node=2, length_m=700.0, surface="asphalt"),
        make_edge(edge_id=1, start_node=2, end_node=3, length_m=300.0, surface="dirt"),
    ]
    corridor = make_corridor(edges)
    profile = make_profile([(1000.0, 5.0)], corridor=corridor)

    match = _materialise(profile, make_target(), 0, profile.size - 1, 5.0)

    assert match.surface == "unpaved"
    assert match.paved_fraction == pytest.approx(0.7, abs=0.01)


def test_match_surface_paved_when_fully_sealed():
    edges = [
        make_edge(edge_id=0, start_node=1, end_node=2, length_m=500.0, surface="asphalt"),
        make_edge(edge_id=1, start_node=2, end_node=3, length_m=500.0, surface="concrete"),
    ]
    profile = make_profile([(1000.0, 5.0)], corridor=make_corridor(edges))

    match = _materialise(profile, make_target(), 0, profile.size - 1, 5.0)

    assert match.surface == "paved"
    assert match.paved_fraction == pytest.approx(1.0)


def test_match_surface_unknown_when_untagged():
    edges = [make_edge(edge_id=0, length_m=1000.0, surface="")]
    profile = make_profile([(1000.0, 5.0)], corridor=make_corridor(edges))

    match = _materialise(profile, make_target(), 0, profile.size - 1, 5.0)

    assert match.surface == "unknown"
    assert match.paved_fraction == pytest.approx(0.0)


def test_bridge_is_flagged_unreliable():
    """The DEM reads the ground under a bridge, so bridges are not trusted."""
    edges = [make_edge(edge_id=0, length_m=1000.0, bridge=True)]
    profile = make_profile([(1000.0, 5.0)], corridor=make_corridor(edges))

    match = _materialise(profile, make_target(), 0, profile.size - 1, 5.0)

    assert match.unreliable
    assert "bridge" in match.unreliable_reasons


def test_tunnel_is_flagged_unreliable():
    edges = [make_edge(edge_id=0, length_m=1000.0, tunnel=True)]
    profile = make_profile([(1000.0, 5.0)], corridor=make_corridor(edges))

    match = _materialise(profile, make_target(), 0, profile.size - 1, 5.0)

    assert match.unreliable and "tunnel" in match.unreliable_reasons


def test_waterway_crossing_is_flagged_unreliable():
    edges = [make_edge(edge_id=7, length_m=1000.0)]
    profile = make_profile([(1000.0, 5.0)], corridor=make_corridor(edges))

    match = _materialise(
        profile, make_target(), 0, profile.size - 1, 5.0,
        waterway_edges=frozenset({7}),
    )

    assert match.unreliable
    assert "crosses waterway" in match.unreliable_reasons


def test_dem_spike_is_flagged_unreliable():
    """A one-cell cliff in the raw DEM marks the segment as untrustworthy."""
    profile = make_profile([(500.0, 1.0)])
    profile.elev_raw_m[12] += 30.0  # a 150% single-sample grade

    match = _materialise(profile, make_target(), 0, profile.size - 1, 1.0)

    assert match.unreliable
    assert "DEM spike" in match.unreliable_reasons


def test_clean_segment_is_not_flagged():
    profile = make_profile([(1000.0, 5.0)])
    match = _materialise(profile, make_target(), 0, profile.size - 1, 5.0)
    assert not match.unreliable
    assert match.unreliable_reasons == []


def test_match_distance_from_reference_is_measured_from_the_start():
    profile = make_profile([(1000.0, 5.0)])
    match = _materialise(
        profile, make_target(), 0, profile.size - 1, 5.0,
        reference_point=(35.0, 0.0),
    )
    assert match.distance_from_reference_mi == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in exposing only what :func:`score_match` reads."""

    scoring = {
        "grade_weight": 45.0,
        "length_weight": 25.0,
        "paved_bonus": 15.0,
        "proximity_bonus": 15.0,
        "proximity_decay_mi": 8.0,
        "unreliable_penalty": 25.0,
        "unpaved_penalty": 20.0,
    }


def _scored(target, **overrides):
    """Score a synthetic match built from keyword overrides."""
    profile = make_profile([(0.5 * MILE, overrides.get("grade", 5.5))])
    match = _materialise(
        profile, target, 0, profile.size - 1, overrides.get("grade", 5.5)
    )
    for key, value in overrides.items():
        if key != "grade":
            setattr(match, key, value)
    return score_match(match, target, _FakeConfig())


def test_ideal_grade_scores_above_a_band_edge_grade():
    target = make_target(grade_pct=(4.0, 7.0), ideal_grade_pct=5.5)
    assert _scored(target, grade=5.5) > _scored(target, grade=4.1)


def test_paved_outranks_unknown_which_outranks_unpaved():
    """Surface ordering must be strict for targets that ask for pavement."""
    target = make_target(grade_pct=(4.0, 7.0), ideal_grade_pct=5.5, prefer_paved=True)

    paved = _scored(target, surface="paved", paved_fraction=1.0)
    unknown = _scored(target, surface="unknown", paved_fraction=0.0)
    unpaved = _scored(target, surface="unpaved", paved_fraction=0.0)

    assert paved > unknown > unpaved


def test_closer_to_the_reference_point_scores_higher():
    target = make_target()
    near = _scored(target, distance_from_reference_mi=0.5)
    far = _scored(target, distance_from_reference_mi=7.5)
    assert near > far


def test_proximity_bonus_bottoms_out_rather_than_going_negative():
    target = make_target()
    at_decay = _scored(target, distance_from_reference_mi=8.0)
    way_out = _scored(target, distance_from_reference_mi=80.0)
    assert at_decay == pytest.approx(way_out)


def test_unreliable_segments_are_penalised_but_kept():
    target = make_target()
    clean = _scored(target, unreliable=False)
    flagged = _scored(target, unreliable=True)
    assert flagged < clean
    assert flagged >= 0.0


def test_score_never_goes_negative():
    target = make_target(grade_pct=(4.0, 7.0), ideal_grade_pct=5.5)
    score = _scored(
        target,
        grade=4.0,
        surface="unpaved",
        unreliable=True,
        distance_from_reference_mi=50.0,
    )
    assert score == 0.0


def test_longer_segments_score_higher_within_the_band():
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(4.0, 7.0))
    short = make_profile([(0.25 * MILE, 5.5)])
    long = make_profile([(0.5 * MILE, 5.5)])

    short_match = _materialise(short, target, 0, short.size - 1, 5.5)
    long_match = _materialise(long, target, 0, long.size - 1, 5.5)

    assert score_match(long_match, target, _FakeConfig()) > score_match(
        short_match, target, _FakeConfig()
    )


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------


def _way(way_id, nodes, coords, **tags):
    """Build an Overpass-shaped way element."""
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "nodes": nodes,
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in coords],
    }


def test_ways_are_split_at_shared_nodes():
    """A way crossed in the middle becomes two edges."""
    main = _way(
        1,
        [10, 11, 12],
        [(0.0, 35.0), (0.005, 35.0), (0.010, 35.0)],
        highway="residential",
        name="Main",
    )
    side = _way(
        2,
        [20, 11],
        [(0.005, 34.995), (0.005, 35.0)],
        highway="residential",
        name="Side",
    )

    edges, nodes = build_edges({"elements": [main, side]})

    main_edges = [edge for edge in edges if edge.way_id == 1]
    assert len(main_edges) == 2
    assert {edge.start_node for edge in main_edges} == {10, 11}
    assert 11 in nodes


def test_way_without_a_junction_stays_one_edge():
    lone = _way(
        1,
        [10, 11, 12],
        [(0.0, 35.0), (0.005, 35.0), (0.010, 35.0)],
        highway="residential",
    )
    edges, _ = build_edges({"elements": [lone]})
    assert len(edges) == 1
    assert edges[0].start_node == 10 and edges[0].end_node == 12


def test_tags_are_carried_onto_edges():
    way = _way(
        1,
        [10, 11],
        [(0.0, 35.0), (0.005, 35.0)],
        highway="tertiary",
        name="Reservoir Canyon Road",
        surface="gravel",
        bridge="yes",
    )
    edges, _ = build_edges({"elements": [way]})
    edge = edges[0]
    assert edge.name == "Reservoir Canyon Road"
    assert edge.highway == "tertiary"
    assert edge.surface_class == "unpaved"
    assert edge.bridge is True


def test_bridge_no_is_not_a_bridge():
    way = _way(1, [10, 11], [(0.0, 35.0), (0.005, 35.0)], highway="residential", bridge="no")
    edges, _ = build_edges({"elements": [way]})
    assert edges[0].bridge is False


def test_degenerate_ways_are_dropped():
    bad = [
        {"type": "way", "id": 1, "nodes": [1], "geometry": [{"lat": 35.0, "lon": 0.0}]},
        {"type": "way", "id": 2, "nodes": [], "geometry": []},
        {"type": "node", "id": 3},
    ]
    edges, _ = build_edges({"elements": bad})
    assert edges == []


def test_collinear_edges_chain_into_one_corridor():
    """Two straight segments meeting end to end form a single corridor.

    This is the behaviour the long targets depend on: a two-mile climb is
    never one OSM edge, so without chaining it could not be found at all.
    """
    first = _way(
        1, [10, 11], [(0.0, 35.0), (0.010, 35.0)], highway="residential", name="Long Road"
    )
    second = _way(
        2, [11, 12], [(0.010, 35.0), (0.020, 35.0)], highway="residential", name="Long Road"
    )

    edges, _ = build_edges({"elements": [first, second]})
    corridors = build_corridors(edges)

    assert len(corridors) == 1
    assert len(corridors[0].edges) == 2
    assert corridors[0].length_m == pytest.approx(
        sum(edge.length_m for edge in edges)
    )


def test_a_sharp_turn_does_not_chain():
    """A right-angle junction is two corridors, not one."""
    east = _way(1, [10, 11], [(0.0, 35.0), (0.010, 35.0)], highway="residential")
    north = _way(2, [11, 12], [(0.010, 35.0), (0.010, 35.010)], highway="residential")

    edges, _ = build_edges({"elements": [east, north]})
    corridors = build_corridors(edges, max_turn_deg=60.0)

    assert len(corridors) == 2


def test_every_edge_lands_in_exactly_one_corridor():
    ways = [
        _way(1, [10, 11], [(0.0, 35.0), (0.005, 35.0)], highway="residential"),
        _way(2, [11, 12], [(0.005, 35.0), (0.010, 35.0)], highway="residential"),
        _way(3, [11, 20], [(0.005, 35.0), (0.005, 35.005)], highway="residential"),
    ]
    edges, _ = build_edges({"elements": ways})
    corridors = build_corridors(edges)

    seen = [edge.edge_id for corridor in corridors for edge in corridor.edges]
    assert sorted(seen) == sorted(edge.edge_id for edge in edges)
    assert len(seen) == len(set(seen)), "no edge may appear twice"


def test_corridor_length_cap_is_respected():
    ways = [
        _way(
            index,
            [index * 10, index * 10 + 10],
            [(index * 0.01, 35.0), ((index + 1) * 0.01, 35.0)],
            highway="residential",
            name="Endless Road",
        )
        for index in range(1, 8)
    ]
    edges, _ = build_edges({"elements": ways})
    corridors = build_corridors(edges, max_corridor_m=2000.0)

    assert len(corridors) > 1
    for corridor in corridors:
        assert corridor.length_m <= 2000.0 + max(
            edge.length_m for edge in edges
        )


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------


def test_a_descent_that_climbs_back_is_rejected():
    """Mean grade is not enough: the window must actually be a descent.

    A drop followed by a partial climb can average into the target band while
    being nothing like the continuous descent the specification asks for.
    """
    profile = make_profile([(400.0, -10.0), (400.0, 10.0)])
    target = make_target(
        key="descent", length_mi=(0.3, 0.6), grade_pct=(-8.0, -1.0),
        ideal_grade_pct=-5.0,
    )

    for start, end, _ in search_profile(profile, target, 100.0, 1):
        # Every accepted window must sit in the descending half.
        assert profile.dist_m[end] <= 400.0 + 1e-6


def test_a_steady_descent_is_still_accepted():
    profile = make_profile([(0.5 * MILE, -5.1)])
    target = make_target(
        key="descent", length_mi=(0.3, 0.6), grade_pct=(-6.5, -4.0),
        ideal_grade_pct=-5.1,
    )
    assert search_profile(profile, target, 100.0, 1)


def test_small_undulations_are_tolerated():
    """A real road is never perfectly smooth; minor dips must not disqualify."""
    runs = []
    for _ in range(10):
        runs.append((60.0, 6.0))
        runs.append((20.0, -1.0))
    profile = make_profile(runs)
    target = make_target(length_mi=(0.2, 0.5), grade_pct=(3.0, 7.0))

    assert search_profile(profile, target, 100.0, 1)


def test_flat_target_ignores_the_continuity_rule():
    """A flat target spans zero grade, where "against the trend" is meaningless."""
    runs = []
    for _ in range(40):
        runs.append((40.0, 1.5))
        runs.append((40.0, -1.5))
    profile = make_profile(runs)
    target = make_target(
        key="flat", length_mi=(1.0, 2.0), grade_pct=(-1.0, 1.0), ideal_grade_pct=0.0
    )

    assert search_profile(profile, target, 100.0, 1)


def test_flat_target_rejects_steep_rollers():
    """A road of rollers that ends where it started is not a flat finish.

    Net grade alone would accept it, which is why the flat target is capped
    on total climbing per kilometre instead of on monotonicity.
    """
    runs = []
    for _ in range(20):
        runs.append((100.0, 6.0))
        runs.append((100.0, -6.0))
    profile = make_profile(runs)
    target = make_target(
        key="flat", length_mi=(1.0, 2.0), grade_pct=(-1.0, 1.0), ideal_grade_pct=0.0
    )

    assert search_profile(profile, target, 100.0, 1, 0.25, 20.0) == []


def test_flat_target_accepts_a_genuinely_flat_road():
    profile = make_profile([(2.0 * MILE, 0.3)])
    target = make_target(
        key="flat", length_mi=(1.0, 2.0), grade_pct=(-1.0, 1.0), ideal_grade_pct=0.0
    )

    assert search_profile(profile, target, 100.0, 1, 0.25, 20.0)


def test_flat_target_tolerates_gentle_undulation():
    runs = []
    for _ in range(20):
        runs.append((100.0, 1.0))
        runs.append((100.0, -1.0))
    profile = make_profile(runs)
    target = make_target(
        key="flat", length_mi=(1.0, 2.0), grade_pct=(-1.0, 1.0), ideal_grade_pct=0.0
    )

    assert search_profile(profile, target, 100.0, 1, 0.25, 20.0)
