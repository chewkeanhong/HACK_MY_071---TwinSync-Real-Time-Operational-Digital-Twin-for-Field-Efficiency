"""ST-DBSCAN must discriminate, not just group.

The stub this replaced put every failed tower in one cluster. These tests pin down the
four ways the real algorithm is allowed to say "no": too far apart, too far apart in
time, too few neighbours, and wrong kind of fault.
"""

from __future__ import annotations

from twinsync.stdbscan import (
    NOISE,
    Alarm,
    AlarmClusterer,
    families_compatible,
    st_dbscan,
)

from .conftest import make_world


def alarm(tower_id, x, y, t, kind="amplifier_degradation", z=50.0):
    return Alarm(tower_id=tower_id, x=x, y=y, z=z, t=t, alarm_type=kind)


# -- the core algorithm ------------------------------------------------


def test_co_located_simultaneous_alarms_form_one_cluster():
    labels = st_dbscan([alarm("A", 0.0, 0.0, 100.0), alarm("B", 200.0, 0.0, 130.0)])
    assert labels[0] == labels[1]
    assert labels[0] != NOISE


def test_spatially_distant_alarms_do_not_cluster():
    """Same second, five kilometres apart. Not one incident."""
    labels = st_dbscan([alarm("A", 0.0, 0.0, 100.0), alarm("B", 5000.0, 0.0, 100.0)])
    assert labels == [NOISE, NOISE]


def test_temporally_distant_alarms_do_not_cluster():
    """Same street, an hour apart. Not one incident either."""
    labels = st_dbscan([alarm("A", 0.0, 0.0, 100.0), alarm("B", 200.0, 0.0, 3700.0)])
    assert labels == [NOISE, NOISE]


def test_lone_alarm_is_noise():
    assert st_dbscan([alarm("A", 0.0, 0.0, 100.0)]) == [NOISE]


def test_empty_input():
    assert st_dbscan([]) == []


def test_elevation_separates_alarms_that_look_close_on_a_flat_map():
    """The Z axis has to actually count, or the DEM is decoration.

    Two sites at the same map position, one on a 700 m hilltop relay. A 2D clusterer
    calls that distance zero.
    """
    flat = st_dbscan([alarm("A", 0.0, 0.0, 100.0, z=40.0),
                      alarm("B", 0.0, 0.0, 100.0, z=40.0)])
    assert flat[0] == flat[1] != NOISE

    stacked = st_dbscan([alarm("A", 0.0, 0.0, 100.0, z=40.0),
                         alarm("B", 0.0, 0.0, 100.0, z=1000.0)])
    assert stacked == [NOISE, NOISE]


def test_incompatible_fault_families_stay_separate():
    """Co-located and simultaneous, but a bent antenna is not a congested backhaul."""
    labels = st_dbscan([
        alarm("A", 0.0, 0.0, 100.0, kind="antenna_misalignment"),
        alarm("B", 150.0, 0.0, 110.0, kind="backhaul_congestion"),
    ])
    assert labels == [NOISE, NOISE]


def test_compatible_families_do_cluster():
    """Power loss congesting a neighbour's backhaul is the textbook cascade."""
    labels = st_dbscan([
        alarm("A", 0.0, 0.0, 100.0, kind="power_failure"),
        alarm("B", 150.0, 0.0, 110.0, kind="backhaul_congestion"),
    ])
    assert labels[0] == labels[1] != NOISE


def test_family_compatibility_is_symmetric():
    """DBSCAN core/border reasoning breaks if the neighbour relation is not symmetric."""
    kinds = list({"amplifier_degradation", "antenna_misalignment",
                  "power_failure", "backhaul_congestion"})
    for a in kinds:
        for b in kinds:
            assert families_compatible(a, b) == families_compatible(b, a)


def test_min_pts_three_rejects_a_pair():
    pair = [alarm("A", 0.0, 0.0, 100.0), alarm("B", 100.0, 0.0, 110.0)]
    assert st_dbscan(pair, min_pts=3) == [NOISE, NOISE]
    assert st_dbscan(pair, min_pts=2) != [NOISE, NOISE]


def test_two_separate_cascades_get_separate_labels():
    labels = st_dbscan([
        alarm("A", 0.0, 0.0, 100.0), alarm("B", 200.0, 0.0, 120.0),
        alarm("C", 8000.0, 0.0, 100.0), alarm("D", 8200.0, 0.0, 120.0),
    ])
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert NOISE not in labels


# -- the streaming wrapper ---------------------------------------------


def clusterer_world():
    """Three towers: two neighbours and one far away, all at equal elevation."""
    return make_world(
        specs=[{"id": "b1", "cx": 0.0, "cy": 0.0, "size": 20.0, "height": 20.0}],
        towers=[
            {"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 50.0},
            {"id": "T2", "x": 300.0, "y": 0.0, "antenna_height": 50.0},
            {"id": "FAR", "x": 6000.0, "y": 0.0, "antenna_height": 50.0},
        ],
    )


def test_streaming_first_alarm_is_isolated_then_pairs_up():
    clusterer = AlarmClusterer(clusterer_world())

    first = clusterer.observe("T1", "amplifier_degradation", 100.0)
    assert first.is_noise is True
    assert first.cluster_id == "ISOLATED"

    second = clusterer.observe("T2", "antenna_misalignment", 200.0)
    assert second.is_noise is False
    assert second.members == ["T1", "T2"]
    assert second.span_s == 100.0
    assert 290.0 < second.span_m < 310.0
    assert second.model == "st-dbscan-v1"


def test_streaming_distant_alarm_stays_isolated():
    clusterer = AlarmClusterer(clusterer_world())
    clusterer.observe("T1", "amplifier_degradation", 100.0)
    clusterer.observe("T2", "amplifier_degradation", 200.0)

    far = clusterer.observe("FAR", "amplifier_degradation", 250.0)
    assert far.is_noise is True
    assert far.members == ["FAR"]


def test_cluster_id_is_stable_when_a_member_is_added():
    """The id an operator is working against must not mutate under them."""
    world = make_world(
        specs=[{"id": "b1", "cx": 0.0, "cy": 0.0, "size": 20.0, "height": 20.0}],
        towers=[
            {"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 50.0},
            {"id": "T2", "x": 300.0, "y": 0.0, "antenna_height": 50.0},
            {"id": "T3", "x": 600.0, "y": 0.0, "antenna_height": 50.0},
        ],
    )
    clusterer = AlarmClusterer(world)
    clusterer.observe("T1", "amplifier_degradation", 100.0)
    original = clusterer.observe("T2", "amplifier_degradation", 150.0).cluster_id

    grown = clusterer.observe("T3", "amplifier_degradation", 200.0)
    assert grown.cluster_id == original, "adding a third site must not renumber it"
    assert grown.members == ["T1", "T2", "T3"]


def test_retention_window_expires_old_alarms():
    clusterer = AlarmClusterer(clusterer_world(), retention_s=300.0)
    clusterer.observe("T1", "amplifier_degradation", 100.0)

    # Far outside both the retention window and the temporal epsilon.
    late = clusterer.observe("T2", "amplifier_degradation", 5000.0)
    assert late.is_noise is True


def test_forget_removes_a_resolved_tower():
    clusterer = AlarmClusterer(clusterer_world())
    clusterer.observe("T1", "amplifier_degradation", 100.0)
    clusterer.forget("T1")

    alone = clusterer.observe("T2", "amplifier_degradation", 150.0)
    assert alone.is_noise is True


def test_repeat_alarm_from_one_tower_does_not_self_cluster():
    """A tower re-alarming is one alarm, not a two-site cascade."""
    clusterer = AlarmClusterer(clusterer_world())
    clusterer.observe("T1", "amplifier_degradation", 100.0)
    again = clusterer.observe("T1", "amplifier_degradation", 150.0)
    assert again.is_noise is True
    assert again.members == ["T1"]
