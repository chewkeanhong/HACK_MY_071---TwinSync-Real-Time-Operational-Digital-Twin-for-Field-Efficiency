"""Root-cause attribution: which member of a cluster is the head, and when to decline.

ST-DBSCAN decides that a set of alarms is one incident. This decides which of them
caused the rest, off a structure ST-DBSCAN cannot see -- the transport graph. The tests
that matter most here are the ones pinning the *refusal*: a clusterer that always names
a culprit is worse than useless, because an operator cannot tell its confident answers
from its invented ones.
"""

from __future__ import annotations

import pytest

from twinsync.rootcause import (
    BASIS_TOPOLOGICAL, ROLE_DOWNSTREAM, ROLE_PEER, ROLE_SOURCE, attribute,
)
from twinsync.world import World

from .conftest import make_world


def chain_world() -> World:
    """hub -- relay -- two edge nodes, west to east. Same shape as test_asset_graph."""
    return make_world(
        specs=[{"id": "b1", "cx": 0.0, "cy": 0.0, "size": 10.0, "height": 10.0}],
        towers=[
            {"id": "H", "x": 0.0, "y": 0.0, "antenna_height": 90.0},
            {"id": "R", "x": 400.0, "y": 0.0, "antenna_height": 60.0},
            {"id": "E1", "x": 800.0, "y": 0.0, "antenna_height": 30.0},
            {"id": "E2", "x": 1200.0, "y": 0.0, "antenna_height": 20.0},
        ],
    )


@pytest.fixture(scope="module")
def chain() -> World:
    return chain_world()


@pytest.fixture(scope="module")
def real_world() -> World:
    return World.load("data", require_towers=True)


# -- the basic verdict ---------------------------------------------------

def test_upstream_site_is_named_the_source(chain):
    """The parent of an alarming child is the head, and the child is its symptom."""
    verdict = attribute(chain.asset_graph, "CL-001", ["R", "E1"],
                        {"R": 100.0, "E1": 250.0})
    assert verdict.source == "R"
    assert verdict.downstream == ("E1",)
    assert verdict.role_of("R") == ROLE_SOURCE
    assert verdict.role_of("E1") == ROLE_DOWNSTREAM
    assert verdict.hops["E1"] == 1


def test_ancestry_outranks_an_earlier_alarm(chain):
    """A site cannot be the cause of the site that feeds it, however early it alarmed.

    This is the ordering that stops the twin blaming the first symptom to be noticed --
    which, on a network where failures propagate downhill, is exactly the wrong end.
    """
    verdict = attribute(chain.asset_graph, "CL-001", ["H", "E1"],
                        {"E1": 10.0, "H": 900.0})
    assert verdict.source == "H"
    assert verdict.basis == BASIS_TOPOLOGICAL


def test_multi_hop_chains_report_their_distance(chain):
    """Two hops down is still the same fault, and the card should say how far."""
    verdict = attribute(chain.asset_graph, "CL-001", ["H", "R", "E2"],
                        {"H": 10.0, "R": 20.0, "E2": 30.0})
    assert verdict.source == "H"
    assert set(verdict.downstream) == {"R", "E2"}
    assert verdict.hops["R"] == 1
    assert verdict.hops["E2"] == 3
    # Nearest symptom first, so the dispatcher reads the chain in the order it spread.
    assert verdict.downstream == ("R", "E2")


def test_a_healthy_intermediate_hop_does_not_break_the_chain(chain):
    """H -> E2 is still H's fault even when R and E1, the sites between, are fine."""
    verdict = attribute(chain.asset_graph, "CL-001", ["H", "E2"],
                        {"H": 10.0, "E2": 30.0})
    assert verdict.source == "H"
    assert verdict.hops["E2"] == 3


# -- the refusal ---------------------------------------------------------

def test_siblings_with_no_dependency_get_no_source(real_world):
    """Two sites that feed neither other are two jobs, however close they are.

    KL-04 and KL-07 both hang off the hierarchy without either being upstream of the
    other. Naming one of them would be inventing a finding, so the attributor declines.
    """
    verdict = attribute(real_world.asset_graph, "CL-009", ["KL-04", "KL-07"],
                        {"KL-04": 10.0, "KL-07": 20.0})
    assert verdict.source is None
    assert verdict.downstream == ()
    assert verdict.symptoms == 0
    assert verdict.role_of("KL-04") == ROLE_PEER
    assert "no member feeds another" in verdict.reason


def test_an_empty_cluster_is_not_an_error(chain):
    verdict = attribute(chain.asset_graph, "CL-000", [], {})
    assert verdict.source is None


def test_a_single_member_has_nothing_to_blame(chain):
    """One alarm cannot be a chain, even though it is trivially its own ancestor."""
    verdict = attribute(chain.asset_graph, "CL-001", ["R"], {"R": 10.0})
    assert verdict.source is None


# -- determinism ---------------------------------------------------------

def test_ties_resolve_on_id_not_input_order(chain):
    """Same cluster, two orderings, one answer.

    Input order changing the verdict would make the incident card flicker between
    re-clusterings, the same failure AlarmClusterer._stable_id exists to prevent.
    """
    times = {"H": 10.0, "R": 10.0, "E1": 10.0}
    first = attribute(chain.asset_graph, "CL-001", ["E1", "R", "H"], times)
    second = attribute(chain.asset_graph, "CL-001", ["H", "E1", "R"], times)
    assert first.source == second.source == "H"
    assert first.downstream == second.downstream


def test_basis_reports_which_evidence_decided_it(chain):
    """'Upstream of' and 'alarmed first' are different strengths of claim."""
    topological = attribute(chain.asset_graph, "CL-001", ["R", "E1"],
                            {"R": 10.0, "E1": 20.0})
    assert topological.basis == BASIS_TOPOLOGICAL


# -- the committed scenario ----------------------------------------------

def test_the_demo_chain_attributes_to_kl03(real_world):
    """Beat 5. KL-03 feeds KL-13 at 290 m; when both alarm, KL-03 is the head.

    Pins the beat itself: a re-derivation that re-parented KL-13 would leave the
    narration claiming a dependency the graph no longer has.
    """
    verdict = attribute(real_world.asset_graph, "CL-001", ["KL-03", "KL-13"],
                        {"KL-03": 183.0, "KL-13": 333.0},
                        profiles={"KL-03": "backhaul_congestion",
                                  "KL-13": "backhaul_congestion"})
    assert verdict.source == "KL-03"
    assert verdict.downstream == ("KL-13",)
    assert verdict.hops["KL-13"] == 1
    assert verdict.symptoms == 1
    assert "primary feed" in verdict.reason


# -- retroactive relabelling ---------------------------------------------
#
# Attribution is asked the moment a new alarm joins a cluster, which means the verdict
# for an alarm that arrived minutes earlier can change under it. KL-03 is not "the
# source" of anything until KL-13 shows up. Simulation._attribute walks the whole
# cluster for exactly this reason, and these pin that it does.

class _Intelligence:
    """The attribution half of IntelligenceLayer, against a hand-built graph."""

    def __init__(self, world):
        self.world = world

    def attribute(self, cluster_id, members, alarm_times, *, profiles=None):
        return attribute(self.world.asset_graph, cluster_id, members, alarm_times,
                         profiles=profiles)


class _Localisation:
    def __init__(self, cluster_id, members, is_noise=False):
        self.cluster_id = cluster_id
        self.members = members
        self.is_noise = is_noise
        self.span_m = 301.0
        self.span_s = 148.0


def _sim(chain, towers):
    """A Simulation stripped to what _attribute touches, so the test stays fast."""
    from types import SimpleNamespace

    from twinsync.dispatch import Incident
    from twinsync.sim import Simulation

    incidents, by_tower = {}, {}
    for n, tower in enumerate(towers, start=1):
        incident_id = f"INC-{n:03d}"
        incidents[incident_id] = Incident(
            id=incident_id, tower_id=tower, severity="degraded",
            detected_at=0.0, impact=None, xy=None,
        )
        by_tower[tower] = incident_id

    sim = SimpleNamespace(
        intelligence=_Intelligence(chain),
        dispatch=SimpleNamespace(incidents=incidents),
        _incident_by_tower=by_tower,
        _alarm_at={t: 100.0 * i for i, t in enumerate(towers, start=1)},
        _alarm_profile={},
        events=[],
    )
    sim._log = sim.events.append
    sim._attribute = Simulation._attribute.__get__(sim, SimpleNamespace)
    sim._set_role = Simulation._set_role.__get__(sim, SimpleNamespace)
    return sim


def test_the_earlier_incident_is_relabelled_when_a_symptom_joins(chain):
    """R's card must flip to ROOT CAUSE when E1 alarms underneath it.

    Without this the dispatcher reads a stale card for the one site they are about to
    drive to -- the site the whole verdict is telling them to go to.
    """
    sim = _sim(chain, ["R", "E1"])
    sim._attribute(_Localisation("CL-001", ["R", "E1"]))

    assert sim.dispatch.incidents["INC-001"].root_cause_role == ROLE_SOURCE
    assert sim.dispatch.incidents["INC-002"].root_cause_role == ROLE_DOWNSTREAM
    assert sim.dispatch.incidents["INC-002"].root_cause_id == "R"
    assert sim.dispatch.incidents["INC-002"].root_cause_hops == 1


def test_the_root_line_tells_the_crew_where_to_start(chain):
    """The log line is a work order, not a savings claim -- see Attribution.symptoms."""
    sim = _sim(chain, ["H", "R", "E1"])
    sim._attribute(_Localisation("CL-001", ["H", "R", "E1"]))
    line = next(e for e in sim.events if e.startswith("ROOT"))
    assert "fix H first" in line
    assert "R, E1 clears with it" in line
    assert "van" not in line


def test_an_isolated_alarm_is_marked_sole_and_logs_nothing(chain):
    """ST-DBSCAN's noise verdict short-circuits attribution entirely."""
    sim = _sim(chain, ["R"])
    sim._attribute(_Localisation("ISOLATED", ["R"], is_noise=True))
    assert sim.dispatch.incidents["INC-001"].root_cause_role == "sole"
    assert sim.dispatch.incidents["INC-001"].root_cause_id is None
    assert not [e for e in sim.events if e.startswith("ROOT")]


def test_a_cluster_with_no_dependency_leaves_every_card_a_peer(real_world):
    """The refusal has to survive the trip through the simulation, not just the unit.

    KL-04 and KL-07 cluster in ST-DBSCAN's terms but neither feeds the other, so both
    stay ordinary standalone jobs and neither card claims a cause.
    """
    sim = _sim(real_world, ["KL-04", "KL-07"])
    sim._attribute(_Localisation("CL-004", ["KL-04", "KL-07"]))

    for incident in sim.dispatch.incidents.values():
        assert incident.root_cause_role == ROLE_PEER
        assert incident.root_cause_id is None
    line = next(e for e in sim.events if e.startswith("ROOT"))
    assert "no transport dependency" in line


def test_the_earlier_member_inherits_the_cluster_id(chain):
    """R alarmed alone and was ISOLATED; once E1 joins it, R is in CL-001 too.

    The localiser is only asked about the alarm that just arrived, so the earlier
    incident keeps whatever verdict it earned when it really was on its own. Left
    unpatched that puts "cluster ISOLATED" on the same card as a ROOT CAUSE badge
    naming it the head of CL-001, and invents a phantom cluster in /api/rootcause.
    """
    sim = _sim(chain, ["R", "E1"])
    first = sim.dispatch.incidents["INC-001"]
    first.ai_cluster_id, first.ai_cluster_noise = "ISOLATED", True

    sim._attribute(_Localisation("CL-001", ["R", "E1"]))

    assert first.ai_cluster_id == "CL-001"
    assert first.ai_cluster_noise is False
    assert first.ai_cluster_members == ["R", "E1"]
