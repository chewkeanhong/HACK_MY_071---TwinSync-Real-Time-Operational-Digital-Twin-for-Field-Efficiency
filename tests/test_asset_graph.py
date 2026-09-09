"""The transport dependency graph, and what a failure does downstream of it.

Two kinds of test here. The hand-built ones pin the *behaviour* -- a chain darkens, a
protected node survives, a cycle terminates -- on graphs small enough to check on paper.
The ones against `data/` pin the *derivation*, because the hub set is the demo's
narrative and a silent re-tiering would rewrite the story without failing anything else.
"""

from __future__ import annotations

import networkx as nx
import pytest

from twinsync.world import (
    BATTERY_AUTONOMY_S, TIER_EDGE, TIER_HUB, TIER_RELAY, AssetGraph, World,
)

from .conftest import make_world


def chain_world() -> World:
    """hub -- relay -- two edge nodes, strung out west to east.

        H(0)      R(400)      E1(800)   E2(1200)
        tall       mid          low       low

    Ranked by antenna altitude the order is H, R, E1, E2, and each takes the nearest
    higher-ranked site as its parent, so the topology is the chain it looks like.
    """
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
def real_world() -> World:
    return World.load("data", require_towers=True)


# -- derivation against the committed scene ----------------------------


def test_graph_is_acyclic(real_world):
    """Every edge runs high-rank to low-rank, so a cycle would be a derivation bug."""
    assert nx.is_directed_acyclic_graph(real_world.asset_graph.graph)


def test_hub_set_is_pinned(real_world):
    """The hubs are the demo narrative -- north, centre and west, not three in a row.

    Height ranking alone puts KL-01, KL-02 and KL-04 in the core, all in the north.
    The spacing rule is what replaces KL-04 with KL-11 out west, and that is the whole
    reason the derived topology is defensible rather than merely automatic.
    """
    assert real_world.asset_graph.hubs == ("KL-01", "KL-02", "KL-11")


def test_every_site_has_a_tier_and_a_battery(real_world):
    graph = real_world.asset_graph
    assert set(graph.tier) == {t.id for t in real_world.towers}
    assert set(graph.battery_s) == set(graph.tier)
    for site, tier in graph.tier.items():
        assert graph.battery_s[site] == BATTERY_AUTONOMY_S[tier]


def test_tier_counts_follow_the_fractions(real_world):
    tiers = real_world.asset_graph.tier
    counts = {tier: sum(1 for v in tiers.values() if v == tier)
              for tier in (TIER_HUB, TIER_RELAY, TIER_EDGE)}
    assert counts == {TIER_HUB: 3, TIER_RELAY: 5, TIER_EDGE: 7}


def test_relay_edge_boundary_has_real_margin(real_world):
    """The tier split must not hinge on a rounding error in the DEM.

    Two sites sit 0.07 m apart in antenna altitude, which sounds alarming until you
    notice they are both relays. The boundary that decides a tier -- the last relay
    against the first edge node -- has metres of daylight, so re-baking the DEM will
    not silently re-tier the fleet.
    """
    graph = real_world.asset_graph
    altitude = {t.id: t.antenna_z for t in real_world.towers}
    lowest_relay = min(altitude[s] for s, v in graph.tier.items() if v == TIER_RELAY)
    highest_edge = max(altitude[s] for s, v in graph.tier.items() if v == TIER_EDGE)
    assert lowest_relay - highest_edge > 1.0


def test_derivation_is_deterministic(real_world):
    """Same world in, byte-identical topology out. Guards the A/B contract."""
    first = AssetGraph.derive(real_world.towers, real_world.terrain)
    second = AssetGraph.derive(real_world.towers, real_world.terrain)
    assert (first.to_dict(real_world.frame, real_world.towers)
            == second.to_dict(real_world.frame, real_world.towers))
    assert first.hubs == second.hubs


def test_protection_paths_are_rooted_elsewhere(real_world):
    """A standby link back into the same subtree protects against nothing."""
    graph = real_world.asset_graph
    protect = [link for link in graph.links if link.role == "protect"]
    assert protect, "no protection paths derived at all"
    for link in protect:
        primary = next(other for other in graph.links
                       if other.child == link.child and other.role == "primary")
        assert link.parent != primary.parent


def test_single_hub_failure_does_not_black_out_the_fleet(real_world):
    """Losing one hub is survivable -- that is what the protection paths are for."""
    impact = real_world.calculate_cascade_impact("KL-01")
    assert impact.dark == ("KL-01",)
    # But sites that were dual-fed are now single-fed, which is the warning that fires.
    assert impact.unprotected


def test_two_hub_failure_cascades(real_world):
    """Two of three hubs is not survivable, and the blast radius is pinned."""
    impact = real_world.calculate_cascade_impact(["KL-02", "KL-11"])
    assert len(impact.dark) == 8
    assert set(impact.origin) <= set(impact.dark)
    # Everything isolated is reachable from the origin, and depth records how far.
    for site in impact.cascaded:
        assert impact.depth[site] >= 1


# -- behaviour, on graphs small enough to check by hand -----------------


def test_chain_failure_darkens_everything_downstream():
    world = chain_world()
    graph = world.asset_graph
    assert graph.tier["H"] == TIER_HUB
    assert list(graph.graph.successors("H")) == ["R"]

    impact = world.calculate_cascade_impact("R")
    assert set(impact.dark) == {"R", "E1", "E2"}
    assert impact.depth == {"R": 0, "E1": 1, "E2": 2}


def test_failing_a_leaf_takes_nothing_with_it():
    impact = chain_world().calculate_cascade_impact("E2")
    assert impact.dark == ("E2",)
    assert impact.cascaded == ()


def test_severed_links_are_reported():
    impact = chain_world().calculate_cascade_impact("R")
    assert ("H", "R") in impact.severed_links
    assert ("R", "E1") in impact.severed_links


def test_cycle_terminates_instead_of_recursing():
    """The derivation cannot produce a cycle, but the traversal must survive one.

    A future ring topology, or a hand-built graph like this one, must get an answer
    rather than a RecursionError or a hang.
    """
    world = chain_world()
    cyclic = nx.DiGraph()
    cyclic.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    world.asset_graph.graph = cyclic
    world.asset_graph.hubs = ("A",)
    world.asset_graph.battery_s = {"A": 60.0, "B": 60.0, "C": 60.0}

    impact = world.calculate_cascade_impact("A")
    assert set(impact.dark) == {"A", "B", "C"}


def test_battery_wave_is_staggered_by_tier(real_world):
    """A substation trip is not one cliff: edge sites die first, then relays, then hubs.

    This is the whole reason `battery_depletion_timer` exists. Without it a cascade is
    instantaneous and there is nothing for a dispatcher to get ahead of.
    """
    everything = [t.id for t in real_world.towers]
    impact = real_world.calculate_cascade_impact((), on_battery=everything)
    graph = real_world.asset_graph

    by_tier: dict[str, list[float]] = {}
    for site in impact.at_risk:
        by_tier.setdefault(graph.tier[site], []).append(impact.time_to_dark_s[site])

    assert max(by_tier[TIER_EDGE]) < min(by_tier[TIER_RELAY])
    assert max(by_tier[TIER_RELAY]) < min(by_tier[TIER_HUB])
    # at_risk is a schedule: soonest first.
    times = [impact.time_to_dark_s[s] for s in impact.at_risk]
    assert times == sorted(times)


def test_mains_powered_sites_are_never_at_risk(real_world):
    """Nothing on mains has a finite time to dark, however deep in the tree it sits."""
    impact = real_world.calculate_cascade_impact(())
    assert impact.at_risk == ()
    assert all(v == float("inf") for v in impact.time_to_dark_s.values())


def test_battery_remainder_overrides_the_tier_default():
    world = chain_world()
    impact = world.calculate_cascade_impact(
        (), on_battery=["R"], battery_remaining_s={"R": 120.0})
    # R dies in two minutes, and everything behind it dies with it, not sooner.
    assert impact.time_to_dark_s["R"] == 120.0
    assert impact.time_to_dark_s["E1"] == 120.0
    assert impact.time_to_dark_s["E2"] == 120.0
    assert impact.time_to_dark_s["H"] == float("inf")


def test_restoration_order_fixes_the_most_valuable_site_first():
    world = chain_world()
    order = world.asset_graph.restoration_order(["R", "E2"])
    assert order == ["R", "E2"], "restoring the relay relights more of the chain"


# -- degenerate scenes --------------------------------------------------


def test_world_with_no_towers_has_an_empty_graph():
    world = make_world(specs=[{"id": "b1", "cx": 0.0, "cy": 0.0,
                               "size": 10.0, "height": 10.0}])
    assert world.asset_graph.graph.number_of_nodes() == 0
    impact = world.calculate_cascade_impact("nobody")
    assert impact.dark == () and impact.at_risk == ()


def test_single_tower_world_is_its_own_hub():
    world = make_world(
        specs=[{"id": "b1", "cx": 0.0, "cy": 0.0, "size": 10.0, "height": 10.0}],
        towers=[{"id": "only", "x": 0.0, "y": 0.0, "antenna_height": 30.0}],
    )
    assert world.asset_graph.hubs == ("only",)
    assert world.calculate_cascade_impact("only").dark == ("only",)
