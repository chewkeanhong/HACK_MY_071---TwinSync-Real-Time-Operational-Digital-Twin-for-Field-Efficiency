"""Root-cause attribution over a localised cluster.

ST-DBSCAN answers *"are these alarms one incident?"*. It cannot answer *"which one is
the cause?"*, and that is not an oversight to be patched -- it is structural. DBSCAN's
core/border reasoning requires a **symmetric** neighbour relation, which is exactly why
:data:`twinsync.stdbscan.COMPATIBLE_FAMILIES` is forced symmetric and says so. A
relation that could express "A caused B" would make cluster membership depend on visit
order and stop being DBSCAN.

So direction has to come from a second question, asked of a different structure. The
twin already holds the answer: :class:`twinsync.world.AssetGraph` knows who feeds whom,
and every alarm carries the instant it fired. Between them they give the two things a
cause must satisfy:

* **Topology.** A cause sits upstream of its effect. If KL-03 feeds KL-13 and both are
  alarming, KL-03 is the candidate head and KL-13 is a candidate symptom -- never the
  reverse, because the dependency only points one way.
* **Time.** A cause cannot post-date its effect.

The operational payoff is a crew instruction rather than a label: **go to the head**.
Two alarms with a dependency between them are one fault and one symptom, so a crew that
starts at the symptom repairs a site that was never broken and watches it re-alarm
behind them. Note what this does *not* claim -- the second truck roll is saved by the
dispatcher's existing proximity batching, which books that saving itself. What
attribution adds is the order the crew works in, and the confidence that batching these
two is safe rather than merely convenient.

**What this module refuses to do** matters as much as what it does. If no member of a
cluster is upstream of any other, there is no causal story and :func:`attribute` returns
``source=None``. Proximity is not causation, and a cluster of two co-located faults with
no dependency between them really is two jobs. That refusal is the counterpart to
ST-DBSCAN's ``ISOLATED`` label: both exist so the system can say "I have nothing to add
here" instead of manufacturing a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

# Hubs explain more than relays, relays more than edge sites. Only consulted when
# topology and time have both failed to separate two candidates, which on a derived
# hierarchy means two sites that alarmed in the same instant and feed neither
# other -- rare, but it must resolve to something rather than to input order.
TIER_RANK = {"hub": 0, "relay": 1, "edge": 2}
DEFAULT_TIER_RANK = 2

# How the winner was decided. Surfaced to the UI because "KL-03 is upstream of KL-13"
# and "KL-03 alarmed first" are very different strengths of claim, and an operator is
# entitled to know which one they are being handed.
BASIS_TOPOLOGICAL = "topological"
BASIS_TEMPORAL = "temporal"
BASIS_TIER = "tier"
BASIS_TIE = "tie"

# Where one incident sits in its cluster's chain.
#
# * source     -- has symptoms under it. This is the site the van goes to.
# * downstream -- a symptom of a source. Fixing the head clears it; no van.
# * peer       -- clustered, but no dependency was found. Its own job.
# * sole       -- not in any cluster at all. ST-DBSCAN's ISOLATED verdict. Set by the
#                 caller, since attribution is never asked about a cluster of one.
ROLE_SOURCE = "source"
ROLE_DOWNSTREAM = "downstream"
ROLE_PEER = "peer"
ROLE_SOLE = "sole"


@dataclass(frozen=True)
class Attribution:
    """Which member of a cluster is the head, and why.

    ``source`` is ``None`` when the cluster has no internal dependency -- see the module
    docstring. ``downstream`` is then empty and every member is a ``peer``: still one
    incident in ST-DBSCAN's sense, but not one job.
    """

    cluster_id: str
    source: str | None
    downstream: tuple[str, ...] = ()
    hops: dict[str, int] = field(default_factory=dict)
    basis: str = BASIS_TIE
    reason: str = ""

    @property
    def attributed(self) -> bool:
        """True when a head was actually identified."""
        return self.source is not None

    @property
    def symptoms(self) -> int:
        """How many alarms in this cluster are consequences rather than causes.

        Deliberately *not* called a truck-roll saving. The dispatcher already batches
        nearby jobs onto one trip on distance alone and books that saving itself; what
        attribution adds on top is the repair order -- go to the head, because fixing a
        symptom first leaves the fault in place and the site re-alarms behind you.
        """
        return len(self.downstream)

    def role_of(self, tower_id: str) -> str:
        """This site's part in the chain. ``attribute`` never returns a source with no
        downstream, so every attributed cluster has a head and at least one symptom."""
        if self.source is None:
            return ROLE_PEER
        if tower_id == self.source:
            return ROLE_SOURCE
        if tower_id in self.downstream:
            return ROLE_DOWNSTREAM
        return ROLE_PEER

    def describe(self) -> str:
        if self.source is None:
            return "no transport dependency -- co-located, not one job"
        if not self.downstream:
            return f"source {self.source}"
        tail = ", ".join(self.downstream)
        return f"source {self.source} -- {tail} downstream, fix the head first"


def _descendants_within(graph: nx.DiGraph, node: str, members: set[str]) -> set[str]:
    """Cluster members strictly downstream of ``node``.

    Uses the full graph rather than the subgraph induced on the cluster: a cause can
    reach its symptom through sites that are perfectly healthy, and usually does. KL-03
    feeding KL-13 directly is the easy case; KL-03 feeding it through a relay nobody has
    complained about is the same causal story and must not be missed because the
    intermediate hop is not itself alarming.
    """
    if node not in graph:
        return set()
    return (nx.descendants(graph, node) & members) - {node}


def attribute(asset_graph, cluster_id: str, members, alarm_times: dict[str, float],
              *, profiles: dict[str, str] | None = None) -> Attribution:
    """Name the head of a localised cluster, or decline to.

    ``asset_graph`` is a :class:`twinsync.world.AssetGraph`; ``members`` are the tower
    ids ST-DBSCAN grouped; ``alarm_times`` maps each to the instant it alarmed.

    Candidates are ranked lexicographically on four keys, strongest evidence first:

    1. **How many other members it is upstream of.** Most wins. This is the only key
       that can establish causation at all; the rest merely break ties beneath it.
    2. **Alarm time.** Earliest wins -- a cause cannot post-date its effect.
    3. **Tier.** Hub over relay over edge.
    4. **Id.** So the answer is deterministic rather than dependent on input order, the
       same reason :meth:`twinsync.world.AssetGraph.derive` breaks its altitude ties
       on id.

    The result is only reported as a cause if the winner is genuinely upstream of
    something. A cluster whose members do not depend on each other gets ``source=None``.
    """
    member_set = {m for m in members}
    if not member_set:
        return Attribution(cluster_id=cluster_id, source=None,
                           reason="empty cluster")

    graph = asset_graph.graph
    tiers = asset_graph.tier
    profiles = profiles or {}

    reach = {m: _descendants_within(graph, m, member_set) for m in member_set}

    def rank(member: str) -> tuple:
        return (
            -len(reach[member]),
            alarm_times.get(member, float("inf")),
            TIER_RANK.get(tiers.get(member, ""), DEFAULT_TIER_RANK),
            member,
        )

    ordered = sorted(member_set, key=rank)
    head = ordered[0]
    downstream = reach[head]

    if not downstream:
        # Nobody in this cluster is upstream of anybody else. ST-DBSCAN was right that
        # these alarms belong together in space and time; there is still no chain here,
        # and saying otherwise would be inventing a finding.
        return Attribution(
            cluster_id=cluster_id,
            source=None,
            basis=BASIS_TIE,
            reason=("no member feeds another -- co-located in space and time, "
                    "but independent jobs"),
        )

    basis = _basis(head, ordered, reach, alarm_times, tiers)
    hops = _hops_from(graph, head, downstream)
    ordered_downstream = tuple(sorted(downstream, key=lambda m: (hops.get(m, 99), m)))

    return Attribution(
        cluster_id=cluster_id,
        source=head,
        downstream=ordered_downstream,
        hops=hops,
        basis=basis,
        reason=_reason(head, ordered_downstream, hops, basis, alarm_times, profiles,
                       graph),
    )


def _basis(head: str, ordered: list[str], reach: dict[str, set[str]],
           alarm_times: dict[str, float], tiers: dict[str, str]) -> str:
    """Which key actually separated the winner from the runner-up."""
    if len(ordered) < 2:
        return BASIS_TOPOLOGICAL
    rival = ordered[1]
    if len(reach[head]) != len(reach[rival]):
        return BASIS_TOPOLOGICAL
    if alarm_times.get(head) != alarm_times.get(rival):
        return BASIS_TEMPORAL
    if tiers.get(head) != tiers.get(rival):
        return BASIS_TIER
    return BASIS_TIE


def _hops_from(graph: nx.DiGraph, source: str, targets: set[str]) -> dict[str, int]:
    """Shortest dependency distance from the head to each symptom.

    One hop means the symptom sits directly on the head's feed, which is the case an
    operator can act on without reading a diagram.
    """
    if source not in graph:
        return {}
    lengths = nx.single_source_shortest_path_length(graph, source)
    return {t: int(lengths[t]) for t in sorted(targets) if t in lengths}


def _reason(head: str, downstream: tuple[str, ...], hops: dict[str, int], basis: str,
            alarm_times: dict[str, float], profiles: dict[str, str],
            graph: nx.DiGraph) -> str:
    """One sentence a dispatcher can act on without opening the topology."""
    profile = profiles.get(head)
    head_label = f"{head} ({profile})" if profile else head

    parts = []
    for member in downstream:
        hop = hops.get(member)
        if hop == 1:
            role = graph.edges[head, member].get("role", "primary") \
                if graph.has_edge(head, member) else "primary"
            parts.append(f"{member} sits on its {role} feed")
        else:
            parts.append(f"{member} is {hop} hops downstream")

    lead = f"{head_label} is the head"
    if basis == BASIS_TEMPORAL:
        lead += ", upstream and alarmed first"
    elif basis == BASIS_TOPOLOGICAL:
        lead += ", upstream of every other alarm here"

    # Deliberately stops at the causal explanation. What it is worth in truck rolls is
    # the caller's sentence to write, because the same reason reads differently in a
    # dispatcher's log, an incident card and a KPI tile.
    return f"{lead}: {'; '.join(parts)}"
