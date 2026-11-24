"""NOMEN lineage graph: evolution is a GRAPH, not a ladder.

OWNER DESIGN LAW (NOMEN, 2026-11-25): AION records its own transitions; developments
can be simultaneous, reverse, branch, and merge. There is NO stage sequence here and no
ladder API — no "stage N" exists anywhere in this module. The owner's example chain
(observed -> recognised -> connected -> consolidated) is ONE possible path that EMERGES
from successive recorded transitions; nothing pre-orders them.

Representation:
  nodes  = NAMED STATES. A node is (entity_id, name) — one named state of one entity.
           Revisiting a name returns to the SAME node, so a reversal literally walks
           back to the earlier named state and the graph grows a cycle.
  edges  = TRANSITIONS between named states, each carrying a GENERATED label (a verb
           phrase produced from the entity + endpoint states hash — per-entity, not a
           hardcoded table) plus a kind: advance | branch | merge | reverse.

  branch       a node gains a second (or more) outgoing branch edge; a BRANCH POINT is
               any node with >= 2 outgoing branch edges (is_branch_point). No ordering
               between siblings is implied.
  merge        merged node records merged_from = [a, b]; BOTH source nodes and their
               full lineages survive untouched — merge joins developments, it does not
               delete either.
  reverse      adds the dst->src edge of kind "reverse". The original edge is kept:
               history is never erased, the cycle simply exists.
  simultaneous edges created with the SAME group_id (record_simultaneous assigns one)
               share a group and are created at the same logical tick; that is the whole
               representation — simultaneous developments are siblings of one tick, not
               ordered steps.

Determinism: node ids are sequential (n000001...), edge labels come from md5 hashes of
(entity_id, endpoints, kind) — same inputs -> same label, everywhere.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

EDGE_KINDS = ("advance", "branch", "merge", "reverse")

# primitive verb vocabulary per edge kind. The LABEL itself is GENERATED per edge:
# the hash of (entity_id, from, to, kind) picks the verb, so two entities passing
# through the same state names do not produce identical transition records.
VERBS = {
    "advance": ("observed", "recognised", "connected", "consolidated", "refined",
                "stabilised", "extended", "deepened"),
    "branch": ("diverged", "branched", "explored", "split"),
    "merge": ("merged", "converged", "unified", "absorbed"),
    "reverse": ("reverted", "receded", "unbound", "reconsidered"),
}


def _hash_int(*parts) -> int:
    return int(hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest(), 16)


def generate_edge_label(kind: str, entity_id: str, from_name: str, to_name: str) -> str:
    """Generated transition label — per-entity, deterministic, never a fixed ladder."""
    verbs = VERBS[kind]
    return verbs[_hash_int(entity_id, from_name, to_name, kind) % len(verbs)]


@dataclass
class GraphNode:
    """One named state of one entity."""

    node_id: str
    entity_id: str
    name: str
    tick: int                                  # logical creation time
    first_seq: int
    note: str = ""
    merged_from: List[str] = field(default_factory=list)   # node ids (for merge nodes)
    events: List[Dict] = field(default_factory=list)       # developmental events recorded
    # against this named state (written by evolution_tracker; the naming layer only)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.entity_id, self.name)

    def to_dict(self) -> Dict:
        return {"node_id": self.node_id, "entity_id": self.entity_id, "name": self.name,
                "tick": int(self.tick), "first_seq": int(self.first_seq), "note": self.note,
                "merged_from": list(self.merged_from), "events": [dict(e) for e in self.events]}

    @classmethod
    def from_dict(cls, d: Dict) -> "GraphNode":
        return cls(node_id=d["node_id"], entity_id=d["entity_id"], name=d["name"],
                   tick=int(d["tick"]), first_seq=int(d["first_seq"]),
                   note=d.get("note", ""), merged_from=list(d.get("merged_from", [])),
                   events=[dict(e) for e in d.get("events", [])])


@dataclass
class GraphEdge:
    """One recorded transition between two named states."""

    edge_id: str
    src: str
    dst: str
    entity_id: str
    kind: str                       # advance | branch | merge | reverse
    label: str                      # GENERATED per-edge verb phrase
    reason: str
    group_id: Optional[str] = None  # shared by simultaneously recorded edges
    tick: int = 0

    def to_dict(self) -> Dict:
        return {"edge_id": self.edge_id, "src": self.src, "dst": self.dst,
                "entity_id": self.entity_id, "kind": self.kind, "label": self.label,
                "reason": self.reason, "group_id": self.group_id, "tick": int(self.tick)}

    @classmethod
    def from_dict(cls, d: Dict) -> "GraphEdge":
        return cls(edge_id=d["edge_id"], src=d["src"], dst=d["dst"],
                   entity_id=d["entity_id"], kind=d["kind"], label=d["label"],
                   reason=d.get("reason", ""), group_id=d.get("group_id"),
                   tick=int(d.get("tick", 0)))


class LineageGraph:
    """The evolution graph over named states. Deliberately NOT a ladder API."""

    def __init__(self):
        self._nodes: Dict[Tuple[str, str], GraphNode] = {}      # (entity_id, name) -> node
        self._by_node_id: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._by_edge_id: Dict[str, GraphEdge] = {}
        self._tick = 0
        self._node_seq = 0
        self._edge_seq = 0
        self._group_seq = 0

    # -- nodes ------------------------------------------------------------------ #

    def ensure_node(self, entity_id: str, name: str, tick: Optional[int] = None,
                    note: str = "") -> GraphNode:
        """The named state (entity_id, name): created once, reused forever after —
        which is what makes reversal a genuine return to the earlier state."""
        key = (entity_id, name)
        node = self._nodes.get(key)
        if node is not None:
            return node
        self._tick = self._tick + 1 if tick is None else max(self._tick, int(tick))
        self._node_seq += 1
        node = GraphNode(node_id=f"n{self._node_seq:06d}", entity_id=entity_id, name=name,
                         tick=self._tick, first_seq=self._node_seq, note=note)
        self._nodes[key] = node
        self._by_node_id[node.node_id] = node
        return node

    def node(self, node_id: str) -> GraphNode:
        if node_id not in self._by_node_id:
            raise KeyError(f"no node {node_id!r}")
        return self._by_node_id[node_id]

    def node_of(self, entity_id: str, name: str) -> Optional[GraphNode]:
        return self._nodes.get((entity_id, name))

    def nodes_of(self, entity_id: str) -> List[GraphNode]:
        """All named states of one entity, in first-recorded order."""
        return sorted((n for n in self._nodes.values() if n.entity_id == entity_id),
                      key=lambda n: n.first_seq)

    # -- transitions -------------------------------------------------------------- #

    def record_transition(self, entity_id: str, from_name: str, to_name: str,
                          reason: str = "", kind: str = "advance",
                          group_id: Optional[str] = None,
                          tick: Optional[int] = None) -> GraphNode:
        """Record one transition between named states; returns the destination node.
        `tick` pins one logical time across a batch (record_simultaneous uses this so
        simultaneous transitions share their tick as well as their group)."""
        if kind not in EDGE_KINDS:
            raise ValueError(f"edge kind must be one of {EDGE_KINDS}, got {kind!r}")
        t = self._tick + 1 if tick is None else max(self._tick, int(tick))
        self._tick = t
        src = self.ensure_node(entity_id, from_name, tick=t)
        dst = self.ensure_node(entity_id, to_name, tick=t)
        if src.node_id == dst.node_id:
            raise ValueError("a transition needs two distinct named states; non-renaming "
                             "events attach to the current node via tracker events")
        self._edge_seq += 1
        edge = GraphEdge(
            edge_id=f"e{self._edge_seq:06d}", src=src.node_id, dst=dst.node_id,
            entity_id=entity_id, kind=kind, reason=reason,
            label=generate_edge_label(kind, entity_id, from_name, to_name),
            group_id=group_id, tick=t)
        self._edges.append(edge)
        self._by_edge_id[edge.edge_id] = edge
        return dst

    def record_simultaneous(self, transitions: List[Dict]) -> Tuple[GraphNode, ...]:
        """Record several transitions as SIMULTANEOUS: they share one group_id AND one
        logical tick. Representation law: simultaneity = shared group + shared tick,
        never an implied ordering."""
        self._group_seq += 1
        group = f"g{self._group_seq:06d}"
        shared_tick = self._tick + 1
        out = []
        for t in transitions:
            out.append(self.record_transition(group_id=group, tick=shared_tick, **t))
        return tuple(out)

    # -- graph operations (branch / merge / reverse) -------------------------------- #

    def branch(self, parent_node_id: str, new_name: str, reason: str = "") -> GraphNode:
        """Grow one branch from a named state. Call it twice from the same parent and
        the parent is a branch point — simultaneous/parallel developments, no order."""
        parent = self.node(parent_node_id)
        return self.record_transition(parent.entity_id, parent.name, new_name,
                                      reason=reason, kind="branch")

    def is_branch_point(self, node_id: str) -> bool:
        return sum(1 for e in self._edges if e.src == node_id and e.kind == "branch") >= 2

    def merge(self, node_a_id: str, node_b_id: str, merged_name: str,
              reason: str = "") -> GraphNode:
        """Merge two developments into one new named state. BOTH source nodes and all
        their edges/lineages survive; the merged node records merged_from = [a, b]."""
        a, b = self.node(node_a_id), self.node(node_b_id)
        merged = self.ensure_node(a.entity_id, merged_name,
                                  note=f"merge of {a.name!r} and {b.name!r}")
        merged.merged_from = sorted(set(merged.merged_from) | {a.node_id, b.node_id})
        for src in (a, b):
            if src.node_id == merged.node_id:
                continue
            self._edge_seq += 1
            edge = GraphEdge(
                edge_id=f"e{self._edge_seq:06d}", src=src.node_id, dst=merged.node_id,
                entity_id=src.entity_id, kind="merge", reason=reason,
                label=generate_edge_label("merge", src.entity_id, src.name, merged.name),
                tick=self._tick)
            self._edges.append(edge)
            self._by_edge_id[edge.edge_id] = edge
        return merged

    def reverse(self, edge_id: str, reason: str = "") -> GraphEdge:
        """Reverse one recorded transition: adds the opposite edge of kind 'reverse'.
        The original edge is KEPT — history is never erased, the graph simply cycles."""
        edge = self._by_edge_id.get(edge_id)
        if edge is None:
            raise KeyError(f"no edge {edge_id!r}")
        src = self.node(edge.src)
        dst = self.node(edge.dst)
        self._edge_seq += 1
        back = GraphEdge(
            edge_id=f"e{self._edge_seq:06d}", src=dst.node_id, dst=src.node_id,
            entity_id=edge.entity_id, kind="reverse", reason=reason,
            label=generate_edge_label("reverse", edge.entity_id, dst.name, src.name),
            tick=self._tick)
        self._edges.append(back)
        self._by_edge_id[back.edge_id] = back
        return back

    # -- reading ------------------------------------------------------------------- #

    def edges(self) -> List[GraphEdge]:
        return list(self._edges)

    def outgoing(self, node_id: str) -> List[GraphEdge]:
        return [e for e in self._edges if e.src == node_id]

    def incoming(self, node_id: str) -> List[GraphEdge]:
        return [e for e in self._edges if e.dst == node_id]

    def history(self, entity_id: str) -> List[GraphNode]:
        """Named states of one entity in first-recorded order (NOT a stage order —
        the order the entity itself walked)."""
        return self.nodes_of(entity_id)

    def path(self, src_node_id: str, dst_node_id: str) -> Optional[List[str]]:
        """BFS path of node ids from src to dst along recorded transitions (directed);
        None when no recorded path exists — which is an honest answer, not an error."""
        if src_node_id == dst_node_id:
            return [src_node_id]
        from collections import deque
        q, seen, parent = deque([src_node_id]), {src_node_id}, {}
        while q:
            cur = q.popleft()
            for edge in self.outgoing(cur):
                nxt = edge.dst
                if nxt in seen:
                    continue
                seen.add(nxt)
                parent[nxt] = cur
                if nxt == dst_node_id:
                    path, cur2 = [nxt], nxt
                    while cur2 != src_node_id:
                        cur2 = parent[cur2]
                        path.append(cur2)
                    return list(reversed(path))
                q.append(nxt)
        return None

    def simultaneous_groups(self) -> Dict[str, List[str]]:
        """group_id -> edge ids that were recorded as simultaneous."""
        groups: Dict[str, List[str]] = {}
        for e in self._edges:
            if e.group_id:
                groups.setdefault(e.group_id, []).append(e.edge_id)
        return groups

    # -- persistence (exact; file I/O lives in name_memory) -------------------------- #

    def to_dict(self) -> Dict:
        return {
            "tick": self._tick, "node_seq": self._node_seq, "edge_seq": self._edge_seq,
            "group_seq": self._group_seq,
            "nodes": [n.to_dict() for n in
                      sorted(self._nodes.values(), key=lambda n: n.first_seq)],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LineageGraph":
        g = cls()
        for nd in d.get("nodes", []):
            node = GraphNode.from_dict(nd)
            g._nodes[node.key] = node
            g._by_node_id[node.node_id] = node
        for ed in d.get("edges", []):
            edge = GraphEdge.from_dict(ed)
            g._edges.append(edge)
            g._by_edge_id[edge.edge_id] = edge
        g._tick = int(d.get("tick", 0))
        g._node_seq = int(d.get("node_seq", 0))
        g._edge_seq = int(d.get("edge_seq", 0))
        g._group_seq = int(d.get("group_seq", 0))
        return g

    def equals(self, other: "LineageGraph") -> bool:
        return self.to_dict() == other.to_dict()
