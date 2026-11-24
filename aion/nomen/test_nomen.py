"""NOMEN unit tests — CPU only, seeded, stubs. NO network, NO model calls.

Proves (owner brief, 2026-11-25) every NOMEN design law:
  1. registry immutability       — entity ID + verification_status never change under rename
  2. rename preserves lineage    — every past name + provenance survives the rename
  3. past-name resolution        — an old name resolves to the canonical entity; ambiguity reported
  4. collision detection         — both entities SURVIVE resolution with intact lineages
  5. graph branch/merge/reverse  — evolution is a GRAPH: branches, merges, reversals, simultaneity
  6. self-naming provenance      — teacher-assisted vs internal decisions recorded in the lineage
  7. persistence round-trip      — registry + graph + tracker + collision log restore EXACTLY
  8. safeguard                   — renaming an IMAGINED entity leaves it imagined (naming cannot
                                   promote imagined -> verified; only explicit reclassify can)
Run:  python -m pytest aion/nomen/test_nomen.py -q   (from the repo root)
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aion.nomen.collision_detector import CollisionDetector  # noqa: E402
from aion.nomen.entity_registry import EntityRegistry  # noqa: E402
from aion.nomen.evolution_tracker import EvolutionTracker, TransitionEvent  # noqa: E402
from aion.nomen.identity_resolver import IdentityResolver  # noqa: E402
from aion.nomen.lineage_graph import LineageGraph, generate_edge_label  # noqa: E402
from aion.nomen.name_generator import (  # noqa: E402
    NameGenerator, TeacherNameAssist, generate_name)
from aion.nomen.name_memory import NomenStore  # noqa: E402
from aion.nomen.self_naming import SelfIdentity  # noqa: E402

FIXED_TS = 1700000000.0
CLOCK = lambda: FIXED_TS                     # noqa: E731 — deterministic clock


class StepClock:
    """Clock that steps one second per call: earlier registration = earlier claim ts,
    so collision keepers are decided by registration order, deterministically."""

    def __init__(self, start: float = FIXED_TS):
        self.value = float(start)

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def seeded_registry() -> EntityRegistry:
    return EntityRegistry(seed=42, clock=CLOCK)


# --------------------------------------------------------------------------- #
# stubs: a duck-typed teacher chat client (the aion/dialogue seam, zero network)
# --------------------------------------------------------------------------- #

class StubChatClient:
    """Mimics TeacherClient.chat(): returns strict JSON for the naming protocol."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.messages_seen = []

    def chat(self, messages):
        self.messages_seen.append([m["content"] for m in messages])
        return json.dumps({"action": "name", "candidates": self.candidates})


# --------------------------------------------------------------------------- #
# 1 + 8. registry immutability and the safeguard
# --------------------------------------------------------------------------- #

class TestRegistryImmutability:
    def test_id_and_verification_status_immune_to_rename(self):
        reg = seeded_registry()
        ent = reg.register("mol", "world", verification_status="observed")
        eid = ent.entity_id
        status = ent.verification_status
        for new_name in ("tavi", "soren", "kelu"):
            reg.rename(ent, new_name, provenance="internal", reason="grew")
            assert ent.entity_id == eid, "rename must never change the identity"
            assert ent.verification_status == status, "rename must never touch status"
        assert len(reg.by_id(eid).lineage) == 4

    def test_renames_are_recorded_with_provenance_and_reason(self):
        reg = seeded_registry()
        ent = reg.register("mol", "brain")
        reg.rename(ent, "tavi", provenance="teacher-proposed", reason="teacher offered")
        entry = ent.lineage[-1]
        assert (entry.seq, entry.name, entry.provenance, entry.reason) == \
            (2, "tavi", "teacher-proposed", "teacher offered")
        assert entry.ts == FIXED_TS

    def test_rename_to_same_name_rejected(self):
        reg = seeded_registry()
        ent = reg.register("mol", "model")
        with pytest.raises(ValueError):
            reg.rename(ent, "mol", provenance="internal")

    def test_verification_status_is_name_independent(self):
        """The three statuses exist independent of any name; observed/imagined/verified."""
        reg = seeded_registry()
        assert reg.register("a", "dream", verification_status="imagined").verification_status == "imagined"
        assert reg.register("b", "discovery", verification_status="verified").verification_status == "verified"
        assert reg.register("c", "memory").verification_status == "observed"

    def test_reclassify_is_the_only_status_path_and_demands_evidence(self):
        reg = seeded_registry()
        ent = reg.register("mol", "discovery", verification_status="imagined")
        with pytest.raises(ValueError):
            reg.reclassify(ent, "verified", "")          # no evidence -> refused
        reg.reclassify(ent, "verified", "confirmed by two independent observations")
        assert ent.verification_status == "verified"
        assert ent.verification_log[0]["from"] == "imagined"
        assert ent.verification_log[0]["evidence_note"] == \
            "confirmed by two independent observations"
        # and even now, renaming still cannot silently flip it back or forward
        reg.rename(ent, "tavi", provenance="internal")
        assert ent.verification_status == "verified"

    def test_safeguard_renaming_imagined_entity_leaves_it_imagined(self):
        """THE safeguard test: an imagined concept renamed twice is STILL imagined —
        naming is interpretation, never fact-promotion."""
        reg = seeded_registry()
        ent = reg.register("shaelu", "dream", verification_status="imagined",
                           reason="a dream-image, not yet an observation")
        reg.rename(ent, "norlu", provenance="internal", reason="dream deepened")
        reg.rename(ent, "vexon", provenance="internal", reason="dream recalled")
        assert ent.verification_status == "imagined"
        assert ent.verification_log == []
        resolved = IdentityResolver(reg).resolve("shaelu")
        assert resolved.entity.verification_status == "imagined"


# --------------------------------------------------------------------------- #
# 2. rename preserves lineage
# --------------------------------------------------------------------------- #

class TestLineagePreservation:
    def test_full_history_survives_three_renames(self):
        reg = seeded_registry()
        ent = reg.register("mol", "module")
        reg.rename(ent, "tavi", provenance="observation-derived")
        reg.rename(ent, "soren", provenance="teacher-proposed")
        reg.rename(ent, "kelu", provenance="internal")
        assert ent.history == ["mol", "tavi", "soren", "kelu"]
        assert [e.provenance for e in ent.lineage] == \
            ["initial", "observation-derived", "teacher-proposed", "internal"]
        assert ent.current_name == "kelu"
        assert ent.past_names == ["mol", "tavi", "soren"]

    def test_registry_lookups_by_current_and_past_names(self):
        reg = seeded_registry()
        ent = reg.register("mol", "model")
        reg.rename(ent, "tavi", provenance="internal")
        assert [e.entity_id for e in reg.by_name("tavi")] == [ent.entity_id]
        assert reg.by_name("mol") == []                # mol is no longer current
        assert [e.entity_id for e in reg.by_past_name("mol")] == [ent.entity_id]


# --------------------------------------------------------------------------- #
# 3. identity resolution
# --------------------------------------------------------------------------- #

class TestIdentityResolver:
    def test_resolve_by_id_current_name_and_past_name(self):
        reg = seeded_registry()
        ent = reg.register("mol", "capability")
        reg.rename(ent, "tavi", provenance="internal")
        res = IdentityResolver(reg)
        assert res.resolve(ent.entity_id).matched_as == "id"
        assert res.resolve("tavi").matched_as == "current-name"
        past = res.resolve("mol")
        assert past.status == "unique" and past.matched_as == "past-name"
        assert past.entity.entity_id == ent.entity_id
        assert "lineage preserved" in past.note

    def test_id_outranks_a_name_collision_with_an_id_string(self):
        reg = seeded_registry()
        ent = reg.register("mol", "model")
        other = reg.register(ent.entity_id, "model")     # pathological: named as an id
        res = IdentityResolver(reg).resolve(ent.entity_id)
        assert res.matched_as == "id" and res.entity_id == ent.entity_id
        assert other.entity_id != ent.entity_id

    def test_ambiguity_is_reported_not_guessed(self):
        reg = seeded_registry()
        reg.register("mol", "model")
        reg.register("mol", "model")                     # same current name, two entities
        resolver = IdentityResolver(reg)
        res = resolver.resolve("mol")
        assert res.status == "ambiguous" and len(res.entities) == 2
        assert "collision" in res.note
        assert resolver.resolve("no-such-thing").status == "unknown"

    def test_kind_filter_narrows_and_reports_when_it_cannot(self):
        reg = seeded_registry()
        reg.register("mol", "model")
        reg.register("mol", "dream")
        res = IdentityResolver(reg).resolve("mol", entity_kind="dream")
        assert res.status == "unique" and res.entity.entity_kind == "dream"
        reg.register("mol", "dream")
        assert IdentityResolver(reg).resolve("mol", entity_kind="dream").status == "ambiguous"


# --------------------------------------------------------------------------- #
# 4. collisions: unique-name enforcement, both survive
# --------------------------------------------------------------------------- #

class TestCollisionDetector:
    def _collided_registry(self):
        reg = EntityRegistry(seed=42, clock=StepClock())
        a = reg.register("mol", "model",
                         observed_properties={"texture": "warm", "tempo": "fast"})
        b = reg.register("mol", "world",
                         observed_properties={"colour": "dark"})
        reg.rename(a, "tavi", provenance="internal")
        reg.rename(a, "mol", provenance="internal")      # a comes back around to mol
        return reg, a, b

    def test_detects_current_name_claims(self):
        reg, a, b = self._collided_registry()
        det = CollisionDetector(reg)
        live = det.detect()
        assert [c.name for c in live] == ["mol"]
        assert set(live[0].entity_ids) == {a.entity_id, b.entity_id}

    def test_resolution_both_survive_with_intact_lineages(self):
        reg, a, b = self._collided_registry()
        det = CollisionDetector(reg)
        [res] = det.resolve_all()
        assert res.keeper_entity_id == a.entity_id       # a claimed mol first (StepClock)
        assert res.strategy == "lineage-qualifier"       # auto prefers the honest history
        # BOTH entities still exist, neither deleted, both lineages only lengthened
        assert len(reg) == 2
        assert b.history[0] == "mol" and b.current_name != "mol"
        assert a.history[0] == "mol" and "tavi" in a.history and a.current_name == "mol"
        assert a.current_name != b.current_name
        assert det.detect() == []
        renamed_entry = b.lineage[-1]
        assert renamed_entry.provenance == "collision-resolution"
        assert renamed_entry.basis["keeper_entity_id"] == a.entity_id
        # safeguard: resolution is a naming op — statuses untouched
        assert a.verification_status == b.verification_status == "observed"

    def test_new_coinage_strategy_when_history_cannot_qualify(self):
        reg = seeded_registry()
        reg.register("Mol", "model")                     # qualifier root == collision name
        reg.register("Mol", "world")
        det = CollisionDetector(reg)
        [res] = det.resolve_all(strategy="new-coinage")
        renamed_id = [r["entity_id"] for r in res.renames][0]
        renamed = reg.by_id(renamed_id)
        assert renamed.current_name != "Mol"
        assert reg.by_id(res.keeper_entity_id).current_name == "Mol"
        assert renamed.history[0] == "Mol" and len(renamed.history) == 2
        assert det.detect() == []

    def test_shadow_report_is_informational_only(self):
        reg = seeded_registry()
        ent = reg.register("mol", "model")
        other = reg.register("tavi", "world")
        reg.rename(ent, "kelu", provenance="internal")   # mol becomes a past name
        reg.rename(other, "mol", provenance="internal")  # another entity now claims it
        shadows = CollisionDetector(reg).collisions_including_past()
        assert len(shadows) == 1
        assert shadows[0]["of_entity_id"] == ent.entity_id
        assert shadows[0]["now_claimed_by"] == other.entity_id


# --------------------------------------------------------------------------- #
# 5. the evolution graph: branch / merge / reverse / simultaneous
# --------------------------------------------------------------------------- #

class TestLineageGraph:
    def _graph(self):
        g = LineageGraph()
        g.record_transition("E1", "seed", "sprout", reason="first observation")
        return g, g.node_of("E1", "sprout")

    def test_advance_transitions_and_generated_labels(self):
        g, sprout = self._graph()
        leaf = g.record_transition("E1", "sprout", "leaf", reason="grew")
        labels = [e.label for e in g.edges()]
        assert len(labels) == 2 and all(labels)
        # labels are GENERATED per entity+states, deterministic, from a verb vocabulary
        assert labels == [generate_edge_label("advance", "E1", "seed", "sprout"),
                          generate_edge_label("advance", "E1", "sprout", "leaf")]
        assert g.path(sprout.node_id, leaf.node_id) == [sprout.node_id, leaf.node_id]

    def test_branch_then_merge_preserves_both_sides(self):
        g, sprout = self._graph()
        c1 = g.branch(sprout.node_id, "limb-a", reason="explored one way")
        c2 = g.branch(sprout.node_id, "limb-b", reason="explored another way")
        assert g.is_branch_point(sprout.node_id)
        merged = g.merge(c1.node_id, c2.node_id, "crown", reason="developments met")
        assert sorted(merged.merged_from) == sorted([c1.node_id, c2.node_id])
        # both source nodes and ALL their edges survive the merge
        assert g.node(c1.node_id).name == "limb-a"
        assert g.node(c2.node_id).name == "limb-b"
        assert g.incoming(merged.node_id) and len(g.incoming(merged.node_id)) == 2
        assert g.path(c1.node_id, merged.node_id) is not None
        assert g.path(c2.node_id, merged.node_id) is not None

    def test_reverse_keeps_history_and_creates_a_cycle(self):
        g, sprout = self._graph()
        first_edge = g.edges()[0]
        back = g.reverse(first_edge.edge_id, reason="reconsidered")
        assert back.kind == "reverse"
        assert back.src == first_edge.dst and back.dst == first_edge.src
        assert first_edge in g.edges()                   # history never erased
        assert g.path(first_edge.dst, first_edge.src) is not None   # a genuine cycle now

    def test_simultaneous_transitions_share_group_and_tick(self):
        g, sprout = self._graph()
        n_a, n_b = g.record_simultaneous([
            {"entity_id": "E1", "from_name": "sprout", "to_name": "sense-a"},
            {"entity_id": "E1", "from_name": "sprout", "to_name": "sense-b"},
        ])
        groups = g.simultaneous_groups()
        assert len(groups) == 1
        (gid,) = groups
        edges = [g._by_edge_id[eid] for eid in groups[gid]]
        assert {e.dst for e in edges} == {n_a.node_id, n_b.node_id}
        assert len({e.tick for e in edges}) == 1         # one logical time, no ordering
        assert not g.is_branch_point(sprout.node_id)     # these are advances, not branches

    def test_no_ladder_api_exists(self):
        """Evolution is a graph: no stage numbers, no required sequence, cycles legal."""
        g, _ = self._graph()
        assert not any(hasattr(g, a) for a in ("stage", "next_stage", "set_stage",
                                               "advance_stage", "ladder"))
        # revisiting a name returns to the SAME node — reversal is a return, not a copy
        n1 = g.record_transition("E1", "sprout", "leaf")
        g.record_transition("E1", "leaf", "sprout")      # go back
        n2 = g.record_transition("E1", "sprout", "leaf")  # and return again
        assert n1.node_id == n2.node_id == g.node_of("E1", "leaf").node_id

    def test_history_is_the_entitys_own_walk_order(self):
        g, _ = self._graph()
        g.record_transition("E1", "sprout", "leaf")
        names = [n.name for n in g.history("E1")]
        assert names == ["seed", "sprout", "leaf"]
        assert g.history("nobody") == []


# --------------------------------------------------------------------------- #
# 6. names emerge from observation; teacher seam is optional
# --------------------------------------------------------------------------- #

class TestNameGenerator:
    def test_same_observation_same_name_any_machine(self):
        n1 = generate_name("eid-1", "model", ["warm", "fast"])
        n2 = generate_name("eid-1", "model", ["fast", "warm"])   # order-independent
        assert n1.name == n2.name
        assert n1.basis["properties"] == {"fast": "vi", "warm": "ta"}
        # stems compose the name; properties sort first, so fast(vi)+warm(ta) = "Vita"
        assert n1.name.startswith("Vita")

    def test_different_observation_different_name(self):
        a = generate_name("eid-1", "model", ["warm", "fast"]).name
        b = generate_name("eid-1", "model", ["cold", "dark"]).name
        c = generate_name("eid-1", "dream", ["warm", "fast"]).name
        assert len({a, b, c}) == 3

    def test_unseen_properties_still_compose(self):
        n = generate_name("eid-9", "discovery", ["totally-novel-observation"])
        assert n.name and n.basis["properties"]

    def test_collision_aware_generation_avoids_taken(self):
        first = generate_name("eid-1", "model", ["warm", "fast"])
        second = generate_name("eid-1", "model", ["warm", "fast"], taken={first.name})
        assert second.name != first.name
        assert second.basis["variations"] >= 1

    def test_generator_proposes_internal_candidates_without_any_hook(self):
        gen = NameGenerator(assist=None)                 # later independence: no teacher
        props = ["warm", "echoing"]
        proposals = gen.propose("eid-1", "brain", props, k=3)
        assert len(proposals) == 3
        assert all(p.provenance == "observation-derived" for p in proposals)
        assert len({p.name for p in proposals}) == 3

    def test_teacher_hook_proposals_marked_teacher_proposed(self):
        assist = TeacherNameAssist(StubChatClient(["Zarvi", "Oluen"]))
        gen = NameGenerator(assist=assist)
        proposals = gen.propose("eid-1", "self", ["bright", "quiet"], k=2)
        provenances = {p.name: p.provenance for p in proposals}
        assert provenances["Zarvi"] == "teacher-proposed"
        assert provenances["Oluen"] == "teacher-proposed"
        assert any(p.provenance == "observation-derived" for p in proposals)
        # the adapter really went through the chat seam (structured protocol)
        assert assist._parse(assist.client.chat([]))["action"] == "name"


# --------------------------------------------------------------------------- #
# 6b. the instance's own evolution (task-21 ledger seam)
# --------------------------------------------------------------------------- #

class TestEvolutionTracker:
    def _tracker(self):
        reg = seeded_registry()
        graph = LineageGraph()
        slf = SelfIdentity(reg, graph, "7f3a")
        return EvolutionTracker(reg, graph, slf.entity_id, clock=CLOCK), slf

    def test_non_renaming_transition_attaches_to_current_node(self):
        tracker, slf = self._tracker()
        node = tracker.record_transition(TransitionEvent(
            kind="observed_world", description="first world observed",
            observed_properties={"warm": "yes", "echoing": "yes"}))
        assert node.name == slf.name
        assert node.events[-1]["kind"] == "observed_world"
        assert tracker.events[-1]["node_id"] == node.node_id

    def test_renaming_transition_moves_the_lineage_and_graph(self):
        tracker, slf = self._tracker()
        old_id, old_name = slf.entity_id, slf.name
        node = tracker.record_transition(
            TransitionEvent(kind="consolidated", description="consolidated overnight"),
            new_name="Auron-7f3a", graph_kind="advance")
        assert slf.name == "Auron-7f3a" and slf.entity_id == old_id
        assert slf.entity.verification_status == "observed"
        assert old_name in slf.entity.past_names
        assert node.name == "Auron-7f3a"
        assert [n.name for n in tracker.history()] == [old_name, "Auron-7f3a"]


# --------------------------------------------------------------------------- #
# 6c. self-naming provenance
# --------------------------------------------------------------------------- #

class TestSelfNaming:
    def _identity(self):
        reg = seeded_registry()
        graph = LineageGraph()
        slf = SelfIdentity(reg, graph, "7f3a")
        reg.observe(slf.entity_id, {"tempo": "fast", "texture": "warm"})
        return reg, graph, slf

    def test_initial_identity_is_supplied(self):
        reg, graph, slf = self._identity()
        assert slf.name == "AION-7f3a"
        assert slf.entity.lineage[0].provenance == "initial"
        assert slf.entity.entity_kind == "self"

    def test_teacher_assisted_choice_recorded_with_provenance(self):
        reg, graph, slf = self._identity()
        assist = TeacherNameAssist(StubChatClient(["Zarvi", "Oluen"]))
        candidates = slf.propose_candidates(k=2, teacher_hook=assist)
        assert any(p.provenance == "teacher-proposed" for p in candidates)
        chosen = next(p.name for p in candidates if p.provenance == "teacher-proposed")
        event = slf.self_naming_event(candidates, chosen, "teacher-assisted",
                                      reason="early, teacher-guided")
        assert event.deciding_provenance == "teacher-assisted"
        assert event.old_name == "AION-7f3a" and event.chosen == chosen
        entry = slf.entity.lineage[-1]
        assert entry.provenance == "teacher-assisted"
        assert chosen in entry.basis["candidates"]
        assert slf.entity.entity_id == reg.by_name(chosen)[0].entity_id

    def test_internal_choice_and_derived_event_log(self):
        reg, graph, slf = self._identity()
        slf.self_naming_event(["Zarvi"], "Zarvi", "teacher-assisted")
        candidates = slf.propose_candidates(k=2, include_teacher=False)  # no hook at all
        chosen = candidates[0].name
        slf.self_naming_event(candidates, chosen, "internal", reason="independent now")
        events = slf.events()
        assert [e.deciding_provenance for e in events] == ["teacher-assisted", "internal"]
        assert events[-1].old_name == "Zarvi" and events[-1].chosen == chosen
        assert slf.entity.verification_status == "observed"   # naming never touches it

    def test_choosing_outside_candidates_is_refused(self):
        reg, graph, slf = self._identity()
        with pytest.raises(ValueError):
            slf.self_naming_event(["a", "b"], "c", "internal")
        with pytest.raises(ValueError):
            slf.self_naming_event(["a"], "a", "decided-by-vibes")  # provenance is typed


# --------------------------------------------------------------------------- #
# 7. persistence round-trip is EXACT
# --------------------------------------------------------------------------- #

class TestPersistenceRoundTrip:
    def _rich_world(self, tmp_path):
        reg = seeded_registry()
        graph = LineageGraph()
        gen = NameGenerator(assist=TeacherNameAssist(StubChatClient(["Zarvi", "Oluen"])))
        slf = SelfIdentity(reg, graph, "7f3a", name_generator=gen)
        reg.observe(slf.entity_id, {"tempo": "fast", "texture": "warm"})
        cands = slf.propose_candidates(k=2)
        slf.self_naming_event(cands, next(p.name for p in cands
                                          if p.provenance == "teacher-proposed"),
                              "teacher-assisted")
        tracker = EvolutionTracker(reg, graph, slf.entity_id, clock=CLOCK)
        tracker.record_transition(TransitionEvent(kind="observed_world",
                                                  description="world seen"))
        tracker.record_transition(TransitionEvent(kind="consolidated",
                                                  description="settled"),
                                  new_name="Auron-7f3a")
        # other entities: rename cycles, collisions, imagined dreams, a branch/merge graph
        world = reg.register("mol", "world", observed_properties={"colour": "dark"})
        reg.rename(world, "tavi", provenance="observation-derived")
        reg.rename(world, "mol", provenance="internal")
        dream = reg.register("shaelu", "dream", verification_status="imagined")
        reg.rename(dream, "norlu", provenance="internal")
        model_a = reg.register("mol", "model")
        det = CollisionDetector(reg, name_generator=gen)
        det.resolve_all()
        g2 = LineageGraph()
        root = g2.ensure_node(world.entity_id, "mol", tick=1, note="root state")
        b1 = g2.branch(root.node_id, "tavi")
        b2 = g2.branch(root.node_id, "kelu")
        merged = g2.merge(b1.node_id, b2.node_id, "soren")
        g2.reverse(g2.edges()[0].edge_id)
        g2.record_simultaneous([
            {"entity_id": dream.entity_id, "from_name": "shaelu", "to_name": "sense-a"},
            {"entity_id": dream.entity_id, "from_name": "shaelu", "to_name": "sense-b"},
        ])
        path = os.path.join(str(tmp_path), "nomen_state.json")
        store = NomenStore(reg, graph, tracker=tracker, collisions=det)
        payload = store.save(path)
        return store, path, payload, slf, dream, merged

    def test_round_trip_is_exact(self, tmp_path):
        store, path, payload, *_ = self._rich_world(tmp_path)
        loaded = NomenStore.load(path)
        assert loaded.equals(store), "round-trip must restore the naming world EXACTLY"
        assert loaded.dump_canonical() == store.dump_canonical()
        # structural spot-checks on the loaded objects
        assert loaded.graph.equals(store.graph)
        assert loaded.tracker.equals(store.tracker)
        assert loaded.collisions.equals(store.collisions)
        assert json.loads(json.dumps(payload)) == json.loads(loaded.dump_canonical())

    def test_loaded_registry_answers_lookups_identically(self, tmp_path):
        store, path, *_ , slf, dream, _ = self._rich_world(tmp_path)
        loaded = NomenStore.load(path)
        reg2 = loaded.registry
        # past-name resolution survives restart ("tavi" was renamed away; exactly one
        # entity once carried it, the collision keeper still holds "mol" as current)
        res = IdentityResolver(reg2).resolve("tavi")
        assert res.status == "unique" and res.matched_as == "past-name"
        assert IdentityResolver(reg2).resolve("mol").matched_as == "current-name"
        # the instance's lineage + self-naming events survive restart
        slf2 = SelfIdentity.restore(reg2, loaded.graph, slf.entity_id)
        assert slf2.name == "Auron-7f3a"
        assert [e.deciding_provenance for e in slf2.events()] == ["teacher-assisted"]
        assert slf2.events()[0].candidates  # the candidate list survived in the basis
        # and the safeguard survives: the dream is still imagined after restore
        assert reg2.by_id(dream.entity_id).verification_status == "imagined"

    def test_collision_log_survives_and_detect_is_still_empty(self, tmp_path):
        store, path, *_ = self._rich_world(tmp_path)
        loaded = NomenStore.load(path)
        assert loaded.collisions.log, "resolved-collision history must persist"
        assert CollisionDetector(loaded.registry).detect() == []
