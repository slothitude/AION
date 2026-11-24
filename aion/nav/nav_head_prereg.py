"""NAV-DOMAIN decision head + d1-as-is decomposition - FROZEN PREREGISTRATION.

# ============================================================================
# PREREGISTRATION (frozen 2026-11-25, BEFORE dataset build, BEFORE training,
# BEFORE any eval run - do not edit after runs; amendments are appended with
# their own date, never folded silently)
# ============================================================================
#
# PURPOSE  Fix the measured NAV exam gap with two arms, both declared here
#          BEFORE any run:
#            ARM 1 (PRIMARY, owner amendment 2026-11-25): d1-2B AS-IS, zero
#                 training. The 5-way single-choice exam usage left d1 with a
#                 QUERY blind spot; the fix is USAGE - decompose the NAV
#                 decision into the card's typed primitives (noul gates +
#                 residual choice), one system_one call, one forward pass.
#            ARM 2 (SECONDARY): an economical NAV-domain trained head
#                 (logistic regression / gradient boosting, CPU, minutes)
#                 filling the domain gap the larql M6 head showed (0.200).
#
# OWNER AMENDMENTS IN EFFECT (2026-11-25, logged before any run):
#   A1. d1-as-is with decomposed card usage is the PRIMARY arm; training is
#       NOT a prerequisite for using d1.
#   A2. REDO RULE (applies to ALL scored backends): a scored decision whose
#       calibrated confidence lands in [0.49, 0.51] is a REDO QUESTION - the
#       item is re-presented with a fresh formulation from the frozen pool
#       below, at most 2 re-presentations (3 attempts max); the FINAL attempt
#       is scored, whatever it gives. Redos never change gold. Reported
#       per backend: redo_count (total re-presentations) and ambiguity_rate
#       (fraction of items that hit the band at least once). A high ambiguity
#       rate is itself a finding about the question's exactness.
#
# CLOSED ENUM: aion.nav.state.NAV_ACTIONS (frozen): QUERY|INFER|ANSWER|
#   ASK_TEACHER|STOP. Gold = aion.nav.backends.nav_decision_policy (the
#   preregistered graph rule table, imported read-only). Gold is NEVER shown
#   to any backend at decision time.
#
# ---------------------------------------------------------------------------
# PART 1 - DATASET (ARM 2 training data)
# ---------------------------------------------------------------------------
# Derivation: every state is a (world, memory regime, goal) triple over a
# generated chain world; features are STRUCTURAL (graph truth + memory
# coverage), never entity names. Gold per state comes from the graph rule
# table alone.
#
# WORLDS (frozen seed families; disjoint by construction):
#   TRAIN   worlds: 24 seeds, seed = 310001 + 37*i, i in 0..23, n_chains=8
#   HOLDOUT worlds: 12 seeds, seed = 420001 + 53*i, i in 0..11, n_chains=10
#   DEMO    worlds:  6 seeds, seed = 530001 + 29*i, i in 0..5,  n_chains=8
#   (outcome-feedback demo stream; never used for fitting)
#   SPLIT FREEZE DATE: 2026-11-25. The split is BY WORLD SEED: no holdout
#   example shares a world with any train example, and n_chains differs
#   (train 8 -> holdout 10) so holdout is a fresh formulation.
#
# MEMORY REGIMES (frozen weights, drawn per example from the example RNG):
#   empty    w=0.05  memory = {}
#   full_k   w=0.20  first k chains fully covered, k ~ U[1, n_chains]
#   prefix   w=0.20  first k chains fully covered (k ~ U[1, n//2]) plus the
#                    next j chains with their first two edges (j ~ U[0, 2])
#   sparse   w=0.35  each world edge included independently p ~ U[0.2, 0.8]
#   exam_like w=0.20 first ceil(n/2) chains full, next third prefix-2, rest
#                    uncovered (the original exam's shape)
#   Memory contents are ALWAYS true world relationships (no invented facts).
#
# GOAL DRAW (per example): fact goal with relation ~ U{operates, routes-to,
#   controls, produces} with p=0.6; reach goal with p=0.4. Subject ~ uniform
#   over (all world entities) + OUT_OF_WORLD_NAMES.
#
# OUT_OF_WORLD_NAMES (frozen, disjoint from the original exam's Velrix/
#   Ombaru/Skelth and from the fresh-variant names): Velmora, Quithra,
#   Zandrel, Ophrix, Bralune, Skarnet, Yolveth, Cindrath.
#
# SIZE + SPLIT (frozen): total target 4000 examples = train 3200 (640 per
#   class x 5) + holdout 800 (160 per class x 5). Class quotas are filled by
#   rejection sampling with the frozen RNG; duplicate (world_seed, goal_kind,
#   subject, relation, memory_digest) keys are dropped (first kept). Sampling
#   order: train split first, then holdout, separate RNG streams.
#   RNG SEEDS: train stream 90210, holdout stream 90211, demo stream 90212.
#
# FEATURES (frozen order; all graph-derivable, no identity features):
#   1  in_world              1/0
#   2  role_index            -1 out-of-world, 0 actor..4 output
#   3  goal_is_reach         1/0
#   4  goal_relation_index   0 operates,1 routes-to,2 controls,3 produces,
#                            -1 reach
#   5  relation_possible     1/0 role can emit the goal relation (reach:
#                            any outgoing)
#   6  next_hop_covered      1/0 goal's first-hop edge in memory
#   7  walk_hops_total       edges on the TRUE graph walk from subject
#   8  walk_hops_missing     of those, missing from memory
#   9  walk_fully_covered    1 iff total>0 and missing==0
#   10 n_unknown_evidence    exam/larql socket semantics: fact 0/1 by
#                            single-edge coverage; reach = missing hops
#   11 n_known_facts         memory size
#   12 memory_edge_coverage  |memory| / world edge count
#   13 subject_edges_in_memory  memory edges whose subject is this subject
#
# ---------------------------------------------------------------------------
# PART 2 - MODEL (ARM 2)
# ---------------------------------------------------------------------------
# Two preregistered candidates, BOTH trained on the train split only:
#   M1 logistic: sklearn Pipeline(StandardScaler, LogisticRegression(multi
#      class multinomial, lbfgs, C=1.0, max_iter=2000, random_state=90210))
#   M2 boost:    sklearn HistGradientBoostingClassifier(random_state=90210)
# SELECTION RULE (frozen BEFORE training): winner = higher accuracy on a
#   validation slice carved from TRAIN worlds only (last 15% of train rows,
#   deterministic index slice); ties -> M1 (simpler). The frozen HOLDOUT is
#   scored ONCE per candidate and never used for selection. Artifact saved to
#   aion/nav/artifacts/nav_decision_head.pkl (+ .meta.json).
#
# ---------------------------------------------------------------------------
# PART 3 - EXAMS (both arms)
# ---------------------------------------------------------------------------
# E1 ORIGINAL: aion.nav.exam.build_scenarios() verbatim (seed 20261125,
#    n_chains=8, 30 scenarios, 6 per class). read-only import.
# E2 FRESH-SEED HOLDOUT VARIANT (frozen recipe, mirrors E1's construction
#    scaled to a fresh world it has never seen):
#      world  = generate_world(seed=20261126, n_chains=10)
#      memory = chains 0-4 FULL (20 facts); chains 5-7 first two edges (6
#               facts); chains 8-9 uncovered
#      classes (6 each, 30 total, positional picks):
#        ANSWER       fact goals on fully covered chains 0-4: picks
#                     [(0,operates),(0,controls),(1,routes-to),(1,produces),
#                      (2,operates),(3,produces)]
#        INFER        reach goals, subjects = actors[0..3] + conduits[0..1]
#        QUERY        3 fact goals on uncovered chain 9 (operates, routes-to,
#                     controls) + 3 reach goals on actors of partial chains
#                     5,6,7
#        STOP         outputs of chains 0-5, alternating produces/controls
#        ASK_TEACHER  invented names FRESH POOL: (Quorvane,operates),
#                     (Quorvane,controls), (Threnn,routes-to),
#                     (Threnn,produces), (Malzik,operates), (Malzik,controls)
#    gold via nav_decision_policy; assert all five classes appear.
# SCORING (both exams): accuracy = decided-and-correct / scored; misses
#   (no_answer / unavailable / license_blocked) are MISSES, never guessed;
#   enum violations counted; redo_count and ambiguity_rate reported per
#   backend (amendment A2). Confidence per backend:
#     d1_asis: probability of the DECIDING answer (fired gate's P(yes), or
#              chosen residual option's probability)
#     trained head / larql head: max predict_proba of the chosen class
#     heuristic_floor: rule-determined, confidence = None -> exempt from the
#              redo band (no calibrated probability exists; declared)
#     tev1_0p8b: LICENSE-BLOCKED (card: weights license being finalized);
#              never runs, recorded as license_blocked
#
# ARM 1 DECOMPOSITION (d1-as-is, frozen BEFORE any run; card usage only):
#   ONE model.system_one(state_text, questions) call per attempt, four named
#   questions, single forward pass, zero generated tokens:
#     gate_in_world  (noul)  "does the subject exist in this world?"
#     gate_path      (noul)  "can the subject's role emit the goal relation?"
#     gate_covered   (noul)  "is what the goal needs already in memory?"
#     residual       (choice over ANSWER, INFER)
#   ROUTING (application logic, card-typed answers, threshold 0.5):
#     P(gate_in_world yes) < 0.5            -> ASK_TEACHER
#     else P(gate_path yes) < 0.5           -> STOP
#     else P(gate_covered yes) < 0.5        -> QUERY   (the blind-spot fix:
#                                            information-seeking is structural)
#     else residual choice winner           -> ANSWER / INFER
#   NOTE vs the amendment's illustrative wording ("choice over
#   ANSWER/INFER/STOP/ASK_TEACHER"): with the three structural gates, STOP and
#   ASK_TEACHER are already routed; the exactness law (A2 rationale: a
#   well-formed question should not offer impossible options) narrows the
#   residual to the only two remaining actions. Rationale frozen here.
#   REDO (A2): if the DECIDING probability is in [0.49, 0.51], re-present the
#   item with the next formulation (form 0 -> 1 -> 2, max 2 re-presentations)
#   and score the final attempt. All four probabilities recorded per attempt.
#
# REDO FORMULATION POOL (frozen; same pool for every text backend):
#   form 0  canonical: aion.nav.state.build_state_text + canonical questions
#   form 1  re-derived reverse-sorted memory listing + alternate phrasings
#   form 2  memory grouped by subject in world role order + alternate phrasings
#   Feature-vector heads (trained head, larql head) are formulation-INVARIANT:
#   their redo is recorded and bounded but cannot change the vector (declared;
#   expectation: redo_count>0 with unchanged choices).
#
# BACKEND ROWS (both exams, frozen):
#   d1_asis (PRIMARY)      Lappy only (fp32 CPU ~7.7 GB; card compliance:
#                          system_one, trust_remote_code=True, no sampling)
#   trained_head (ARM 2)   Rog, sklearn, CPU
#   heuristic_floor        reference line (validates the exam)
#   larql_classifier       reference: the larql M6 head via the owner-approved
#                          socket (expected ~0.2 - wrong domain, reported as is)
#   tev1_0p8b              reference row, LICENSE-BLOCKED, never runs
#
# LAWS HONORED: no fabricated numbers (every accuracy below is measured);
#   cards-only for Tev1/d1 references; CPU-only training (minutes); prereg
#   frozen before training; results written to aion/nav/artifacts/ (the
#   original aion/nav/results/ preregistered outputs stay untouched).
# ============================================================================
# END PREREGISTRATION
# ============================================================================

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from aion.worlds.generator import DIRECT_ASK, World, generate_world
from aion.nav.state import NavState, fact_goal_text, reach_goal_text
from aion.nav.backends import nav_decision_policy

# ---------------------------------------------------------------------------
# frozen constants (mirrored from the block above)
# ---------------------------------------------------------------------------

PREREG_DATE = "2026-11-25"
AMENDMENTS = (
    "2026-11-25 A1: d1-as-is decomposed usage is the PRIMARY arm (zero training).",
    "2026-11-25 A2: redo rule - calibrated confidence in [0.49,0.51] triggers a "
    "bounded re-presentation (max 2), final attempt scored; redo_count and "
    "ambiguity_rate reported per backend.",
)

# exam versions / artefact names
HEAD_EXAM_VERSION = "nav-head-exam-1.0"
ARTIFACT_DIR_NAME = "artifacts"

# world seed families (PART 1)
TRAIN_WORLD_SEEDS: Tuple[int, ...] = tuple(310001 + 37 * i for i in range(24))
HOLDOUT_WORLD_SEEDS: Tuple[int, ...] = tuple(420001 + 53 * i for i in range(12))
DEMO_WORLD_SEEDS: Tuple[int, ...] = tuple(530001 + 29 * i for i in range(6))
TRAIN_N_CHAINS = 8
HOLDOUT_N_CHAINS = 10
DEMO_N_CHAINS = 8
SPLIT_FREEZE_DATE = "2026-11-25"

# rng streams
TRAIN_RNG_SEED = 90210
HOLDOUT_RNG_SEED = 90211
DEMO_RNG_SEED = 90212
MODEL_SEED = 90210
VAL_SLICE_FRACTION = 0.15          # last 15% of train rows -> validation slice

# sizes (PART 1)
TRAIN_PER_CLASS = 640
HOLDOUT_PER_CLASS = 160
CLASSES: Tuple[str, ...] = ("ANSWER", "INFER", "QUERY", "STOP", "ASK_TEACHER")

OUT_OF_WORLD_NAMES: Tuple[str, ...] = (
    "Velmora", "Quithra", "Zandrel", "Ophrix", "Bralune", "Skarnet",
    "Yolveth", "Cindrath")
FRESH_TEACHER_NAMES: Tuple[str, ...] = ("Quorvane", "Threnn", "Malzik")

# memory regimes (PART 1): name -> weight
REGIME_WEIGHTS: Dict[str, float] = {
    "empty": 0.05, "full_k": 0.20, "prefix": 0.20, "sparse": 0.35,
    "exam_like": 0.20,
}
REGIME_NAMES: Tuple[str, ...] = tuple(REGIME_WEIGHTS)

# E2 fresh variant constants (PART 3)
FRESH_WORLD_SEED = 20261126
FRESH_N_CHAINS = 10

# redo rule (amendment A2)
REDO_BAND_LOW = 0.49
REDO_BAND_HIGH = 0.51
REDO_MAX_REPRESENTATIONS = 2      # => max 3 attempts
N_FORMS = 3

# PART 2 model spec strings (documentation; the trainer binds them)
MODEL_M1 = ("Pipeline(StandardScaler, LogisticRegression(multinomial, lbfgs, "
            "C=1.0, max_iter=2000, random_state=90210))")
MODEL_M2 = "HistGradientBoostingClassifier(random_state=90210)"

ADAPTATION_LR = 0.25               # frozen outcome-feedback learning rate

# PART 1 frozen feature order
FEATURE_NAMES: Tuple[str, ...] = (
    "in_world", "role_index", "goal_is_reach", "goal_relation_index",
    "relation_possible", "next_hop_covered", "walk_hops_total",
    "walk_hops_missing", "walk_fully_covered", "n_unknown_evidence",
    "n_known_facts", "memory_edge_coverage", "subject_edges_in_memory")

ROLE_ORDER: Tuple[str, ...] = ("actor", "conduit", "relay", "device", "output")
RELATION_ORDER: Tuple[str, ...] = ("operates", "routes-to", "controls", "produces")


# ---------------------------------------------------------------------------
# graph-derived state features (PART 1; the ONLY model inputs)
# ---------------------------------------------------------------------------

def _true_walk(world: World, subject: str) -> List[Tuple[str, str, str]]:
    """The true chain walk from subject as (subject, relation, object) edges;
    empty when the subject is out-of-world or terminal."""
    edges: List[Tuple[str, str, str]] = []
    node = subject
    seen = {node}
    while node in world.graph:
        rel, obj = world.graph[node][0]
        edges.append((node, rel, obj))
        node = obj
        if node in seen:
            break
        seen.add(node)
    return edges


def state_features(state: NavState) -> List[float]:
    """Frozen 13-feature structural vector for one NAV decision point.
    Everything is derived from the world graph + the declared memory set;
    entity names never enter the vector."""
    world = state.world
    in_world = 1.0 if state.subject_role() is not None else 0.0
    role_index = -1.0
    for i, role in enumerate(ROLE_ORDER):
        if in_world and state.subject_role() == role:
            role_index = float(i)
            break
    goal_is_reach = 1.0 if state.goal_kind == "reach" else 0.0
    rel_idx = -1.0
    if state.goal_kind == "fact" and state.relation in RELATION_ORDER:
        rel_idx = float(RELATION_ORDER.index(state.relation))

    from aion.nav.state import possible_relations
    poss = possible_relations(state.subject_role())
    if state.goal_kind == "fact":
        relation_possible = 1.0 if (poss and state.relation == poss[0]) else 0.0
    else:
        relation_possible = 1.0 if poss else 0.0

    walk = _true_walk(world, state.subject)
    n_hops = float(len(walk))
    n_missing = float(sum(1 for e in walk if e not in state.known_facts))
    next_hop_covered = 1.0 if (walk and walk[0] in state.known_facts) else 0.0
    walk_full = 1.0 if (n_hops > 0 and n_missing == 0.0) else 0.0
    if state.goal_kind == "fact":
        n_unknown = 0.0 if next_hop_covered else 1.0
    else:
        n_unknown = n_missing

    total_edges = float(len(world.relationships))
    n_known = float(len(state.known_facts))
    subj_edges = float(sum(1 for (s, _r, _o) in state.known_facts
                           if s == state.subject))

    return [
        in_world, role_index, goal_is_reach, rel_idx, relation_possible,
        next_hop_covered, n_hops, n_missing, walk_full, n_unknown,
        n_known, (n_known / total_edges if total_edges else 0.0), subj_edges,
    ]


# ---------------------------------------------------------------------------
# frozen formulation pool (PART 3 redo rule) - state text + question phrasings
# ---------------------------------------------------------------------------

_FACT_ALT: Dict[str, str] = {
    "operates": "{subject} operates whom?",
    "routes-to": "Where does {subject} route to?",
    "controls": "{subject} controls whom?",
    "produces": "What does {subject} produce?",
}
_FACT_ALT2: Dict[str, str] = {
    "operates": "Identify the entity that {subject} operates.",
    "routes-to": "Identify the relay that {subject} routes to.",
    "controls": "Identify the entity that {subject} controls.",
    "produces": "Identify what {subject} produces.",
}
_REACH_FORMS: Tuple[str, str, str] = (
    "What does {subject} reach?",
    "What is reachable from {subject}?",
    "Follow {subject}'s chain: where does it end?",
)


def goal_question_text(state: NavState, form: int) -> str:
    """Goal question under the frozen formulation pool (form 0 = canonical)."""
    if state.goal_kind == "fact":
        base = DIRECT_ASK[state.relation].format(subject=state.subject)
        if form == 1:
            return _FACT_ALT[state.relation].format(subject=state.subject)
        if form == 2:
            return _FACT_ALT2[state.relation].format(subject=state.subject)
        return base
    return _REACH_FORMS[min(form, 2)].format(subject=state.subject)


def render_state_text(state: NavState, form: int) -> str:
    """State text under the frozen formulation pool. All three forms present
    the SAME facts and the SAME goal; only ordering / wording differ."""
    lines: List[str] = []
    if form == 0:                                   # canonical (exam verbatim)
        if state.known_facts:
            lines.append("Known facts (memory):")
            for (s, r, o) in sorted(state.known_facts):
                lines.append(f"- {s} {r} {o}.")
        else:
            lines.append("Known facts (memory): none.")
        lines.append(f"Goal question: {goal_question_text(state, 0)}")
        return "\n".join(lines)
    if form == 1:                                   # re-derived, reverse order
        if state.known_facts:
            lines.append("Memory facts on record (re-derived, reverse order):")
            for (s, r, o) in sorted(state.known_facts, reverse=True):
                lines.append(f"- {s} {r} {o}.")
        else:
            lines.append("Memory facts on record: none.")
        lines.append(f"Question under evaluation: {goal_question_text(state, 1)}")
        return "\n".join(lines)
    # form 2: grouped by subject in world role order
    lines.append("Covered memory, grouped by subject:")
    grouped = False
    for role in ROLE_ORDER:
        for entity in getattr(state.world, role + "s"):
            edges = sorted((r, o) for (s, r, o) in state.known_facts
                           if s == entity)
            if edges:
                grouped = True
                lines.append(f"- {entity}: "
                             + "; ".join(f"{r} {o}" for (r, o) in edges) + ".")
    if not grouped:
        lines.append("Covered memory: empty.")
    lines.append(f"Goal being decided: {goal_question_text(state, 2)}")
    return "\n".join(lines)


_GATE_IN_WORLD: Tuple[str, str, str] = (
    "Is {subject} an entity that exists in this world?",
    "Does the entity named {subject} belong to this world?",
    "Based on the world described, is {subject} one of its entities?",
)
_GATE_PATH_FACT: Tuple[str, str, str] = (
    "In this world's role structure, can {subject} have a '{relation}' relationship?",
    "Is a '{relation}' relationship possible for the role of {subject}?",
    "Could {subject} ever be linked by '{relation}' in this world?",
)
_GATE_PATH_REACH: Tuple[str, str, str] = (
    "In this world's role structure, can {subject} have an outgoing relationship (a path)?",
    "Is any outgoing relationship possible for the role of {subject}?",
    "Could {subject} ever start a path in this world?",
)
_GATE_COVERED_FACT: Tuple[str, str, str] = (
    "Is the fact needed to answer the goal question already among the known memory facts?",
    "Do the known memory facts already contain the fact that answers the goal question?",
    "Is answering the goal question possible using only the listed memory facts?",
)
_GATE_COVERED_REACH: Tuple[str, str, str] = (
    "Is every hop of the path needed to answer the goal question already among the known memory facts?",
    "Do the known memory facts cover the whole path the goal question asks to walk?",
    "Can the full path for the goal question be walked using only the listed memory facts?",
)
_RESIDUAL_INSTRUCTIONS: Tuple[str, str, str] = (
    "The memory covers what the goal needs. Which operation serves the goal?",
    "Given that memory covers the goal's needs, choose the serving operation.",
    "With the goal's needs already covered, which single operation should run?",
)
RESIDUAL_CRITERIA: Dict[str, str] = {
    "ANSWER": "Answer directly from a single covered fact.",
    "INFER": "Compose the answer by walking covered multi-hop facts.",
}


def gate_in_world_text(state: NavState, form: int) -> str:
    return _GATE_IN_WORLD[min(form, 2)].format(subject=state.subject)


def gate_path_text(state: NavState, form: int) -> str:
    tpl = _GATE_PATH_REACH if state.goal_kind == "reach" else _GATE_PATH_FACT
    return tpl[min(form, 2)].format(subject=state.subject,
                                    relation=state.goal_relation())


def gate_covered_text(state: NavState, form: int) -> str:
    tpl = (_GATE_COVERED_REACH if state.goal_kind == "reach"
           else _GATE_COVERED_FACT)
    return tpl[min(form, 2)].format(subject=state.subject)


def residual_instructions(form: int) -> str:
    return _RESIDUAL_INSTRUCTIONS[min(form, 2)]


# ---------------------------------------------------------------------------
# E2 fresh-seed holdout variant (frozen recipe from PART 3)
# ---------------------------------------------------------------------------

def build_fresh_scenarios() -> List:
    """The fresh-seed holdout exam variant, exactly per the frozen recipe:
    seed 20261126, n_chains=10, exam-shaped construction, 30 scenarios."""
    from aion.nav.exam import Scenario

    world = generate_world(seed=FRESH_WORLD_SEED, n_chains=FRESH_N_CHAINS)
    full: set = set()
    for c in range(5):                                   # chains 0-4 full
        for r in world.relationships_of_chain(c):
            full.add((r.subject, r.relation, r.object))
    partial: set = set(full)
    for c in range(5, 8):                                # chains 5-7 prefix-2
        for r in world.relationships_of_chain(c)[:2]:
            partial.add((r.subject, r.relation, r.object))
    uncovered_chain = 9

    scenarios: List[Scenario] = []

    def mk(sid: str, goal_kind: str, subject: str, relation, facts: set) -> None:
        text = (fact_goal_text(subject, relation) if goal_kind == "fact"
                else reach_goal_text(subject))
        st = NavState(scenario_id=sid, goal_kind=goal_kind, subject=subject,
                      relation=relation, goal_text=text, world=world,
                      known_facts=set(facts))
        scenarios.append(Scenario(state=st, gold=nav_decision_policy(st)))

    # ANSWER: 6 fact goals on fully covered chains 0-4
    picks = [(0, "operates"), (0, "controls"), (1, "routes-to"),
             (1, "produces"), (2, "operates"), (3, "produces")]
    for i, (c, rel) in enumerate(picks):
        r = world.relationships_of_chain(c)
        subj = next(x.subject for x in r if x.relation == rel)
        mk(f"ans-{i:02d}", "fact", subj, rel, full)

    # INFER: 6 reach goals on fully covered walks
    subjects = ([world.actors[c] for c in range(4)]
                + [world.conduits[c] for c in range(2)])
    for i, subj in enumerate(subjects):
        mk(f"inf-{i:02d}", "reach", subj, None, full)

    # QUERY: 3 fact goals on uncovered chain 9 + 3 reach goals on partial 5-7
    for i, rel in enumerate(("operates", "routes-to", "controls")):
        r = world.relationships_of_chain(uncovered_chain)
        subj = next(x.subject for x in r if x.relation == rel)
        mk(f"qry-{i:02d}", "fact", subj, rel, partial)
    for i, c in enumerate(range(5, 8)):
        mk(f"qry-{3 + i:02d}", "reach", world.actors[c], None, partial)

    # STOP: outputs of chains 0-5, alternating produces/controls
    for i, c in enumerate(range(6)):
        rel = "produces" if i % 2 == 0 else "controls"
        mk(f"stp-{i:02d}", "fact", world.outputs[c], rel, full)

    # ASK_TEACHER: fresh invented names x fact-goal forms
    forms = [("Quorvane", "operates"), ("Quorvane", "controls"),
             ("Threnn", "routes-to"), ("Threnn", "produces"),
             ("Malzik", "operates"), ("Malzik", "controls")]
    for i, (name, rel) in enumerate(forms):
        mk(f"tea-{i:02d}", "fact", name, rel, partial)

    assert len(scenarios) == 30
    assert {s.gold for s in scenarios} == set(CLASSES)
    return scenarios
