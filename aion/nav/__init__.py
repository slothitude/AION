"""AION-NAV — the NAV decision consumer (owner lane, 2026-11-25).

Consumes the NAV closed-enum decision over a chain-world state via pluggable
backends, each used exactly per its public model card:

- HeuristicFloor        local rules (the zero-model floor)
- D1Backend             LiquidAI/d1-2B via model.system_one() (card: choice query)
- Tev1Backend           togethercomputer/Tev1-0.8B-experimental (card: structured
                        chat-completions; LICENSE-BLOCKED at 2026-11-25 - repo
                        carries no license tag, card says the fine-tuned-weights
                        release license is being finalized)
- LarqlClassifierBackend  larql-d1 fitted head, consumed read-only through the
                        owner-approved socket (reports/d1_adapters.md §2.3
                        portable artifact contract)

Boundary: this package only imports aion.worlds.generator (read-only) and the
larql-d1 socket per its documented contract. It never touches src/nnfs,
dreaming, or any larql-d1 internal beyond the documented socket names.
"""
