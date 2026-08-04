status: historical
valid_for_commit: 0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781
superseded_by: docs/ARCHITECTURE.md
allowed_use: provenance only

# Target architecture decision

The detailed design evidence is
`audits/repository_consolidation/20260730T083258Z/04_design/TARGET_ARCHITECTURE.md`.

The implemented decision keeps one mathematical package, one canonical
benchmark source, one supported runner, one versioned schema/eligibility
implementation, one report/figure path, and one independent auditor. Historical
adapters may be imported as implementation support, but their old runners and
result directories are not active sources of truth.
