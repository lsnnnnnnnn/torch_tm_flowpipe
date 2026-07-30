# Target architecture decision

The detailed design evidence is
`audits/repository_consolidation/20260730T083258Z/04_design/TARGET_ARCHITECTURE.md`.

The implemented decision keeps one mathematical package, one canonical
benchmark source, one supported runner, one versioned schema/eligibility
implementation, one report/figure path, and one independent auditor. Historical
adapters may be imported as implementation support, but their old runners and
result directories are not active sources of truth.
