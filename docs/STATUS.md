# Status

Current state: `ready_for_external_pytorch_audit_repository_contract_only`.

Completed repository-contract work:

- exhaustive branch/worktree/tag and repository-content inventories;
- clean base selection at `08b6f2416122cbf4220ff351e663caa1a0af13a2`;
- baseline install and test evidence;
- one numerical core, canonical benchmark source, supported comparison CLI,
  and supported order-2 diagnostic;
- fail-closed backend identity and explicit bound/order/completion contracts;
- withdrawal registries without modifying frozen artifacts;
- formal cross-tool execution blocked on eight explicit gates.

Final verification passed: editable install, compileall, 228-test full suite,
unit/integration/Flowstar/DiffReach marker suites, CLI help, README example,
invalid-backend and output-safety checks, structured-file and path scans,
frozen-artifact identity, and Git whitespace checks. The order-2 smoke is
recorded as a validation rejection rather than a crash.

`main` is not changed. No branch or tag is deleted. External repositories are
read-only. No formal three-tool matrix is rerun.

This readiness label covers repository organization and the comparison
contract. The external PyTorch implementation is still unidentified, and all
eight formal cross-tool comparison gates remain pending.
