# Start-state provenance

This directory records the clean isolated Torch starting state, all pinned
external SHAs, the incompatible dirty original worktrees that were preserved,
the actual software/GPU environment, the Xiangru private-remote fetch blocker,
the editable install log, and the pre-change full test log.

The Flow* and DiffReach native raw logs are stored separately under
`../01_native_baselines/`.  Build products and native artifacts are hashed in
the final run manifest; no dirty original worktree supplies a formal result.
