# S1 terminal causal gate

Date: 2026-08-10

Status: `not_run_after_stop`  
Primary outcome: `S1_PREFIX_REJECTS_BEFORE_TERMINAL`

The historical terminal prestate is at `t=6.397083942944808`, after 307
accepted boundaries, with proposed `h=0.003623635847674574`. L2 stops on the
frozen schedule after boundary 164 at `t=4.738198114669049`. Therefore no S1
checkpoint exists at the historical terminal and a same-pre-state A/B there
would require inventing 143 missing accepted boundaries.

The exact boundary-164 checkpoint was still used for a causal replay of the
first failed frozen obligation. At `h=0.03661680691961388`, the historical
baseline accepts with no shrink. L2's unchanged raw-compatible target has
subset margins
`[+9.633831630803861e-5, -3.773875528686747e-6]`, so y fails. The adaptive
helper subsequently returns a half-step, but that state is discarded and the
frozen pre/post hash remains unchanged.

Thus the earlier empty-history terminal ordinary margin is not promoted. The
terminal GO checklist was not evaluated as a numerical A/B; its gate file
records `authorized=false` and `passed=false` solely because the prerequisite
prefix was absent.
