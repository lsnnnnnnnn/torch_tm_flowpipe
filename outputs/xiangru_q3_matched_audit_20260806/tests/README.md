# Final test evidence

Each subdirectory contains `command.json`, raw `stdout.log`, and raw `stderr.log`. The command record includes the exact argv, cwd, start/end times and exit code.

- Torch editable install: exit 0.
- Torch full pytest: 455 passed, 2 skipped, exit 0.
- Xiangru remainder-ablation tests: 51 passed, 4 skipped, exit 0.
- Xiangru combined tests: 77 passed, 4 skipped, one upstream inventory failure, exit 1. The missing historical `run.json` path is preserved in the raw traceback.
- Final repository checks record exact commits, branches/status, Python environments and `git diff --check` exit codes.
