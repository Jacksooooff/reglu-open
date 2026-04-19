# Code vs Paper

This repository follows the current effective code path as the implementation source of truth.

That means:

- v1 does not ship separate `paper` and `code` modes
- any differences between the paper description and the current implementation should be documented here
- public CLI and configs remain stable even if internal experimental details change later

Known current v1 simplification:

- `RILA` is exposed as a cache-backed initialization path in the clean repo interface
- historical baseline branches such as FILA, KL, NPO, and VILA are not exposed publicly
