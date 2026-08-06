# TORA-Q3 contract field map

| Field | Frozen evidence | Torch implementation |
|---|---|---|
| Dynamics/state order | AST-hashed `tm_tora_rhs` at Xiangru `27d29050` | `torch_tm_flowpipe.tora_q3.tora_q3_rhs` |
| Q3 basis/order | live Xiangru exporter header | `BatchedMonomialBasis.build(6,3)`; identical 84-slot order |
| Workload | hash-pinned resolved config | native B48 runner |
| Controller assets | parsed ONNX hashes and observation trace | external `TORA_CONTROLLER_PATH`; bytes excluded from Git |
| Endpoint/tube | explicit tagged raw fields | distinct endpoint substitution and full-domain range |
| Replay reset | observed pre-controller leaf boxes each period | period-local observation restart; not independent closed loop |

Fields described for readability rather than directly parsed are listed in each
JSON file under `manually_transcribed_fields` or `manually_transcribed`.
