# Missing requested PDF materials

Audit time: 2026-07-29 UTC.

An exact-name search was run first under `/srv/local/shengenli` and then over
the readable server filesystem.  None of the 16 requested files was present:

1. `2312.16760v1.pdf`
2. `2103.06624v2.pdf`
3. `1811.00866v1.pdf`
4. `1902.08722v5.pdf`
5. `1711.07356v3.pdf`
6. `19-468.pdf`
7. `1810.12715v4.pdf`
8. `1706.06083v4.pdf`
9. `1711.00851v3.pdf`
10. `Week-4-2-3.pdf`
11. `Week-4-1-2.pdf`
12. `Lecture-9_-Neural-Network-Verification-Bound-Propagation-2.pdf`
13. `Lecture-10_-Neural-Network-Verification-Bound-Propagation.pdf`
14. `Lecture-11_-Neural-Network-Verification-Bound-Propagation.pdf`
15. `Lecture-12.pdf`
16. `584_homework2.pdf`

Consequences:

- no attachment-specific page number, diagram, exercise, or quotation is
  claimed in `LITERATURE_MAP.md`;
- public paper landing pages are mapped as public literature, not falsely
  marked as locally read attachments;
- the course-attachment catalog is kept separate from the public ECE/CS 584
  schedule;
- the supplied metadata correction is retained: attachment `Lecture-12.pdf`
  is **Modeling Physics** and covers dynamical systems, stability, and
  Lyapunov reasoning;
- the requested assignment is `584_homework2.pdf` (Homework 2), not an
  unavailable Homework 1;
- `Week-4-1-2.pdf` and `Week-4-2-3.pdf` are two distinct attachments and are
  never merged into one item.

If these exact PDFs are later restored, the literature audit must record each
file's SHA-256, PDF title metadata, page count, relevant page numbers, and the
specific plant-only/NNCS relationship before claiming that it was read.
