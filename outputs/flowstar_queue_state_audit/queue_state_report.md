# Flowstar Queue State Audit

This audit reads normalized-insertion no-queue, split-symbolic-queue, and v2 symbolic-queue outputs.
It is a raw diagnostics audit; missing raw fields are reported as blanks rather than reconstructed from Flow* source.
Audit status: complete three-way input set loaded.

## Answers

Did v2 reach h10? no; best v2 t=`7.4960392581387341`.
Did v2 beat no_queue on last_validated_t? no; no_queue best t=`7.4960392581387341`.
Did v2 beat split_symqueue on last_validated_t? no; split best t=`7.4960392581387341`.
Common horizon for best-run width comparisons: `7.4960392581387341`.
Did v2 reduce max reset width before the common horizon? no; no_queue=`21.902999702465412`, split=`21.902999702465412`, v2=`21.902999702465415`.
Did v2 reduce max right-map range width before the common horizon? not comparable; no_queue=`None`, split=`None`, v2=`21.883875748631645`.
Did v2 move Flow* width-ratio crossings later? no_queue first width_ratio>=1 at 4.0459460349724967, split_symqueue first width_ratio>=1 at 4.0459460349724967, v2_symqueue first width_ratio>=1 at 3.206019346657988.
Did v2 keep symbolic width out of target check? yes; max target contribution=`9.8813129168249309e-324`.
Did v2 add symbolic width only to output/range boxes? yes; max output-only symbolic=`0.24507717887847935`; output flags all true=True.

## Trace Columns

See `queue_state_trace.csv` for per-segment J/Phi_L/scalar, target, reset, right-map, and output-only channels.
See `queue_state_summary.csv` for source/run-level metrics.

## Run Summary

| source | run_id | status | last_validated_t | max_queue | max_reset_common | max_right_common | max_output_only_common |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| no_queue | flowstar_style_o4_target_insert | failed | 6.4730088058091901 | 0 | 4.820920894732831 | None | None |
| no_queue | flowstar_style_o6_candidate8_output6_insert | failed | 7.4960392581387341 | 0 | 21.902999702465412 | None | None |
| split_symqueue | flowstar_style_o4_target_insert_symqueue_split | failed | 6.4730088058091901 | 99 | 4.820920894732831 | None | 0.30285839887550597 |
| split_symqueue | flowstar_style_o6_candidate8_output6_insert_symqueue_split | failed | 7.4960392581387341 | 99 | 21.902999702465412 | None | 5.0175141128736733 |
| v2_symqueue | flowstar_style_o4_target_insert_symqueue_v2 | failed | 6.4730088058091901 | 99 | 4.8209208947328319 | 4.7809633686058826 | 0.17422689579314798 |
| v2_symqueue | flowstar_style_o6_candidate8_output6_insert_symqueue_v2 | failed | 7.4960392581387341 | 99 | 21.902999702465415 | 21.883875748631645 | 0.24507717887847935 |
