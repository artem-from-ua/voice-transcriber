# Proofread context-size sweep

Audio: `/Users/artem/Downloads/two-speakers-diar-test-ukr.m4a`
Grid: [0, 1, 2, 3, 5, 8]

| n_context | total | unchanged | cosmetic | proper_noun_fix | substantive_rewrite | hit_rate_% | wall_clock_s | llm_calls |
|---|---|---|---|---|---|---|---|---|
| 0 | 90 | 59 | 5 | 8 | 18 | 34.4 | 97.63 | 86 |
| 1 | 90 | 57 | 1 | 11 | 21 | 36.7 | 125.76 | 86 |
| 2 | 90 | 62 | 3 | 9 | 16 | 31.1 | 144.93 | 86 |
| 3 | 90 | 63 | 3 | 10 | 14 | 30.0 | 170.05 | 86 |
| 5 | 90 | 65 | 0 | 10 | 15 | 27.8 | 208.17 | 86 |
| 8 | 90 | 66 | 0 | 10 | 14 | 26.7 | 258.34 | 86 |
