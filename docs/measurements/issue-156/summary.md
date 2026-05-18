# ASR chunk-boundary quality — summary

- boundaries judged: 6
- counts: {'clean': 3, 'missing': 3, 'duplicated': 0, 'truncated': 0}
- material issues (missing+duplicated+truncated): 3
- material rate: 0.500 (threshold 0.10)
- decision: **material**

| # | cutoff | verdict | gap_s | jaccard | rms_before | rms_after | note |
|---|--------|---------|-------|---------|------------|-----------|------|
| 0 | 08:00.00 | clean | 3.96 | 0.189 | 0.0004 | 0.0003 | tail ends 'So, um...' + head 'that still going?'. RMS_before=0.00041 and RMS_after=0.00034 are both near silence — the 3.96s gap is a real pause where the interviewer waited. 'Um... that still going?' is plausible spoken English (the copula 'is' often elided in fast conversation). No torn word. |
| 1 | 15:55.00 | clean | -0.16 | 0.087 | 0.0119 | 0.0107 | negative gap (-0.16s) is a tiny overlap absorbed by dedup. 'for designers.' closes the previous thought; 'I also like, if you noticed in my CV...' opens a new one. Both side-tokens are complete words. jaccard 0.087 is in line with the overall baseline. |
| 2 | 23:50.00 | missing | 3.48 | 0.153 | 0.0589 | 0.0674 | tail '...helps you to rank all of the potential.' + head 'of relevant targets is something which is helpful...' fuses into 'rank all of the potential of relevant targets' — one continuous noun phrase split across the cutoff. RMS_after=0.067 is mid-energy (speech, not silence) yet gap_s=3.48 — suspicious. Likely a few words at the rejoin were not transcribed; both side-tokens are complete so this is missing, not truncated. |
| 3 | 31:45.00 | missing | 3.40 | 0.101 | 0.0921 | 0.0887 | tail '...And I need to catch myself.' + head 'to somehow like partially related to this...' — head starts with bare 'to' with no subject, awkwardly attached. RMS_before=0.092 and RMS_after=0.089 are both high (active speech) but gap_s=3.40 — strong signal that words were spoken across the cutoff but not retained. Likely missing a connector like 'related' or '[something] to somehow'. |
| 4 | 39:40.00 | missing | 1.50 | 0.188 | 0.1170 | 0.1313 | tail '...our Forecast product' + head 'scheduling product so it's they go in hand' — fuses into 'Forecast product scheduling product', which is ungrammatical. Almost certainly missing a conjunction ('and') or noun glue. Highest RMS of all boundaries (0.117/0.131) with only 1.5s gap — speech was clearly active across the cutoff. Strongest evidence of a real loss in this dump. |
| 5 | 47:35.00 | clean | 1.82 | 0.100 | 0.0602 | 0.0774 | tail '...but I don't know.' + head 'Cool. Cool. Yeah. All right.' — speaker turn change, complete sentences on both sides. RMS_after rises with the next speaker, gap_s=1.82 is a natural turn pause. last_token 'know' and first_token 'cool' are both complete. |
