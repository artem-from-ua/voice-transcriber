# Proofread spot-check — final table

## Human marks (audio-grounded)

- better: 3
- worse:  4
- neutral: 2

## Claude Code judge marks (text-only)

- better: 2
- worse:  4
- neutral: 3

## Agreement

- Agreed on 4/9 segments (44.4%).

Confusion table (rows = human, columns = judge):

| human \ judge | better | worse | neutral |
|---|---|---|---|
| better | 1 | 2 | 0 |
| worse | 0 | 2 | 2 |
| neutral | 1 | 0 | 1 |

## Per-segment marks

| Segment | Human | Claude | Agree? | Claude rationale |
|---|---|---|---|---|
| 4 | neutral | better | ❌ | ASR misheard 'контролери' (hardware controllers) as 'контролуси' — not a word, not a conversational variant. Context ('де контролери у тебе') makes the fix unambiguous. |
| 9 | neutral | neutral | ✅ | ASR produced 'що', proofread produced 'шо'. Both are valid Ukrainian; 'шо' is the colloquial form people actually use in lively speech. Without audio I cannot tell which the speaker actually said — direction of the change (literary→colloquial) is not by itself a regression because the project goal is to capture real conversation, not enforce literary norms. text-only judge cannot resolve this. |
| 11 | better | worse | ❌ | Speculative semantic change: 'от' (pointing/emphatic particle) → 'а' (contrastive conjunction). These are different parts of speech and change the connective meaning of the sentence. Without audio I cannot verify which was spoken, but proofread is supposed to fix obvious mishearings — when no clear mishearing exists, the default is to trust ASR rather than re-pick a different lexical choice. |
| 51 | better | better | ✅ | Grammatical fix: 'твими' is not a Ukrainian word — neither literary nor colloquial. 'твоїми' is the standard instrumental plural of 'твій'. ASR clearly garbled it; proofread restored the correct form. |
| 69 | worse | neutral | ❌ | ASR hallucinated ('корище' is not a word in any register). Proofread guessed 'користуватися цим' from sparse context. The guess is plausible and grammatical, but I cannot tell from text alone whether the speaker actually said this vs something shorter like 'взяти це'. Not actively wrong, not verified right. |
| 70 | better | worse | ❌ | Two changes; the verdict comes from the second one. (1) 'тусуться' → 'тусуются' could be either a regression (Russianised reflexive form) or a closer phonetic transcription of what the speaker actually said — text-only judgment cannot resolve this; I am neutral on this part alone. (2) 'чуваки' → 'хлопці' is a synonym substitution, not a mishearing fix. 'чуваки' is normal Ukrainian conversational vocabulary — proofread should not be swapping it for a near-synonym regardless of which word was uttered. Synonym rewrites are out of scope for an ASR-error fixer. |
| 75 | worse | neutral | ❌ | 'оця прослойка' (emphatic demonstrative) vs 'ця прослойка' (neutral demonstrative). Both are valid in conversational Ukrainian; the change is stylistic and minimal. |
| 77 | worse | worse | ✅ | Context: 'агента для продуктового інтерв'ю' — Ukrainian IT slang for a product-management interview (from English 'product'). 'продуктового' is the standard Ukrainian form used in this sense across IT conversations. 'продуктівого' is not a conversational variant — it shifts the root to 'продуктивний' (productive/manufacturing-related), which changes the meaning. This is text degradation, not preservation of speech. |
| 81 | worse | worse | ✅ | Same word stem, same speaker, same error as #77: 'продуктовим' → 'продуктівим'. The fact that the model makes this mistake twice in a row on the same word is a sign of a systematic weakness (likely the proofread prompt or model 'corrects' a perceived loanword toward an unrelated Ukrainian root), not random noise. |

> Disclaimer: the Claude Code judge is text-only; the human eval is
> audio-grounded. Agreement here measures whether text-only LLM
> judgment can be trusted for future automated measurements.
