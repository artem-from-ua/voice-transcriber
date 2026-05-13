# Pipeline — step by step

`pipeline.run(PipelineOptions)` is the single entry point used by both the CLI and tests. It runs thirteen sequential stages; each stage gets the previous one's output and adds a layer of information.

## Stages

The component diagram below shows the same thirteen stages as boxes with their data-flow edges: solid orange = audio, solid blue = metadata, solid green = transcript text, dashed = optional CLI overrides. The table that follows it mirrors the diagram one row per box, naming the upstream stage for every input so the dependency graph reads off the rows.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2
skinparam legendBorderColor transparent
skinparam legendBackgroundColor #EEEEEE

<style>
  document {
    Margin 25
  }
  actor {
    LineThickness 4
    Margin 5
    Padding 5
  }
</style>

actor " " as User

component "<b>[1] transcode</b>\n<i><ffmpeg></i>" as WAV #E8E8E8
component "<b>[2] audio_meta</b>\n<i><ffprobe></i>" as FF #E8E8E8
component "<b>[3] diarize_speakers</b>\n<i><pyannote-audio> speaker-diarization-3.1</i>" as Diar #87CEFA
component "<b>[4] lang_detect</b>\n<i><mlx-whisper> Whisper-large-v3</i>" as LangDet #90EE90
component "<b>[5] clear_speech</b>\n<i><soundfile> autogain</i>\n<i><scipy> <s>bandpass</s></i>\n<i><scipy> <s>presence</s></i>\n<i><ffmpeg> <s>denoise</s></i>\n<i><scipy> <s>dereverb</s></i>" as CS #E8E8E8
component "<b>[6] speech2text</b>\n<i><mlx-whisper> Whisper-large-v3</i>" as ASR #90EE90
component "<b>[7] merge</b>" as Merge #E8E8E8
component "<b>[8] proofread</b>\n<i><mlx-lm> Qwen2.5-7B-Instruct</i>" as Post #FFCC66
component "<b>[9] identify_speakers</b>\n<i><mlx-lm> Qwen2.5-7B-Instruct</i>" as Ident #FFCC66
component "<b>[10] speech_structure</b>\n<i><mlx-lm> Qwen2.5-7B-Instruct</i>" as Struct #FFCC66
component "<b>[11] safe_speech</b>\n<i><mlx-lm> Qwen2.5-7B-Instruct</i>" as Safe #FFCC66
component "<b>[12] speech_summary</b>\n<i><mlx-lm> Qwen2.5-7B-Instruct</i>" as TLDR #FFCC66
component "<b>[13] render</b>" as Render #E8E8E8

User -[#FF6B35]-> FF : <color:#404040>  audio file</color>\n<color:#404040>  (wav, m4a, mp3 ...)</color>
User -[#FF6B35]-> WAV : <color:#404040>  audio file</color>\n<color:#404040>  (wav, m4a, mp3 ...)</color>
User -[#3B82F6,dashed]-> LangDet : <color:#404040>  speech language</color>\n<color:#404040>  (optional override)</color>
User -[#3B82F6,dashed]-> Ident : <color:#404040>  speaker names</color>\n<color:#404040>  (optional override)</color>
User -[#3B82F6,dashed]-> Safe : <color:#404040>  sensitive topics</color>\n<color:#404040>  (optional override)</color>

WAV -[#FF6B35]-> Diar : <color:#404040>  16 kHz</color>\n<color:#404040>  mono WAV</color>
WAV -[#FF6B35]-> LangDet : <color:#404040>  16 kHz</color>\n<color:#404040>  mono WAV</color>
WAV -[#FF6B35]-> CS : <color:#404040>  16 kHz</color>\n<color:#404040>  mono WAV</color>
Diar -[#3B82F6]-> LangDet : <color:#404040>  speaker</color>\n<color:#404040>  timecodes</color>
Diar -[#3B82F6]-> CS : <color:#404040>  speaker</color>\n<color:#404040>  timecodes</color>
LangDet -[#3B82F6]-> ASR : <color:#404040>  detected</color>\n<color:#404040>  language</color>
CS -[#FF6B35]-> ASR : <color:#404040>  cleaned</color>\n<color:#404040>  audio</color>

ASR -[#6E9E1F]-> Merge : <color:#404040>  recognized</color>\n<color:#404040>  text</color>
Diar -[#3B82F6]-> Merge : <color:#404040>  speaker</color>\n<color:#404040>  timecodes</color>

Merge -[#6E9E1F]-> Post : <color:#404040>  text with speaker labels</color>
Post -[#6E9E1F]-> Ident : <color:#404040>  proof-read text</color>
Ident -[#6E9E1F]-> Struct : <color:#404040>  text with</color>\n<color:#404040>  speaker names</color>
Ident -[#6E9E1F]-> Safe : <color:#404040>  text with</color>\n<color:#404040>  speaker names</color>
Struct -[#3B82F6]-> Safe : <color:#404040>  topic sections</color>
Safe -[#6E9E1F]-> TLDR : <color:#404040>  sensitive topics</color>\n<color:#404040>  removed</color>

FF -[#3B82F6]-> Render : <color:#404040>  recording date,</color>\n<color:#404040>  duration</color>
TLDR -[#6E9E1F]-> Render : <color:#404040>  + summary</color>
Render -[#6E9E1F]-> User : <color:#404040>  transcript.md</color>

legend top center
  <back:#87CEFA>   </back> <i><color:#404040>AI model: diarization</color></i>
  <back:#90EE90>   </back> <i><color:#404040>AI model: speech2text</color></i>
  <back:#FFCC66>   </back> <i><color:#404040>AI model: LLM</color></i>
  <color:#FF6B35>━━►</color> <i><color:#404040>audio</color></i>
  <color:#3B82F6>━━►</color> <i><color:#404040>metadata</color></i>
  <color:#6E9E1F>━━►</color> <i><color:#404040>transcript text</color></i>
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/jPVzJkD64C3_zrECZdzQ5GT887p8iewFH4K2wGhNtX_N26tiILXXxrgx6pYkgjI-m-aUvXxddgIpkyw7Wvq6IWaWpCxyPdPcFBFxLXkcp2JF0iDDXd0lUOAXKKpeHF4XAlZ-rnSeU84P5mWaFKOTw3ik2gPO3edC2obGc6lpIEeA4yF4ECC5aMEbvCFxMvxS2TGQsWjB6OvHf2TIfQXEPIOLECkqICIdSYov6oiv4QcNtUYvho28j3KU1m2fJ2OvwS8Vz01moTIO2sZlqTEVzCCIGtf-xOBsC_TgTr5-ppsyOsdAnHYyShHL6Wayv0rzc8PVDQeWc4K1taJ3-EFEmEyZaIb6MyFuTn7nE1gDyWB7SRJ5OwVwuVWtsiA1_Im3sWDWavJBcnmDMoGKIWvnZkZtcmYT0QISAVuPRtI1x0wLddEAHoQ4D1Ww8p6K4g7NO8PB4NPQEpCpP_H_s3ZOF-trZvSDxGuWeupVf6WeztCRUVOfVBZbX5OnVF1_X1d55yFxpenyGQfdQ63ZSBlNEznUXkyD8CcGARi7J6xdT6shOCGf7nGe8yUC2yii5nDUFCOGwNZ8H5emJLlNSTrweL2ZIB2wNYRA2gGe9DVOf4zbY_UeXjDrjwVJwwQaT0VWDz8s-EdPiJg-lcgAq_u0SYHnYtIob_QfoOs30L0ToP52bbQToF8OVdb0qMxjXViduRdGHariLaiNtabDkUhtJq-xtMNguG0uXSlmqMDDoQr3FxVgJVYTxMdmRhpQHE7pyDVkiP5FBwrc8tnIQckXIR4Ht9uxFibpfXwVntv_SNRLoARNd5ebbU4iyrVkSPRwm3Oe23yIe7lIsHk4iMqQHn0bjeKURUnksqyClj-0VQMYBRTcNLmM-k61tMz2lileLz61LglrurIwnf1jSF-dfSx9GRlVtKoPliNK6fnsangZFWEkLKtOUALbMTW6o3AGz6ehgkarBFlIhRThtmGGB4VzsbPTsTKPHQ6vuVS8HXOyUQdTm2Qmab4t26eCxdJXxkVFAypaKaXR3pFs4_IAtBq2dNhogu1T06Qf-FUQixbVmJSyHtj8q2iCrBl-4lRKrGhUJfKQlX_dcAumiFmk1UHf9UWDQ3lHnKgown3pChGS0dTxXxsTlWNxwLQ3fkEd70iwzAoY-zdR6Ez6-4j27dXQnNitH6lurYzuuEPsrZGoDiHi3dEA5LHZst7pFRG3lhfThr1Xb6DnbKChjbpRtshDDFIgbnif7Qyahz6AxOFK5nFRx1OGLhpYffks_wsbAiofbwRp6g21Mt6o7D4DzQlSfIDb1ZTN64adoftJPrQSuvMjD9lv2MOdaLAxbAtekx5J5yloSbYOLhwmItz7i-61XB9FroA0Q4XNjgFova2gGCSK-W-TdUdmK-KUdrEtJJ4xWeLRo9Hk3qDpd3y3hukhdBZhSFvejIxkuk9o6LEA-iONV__oj_t--cqgLeEhDBWbZA-JjJ3sTaY5mXf8FfLhaUOvXIU1ibdroGtUqgz9dlq3)


| # | Stage | Input (from) | Action | Output |
|---|-------|--------------|--------|--------|
| **1** | `transcode` | Original audio *(from user)* | **Normalize audio format**<br>via `ffmpeg` | WAV (16 kHz mono PCM) |
| **2** | `audio_meta` | Original audio *(from user)* | **Extract recording start/end/duration**<br>via `ffprobe`, `birthtime`, `mtime` | <ul><li>Recording date</li><li>Duration</li></ul> |
| **3** | `diarize_speakers` | WAV *(from `transcode`)* | **Who speaks when**<br>AI model `speaker-diarization-3.1` | Turns (timecodes for each distinct speaker) |
| **4** | `lang_detect` | <ul><li>WAV *(from `transcode`)*</li><li>Turns *(from `diarize_speakers`)*</li><li>Language code *(optional, from user)*</li></ul> | **Auto-detect language**<br>longest turn audio → AI model `Whisper-large-v3` | Language code |
| **5** | `clear_speech` | <ul><li>WAV *(from `transcode`)*</li><li>Turns *(from `diarize_speakers`)*</li></ul> | **Clean the audio for speech recognition**<br>Apply a chain of audio effects (`autogain`, `bandpass`, `presence`, `denoise`, `dereverb`) | Cleaned WAV |
| **6** | `speech2text` | <ul><li>Cleaned WAV *(from `clear_speech`)*</li><li>Language code *(from `lang_detect`)*</li></ul> | **Convert audio to text**<br>AI model `Whisper-large-v3` | Text segments (unattributed to speakers) |
| **7** | `merge` | <ul><li>Text segments *(from `speech2text`)*</li><li>Turns *(from `diarize_speakers`)*</li></ul> | **Attach speakers to text segments**<br>Per-segment max-overlap mapping | Text segments (attributed to anonymous distinct speakers) |
| **8** | `proofread` | Text segments *(from `merge`)* | **Fix Automatic Speech Recognition mishearings**<br>Per segment → AI model `Qwen2.5-7B-Instruct` | Text segments (proofread) |
| **9** | `identify_speakers` | <ul><li>Text segments *(from `proofread`)*</li><li>Speaker names *(optional override, from user)*</li></ul> | **Anonymous speaker labels → real speaker names**<br>For each distinct speaker, identify how they introduce themselves in their first 60 seconds of speech to extract their name using AI model `Qwen2.5-7B-Instruct` | Text segments (attributed to named speakers) |
| **10** | `speech_structure` | Text segments *(from `identify_speakers`)* | **Identify and split the speech into thematic sections**<br>For each group of segments → AI model `Qwen2.5-7B-Instruct` to detect topic-based sections | Topic sections |
| **11** | `safe_speech` | <ul><li>Text segments *(from `identify_speakers`)*</li><li>Topic sections *(from `speech_structure`)*</li></ul> | **Redact sensitive topics**<br>For each section → AI model `Qwen2.5-7B-Instruct` to redact utterances on sensitive topics | Text segments (redacted) |
| **12** | `speech_summary` | Text segments *(from `safe_speech`)* | **Produce a TL;DR with overview, key points and action items**<br>Whole flat transcript → AI model `Qwen2.5-7B-Instruct` | Markdown string |
| **13** | `render` | <ul><li>Recording date *(from `audio_meta`)*</li><li>Duration *(from `audio_meta`)*</li><li>Text segments *(from `safe_speech`)*</li><li>Topic sections *(from `speech_structure`)*</li><li>TL;DR *(from `speech_summary`)*</li></ul> | **Build final document**<br>via markdown templating | Final markdown document |
