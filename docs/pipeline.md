# Pipeline — step by step

`pipeline.run(PipelineOptions)` is the single entry point used by both the CLI and tests. It runs thirteen sequential stages; each stage gets the previous one's output and adds a layer of information.

## Stages

The component diagram below shows the same thirteen stages as boxes with their data-flow edges: solid orange = audio, solid blue = metadata, solid green = transcript text, dashed blue = optional CLI overrides (including `--language`, `--names`, `--safe-speech-topics`, and `--user-context`, which seeds the system prompt of every LLM stage — see ADR 0033). The table that follows it mirrors the diagram one row per box, naming the upstream stage for every input so the dependency graph reads off the rows.

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
User -[#FF6B35,thickness=5]-> WAV : <color:#404040>  audio file</color>\n<color:#404040>  (wav, m4a, mp3 ...)</color>
User -[#3B82F6,dashed]-> LangDet : <color:#404040>  speech language</color>\n<color:#404040>  (optional override)</color>
User -[#3B82F6,dashed]-> Ident : <color:#404040>  speaker names</color>\n<color:#404040>  (optional override)</color>
User -[#3B82F6,dashed]-> Safe : <color:#404040>  sensitive topics</color>\n<color:#404040>  (optional override)</color>

User -[#3B82F6,dashed]-> Post : <color:#404040>  user context</color>\n<color:#404040>  (optional)</color>
User -[#3B82F6,dashed]-> Ident : <color:#404040>  user context</color>\n<color:#404040>  (optional)</color>
User -[#3B82F6,dashed]-> Struct : <color:#404040>  user context</color>\n<color:#404040>  (optional)</color>
User -[#3B82F6,dashed]-> Safe : <color:#404040>  user context</color>\n<color:#404040>  (optional)</color>
User -[#3B82F6,dashed]-> TLDR : <color:#404040>  user context</color>\n<color:#404040>  (optional)</color>

WAV -[#FF6B35,thickness=5]-> Diar : <color:#404040>  16 kHz</color>\n<color:#404040>  mono WAV</color>
WAV -[#FF6B35]-> LangDet : <color:#404040>  16 kHz</color>\n<color:#404040>  mono WAV</color>
WAV -[#FF6B35,thickness=5]-> CS : <color:#404040>  16 kHz</color>\n<color:#404040>  mono WAV</color>
Diar -[#3B82F6]-> LangDet : <color:#404040>  speaker</color>\n<color:#404040>  timecodes</color>
Diar -[#3B82F6]-> CS : <color:#404040>  speaker</color>\n<color:#404040>  timecodes</color>
LangDet -[#3B82F6]-> ASR : <color:#404040>  speech</color>\n<color:#404040>  language</color>
CS -[#FF6B35,thickness=5]-> ASR : <color:#404040>  cleaned</color>\n<color:#404040>  audio</color>

ASR -[#6E9E1F,thickness=5]-> Merge : <color:#404040>  recognized</color>\n<color:#404040>  text</color>
Diar -[#3B82F6,thickness=5]-> Merge : <color:#404040>  speaker</color>\n<color:#404040>  timecodes</color>

Merge -[#6E9E1F,thickness=5]-> Post : <color:#404040>  text with</color>\n<color:#404040>  speaker labels</color>
Post -[#6E9E1F,thickness=5]-> Ident : <color:#404040>  proof-read text</color>
Ident -[#6E9E1F]-> Struct : <color:#404040>  text with</color>\n<color:#404040>  speaker names</color>
Ident -[#6E9E1F,thickness=5]-> Safe : <color:#404040>  text with</color>\n<color:#404040>  speaker names</color>
Struct -[#3B82F6]-> Safe : <color:#404040>  topic sections</color>
Safe -[#6E9E1F,thickness=5]-> TLDR : <color:#404040>  sensitive topics</color>\n<color:#404040>  removed</color>

FF -[#3B82F6]-> Render : <color:#404040>  recording date,</color>\n<color:#404040>  duration</color>
TLDR -[#6E9E1F,thickness=5]-> Render : <color:#404040>  + summary</color>
Render -[#6E9E1F,thickness=5]-> User : <color:#404040>  transcript.md</color>

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

![Pipeline stages](https://www.plantuml.com/plantuml/svg/lLXjJoCt4Fw-ls9qV6bLDa02uKNfwdYB3WcaArnx7wuHSdOdmSgklR8TEAwgrFw7KtzCVyxzaewzcxTb7G879H2nyVXvnZFZn-EyrOOfCyxIm72J8jnA7cDe51CwLhoF2hxzzHLodcFA1G9P3r47UiH5pXJB89PPBWKASsNkQRh2s30nJ77Ev50fUVVXXiSZWLf3SsuenI6Av4Yg1DMnJAK2nqo3XFZGZiMZeP9ZaHIsQwVk9mWwsf970K0Yut56S-4FUW2uO6h81JHtwEbF-cEneNd5s3cjP-RLzhpzZdjupvA4Yo4KapiR9KbGWBoX5zFmgqOL1DCem9jEF_gqrIlM4Si4Enlzw7VHuL5dCCXo74QT3HwvgHyFVwCbxDbN5Q3P0pPEkBpEqB1vX5p9FiuWkbqVGgi72MUAVy5hdICReT8pdFoU2I4DXeuaWj8YB6OmmwK8MusjASq9VGvhUxl7fzt3Aj5s3rBA-7M2Xd8_uyZIp-7T3ITe5S77ufymfVXYUDkQ8f_Jr1CqiBQ_UNgwlra5t-b1d29JTWqOtypGjQs20QTyKA2C738kB6Ov6FFyFeAEZlfC93dJj7GTrOtd2ZMA61V7oqHPWmI5v1fzywbiyHPLVpBkrdHyvKjAkmV5Gfe6FpyvLeTNbxuuxVOWGpAtaCxsmZxvtDZh0TMH72XaoQ8JQHR1BtSecesTSFSeF1FQgB7DQkd2UwafLztkyN6xNKNTxm6dS1a-kAyfkLNGp-nq7_pMvYHuryMqiSAdmL-vHoy-RLhD1lYWr5Q2febUuERCyN6MCNN_DEmFvoUNNcpQvYILL8RJp5-wnsdg0ojG47uYWFPHQwSNHbOq3g0JMmazMDlUjAy82hq1kwKw6sxCkbWr-k6EtQv3jiteBMz1ez7uSM9T8Let4sdzsP5QiNjDrjRHNhFRNa-Oli744awKfOQqo8QJhJ4RBcMMkHL3beAaRQwelbTWBigudjVk2X0iG_tIhAu4wqXHQ6xuBOAHEO-Vo-idTffGGpos4s8f2gbxdFHx8liANALQ_4zadkIz0fMJanUb2kpUzcvyrplKy6srOVJkonAcJ0ffPMFAiq3po1R_JlJg8kX8Vn4I5unfLXwNAIiNI_0Dpz3sc7e9GRthpy6UkBe0RviIhyGkWQ-ARq1-UiFl8R6jeS1a2OixNcQ5Qd68f7swVxhLhP8KBLCD3Trfv51G9xsCQNxRLBAmCj5paX8KQDvL-TJP-Wjtt2pBqUIyIbaVqncbW_GIUZNQjPoXxJaNWrLCcE8jrzwdkBrmp5Pfgcxx5FZvXEKY5dQBbyKUpNHKnrPRvo2ikTTbdv8_uSHNcD5HdyogYNh31OVBRjApAvIx7oVCuFeIacIit9Look8SzoxBJ_aJJ1le4gcqzMAvKx0ktkNt6xbfP7EhBxvci663cAg5RlO0dJwBHmVbvPUc0dNQz0bT_wX_NyGzF2F5Jp0zWBcBz0JTzlCpkE8QjohSmgMn3gwu7QmATtv-KOKfJGlvZRvz_Tl-_lFlnAm6Ra5QAp15_Qm4Ox_We09Y7gGYhIiXpN8A3m9bireaDtXBR-Ci_Gy0)


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
