## Nooscro... What's this name?

**Nootropics**: Drug, supplement, or other substance that improves cognitive function. Colloquially brain supplements, smart drugs and cognitive enhancers, are natural, semisynthetic or synthetic compounds which purportedly improve cognitive functions, such as executive functions, attention or memory.

**NootroScripts**: A suite of LLM-powered scripts to help synthesize content in different formats and sources, more efficiently. Cognitive enhancers for curious curators.

## Overview

**Input**: Takes in YouTube video ID, an audio file, or a URL of articles

**Process**: Select a mode that you'd like the Large Language Model (LLM) to go into (see `--help` for each script for a list of modes supported)

**Output**: Generates .txt or .md, on the same path as the input, when an output path is not specified

## Configuration & Installation

1. Set up your virtual env (or not, if you like living dangerously).

2. Run `pip install -r requirements.txt`

3. This project requires an **`OPENAI_KEY`** environment variable with an [OpenAI API key](https://platform.openai.com/api-keys).

4. This project also supports using your own local LLM installed via Ollama.ai.

4.a. You can install ollama by doing `curl https://ollama.ai/install.sh | sh`

4.b. And get the mistral model by running `ollama pull mistral`

5. Install `ffmpeg` on your OS if you do not have it yet. This is required by pytube to download and encode the video from YouTube (to be transcribed locally by Whisper), when transcript is available.

6. You may run `test.sh` to test if the sample commands run successfully. Comment out the line with `--lmodel mistral` if you haven't installed mistral via ollama (step 4), or change the lmodel to other ollama model if mistral is too slow for your machine.

### Large dependencies

The project will also download:

- Chromium via pyppeteer
- if you're on Linux or WSL with NVIDIA, you'll also be downloading large some related libs e.g.
    - nvidia-cublas-cu12
    - nvidia-cuda-cupti-cu12
    - nvidia-cuda-nvrtc-cu12
    - nvidia-cuda-runtime-cu12
    - nvidia-cudnn-cu12
    - nvidia-cufft-cu12
    - nvidia-curand-cu12
    - nvidia-cusolver-cu12
    - nvidia-cusparse-cu12
    - nvidia-nccl-cu12
    - nvidia-nvjitlink-cu12
    - nvidia-nvtx-cu12


## Usage

```
./youtube2llm.py --help
./url2md.py --help
./audio2llm.py --help
./md2notes.py --help
```

Guide for [youtube2llm.py here](https://proses.id/nootroscripts-2)

TODO

## Philosophy

Local first. Have things to reference back to, excerpt, take notes on. Extended insights.

Perfect for journalists and researchers who want to optimise for thoroughness in their workflow when consuming media.

## Backstory

Initially I thought these scripts will save me so much time by allowing me to get the knowledge without having to listen to the podcast episode, watch the video, or read the article.

But over time I realised that these summaries/notes/text help me synthesise the information more. They serve as notes to refer back to, skim, refresh my mind of the ideas introduced.

Yes, these scripts
- help get the nuggets from non-fiction content
- help me persist the ideas and insights into something I can rediscover
- help me manage cognitive load. this is basically a RIL-on-steroid (RIL: Read It Later)
- help me indulge my tsundoku and snoozing/JIC tendency, ^^; (JIC: Just In Case)

But I didn't exactly learn more efficiently. I just learn more effectively. And I think that's a better accidental outcome.

For more context, see blogpost: http://proses.id/nootroscripts/

## Observations and plans

### A. md2notes.py is a subset of audio2llm.py

- Option 1: discard md2notes.py or
- Option 2: refactor each into two scripts as they have their own specific input, moving the common components to a separate piece of code?

### B. the LLM output of youtube2llm.py more high-level, vague, and not as good (refined, granular, precise) as the audio2llm.py

need to check if it's the prompt, the chunking strategy, the context window, or what.

seems like it's a combination all of them sih.

----

I'll eventually need to refactor all these scripts into these specific components:

I. Retrievals:

- **youtube** (am 83% happy with what youtube2llm.py does in terms of retrieval)
- **podcast** (any LOCAL audio stream sih, be it mp3, mp4, m4a, etc).
    - it'd be great to support streams or major podcast syndication services like libsyn, anchor, spotify even though I tend to want to store the audio files locally, and rarely use spotify
    - e.g. [audacy.com](https://www.audacy.com/podcast/tetragrammaton-with-rick-rubin-14b90/episodes/steven-pressfield-1f039)
    - or extract the youtube ID from [this page](https://www.cognitiverevolution.ai/the-ai-email-assistant-ive-been-waiting-for-with-andrew-lee-of-shortwave/)
- **article** (am 98% happy with url2md.py)
- **instagram** (image, video, caption, comments, transcript). am 98% happy with my installm project
- **local notes** (textfiles and markdowns), or popular platforms like notion, obsidian, and perhaps blogging CMS'es exports

II. LLM

- **chunking**: a good chunking strategy would help us get the best level of zoom & coverage, and get a better flow (proper segues and context). the ultimate goal is to appropriately (sometimes fully) and accurately capture and represent the content. common strategies:
    - words # implemented
    - sentences # implemented
    - semantic # implemented in chunking_test.py
    - specific e.g. podcast vs article vs essay # WiP #1
    - dynamic aka different chunk size based on the podcast/article/essay content e.g. average length of argument, the speaker's tendency, monologue, dialogue, breaking news # WiP #2
- **prompts** (the collection of "modes" i.e. prompts I supported here)
- **model configuration**
    - cloud: OpenAI, MistralAI, Anthropic, Google Gemini, AWS BedRock
    - local: ollama, huggingface, and other providers

III. Post processing

- **embedding**: BGE, lv2, openAI's, mistralAI's, claude's, instructor-xl
- **indexing**: llama index, langchain?
- **querying**


IV. Creation / Production

- **curation**
- **generate drafts**