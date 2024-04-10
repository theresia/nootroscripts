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

3. This project requires an `OPENAI_KEY` environment variable with an [OpenAI API key](https://platform.openai.com/api-keys).

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

TODO

## Philosophy

Local first. Have things to reference back to, excerpt, take notes on. Extended insights.

Perfect for journalists and researchers who want to optimise for thoroughness in their workflow when consuming media.

These scripts
- help get the nuggets from non-fiction content
- help me persist the ideas and insights into something I can rediscover

For more context, see blogpost: http://proses.id/nootroscripts/
