#!/usr/bin/env python3

'''
Overview:
    Script is developed based on: https://github.com/mirabdullahyaser/Summarizing-Youtube-Videos-with-OpenAI-Whisper-and-GPT-3
    I added the functionality to get the transcript from YouTube when available to save Whisper some work.

Usage
    A. Summarising (after retrieving YouTube's caption/subtitles or transcribing with Whisper)
    
    # process a YouTube video, passing the video ID. will produce a list of Q&As for the video by default when --mode is not specified
    youtube2llm.py analyse --vid 2JmfDKOyQcI
    
    # process a transcript file (e.g. when you already have the caption / transcript file saved locally, saving the web traffic calls to youtube)
    youtube2llm.py analyse --tf output/FbquCdNZ4LM-transcript.txt
    
    # have the LLM use some tones/personalities specified in the system prompts
    youtube2llm.py analyse --nc --lmodel mistral --lmtone doubtful_stylistic_british --vid=6KeiPitz5QM
    
    # don't send chapters, when it's making it too verbose or fragmented
    youtube2llm.py analyse --vid=FbquCdNZ4LM --nc
    
    # produce a note for this video
    youtube2llm.py analyse --vid Lsf166_Rd6M --nc --mode note

    # produce a list of definitions made in this video
    youtube2llm.py analyse --vid Lsf166_Rd6M --nc --mode definition

    # download and transcribe the audio file (don't use YouTube's auto caption)
    youtube2llm.py analyse --vid=MNwdq2ofxoA --nc --lmodel mistral --dla

    # the video has no subtitle, so it falls back to Whisper that transcribes it
    youtube2llm.py analyse --vid=TVbeikZTGKY --nc
    
    # this video in Bahasa Indonesia has no auto caption nor official subtitle, so we specify Indonesian language so Whisper's output is better
    youtube2llm.py analyse --vid PDpyUMOOcyw --lang id
    
    # this video has age filter on and can't be accessed without logging in
    youtube2llm.py analyse --vid=FbquCdNZ4LM --nc
        pytube.exceptions.AgeRestrictedError: FbquCdNZ4LM is age restricted, and can't be accessed without logging in.
    # in this case, I will use `yt-dlp -f` to get the audio file (m4a or mp4) and then run the audio file through audio2llm.py to transcribe it with Whisper
    # another example of such video: https://www.youtube.com/watch?v=77ivEdhHKB0
    
    # # uses mistral (via ollama) for summarisation. the script will use OpenAI's API for the ask and embed mode (still TODO)
    youtube2llm.py analyse --vid=sYmCnngKq00 --nc --lmodel mistral
    
    # run a custom prompt against an audio file (first it will retrieve the transcript from YouTube or generate the transcript using Whisper)
    youtube2llm.py analyse --vid="-3vmxQet5LA" --prompt "all the public speaking techniques and tactics shared"
    
    # run a custom prompt against a transcript file
    youtube2llm.py analyse --tf output/-3vmxQet5LA-transcript.txt --prompt "1. what is the Joke Structure and 2. what is the memory palace technique"
    
    B. Embedding. Generates CSV file with vectors from any transcript (txt) file
    
    # this generates output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3-transcript_embedding.csv
    youtube2llm.py embed --tf output/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3.txt
    
    C. Asking the embedding file a question / query

    # without q(uery) specified, defaults to:
        "what questions can I ask about what's discussed in the video so I understand the main argument and points that the speaker is making? and for each question please answer each and elaborate them in detail in the same response"
    youtube2llm.py ask --ef output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3-transcript_embedding.csv

    # or pass on a query
    youtube2llm.py ask --ef output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3-transcript_embedding.csv --q "what is rule and what is norm"
    youtube2llm.py ask --ef output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media-transcript_embedding.csv --q "if we could distill the transcript into 4 arguments or points, what would they be?"
    
    # batching some video IDs in a text file (line-separated)
    while read -r f || [ -n "$f" ]; do; youtube2llm.py analyse --vid="$f" --nc --lmodel mistral; done < list_of_youtube_video_ids.txt
    
'''

import re
import os
import hashlib
import datetime
import tiktoken
import argparse
import whisper
import openai
import ollama
import ast # for converting embeddings saved as strings back to arrays
import pandas as pd # for storing text and embeddings data

from pathlib import Path
from scipy import spatial  # for calculating vector similarities for search
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL, DownloadError # I only use YoutubeDL to retrieve the chapters and the title of the video
# from openai.embeddings_utils import cosine_similarity, get_embedding # not used yet,
#       see https://platform.openai.com/docs/guides/embeddings/use-cases > text search using embeddings
#       and https://cookbook.openai.com/examples/recommendation_using_embeddings
# for some new methods introduced in https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb
from chunking_evaluation.chunking import (
    ClusterSemanticChunker,
    LLMSemanticChunker,
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker
)
from chunking_evaluation.utils import openai_token_count
from prompts import system_prompts, user_prompts

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v={}"

## summarization-related
GPT_MODEL = 'gpt-3.5-turbo'
GPT_MODEL = 'gpt-4o'

## embedding-related.
## reference: https://platform.openai.com/docs/guides/embeddings/embedding-models
EMBEDDING_MODEL = "text-embedding-ada-002" # 1536 output dimension, "Most capable 2nd generation embedding model, replacing 16 first generation models"
EMBEDDING_MODEL = "text-embedding-3-large" # 3072 output dimension, "Most capable embedding model for both english and non-english tasks"
EMBEDDING_MODEL = "text-embedding-3-small" # 1536 output dimension, "Increased performance over 2nd generation ada embedding model"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = "mxbai-embed-large"
BATCH_SIZE = 1000
# as per https://github.com/openai/openai-cookbook/blob/3f8d3f34054526173c0c9cd110d21d90fe993c3f/examples/Get_embeddings_from_dataset.ipynb or https://cookbook.openai.com/examples/get_embeddings_from_dataset
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191 # and BGE is 8192 https://pyvespa.readthedocs.io/en/latest/examples/mother-of-all-embedding-models-cloud.html
encoding = tiktoken.get_encoding(embedding_encoding)

## prep the LLM for summarization.
##   TODO: should probably put into summarize_text instead,
##   if the `client` object is only used there. but right now I'm using client globally
##   across the different modes supported by this script (summarize, embed, ask).
##   or can just initiate this object separately in each method. but that's sloppy.
##   so this works:
##   - if summarize mode, then GPT_MODEL can be overridden with ollama's 'mistral' for example
##   - if embed or ask mode, then GPT_MODEL stays hardcoded as 'gpt-*'

client = None
use_api = False # set to False to use ollama library or set to True to use ollama API
# I don't merge with the branching under main (to use OpenAI or local ollama (API/library)) because the logic became
#   a bit complicated. and I'm just benchmarking the difference between the two methods for ollama anyway

def extract_video_id(url_or_id):
    if url_or_id.startswith("http"):
        # full YouTube URL
        parsed_url = urlparse(url_or_id)
        if 'youtube.com' in parsed_url.netloc:
            query = parse_qs(parsed_url.query)
            return query.get('v', [None])[0]
        if 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
    return url_or_id  # assume it's already a video ID

def get_captions(video_id, language="en"):
    caption = ''
    raw_caption = ''
    transcripts = []
    
    '''
    Tried 3 methods before deciding on youtube_transcript_api
    
    # method #1: # yt-dlp (the new youtube-dl), blm berhasil, gak tau gmn caranya retrieve the subtitles without downloading them, not sure if it's even supported by yt-dlp
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        # See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    ydl = YoutubeDL(ydl_opts)
    info = ydl.extract_info(url, download=False)
    chapters = [c.get('title') for c in info.get('chapters')]
    ttml_urls = [c.get('url') for c in info.get('automatic_captions').get('en') if c.get('ext') == 'ttml']
    # then, just run requests.get on ttml_urls[0]?
    # e.g. https://www.youtube.com/api/timedtext?v=uA5GV-XmwtM&ei=eTSRZdnHHI2xjuMPyZeK-AE&caps=asr&opi=112496729&xoaf=5&hl=en&ip=0.0.0.0&ipbits=0&expire=1704040169&sparams=ip%2Cipbits%2Cexpire%2Cv%2Cei%2Ccaps%2Copi%2Cxoaf&signature=ED91E4A2584EBD0048F5FD22738F76C5FF1ABF11.364F3A4E50CA060E54F0784709DE38A2229E6757&key=yt8&kind=asr&lang=en&fmt=ttml
    # ydl.list_subtitles(video_id, info.get('automatic_captions'), 'subtitles') # this works, returns a table of available transcripts, but I'm not sure how to then get the transcript text

    # method #2 # with youtube_dl (old youtube-dl)
    ydl = youtube_dl.YoutubeDL({'writesubtitles': True, 'allsubtitles': True, 'writeautomaticsub': True})
    res = ydl.extract_info(url, download=False)
    if res['requested_subtitles'] and (res['requested_subtitles']['en'] or res['requested_subtitles']['a.en']):
        print('Grabbing vtt file from ' + res['requested_subtitles']['en']['url'])
        raw_caption = requests.get(res['requested_subtitles']['en']['url'], stream=True)
        caption = re.sub(r'\d{2}\W\d{2}\W\d{2}\W\d{3}\s\W{3}\s\d{2}\W\d{2}\W\d{2}\W\d{3}','',response.text)
        with f1 = open("caption.vtt", "w"):
            f1.write(caption)
        if len(res['subtitles']) > 0:
            print('manual captions')
        else:
            print('automatic_captions')
    else:
        print('Youtube Video does not have any english captions')

    # method #3: pytube. gak tau jg, sempet bisa tp skrg gak bs, entahlah
    yt = YouTube(YOUTUBE_VIDEO_URL)
    youtube_video.captions.get_by_language_code('a.en')
    '''

    from youtube_transcript_api import YouTubeTranscriptApi # and this for the transcript
    
    # method #4: https://pypi.org/project/youtube-transcript-api/
    # YouTubeTranscriptApi.get_transcript('uA5GV-XmwtM')
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception as e:
        print("An unexpected error occurred:", e)

    for t in transcripts:
        print ("transcript found for language: "+t.language_code)
        # wait, I'm returning the LAST transcript this video has? haha. wrong. but okay works most of the time, for EN language
        if(t.language_code == language.lower()): # ok fixed the above. mestinya ada method yg langsung akses based on language code sih, but I just want this to work for now
            # print(f"{t.language_code}, {language}")
            raw_caption = t.fetch()
    
    if raw_caption:
        caption = ' '.join([t.get('text') for t in raw_caption])

    return caption

def get_chapters(info):
    chapters = []
    if info.get('chapters'):
        chapters = [c.get('title', '') for c in info.get('chapters', {'title': ''})]
    return chapters

def get_youtube_metadata(video_id, save=True):
    ydl_opts = {}
    ydl = YoutubeDL(ydl_opts)
    info = ydl.extract_info(video_id, download=False)
    if(save):
        import json
        with(open('output/'+video_id+'-metadata.json', 'w') as f):
            f.write(json.dumps(info))
    return info

def process_youtube_video(url, video_id, language="en", force_download_audio=False, transcription_model='base'):
    caption = None
    transcript = ''
    video_title = ''
    chapters = []
    
    video_id = extract_video_id(video_id)
    
    if(not force_download_audio):
        caption = get_captions(video_id, language)
    
    import torch
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = whisper.load_model(transcription_model, device=devices)

    youtube_metadata = get_youtube_metadata(video_id, save=False) # returns a dict, optionally saves it into a {video_id}-metadata.json file
    chapters = get_chapters(youtube_metadata)
    video_title = youtube_metadata.get('title')
    
    if not caption or force_download_audio:
        # download the mp4 so Whisper can transcribe them
        from pytube import YouTube
        print(url)
        # trying to fix age restriction-related error
        # from pytube.innertube import _default_clients
        # _default_clients["ANDROID_MUSIC"] = _default_clients["WEB"]
        # youtube_video = YouTube(url, use_oauth=True, allow_oauth_cache=False)
        youtube_video = YouTube(url)
        video_id = youtube_video.vid_info.get('videoDetails').get('videoId')
        if(True): # use pytube
            # this breaks since 2024-05-07. open issue (with a quick fix workaround done on cipher.py) at: https://github.com/pytube/pytube/issues/1918
            streams = youtube_video.streams.filter(only_audio=True)
            stream = streams.first() # taking first object of lowest quality
            OUTPUT_AUDIO = Path(__file__).resolve().parent.joinpath('output', video_id+'.mp4')
            stream.download(filename=OUTPUT_AUDIO)
            transcription = model.transcribe(OUTPUT_AUDIO.as_posix(), verbose=True, fp16=False, language=language) # language="id", or "en" by default
        else: # use yt-dlp, based on one version of workaround in pytube/issues/#1918
            import tempfile
            with tempfile.TemporaryDirectory() as temporary_directory:
                audio_details = download_youtube_mp3(video_id, temporary_directory)
                '''
                with open(audio_details["file_path"], "rb") as f:
                    bytes_data = f.read()
                '''
                transcription = model.transcribe(audio_details["file_path"], verbose=True, fp16=False, language=language) # language="id", or "en" by default
        
        transcript = transcription['text']
    else:
        # youtube caption is available, either auto caption or the one included by the video poster
        transcript = caption

    return transcript, chapters, video_title

def download_youtube_mp3(video_id, temporary_directory):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "format": 'bestaudio/best',
        "max_filesize": 20 * 1024 * 1024,
        "outtmpl": f"{temporary_directory}/%(id)s.%(ext)s",
        "noplaylist": True,
        "verbose": True,
        "postprocessors": [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    ydl = YoutubeDL(ydl_opts)
    try:
        meta = ydl.extract_info(url, download=True)
    except DownloadError as e:
        raise e
    else:
        video_id = meta["id"]
        return {
            "title": meta["title"],
            "file_path": f"{temporary_directory}/{video_id}.mp3"
        }
        
def llm_process(transcript, llm_mode, chapters=[], use_chapters=True, prompt='', video_title='', llm_personality='doubtful_stylistic_british'):
    # need to refactor. I declare this as global because it's used to initiate a client object, which is used by all the llm_process, embed, and ask functionalities supported by this script
    global GPT_MODEL
    
    if llm_mode in['summary', 'kp']:
        llm_mode = 'summary'
    
    if(llm_mode == 'tag'):
        # GPT_MODEL = 'gpt-3.5-turbo-16k'
        GPT_MODEL = 'gpt-4o'
    
    system_prompt = system_prompts[llm_personality]
    
    user_prompt = user_prompts[llm_mode]
    
    if(prompt):
        user_prompt = prompt
    
    if(use_chapters and chapters):
        chapters_string = "\n".join(chapters)
        user_prompt += f" These are the different topics discussed in the conversation:\n{chapters_string}.\nPlease orient the summary and organise them into different sections based on each topic."
        print("user_prompt: "+user_prompt)

    # result=""
    # previous = ""
    result = "System Prompt: "+system_prompt+"\n\n-------\n"
    result = result + "User Prompt: "+user_prompt+"\n\n-------\n"
    if(video_title):
        result += "video_title: " + video_title+"\n\n-------\n"

    ## Chunking. For managing token limit, e.g.
    #   openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 36153 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
    # while the rate limit error is like this
    #   openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-3.5-turbo-16k in organization org-xxxx on tokens_usage_based per min: Limit 60000, Used 22947, Requested 45463. Please try again in 8.41s. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens_usage_based', 'param': None, 'code': 'rate_limit_exceeded'}}
    
    n = 1300
    if(GPT_MODEL == 'gpt-4o' or GPT_MODEL.endswith('-16k') or GPT_MODEL.endswith('-1106')):
        # n = 10000 # maximum context length is 16385 for gpt-3.5-turbo-16k, and 4097 for gpt-3.5-turbo
        n = 5300 # the response was stifled when it was 10k before
    print(f"n used: {n}")
    
    # start chunking
    
    '''
    # v0: my original version
    st = transcript.split()
    snippet= [' '.join(st[i:i+n]) for i in range(0,len(st),n)]
    print(f"snippets generated (old): {len(snippet)}")
    
    # trying two new methods, based on https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb
    
    # v1:
    from chromadb.utils import embedding_functions
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_KEY"], model_name="text-embedding-3-large")
    cluster_chunker = ClusterSemanticChunker(
        embedding_function=embedding_function, 
        max_chunk_size=10000, # if use `n` 1300, will produce 7 chunks (more than 2 the original and recursive one). if set to higher (5k), generates 5 chunks. idk if I set 20k...? still generates 5. interesting. the original video has 8-9 segments btw
        length_function=openai_token_count
    )
    snippet = cluster_chunker.split_text(transcript)
    print(f"snippets generated (cluster_chunker): {len(snippet)}")
    '''
    
    # v2:
    recursive_token_chunker = RecursiveTokenChunker(
        chunk_size=n,
        chunk_overlap=0,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""] # According to Research at https://research.trychroma.com/evaluating-chunking
    )
    snippet = recursive_token_chunker.split_text(transcript)
    print(f"snippets generated (recursive_token_chunker): {len(snippet)}")
    
    # end chunking
    
    ## start sending the values to the LLM
    
    for i in range(0, len(snippet), 1):
        print("Processing transcribed snippet {} of {}".format(i+1,len(snippet)))
        if('gpt' not in GPT_MODEL and use_api is False):
            # result += "use ollama library\n----\n"
            try:
                # to try this, https://github.com/ollama/ollama/issues/2929#issuecomment-2327457681
                options = ollama.Options(temperature=0.0, top_k=30, top_p=0.8, num_thread=8)
                gpt_response = ollama.chat( # err, but I'm still using this endpoint? not the OpenAI's "client.chat.completions.create..." way
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt + "\n\n\"" + "Video Title: " + video_title + "\n\n" + snippet[i] + "\"\n Do not include anything that is not in the transcript."
                            # "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the transcript. For additional context here is the previous written message: \n " + previous
                        }
                    ], options=options
                )
                # previous = gpt_response['message']['content']
                # result += gpt_response.choices[0].message.content + "\n\n-------\n\n(based on the snippet: "+ snippet[i]+")\n\n-------\n\n"
                result += gpt_response['message']['content'] + "\n\n-------\n\n"
            except Exception as e:
                print("An unexpected error occurred:", e)
                print(result)
        else:
            try:
                response = client.chat.completions.create( # new API
                    model=GPT_MODEL,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {"role": "user", "content": user_prompt + "\n\n\"" + "Video Title: " + video_title + "\n\n" + snippet[i]}
                        # {"role": "user", "content": "\"" + snippet[i] + "\"\n For additional context here is the previous written message: \n " + previous}
                    ],
                    # max_tokens=4096,
                    temperature=0
                )
                # previous = response.choices[0].message.content
                result += response.choices[0].message.content+"\n\n-----\n\n"
                # print(f"result appended: {result}")
            except Exception as e:
                print("An unexpected error occurred:", e)
                print(result)

    return result

def create_embedding(transcript, embedding_filename):
    # following and migrated based on
    #   https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_Wikipedia_articles_for_search.ipynb > 
    #       3. Embed document chunks and 4. Store document chunks and embeddings
    # Usage stat: Tokens used: 16,274 from a 17426 words document (76016 chars)
    embeddings = []
    transcripts = []
    SAVE_PATH = "output/embeddings/"+embedding_filename.replace('.txt', '')+"-transcript_embedding.csv"
    
    # ok.... kyknya gini... si transcript ini perlu gw split jadi list of strings dulu... nah yg diproses di bawah ini seharusnya adalah list, NOT the string....
    # https://github.com/Azure/azure-openai-samples/blob/main/use_cases/passage_based_qna/day_in_life_of_data_scientist.ipynb # some good method, with normalize_text
    # atau kyk gini awalnya dah bener ya, cukup looping sekali, tapi bukan extend melainkan append, perhaps that's the only problem earlier?
    # this seems worth checking too, https://cookbook.openai.com/examples/embedding_long_inputs
    
    for batch_start in range(0, len(transcript), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = transcript[batch_start:batch_end]
        # print(f"Chunking batch {batch_start} to {batch_end-1}")
        # print(f"Chunked content: {batch}")
        transcripts.append(batch)

    print(f"{len(transcript)} split into {len(transcripts)} strings.")

    for batch_start in range(0, len(transcripts)): # ini gak perlu pake BATCH_SIZE lagi dong, karena each string in the transcripts list is already < affordable token length
        batch_end = batch_start + 1
        batch = transcripts[batch_start:batch_end]
        # print(f"Running batch {batch_start} to {batch_end-1}")
        # print(f"Batch content: {batch}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # print(f"response {response}")
        for i, be in enumerate(response.data):
            # print(f"(i, be.index): {(i, be.index)} and len(be.embedding): {len(be.embedding)}")
            assert i == be.index  # double check embeddings are in same order as input
        batch_embeddings = [e.embedding for e in response.data]
        # print("len(embeddings) before: {}".format(len(embeddings)))
        embeddings.extend(batch_embeddings)
        # print("len(embeddings) after: {}".format(len(embeddings)))

    df = pd.DataFrame({"text": transcripts, "embedding": embeddings})
    df.to_csv(SAVE_PATH, index=False, mode="w")

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unsupported models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query, )
    query_embedding = query_embedding_response.data[0].embedding
    
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    # print(f"strings_and_relatednesses: {strings_and_relatednesses}")
    # e.g. [(" Welcome back. I took a couple weeks off, but it's good to be back here. .... We'll see you next time.", 0.7000641373282386)]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def strings_ranked_by_relatedness_ollama(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """
    Returns a list of strings and relatednesses, sorted from most related to least.
    Still need a separate method because there's still yet OpenAI-compatible API for ollama embedding (right?)
    """
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    query_embedding = response.get('embedding', [])

    if not query_embedding:
        print("Failed to generate embedding for the query.")
        return

    similarities = [
        (row['text'], relatedness_fn(query_embedding, row['embedding']))
        for _, row in df.iterrows()
    ]

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*similarities)
    return strings[:top_n], relatednesses[:top_n]
    '''
    top_matches = similarities[:top_n]

    # Display results
    print("Top matches:")
    for text, similarity in top_matches:
        print(f"Similarity: {similarity:.4f}\nText: {text}\n")
    '''

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    if('nomic' in EMBEDDING_MODEL or 'mxbai' in EMBEDDING_MODEL):
        strings, relatednesses = strings_ranked_by_relatedness_ollama(query, df, top_n=5)
    else:
        strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=5)
    # print(f"checking strings_ranked_by_relatedness for query: {query}")
    # introduction = 'Use the below transcript of a podcast episode to answer the subsequent question. If the answer cannot be found in the text, write "I could not find an answer."'
    introduction = 'Use the below transcript of a podcast episode to answer the subsequent question."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        # print(f"string: {string} with length of: "+str(len(string)))
        next_article = f'\nTranscript section:\n"""\n{string}\n"""'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            # message += string[:token_budget] # blm yakin gw ini dia lg ngapain ini wkwwk
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT (or ollama? I am not sure if this handles the ollama switch. but perhaps yes because there's already OpenAI-compatible endpoint for chat in ollama, and I have initiated the `client` accordingly as a global thing') and a dataframe of relevant texts and embeddings."""
    response_message = ''
    message = query_message(query, df, model=model, token_budget=token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about the podcast episode."},
        {"role": "user", "content": message},
    ]

    if('gpt' not in model and use_api is False):
        response = ollama.chat(
            model=model, # whichever ollama model was specified
            messages=messages,
            # options=options
        )
        response_message = response['message']['content']
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        response_message = response.choices[0].message.content
        
    if print_message:
        print("ask -> user prompt -> content: " + message)
        print(f"ask's response: {response_message}")
    
    return response_message

def ask_the_embedding(question, embeddings_filename, print_message=False):
    # following this tut https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb?ref=mlq.ai
    df = pd.read_csv(embeddings_filename) # read as dataframe
    # convert embeddings from CSV str type back to list type
    df['embedding'] = df['embedding'].apply(ast.literal_eval) # df has two columns: "text" and "embedding"
    return ask(question, df, model=GPT_MODEL, print_message=print_message)

def create_embedding_ollama(transcript, embedding_filename):
    """Creates embeddings using Ollama's mxbai-embed-large or nomic model and saves to a CSV."""
    SAVE_PATH = "output/embeddings/"+embedding_filename.replace('.txt', '')+"-transcript_embedding.csv"

    embeddings_list = []
    transcripts = []

    for batch_start in range(0, len(transcript), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = transcript[batch_start:batch_end]
        transcripts.append(batch)

    print(f"Transcript split into {len(transcripts)} chunks.")

    for batch in transcripts:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=batch)
        embedding = response.get('embedding', [])
        if embedding:
            embeddings_list.append(embedding)

    # Save embeddings to CSV
    df = pd.DataFrame({"text": transcripts, "embedding": embeddings_list})
    df.to_csv(SAVE_PATH, index=False, mode="w")
    print(f"Embeddings saved to {SAVE_PATH}")

def ask_the_embedding_ollama(question, embeddings_filename, print_message=True):
    """Query embeddings using cosine similarity."""
    # Load the embeddings from CSV
    df = pd.read_csv(embeddings_filename)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return ask(question, df, model=GPT_MODEL, print_message=print_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyse, embed, and ask anything about the content of the youtube video')
    parser.add_argument('action', type=str, default='ask', help='analyse, embed, ask')
    parser.add_argument('--vid', type=str, help='youtube video id') # making it optional so we can create embedding for any transcript
    parser.add_argument('--q', type=str, help='your question')
    parser.add_argument('--tf', type=str, help='transcript filename, if exists, to create embedding for. in this case, value of vid is ignored')
    parser.add_argument('--ef', type=str, help='embedding filename to use')
    # only QnAs mode is implemented at the moment. want to merge with the llm_process method in audio2llm.py
    parser.add_argument('--tmodel', type=str, default='base', help='the Whisper model to use for transcription (tiny/base/small/medium/large. default: base)')
    parser.add_argument('--mode', type=str, default='QnAs', help='QnAs, note, summary/kp, tag, topix, thread, tp, cbb, definition, distinctions, misconceptions, ada, translation')
    parser.add_argument('--lmtone', type=str, default='default', help="customise LLM's tone. doubtful_stylistic_british is one you can use")
    parser.add_argument('--lmodel', type=str, default='gpt-4o', help='the GPT model to use for summarization (default: gpt-4o)')
    parser.add_argument('--prompt', type=str, help='prompt to use, but chapters will be concatenated as well')
    parser.add_argument('--nc', action='store_true', help="don't pass chapters to analyse")
    parser.add_argument('--dla', action='store_true', help="download and transcribe the audio, don't use youtube auto caption")
    parser.add_argument('--lang', type=str, default='en', help='language code of the video')
    
    time_begin = datetime.datetime.now() # in audio2llm.py I put this inside the llm_process method because it saves the text file in the method, unlike here
    
    args = parser.parse_args()
    
    force_download_audio = False
    if(args.dla):
        force_download_audio = True
    
    if(args.lmodel): # hackish because I haven't made GPT_MODEL local (still global)
        GPT_MODEL = args.lmodel

    if('gpt' in GPT_MODEL or 'o3' in GPT_MODEL):
        openai.api_key = os.getenv("OPENAI_KEY")
        client = openai.OpenAI(api_key=openai.api_key)
    else:
        # use ollama. perhaps add support for (simonw's) llm some time. or huggingface's (via sentence_transformer?)
        client = openai.OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )

    if args.action == 'analyse':
        
        chapters = []
        transcript = ''
        llm_result = ''
        
        if(args.tf and os.path.exists(args.tf)): # no need to retrieve the transcript from youtube when it has been provided
            video_id = os.path.basename(args.tf)
            with open(args.tf) as f:
                transcript = f.read()
        else:
            video_id = extract_video_id(args.vid)
            transcript, chapters, video_title = process_youtube_video(YOUTUBE_VIDEO_URL.format(video_id), video_id, language=args.lang, force_download_audio=force_download_audio, transcription_model=args.tmodel)
            with open('output/'+video_id+'-transcript.txt', "w") as f:
                f.write(transcript)
        
        print(f'Transcript acquired of length: \n{str(len(transcript))}')
        with_chapters = True
        if(args.nc):
            with_chapters = False
        
        mode = args.mode
        if(args.prompt):
            llm_result += f"prompt: {args.prompt}\n\n------\n\n"
            print(f"prompt: {args.prompt}")
            mode = f"prompt-{hashlib.md5(args.prompt.encode('utf-8')).hexdigest()}"
        
        llm_result = llm_process(transcript, llm_mode=args.mode, chapters=chapters, use_chapters=with_chapters, prompt=args.prompt, video_title=video_title, llm_personality=args.lmtone)

        time_end = datetime.datetime.now()
    
        print(f"time taken: {str(time_end - time_begin)}")
        llm_result += f"\n\n-----\n\ntime taken: {str(time_end - time_begin)}"

        print(f'LLM result for the Video:\n{llm_result}')
        llm_result_filename = f"output/{video_id}-{mode}-{args.lmodel}.md"
        with open(llm_result_filename, "w") as f:
            f.write(llm_result)
    elif args.action == 'embed': # not strong enough, TODO to refactor
        
        if args.vid:
            transcript_id = extract_video_id(args.vid) # need this to construct the below. TODO: refactor so the file naming is more structured and simple
            transcript_filename = 'output/'+transcript_id+'-transcript.txt'
        
        if(args.tf and os.path.exists(args.tf)): # override the value of vid if tf was provided
            transcript_filename = args.tf
            transcript_id = os.path.basename(transcript_filename)
        
        print(f'Creating embedding from transcript: \n{transcript_filename}, transcript_id: {transcript_id}')
        
        with open(transcript_filename) as f:
            transcript = f.read()
        # print(f'Transcript acquired: \n{transcript}')
        
        if('text-embedding' in EMBEDDING_MODEL): # uses OpenAI's embedding model
            create_embedding(transcript, transcript_id)
        elif('nomic' in EMBEDDING_MODEL or 'mxbai' in EMBEDDING_MODEL): # uses ollama's local embedding model
            create_embedding_ollama(transcript, transcript_id)

    elif args.action == 'ask':
        
        question = "what questions can I ask about what's discussed in the video so I understand the main argument and points that the speaker is making? and for each question please answer each and elaborate them in detail in the same response"
        '''
        question = 'what are the three questions that the video provide answer for? and for each question please answer each and elaborate them in detail in the same response'
        another good one: "what are all arguments that the essay is making?"
        '''

        if args.q:
            question = args.q
        if args.ef and os.path.isfile(args.ef):
            answer = question + "\n======\n"
            # embeddings_filename = "output/embeddings/"+video_id+"-transcript_embedding.csv"
            
            if('nomic' in EMBEDDING_MODEL or 'mxbai' in EMBEDDING_MODEL):
                answer += ask_the_embedding_ollama(question, args.ef)
            else:
                answer += ask_the_embedding(question, args.ef)
            
            embeddings_filename = args.ef
            print(f'Answer:\n{answer}')
            import re
            pattern = r"([^/]+)-transcript-transcript_embedding.csv" # not so robust way of inferring the video_id from the embedding file, which is just gonna be used to save the answer file sih
            match_video_id = re.search(pattern, embeddings_filename.split('/')[-1])

            if match_video_id:
                video_id = match_video_id.group(1)
                with open('output/'+video_id+'-answer-'+hashlib.md5(question.encode('utf-8')).hexdigest()+'.txt', "w") as f:
                    f.write(answer)
            else:
                print(f"Cannot extract video_id from {embeddings_filename}.")

'''
TODO
√ 0. implement some [embedding](https://www.mlq.ai/openai-whisper-gpt-3-fine-tuning-youtube-video/) so I can dig deeper into bits like
    "tell me more about The conversation emphasizes the importance of asking good questions and how they can lead to breakthroughs in writing."
    which, would be a good next step to getting that "living blog" concept implemented
√ 1. add argparse
√ 2. chunk the text properly, consider proper sentences (NLTK) # and clean it up too?
    # I have implemented this in summarize.py and md2notes.py,
    #  but youtube transcripts (auto-caption) is another beast (no punctuations),
    #  so it's sort of useless to do more advanced chunking unless the transcript is generated using Whisper
    #  perhaps some overlap would be beneficial tho? i.e. use the langchain text_splitters
    https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    https://platform.openai.com/tokenizer
√ 3. save the transcript and summary/note to file to save some token and processing costs when doing this trial
√ 4. perhaps for other project: to retrieve web articles and summarise (similar structure with this but use readability and stuff) # done in url2markdown.py
    kyk tool ini https://totheweb.com/learning_center/tools-convert-html-text-to-plain-text-for-content-review/
    read a file containing a list of line-separated URLs, retrieve the text, embed, draw the underlying theme, generate a list of talking points, questions that can be answered from these articles
    help me see wider, deeper, clearer
√ 5. tune the prompt, still seems to be patchy, not as good as audio2llm.py. or is it the temperature?
    https://platform.openai.com/docs/guides/prompt-engineering/strategy-provide-reference-text
    or https://github.com/e-johnstonn/GPT-Doc-Summarizer/blob/master/my_prompts.py
√ 6. get timestamped response to dig in. e.g. "on minute xyz they discussed this point" (perhaps incorporate as answer to my convo as per #1 ?) get ReAct / agent involved?
√ 7. try this https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_graph_db.ipynb # implemented elsewhere (see *embedding*.py)
√ 8. and this https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb # implemented elsewhere (see *embedding*.py)
~ 9. implement / port this on other local model (mistral, llama2, etc) so the cost is managed
    the embed and ask results are pretty shit with ollama mistral. see create_embedding_ollama
        oh, ollama mistral nggak utk embedding kyknya sih. need to probably do the bge version I had working... somewhere... haha
    here, find chromadb https://howaibuildthis.substack.com/p/choosing-the-right-embedding-model
    2024-04-02:
    - done for the `summarize` mode
    - still TODO for the `ask` and `embed` modes
√ 10. image recognition? I want to process my screenshots and stuff # got a version of LLaVA working locally and two scripts using OpenAI's Vision for image and video
    start here:
	* https://huggingface.co/docs/transformers/main/tasks/image_captioning
	* https://llava-vl.github.io/ # the most ready to use. but not perfect
	* https://huggingface.co/blog/blip-2
    * https://simonwillison.net/2023/Sep/12/llm-clip-and-chat/
        CLIP is a pretty old model at this point, and there are plenty of interesting alternatives that are just waiting for someone to wrap them in a plugin.
        I’m particularly excited about Facebook’s ImageBind, which can embed images, text, audio, depth, thermal, and IMU data all in the same vector space!
	* https://github.com/facebookresearch/ImageBind # to try the tutorial https://itnext.io/imagebind-one-embedding-space-to-bind-them-all-b48c8623d39b
	* https://platform.openai.com/docs/guides/vision # reading and processing images
	* https://platform.openai.com/docs/guides/images/introduction?context=node # generating images
√ 11. the LLM output of youtube2llm.py is quite high-level, vague, and not as good (refined, granular, precise) as the audio2llm.py.
    need to check if it's the prompt, the chunking strategy, the context window, or what. seems like it's a combination all of them sih.
    OK refactored the prompts and modes out of audio2llm.py to a separate file (prompts.py) and allow youtube2llm.py to use them.
    chunking method and size is already the same between these scripts
12. refactor the output directory # can't remember what this is about, is it just adding output_dir to the args? or to not hardcode some of the "/output"... paths here?
13. allow running an `ask` mode on a video without having to explicitly call `analyse` and then `embed` beforehand
    i.e.:
        ./youtube2llm.py analyse --vid=zt6i6vVgiO4 --nc --lmodel mistral
        ./youtube2llm.py embed --tf=output/zt6i6vVgiO4-transcript.txt
        ./youtube2llm.py ask --ef=output/embeddings/zt6i6vVgiO4-transcript-transcript_embedding.csv --q "what are the three depression subtypes and the way depression manifests in Earth, Wind, or Fire types?"
    became
        ./youtube2llm.py ask --vid=zt6i6vVgiO4 --nc --lmodel mistral --q "what are the three depression subtypes and the way depression manifests in Earth, Wind, or Fire types?"


14. when trying to use o1-preview, got:
An unexpected error occurred: Error code: 400 - {'error': {'message': "Your organization must qualify for at least usage tier 5 to access 'o1-preview'. See https://platform.openai.com/docs/guides/rate-limits/usage-tiers for more details on usage tiers.", 'type': 'invalid_request_error', 'param': 'model', 'code': 'below_usage_tier'}}
// lol, https://platform.openai.com/docs/guides/rate-limits/usage-tiers > "Tier 5	$1,000 paid and 30+ days since first successful payment"
// and https://platform.openai.com/settings/organization/limits shows I'm currently tier 2. haha
'''
