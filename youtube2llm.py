#!/usr/bin/env python3

'''
Overview:
    Script is developed based on: https://github.com/mirabdullahyaser/Summarizing-Youtube-Videos-with-OpenAI-Whisper-and-GPT-3
    I added the functionality to get the transcript from YouTube when available to save Whisper some work.

Usage
    A. Summarising (after retrieving YouTube's subtitles or transcribing with Whisper)
    
    # summarise a YouTube video, passing the video ID
    youtube2llm.py summarise --vid 2JmfDKOyQcI
    
    # transcribe and summarise a podcast episode (or any audio file)
    youtube2llm.py summarise --af ~/Documents/Podcasts/d7a205f9-90af-44df-a37b-69505a3da691_transcoded.mp3 --mode transcription

    # summarise a transcript file
    youtube2llm.py summarise --tf output/FbquCdNZ4LM-transcript.txt

    # don't send chapters, when it's making it too verbose or fragmented
    youtube2llm.py summarise --vid=FbquCdNZ4LM --nc
    
    youtube2llm.py summarise --vid PDpyUMOOcyw --lang id # vid has no auto caption, so we specify Indonesian language so Whisper's output is better
    youtube2llm.py summarise --vid=TVbeikZTGKY --nc  # the video has no subtitle, so Whisper goes on to transcribe it

    youtube2llm.py summarise --vid=FbquCdNZ4LM --nc # a video that has age filter on
        pytube.exceptions.AgeRestrictedError: FbquCdNZ4LM is age restricted, and can't be accessed without logging in.
        in this case, I will use `yt-dlp -f` to get the audio file (m4a or mp4) and then run the audio file through summarize.py to transcribe it with Whisper
        another example of such video: https://www.youtube.com/watch?v=77ivEdhHKB0
    
    youtube2llm.py summarise --vid=sYmCnngKq00 --nc --lmodel mistral # uses mistral (via ollama) for summarisation. the script will use OpenAI's API for the ask and embed mode (still TODO)
    
    # run a custom prompt against an audio file (first it will retrieve the transcript from YouTube or generate the transcript using Whisper)
    youtube2llm.py summarise --vid="-3vmxQet5LA" --prompt "all the public speaking techniques and tactics shared"
    
    # run a custom prompt against a transcript file
    youtube2llm.py summarise --tf output/-3vmxQet5LA-transcript.txt --prompt "1. what is the Joke Structure and 2. what is the memory palace technique"
    
    B. Embedding. Generates CSV file with vectors from any transcript (txt) file
    
    # this generates output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3-transcript_embedding.csv
    youtube2llm.py embed --tf output/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3.txt
    
    C. Asking the embedding file a query

    # without q(uery) specified, defaults to:
        "what questions can I ask about what's discussed in the video so I understand the main argument and points that the speaker is making? and for each question please answer each and elaborate them in detail in the same response"
    youtube2llm.py ask --ef output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3-transcript_embedding.csv

    # or pass on a query
    youtube2llm.py ask --ef output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media.mp3-transcript_embedding.csv --q "what is rule and what is norm"
    youtube2llm.py ask --ef output/embeddings/452f186b-54f2-4f66-a635-6e1f56afbdd4_media-transcript_embedding.csv --q "if we could distill the transcript into 4 arguments or points, what would they be?"
    
    # batching some video IDs in a text file (line-separated)
    while read -r f || [ -n "$f" ]; do; youtube2llm.py summarise --vid="$f" --nc --lmodel mistral; done < list_of_youtube_video_ids.txt
    
'''

import os
import tiktoken
import argparse
import whisper
import openai
import ast # for converting embeddings saved as strings back to arrays
import pandas as pd # for storing text and embeddings data

from pathlib import Path
from scipy import spatial  # for calculating vector similarities for search
from yt_dlp import YoutubeDL # I only use YoutubeDL to retrieve the chapters and the title of the video

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v={}"

## summarization-related
WHISPER_MODEL = 'base' # tiny base small medium large
GPT_MODEL = 'gpt-3.5-turbo'

## embedding-related
EMBEDDING_MODEL = "text-embedding-ada-002"
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

use_api = False # set to False to use ollama library or set to True to use ollama API

# I don't merge with the branching below (to use OpenAI or local ollama (API/library)) because the logic became
#   a bit complicated. and I'm just benchmarking the difference between the two methods for ollama anyway
if('gpt' in GPT_MODEL):
    openai.api_key = os.getenv("OPENAI_KEY")
    client = openai.OpenAI(api_key=openai.api_key)
else:
    # use ollama. perhaps add support for (simonw's) llm some time. or huggingface's (via sentence_transformer?)
    client = openai.OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )

## begin WiP: prep the LLM for embedding. want to implement ollama-based embedding. see create_embedding_ollama
## status: doesn't work yet, ollama mistral doesn't support embedding. I think I've got it working with bge in one of the llama index scripts tho
## end WiP

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
        if(t.language_code == language): # ok fixed the above. mestinya ada method yg langsung akses based on language code sih, but I just want this to work for now
            raw_caption = t.fetch()
    
    if raw_caption:
        caption = ' '.join([t.get('text') for t in raw_caption])

    return caption

def get_chapters(info):
    chapters = []
    if info.get('chapters'):
        chapters = [c.get('title', '') for c in info.get('chapters', {'title': ''})]
    return chapters

def get_youtube_metadata(url, save=True):
    ydl_opts = {}
    ydl = YoutubeDL(ydl_opts)
    info = ydl.extract_info(url, download=False)
    if(save):
        import json
        with(open('output/'+video_id+'-metadata.json', 'w') as f):
            f.write(json.dumps(info))
    return info

def process_youtube_video(url, video_id, language="en"):
    transcript = ''
    video_title = ''
    chapters = []
    
    caption = get_captions(video_id, language)
    youtube_metadata = get_youtube_metadata(video_id, save=True) # returns a dict, optionally saves it into a {video_id}-metadata.json file
    chapters = get_chapters(youtube_metadata)
    video_title = youtube_metadata.get('title')
    
    if caption:
        # youtube caption is available, either auto caption or the one included by the video poster
        transcript = caption
    else:
        # download the mp4 so Whisper can transcribe them
        from pytube import YouTube
        youtube_video = YouTube(url)
        video_id = youtube_video.vid_info.get('videoDetails').get('videoId')
        streams = youtube_video.streams.filter(only_audio=True)
        # taking first object of lowest quality
        stream = streams.first()
        OUTPUT_AUDIO = Path(__file__).resolve().parent.parent.joinpath('data', video_id+'.mp4')
        stream.download(filename=OUTPUT_AUDIO)
        
        import torch
        devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        model = whisper.load_model(WHISPER_MODEL, device=devices)
        transcription = model.transcribe(OUTPUT_AUDIO.as_posix(), verbose=True, fp16=False, language=language) # language="id", or "en" by default
        transcript = transcription['text']

    return transcript, chapters, video_title

def summarize_text(transcript, chapters, use_chapters=True, prompt='', video_title=''):
    system_prompt = "You are a researcher who needs to generate a concise note out of the transcript below. " \
    "Do not summarize and keep every information. Do not include anything that is not in the transcript. " \
    "Make sure the note has useful and factual information about all the points of the discussion. " \
    "Please use bullet points to list all the relevant details."
    
    if(prompt):
        system_prompt = prompt
    
    if(use_chapters and chapters):
        chapters_string = "\n".join(chapters)
        system_prompt += f" These are the different topics discussed in the conversation:\n{chapters_string}.\nPlease orient the summary and organise them into different sections based on each topic."
        print("system_prompt: "+system_prompt)

    # result=""
    # previous = ""
    result = "System Prompt: "+system_prompt+"\n-------\n"
    if(video_title):
        result += "video_title: " + video_title+"\n-------\n"

    ## Chunking. For managing token limit, e.g.
    #   openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 36153 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
    # while the rate limit error is like this
    #   openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-3.5-turbo-16k in organization org-xxxx on tokens_usage_based per min: Limit 60000, Used 22947, Requested 45463. Please try again in 8.41s. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens_usage_based', 'param': None, 'code': 'rate_limit_exceeded'}}
    
    n = 1300
    if(GPT_MODEL.endswith('-16k') or GPT_MODEL.endswith('-1106')):
        # n = 10000 # maximum context length is 16385 for gpt-3.5-turbo-16k, and 4097 for gpt-3.5-turbo
        n = 5300 # the response was stifled when it was 10k before
    print(f"n used: {n}")
    st = transcript.split()
    snippet= [' '.join(st[i:i+n]) for i in range(0,len(st),n)]
        
    ## start sending the values to the LLM
    
    for i in range(0, len(snippet), 1):
        print("Processing transcribed snippet {} of {}".format(i+1,len(snippet)))
        if('gpt' not in GPT_MODEL and use_api is False):
            import ollama
            # result += "use ollama library\n----\n"
            try:
                gpt_response = ollama.chat(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the transcript."
                            # "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the transcript. For additional context here is the previous written message: \n " + previous
                        }
                    ]
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
                        {"role": "user", "content": "\"" + snippet[i]}
                        # {"role": "user", "content": "\"" + snippet[i] + "\"\n For additional context here is the previous written message: \n " + previous}
                    ],
                    max_tokens=4096,
                    temperature=0
                )
                # previous = response.choices[0].message.content
                result += response.choices[0].message.content+"\n-----\n"
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

def create_embedding_ollama(transcript, embedding_filename):
    # DOESN'T WORK yet. retrieval is so off, irrelevant chunks are returned
    # scratchpad: 20240226-langchain_text_splitter_n_ollama_embedding.py
    '''
    want to adapt to use ollama local (with everythinglm or mistral?)
        https://python.langchain.com/docs/integrations/text_embedding/ollama

    >>> embeddings = OllamaEmbeddings(model="everythinglm")
    >>> doc_result = embeddings.embed_documents([text])
    >>> len(doc_result)
    1
    >>> len(doc_result[0])
    5120
    >>> doc_result[0][:5]
    [-0.4859403967857361, 0.8739838004112244, 0.7551742196083069, -0.16546519100666046, 1.2470595836639404]
    >>> query_result = embeddings.embed_query("some query")

    >>> embeddings_mistral = OllamaEmbeddings(model="mistral")
    >>> doc_result_mistral = embeddings_mistral.embed_documents([text])
    >>> len(doc_result_mistral[0])
    4096
    >>> query_result_mistral = embeddings_mistral.embed_query("gw kemana aja selama bulan ini")
    '''
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS # testing
    from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter

    text = transcript
    embeddings = OllamaEmbeddings(model="mistral")
    # docs = embeddings.embed_documents([text])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap = 100)
    # docs = text_splitter.split_text(text)
    docs = text_splitter.create_documents([text])
    print(f"created {len(docs)} chunks")
    for i, doc in enumerate(docs):
        print(i, len(doc.page_content))
    
    db = FAISS.from_documents(docs, embeddings) # testing
    db.save_local('output/embeddings/'+transcript_id.replace('-transcript.txt', ''))
    '''
    # the searching part
    db = FAISS.load_local(transcript_id.replace('-transcript.txt', ''), embeddings)

    query = "keeper test" # what is being discussed about keeper test (the episode from lenny - netflix CTO elizabeth stone)

    # result = db.similarity_search(query)
    result = db.similarity_search_with_score(query)
    print(result)

    ## kalo pake embedding vector, bukan string
    embedding_vector = embeddings.embed_query(query)
    docs_and_scores = db.similarity_search_by_vector(embedding_vector)
    '''
    return

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
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

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
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
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about the podcast episode."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5
    )
    
    if print_message:
        print("ask -> user prompt -> content: " + message)
        print(f"ask's response: {response}")
    
    response_message = response.choices[0].message.content
    return response_message

def ask_the_embedding(question, embeddings_filename, print_message=False):
    # following this tut https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb?ref=mlq.ai
    df = pd.read_csv(embeddings_filename) # read as dataframe
    # convert embeddings from CSV str type back to list type
    df['embedding'] = df['embedding'].apply(ast.literal_eval) # df has two columns: "text" and "embedding"
    return ask(question, df, print_message=print_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='utub - get transcript, summarise, embed, ask')
    parser.add_argument('mode', type=str, default='ask', help='summarise, embed, ask')
    parser.add_argument('--vid', type=str, help='youtube video id') # making it optional so we can create embedding for any transcript
    parser.add_argument('--q', type=str, help='your question')
    parser.add_argument('--tf', type=str, help='transcript filename, if exists, to create embedding for. in this case, value of vid is ignored')
    parser.add_argument('--ef', type=str, help='embedding filename to use')
    parser.add_argument('--lmodel', type=str, default='gpt-3.5-turbo', help='the GPT model to use for summarization (default: gpt-3.5-turbo)')
    parser.add_argument('--prompt', type=str, help='prompt to use, but chapters will be concatenated as well')
    parser.add_argument('--nc', action='store_true', help="don't pass chapters to analyse")
    parser.add_argument('--lang', type=str, default='en', help='language code of the video')
    
    args = parser.parse_args()
    
    if args.mode == 'summarise':
        if(args.lmodel):
            GPT_MODEL = args.lmodel
        chapters = []
        transcript = ''
        if(args.tf and os.path.exists(args.tf)): # no need to retrieve the transcript from youtube when it has been provided
            video_id = os.path.basename(args.tf)
            with open(args.tf) as f:
                transcript = f.read()
        else:
            video_id = args.vid
            transcript, chapters, video_title = process_youtube_video(YOUTUBE_VIDEO_URL.format(video_id), video_id, language=args.lang)
            with open('output/'+video_id+'-transcript.txt', "w") as f:
                f.write(transcript)
        
        print(f'Transcript acquired of length: \n{str(len(transcript))}')
        with_chapters = True
        if(args.nc):
            with_chapters = False
        
        summary = summarize_text(transcript, chapters, use_chapters=with_chapters, prompt=args.prompt, video_title=video_title)
        print(f'Summary for the Video:\n{summary}')
        with open('output/'+video_id+'-summary.md', "w") as f:
            f.write(summary)
    elif args.mode == 'embed': # not sdtrong enough, TODO to refactor
        if args.vid:
            transcript_id = args.vid # need this to construct the below. TODO: refactor so the file naming is more structured and simple
            transcript_filename = 'output/'+transcript_id+'-transcript.txt'
        if(args.tf and os.path.exists(args.tf)): # override the value of vid if tf was provided
            transcript_filename = args.tf
            transcript_id = os.path.basename(transcript_filename)
        print(f'Creating embedding from transcript: \n{transcript_filename}, transcript_id: {transcript_id}')
        with open(transcript_filename) as f:
            transcript = f.read()
        # print(f'Transcript acquired: \n{transcript}')
        if('gpt' in GPT_MODEL and 'ada-002' in EMBEDDING_MODEL):
            create_embedding(transcript, transcript_id)
        else:
            # still WiP!
            create_embedding_ollama(transcript, transcript_id)
    elif args.mode == 'ask':
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
            answer += ask_the_embedding(question, args.ef)
            embeddings_filename = args.ef
            print(f'Answer:\n{answer}')
            import hashlib, re
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
√ 5. tune the prompt, still seems to be patchy, not as good as the other one (wsgi_new or summarise.py). or temperature ya?
    https://platform.openai.com/docs/guides/prompt-engineering/strategy-provide-reference-text
    or https://github.com/e-johnstonn/GPT-Doc-Summarizer/blob/master/my_prompts.py
√ 6. get timestamped response to dig in. e.g. "on minute xyz they discussed this point" (perhaps incorporate as answer to my convo as per #1 ?) get ReAct / agent involved?
√ 7. try this https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_graph_db.ipynb # implemented elsewhere (see *embedding*.py)
√ 8. and this https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb # implemented elsewhere (see *embedding*.py)
~ 9. implement / port this on other local model (mistral, llama2, etc) so the cost is managed
    2024-04-02:
    - done for the `summarize` mode
    - still TODO for the `ask` and `embed` modes
    here, find chromadb https://howaibuildthis.substack.com/p/choosing-the-right-embedding-model
    the embed and ask results are pretty shit with ollama mistral. see create_embedding_ollama
        oh, ollama mistral nggak utk embedding kyknya sih. need to probably do the bge version I had working... somewhere... haha
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
'''
