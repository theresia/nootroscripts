#!/usr/bin/env python3

'''
Usage:
    
    # process an audio file
    audio2llm.py --mode=note --af ~/Downloads/Podcasts/77e61246-248f-43b1-8a1e-49b404ebc28c_HS9200833232.mp3
    audio2llm.py --mode=summary --af ~/Downloads/Podcasts/77e61246-248f-43b1-8a1e-49b404ebc28c_HS9200833232.mp3
    audio2llm.py --mode=translation --af=~/Downloads/Podcasts/49211664-4cd6-482e-8834-406dc3de665b_ff462ab6-ad81-44d6-a352-900441bfed01.mp3
    audio2llm.py --mode=topix --af ....
    
    # process a text file
    audio2llm.py --tf output/some.md # defaults to generating some QnAs from the text
    # or specify it explicitly
    audio2llm.py --mode QnAs --tf output/some.md
    # the other modes (not exhaustive, see md2notes.py --help for full list)
    audio2llm.py --mode=summary --tf ~/Downloads/Podcasts/77e61246-248f-43b1-8a1e-49b404ebc28c_HS9200833232.txt
    audio2llm.py --mode=note --tf ~/Downloads/Podcasts/49211664-4cd6-482e-8834-406dc3de665b_ff462ab6-ad81-44d6-a352-900441bfed01.txt
    audio2llm.py --mode=translation --tf output/transcript-10000000_831068982109066_8715525788756471886_n.mp4.txt
    audio2llm.py --mode cbb --tf output/mds/bigthink.com/everyone-is-wrong-about-love-languages-heres-why.md
    audio2llm.py --mode thread --lmodel mistral --tf output/mds/terribleminds.com/a-i-and-the-fetishization-of-ideas.md
    audio2llm.py --mode topix --lmodel gpt-3.5-turbo-16k --tf output/mds/qntm.org/responsibility.md # use larger context window to handle tagging (where a zoomed-out view is sufficient)
    
    # pass in a path, will process all .txt and .md in that directory
    audio2llm.py --mode summary output/mds/www.newyorker.com/
     
    # pass a custom prompt
    audio2llm.py --mode note --tf '~/Documents/Screenshots/how_ppl_appreciate_good_writing' --prompt "what are the characteristics of good writing according to the text? don't add anything that's not in the text"
    
    # use ollama's models instead e.g. 'mistral'
    audio2llm.py --mode QnAs output/mds/www.freethink.com/eastern-philosophy-neuroscience-no-self.md --lmodel gpt-4
    
    # pass a glob path
    audio2llm.py --mode note --tf 'output/mds/www.newyorker.com/*.md' # if the glob mode that will return more than 1 file, need to pass it as a string
    # with custom prompt
    audio2llm.py --mode note --tf 'output/mds/www.newyorker.com/*.md' --prompt "what are some underlying themes between these articles?"
    audio2llm.py --mode note --tf 'output/summary-the-*' --prompt "what are some underlying themes between these articles?" # I had two .md summaries of two articles I clipped from newyorker. not sure if there's any shared theme or not, thought I'd ask
    # pass a glob path, of a bunch of job descriptions I've asked mistral to summarise earlier
    audio2llm.py --tf '/path/to/some/JDs/*-mistral.md' --lmodel='mistral' --cm sentence --prompt "list of most common skills and competencies mentioned in the job description"
    # deeper glob
    audio2llm.py --tf '~/Documents/Screenshots/udemy - writing flair/udemy-subtitle-downloads/*/en_*-cleaned-note-mistral.txt' --lmodel mistral --mode note
        
    # use mistral (via ollama) rather than the default gpt-3.5-turbo to process the text
    audio2llm.py --tf ~/Downloads/Podcasts/666d0c61-75c2-4774-b304-a01e9225cac9_transcoded.txt --mode note --lmodel mistral
    
    # zooming into a specific section (3rd one in this case) and pulling out quotes from that section of snippet to keep the soul of the speaker's exact words
    #   i.e. we do a 2nd pass on a transcript; only process that specific section index from the main transcript, using smaller chunk size.
    #   only supports 2 passes at the moment
    audio2llm.py --tf output/hGxZOTobATM-transcript.txt --lmodel mistral --cm word --mode note --specific_snippet 3
    # pass a prompt to drill even further on that section
    audio2llm.py --tf output/hGxZOTobATM-transcript.txt --lmodel mistral --cm word --mode note --specific_snippet 3 --prompt "quotes that reflect the speaker's insight" 

    #### A note on deciding which context window length is appropriate (relevant argument: --lmodel)
    
    # Example 1: zoom-out mode
    # here I use GPT 3.5 turbo's -16k model because I have a specific point I'm interested in, (am in a zoom-out mode)
    #   so I'm OK with having larger chunks and larger context window because it's less of a waste
    # oh, and THIS is the use case where things can be indexed once and queried multiple times via the llama index thing (SummaryIndex?).
    #   so it's like doing multiple "--prompt" but in a way that saves API calls
    audio2llm.py --tf ~/Downloads/Podcasts/0f109354-7839-489e-9ce5-8dd2fb43b8dc_d41cc5ea-41e4-cb35-8648-003762e2ea38.txt --lmodel gpt-3.5-turbo-16k --prompt "how was Mathematics compared to chess with set rules"
    
    # Example 2: zoom-in mode
    # while here, I'm 'zooming in'. I see a point in the QnAs generated that I'd like to expand on,
    #   but I don't want GPT to waste its tokens on irrelevant sections
    audio2llm.py --tf ~/Downloads/Podcasts/3b854098-72d1-4abf-8478-bfa40428cc30_GLT4876505279.txt --specific_snippet 2 --prompt "what is shared about hidden tribes data"
    
    #### A note on chunking methods (relevant argument: --cm)
    
    # this script supports two chunking methods: sentence or word -- the default
    # the former relies on the existence of period and punctuation while the latter is all about word count.
    # note that youtube auto-captions usually have NO punctuations, and Whisper SOMETIMES spits out unpunctuated sections,
    #   so it's more foolproof to use the `words` chunking method.
    #     and `sentence` chunking method is only SLIGHTLY otherwise, for very rare cases of transcripts / text
    audio2llm.py --tf output/GIiaFW874q8-transcript.txt --cm word --lmodel mistral

'''

import os, sys, re, argparse, json, glob, datetime
import torch, openai, whisper, tqdm, inflect

from pathlib import Path
from whisper.utils import get_writer 
from nltk import sent_tokenize # had to run nltk.download('punkt') first

from prompts import prompts # we define the generic prompt for each mode here

current_dir = os.path.dirname(os.path.abspath(__name__))

def convert_to_ordinal(number):
    p = inflect.engine()
    ordinal = p.ordinal(number)
    return ordinal

def split_into_chunks(text, max_length):
    chunks = []
    sentences = sent_tokenize(text)
    current_chunk = ''
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def transcribe(audio, model_type='base', output_dir='output', language='en'):
    class _CustomProgressBar(tqdm.tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._current = self.n
            
        def update(self, n):
            super().update(n)
            self._current += n
            
            print("Audio Transcribe Progress: " + str(round(self._current/self.total*100))+ "%")
    
    if(output_dir == 'output'): # can perhaps drop later, used to refer to Whisper2Summarize/output but now it's gonna output the transcript next to the audio/video file
        output_dir = current_dir+'/output'

    print(f"output_dir: {output_dir}")
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = whisper.load_model(model_type, device=devices)
    transcribe_module = sys.modules['whisper.transcribe']
    transcribe_module.tqdm.tqdm = _CustomProgressBar

    print(f"Transcribing {audio}...")
    result = model.transcribe(audio, verbose=True, fp16=False, language=language) # language="id", or "en" by default
    transcribed = result["text"]
    '''
    timestamped_transcript = ""
    for segment in result.get("segments"):
        timestamped_transcript += "[" + str(timedelta(seconds=int(segment.get("start")))) + " --> " + str(timedelta(seconds=int(segment.get("end")))) + "]  " + segment.get("text") + "\n"
    with open("output/ts_transcript-"+os.path.basename(audio)+".txt", "w",encoding='utf-8') as text_file:
        text_file.write(timestamped_transcript)
        print("Saved timestamped transcript to ts_transcript-"+os.path.basename(audio)+".txt")
    
    with open("output/raw_transcript-"+os.path.basename(audio)+".txt", "w",encoding='utf-8') as text_file:
        text_file.write(json.dumps(result))
        print("Saved raw_transcript to raw_transcript-"+os.path.basename(audio)+".txt")
    
    with open("output/transcript-"+os.path.basename(audio)+".txt", "w",encoding='utf-8') as text_file:
        text_file.write(transcribed)
        print("Saved transcript to transcript-"+os.path.basename(audio)+".txt")
    '''
    # Set VTT Line and words width
    word_options = {
        "highlight_words": True,
        "max_line_count": 1,
        "max_line_width": 42
    }
    # # H/T https://github.com/openai/whisper/discussions/223
    # and https://github.com/openai/whisper/blob/main/whisper/utils.py#L308
    # don't generate vtt's just yet, blm tau mau diapain kok. kalo mau coba2 ya pake vtt yang udah ke-generate aja dulu
    # transcript_writer = get_writer(output_format='vtt', output_dir=current_dir+'/output') # can specify 'all'
    # transcript_writer(result, audio, word_options)
    transcript_writer = get_writer(output_format='txt', output_dir=output_dir)
    transcript_writer(result, audio, word_options)
    return transcribed

def get_transcript(transcript_file, convert_md=False, ignore_links=True):
    html_file = transcript_file
    if convert_md:
        print("converting the md into html first")
        # import here because it might be unnecessary
        from html2text import HTML2Text
        from markdown import markdownFromFile
        html_file = os.path.basename(transcript_file)+".html"
        markdownFromFile(input=transcript_file, output=html_file) # create .html from .md
        h = HTML2Text() # to process the HTML
        h.ignore_links = ignore_links
        with open(html_file, "r", encoding="utf-8") as fh:
            transcript_text = h.handle(fh.read())
    else:
        # print("NOT converting the md into html first")
        with open(html_file, "r", encoding="utf-8") as fh:
            transcript_text = fh.read()

    return transcript_text

def chunk_text(transcript, model_name, chunking_method='word', n_div=1):
    # TOKEN MATH: about 0.75 words per token, in general
    snippet = []
    
    ### sentence chunking method (NLTK sent_tokenize)
    # note that this mode of chunking is not suitable for youtube transcripts (has no punctuations, so it will likely end up as a huge block of text)
    if('sentence' in chunking_method):
        # nltk_n = 6000 # so the response can be longer and more detailed
        nltk_n = int(8000/n_div) # so the response can be longer and more detailed
        # BUT this number can perhaps be further increased as the original code here even allowed for sending previous context to the next GPT call
        if(model_name.endswith('-16k') or model_name.endswith('-1106')):
            nltk_n = int(32000/n_div)
        print(f"nltk_n used: {nltk_n}")
        # Assuming at least 4 hyphens separate sections, if any.
        #   this helps keep semantically related/relevant pieces of text together when there are separators within the transcript itself
        # BUT if you're summarising a summary (rather than a transcript), then you DON'T actually want to have sections split by separators (unlike an article)
        #   so perhaps this line of code fits in md2notes.py but NOT here in summarize.py where I mainly use it to work with podcast transcripts...
        #   but I'll think about how to branch the logic when I decided to merge the scripts
        #   for now, I'm changing the length of the separator from 4 to 10 so there will likely be no sections created
        sections = re.split('-{10,}', transcript)
        for i, section in enumerate(sections):
            snippet.extend(split_into_chunks(section, max_length=nltk_n)) # put the chunks into one big list to be processed further down
        print("number of snippets produced by sentence chunking method: "+str(len(snippet)))
    
    ### word chunking method
    # when working with youtube transcripts, this mode is preferred
    else: # if('word' in chunking_method):
        # n = 700 # so the response can be longer and more detailed
        n = int(1000/n_div)
        if(model_name.endswith('-16k') or model_name.endswith('-1106')):
            n = int(5000/n_div)
        print(f"n used: {n}")
        # 1300 used to work fine for turbo (or any other models that supports 4096 tokens of context window)
        #   but when I summarised some HN threads, it had some hiccups (context limit hit) (can perhaps use the 16k for HN threads?)
        # gpt-3.5-turbo-1106 supports 16,385 tokens (sweet spot for n is 5300 or 5000) just like *-16k. otherwise assume it's 4,096 tokens (sweet spot for n: 1300 or 1000),
        #   which is probably where the original 1300 figure came from (~33% of the max token?) -- I got the initial code from some github project
        #   if I changed this to 3300 for gpt-3.5-turbo-0613, the response would get truncated. ok2 kewl, got it
        # but the result note is not as good when we have more chunks (it's less detailed). hmm. what's the sweet spot?
        #   tergantung use case sih kyknya
        split_transcript = transcript.split()
        print("word count of transcript (split_transcript): "+str(len(split_transcript))) # this is basically the number of space-separated words found in the text kan?
        snippet = [' '.join(split_transcript[i:i+n]) for i in range(0,len(split_transcript),n)] # create x list of words that when concatenated with ' ' will be < n?
        print("number of snippets produced by words chunking method: "+str(len(snippet)))
        
    return snippet

def llm_process(transcript, transcript_file, mode='QnAs', model='gpt-3.5-turbo', context_filename='', prompt='', output_filepath='', specific_snippet=0, chunking_method='sentence', be_thorough = False):
    input_fileext = Path(transcript_file).suffix # to check if it's md (likely an article) or txt (likely a transcript)

    if mode in['summary', 'kp']:
        mode = 'summary'
    
    system_content = prompts[mode]
    
    if(mode == 'tag'):
        model = 'gpt-3.5-turbo-16k' # zoom out, go high-level, as less chunking as possible # but this doesn't perform as well as vanilla turbo??
    
    # if producing a note for markdown, fine tune the prompt
    if(mode == 'note'):
        if(input_fileext == '.md'):
            system_content += " Pay attention to all headings, sections, and table of content that exist in the HTML / rich text / markdown as they could be pointers for the different points and arguments."
    
    # for summary mode, we override the dict value as the prompt is quite specific whether it's a podcast transcript or a markdown from an article
    if(mode == 'summary'):
        if(input_fileext == '.md'):
            system_content = "Summarise the text from the essay or article provided in first-person as if the author has produced a short version of the original text. "
        else:
            system_content = "Summarise the text in first-person as if the speaker has produced a short version of the original conversation. "
        
        system_content += "Include the anecdotes, key points, arguments, and actionable takeaways. " \
                          "Inject some narrative and bullet points as appropriate into your summary so the summary can be easily read. " \
                          "Please use simple language and don't repeat points you have previously made."
    
    # override prompt if mode is topix and be_thorough is set to True
    if(mode == 'topix'):
        # if I am working with one file, I'd like it to be distilled to just three main themes.
        # but when I'm aggregating several text from disparate pieces of texts
        # (as is the case in IG collections analysis), I'd like it to be as thorough as possible
        if(be_thorough): # rather than be succinct, aka, for an atomic piece of text
            system_content = "Aggregate information from the provided texts and extract a comprehensive list of main topics, themes, or concepts. "\
            "Utilize Wikipedia's concept taxonomy to identify and categorize the diverse range of subjects covered. " \
            "Aim for a detailed and nuanced representation of the underlying ideas present in the texts, providing a more exhaustive exploration of the content. "\
            "Then at the end, provide:" \
            "1. the distilled version of the deeper underlying theme of all the ideas." \
            "2. what intersections of topics are being dicussed"
    
    result = ""
    # previous = ""
    
    if('[concatenated text]' in transcript):
        be_thorough = True
    
    if(prompt):
        system_content = prompt
        result += f"prompt: {prompt}\n\n------\n\n"
        print(f"prompt: {prompt}")
        # so the filename contains the unique hash of the prompt and will be unique (running different prompts not gonna overwerite each other)
        # this `mode` object is not being used anymore from this point on other than to name the output file so it's OK to do this here
        import hashlib
        mode = f"prompt-{hashlib.md5(prompt.encode('utf-8')).hexdigest()}"
        
        # some other good prompts
        # - "what are all the observations, key arguments, and suggestions made in the text?" # good for critical essays about science, literature, culture e.g. https://longreads.com/2023/10/24/30-years-internet-online-writing/
        # - "what are some counterarguments for the points made in the article" # I applied it to output/mds/bloomberry.com/how-to-ai-proof-your-career.md
        # - "rewrite in simple language" # applied to whitepapers or academic research like output/mds/hazyresearch.stanford.edu/2023-01-20-h3.md
        # - "what are some useful questions that the speakers recommended to ask" # e.g. infinite loop - cedric chin
        # - my go-to prompt for HN threads: "what are different arguments, facts, and opinions made in the different discussion threads? and what are the topics, concepts, and takeaways?"
        # - "this is a podcast transcript. please retell it highlighting the key points and arguments shared, while sticking to the narrative flow and seques, weaving the arguments together"
        # - "the speaker's insight and the stories shared that illustrates it"
        # - "quotes that reflect the speaker's insight", or "the speaker's insight followed by the quotes that reflect that insight"
    
    # stitch the final output filename
    output_filename = Path(transcript_file).stem
    # output_filename = Path(transcript_file).name
    output_fileext = '.md' # let's hard code it to .md so it shows up larger in quicklook, lol
    if(not output_filepath):
        output_filepath = str(Path(transcript_file).parent)
    
    if(specific_snippet > 0):
        output_filename = f"{output_filename}-{convert_to_ordinal(specific_snippet)}_snippet-{mode}-{model}{output_fileext}"
    else:
        output_filename = f"{output_filename}-{mode}-{model}{output_fileext}"
    if(not output_filename.endswith('.txt') and not output_filename.endswith('.md')): # if no extension (generated through directory-processing, which is not yet implemented here, but eh, TODO)
        output_filename = f"{output_filename}.md" # I like .md better
    
    if(os.path.isfile(output_filepath+'/'+output_filename)):
        print(f"{output_filepath+'/'+output_filename} exists")
        return
    
    ## stitch any additional context needed to be added onto the prompt
    
    additional_context = ''
    if(context_filename and os.path.exists(context_filename)):
        with open(context_filename, 'r') as cf:
            jcf = json.loads(cf.read())
            for k, v in jcf.items():
                if(v):
                    if(type(v) == list):
                        additional_context += k + ': ' + ', '.join(v) + "\n"
                    if(type(v) == str):
                        additional_context += k + ': ' + v + "\n"
        transcript = additional_context + "\n" + transcript

    ## chunking, to manage token limit

    snippet = chunk_text(transcript, model, chunking_method=chunking_method, n_div=1)
    print("Transcript chunked into {} snippets".format(len(snippet)))
    # zoom / go more granular into some section of the snippet (do a 2nd pass on a transcript):
    # only process that specific section index from the main transcript, as well as using smaller chunk size.
    # only supports 2 passes at the moment. idk if it's even worth doing more
    if(specific_snippet > 0):
        # if only want the prompt to focus on one segment in the chunked snippets
        #   (e.g. want to ask about a specific point in the transcript,
        #   without spending the credit to process the whole list of snippets)
        snippet = [s for i,s in enumerate(snippet) if i==(specific_snippet-1)]
        print("Zoom-in resulted in {} sub-snippets".format(len(snippet)))
        zoom_in = True
        # TODO: perhaps allow setting this to False, so just use the original chunk size
        # else, re-chunk the snippet text on the index specified with a smaller n to get more granularity on the specific section
        if(zoom_in):
            transcript = ''.join(snippet)
            snippet = chunk_text(transcript, model, chunking_method=chunking_method, n_div=4)

    ## prep the LLM
    
    use_api = False # set to False to use ollama library or set to True to use ollama API. CURRENTLY JUST HARDCODED
    
    # I don't merge with the branching below (to use OpenAI or local ollama (API/library)) because the logic became
    #   a bit complicated. and I'm just benchmarking the difference between the two methods for ollama anyway
    if('gpt' in model):
        openai.api_key = os.getenv("OPENAI_KEY")
        client = openai.OpenAI(api_key=openai.api_key)
    else:
        # use ollama. perhaps add support for (simonw's) llm some time. or huggingface's (via sentence_transformer?)
        client = openai.OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )

    ## start sending the values to the LLM
    
    time_begin = datetime.datetime.now()
    
    for i in range(0, len(snippet), 1):
        print("Processing transcribed snippet {} of {} with model {}".format(i+1,len(snippet), model))
        if('gpt' not in model and use_api is False):
            import ollama
            # result += "use ollama library\n----\n"
            try:
                gpt_response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {
                            "role": "user",
                            "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the text."
                            # "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the text. For additional context here is the previous written message: \n " + previous
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
            # if('gpt' not in model): # use_api is True but it might still be GPT rather than ollama
            #     result += "use ollama API\n----\n"
            try:
                gpt_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        # {"content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the text. For additional context here is the previous written message: \n " + previous}
                        {"role": "user", "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the text."},
                    ],
                    temperature = 0.2,
                )
                # previous = gpt_response.choices[0].message.content
                # result += gpt_response.choices[0].message.content + "\n\n-------\n\n(based on the snippet: "+ snippet[i]+")\n\n-------\n\n"
                result += gpt_response.choices[0].message.content + "\n\n-------\n\n"
            except Exception as e:
                print("An unexpected error occurred:", e)
                print(result)

    time_end = datetime.datetime.now()
    
    print(f"time taken: {str(time_end - time_begin)}")
    result += f"\n\n-----\n\ntime taken: {str(time_end - time_begin)}"
    
    ## save the result
    with open(output_filepath+'/'+output_filename, "w",encoding='utf-8') as text_file:
        text_file.write(result)
        print("Saved result to "+output_filepath+'/'+output_filename)

if __name__ == '__main__':
    text = ''
    output_filename_path = ''

    parser = argparse.ArgumentParser(description='Whisper2Summarize - a tool for summarizing audio files')
    parser.add_argument('--mode', type=str, default='QnAs', help='transcription, QnAs, note, summary/kp, tag, topix, thread, tp, cbb, definition, translation')
    parser.add_argument('--tf', type=str, help='the transcript to process. can also pass a directory with glob')
    parser.add_argument('--af', type=str, help='the audio file to summarize')
    parser.add_argument('--prompt', type=str, help='the prompt to use in lieu of the default ones')
    parser.add_argument('--be_thorough', action='store_true', help='(currently only) for topix mode, longer prompt, more elaborate analysis')
    parser.add_argument('--lmodel', type=str, default='gpt-3.5-turbo', help='the GPT model to use for summarization (default: gpt-3.5-turbo)')
    parser.add_argument('--tmodel', type=str, default='base', help='the Whisper model to use for transcription (default: base)')
    parser.add_argument('--cf', type=str, help='the additional context file to use (JSON)')
    parser.add_argument('--save', action='store_true', help='if working with multiple files, should it save the concatenated text?')
    parser.add_argument('--output_dir', type=str, help='dir to use to save the output files')
    parser.add_argument('--specific_snippet', type=int, help='just process this part of the snippet (usually used together with --prompt, after seeing a summary/note and want to zoom in on specific point in the transcript)')
    parser.add_argument('--cm', type=str, default='word', help='chunking method to use (sentence or words)') # set to be the default, as Whisper starts to generate unpunctuated transcript in the middle of the conversation on April 6th. observed for the Cal Fussman Frank Blake episode, and the WCDHT - why Liz disappeared episode. but the Whisper transcription invoked by utub.py at around the same time, works as expected, punctuated and all. strange
    parser.add_argument('--lang', type=str, default='en', help='language code of the audio/video')
    
    args = parser.parse_args()
    
    if(args.save):
        save_transcript = False
    else:
        save_transcript = True
    if(args.be_thorough):
        be_thorough = True
    else:
        be_thorough = False
    if(args.specific_snippet):
        specific_snippet = args.specific_snippet
    else:
        specific_snippet = 0

    if(args.tf): # process the transcript file over audio file
        if(os.path.isfile(args.tf)):
            print(f"it's a file, {args.tf}")
            text = get_transcript(args.tf) # we're working with text here, not md as we do in md2notes.py which might contain links
            output_filename_path = args.tf
        elif(os.path.isdir(args.tf)):
            print(f"it's a dir, {args.tf}")
            # but how can I exclude the GPT-generated files? aka "QnAs-","note-","topix-","summary-","definition-","prompt-","concat_transcript-"
            file_list = glob.glob(args.tf+'/*.md') + glob.glob(args.tf+'/*.txt')
            for dmf in file_list:
                print(f"  dmf: {dmf}")
                text += get_transcript(dmf)
                text += "\n\n\n[concatenated text]\n\n\n"
            parent_dir = Path(file_list[0]).parent
            output_filename_path = str(parent_dir) # assign the path to the parent directory of the first file in the list
            print(f"main :: output_filename_path: {output_filename_path}")
            if(save_transcript):
                # for the sake of verbosity
                with(open(output_filename_path+'/concat_transcript-'+parent_dir.name+'.txt', 'w') as f):
                    f.write(text)
        elif(len(glob.glob(args.tf)) > 0):
            matching_files = glob.glob(args.tf)
            # print(f"glob matching_files: {matching_files}")
            if(len(matching_files) == 1 and args.tf == matching_files[0]): # glob returns the folder itself (which exists)
                print("no match jg") # tp ini gak pernah keliatan nge-hit?
            else: # more than 1 file matched the glob pattern
                for gmf in matching_files:
                    print(f"  gmf: {gmf}")
                    text += get_transcript(gmf)
                    text += "\n\n\n[concatenated text]\n\n\n"
                '''
                # this was written for IG shortcodes, but in this script's case, it will just handle some verbose filenames
                file_list = [Path(fn).name for fn in matching_files]
                import re # import di sini aja lah, lebih efisien
                pattern = re.compile(r'-(\\w+)(?:\\.\\w+)?\\.txt')
                output_filename_path = ''.join(set([re.search(pattern, filename).group(1) for filename in file_list if re.search(pattern, filename)]))
                '''
                parent_dir = Path(matching_files[0]).parent
                output_filename_path = str(parent_dir) # assign the path to the parent directory of the first file in the list
                print(f"main :: output_filename_path: {output_filename_path}")
                if(save_transcript):
                    # for audit and posterity
                    with(open(output_filename_path+'/concat_transcript-'+parent_dir.name+'.txt', 'w') as f):
                        f.write(text)
        print("transcript acquired of length: "+str(len(text)))
        
    elif(args.af and os.path.exists(args.af)):
        output_filename_path = args.af
        # output_filename = Path(output_filename_path).name
        output_filepath = Path(output_filename_path).parent
        text = transcribe(args.af, model_type=args.tmodel, output_dir=output_filepath, language=args.lang)
    else:
        print("need to specify either one of tf or af")
    
    if(args.mode != 'transcription' and text and output_filename_path): # if not transcription, then pass this to GPT
        llm_process(
            text, output_filename_path, be_thorough=be_thorough,
            mode=args.mode, model=args.lmodel, context_filename=args.cf, prompt=args.prompt,
            output_filepath=args.output_dir, specific_snippet=specific_snippet, chunking_method=args.cm,
        )

'''
TODO:
0. merge audio2llm.py and md2notes.py
    situation: md2notes.py is 85% similar to audio2llm.py but this one is used for articles (usually retrieved by url2md.py) more while that one I use for podcasts
    differences between these two scripts:
    - the prompt for `note` mode in md2notes.py specifically calls out the markdown structure of headings and other markups, while the audio2llm.py one is more optimised to work on flat transcription of audio content
    - I refer to the text sent to LLM by `text` in md2notes.py and `transcript` in audio2llm.py
    - temperature is set to 0.3 in md2notes.py and 0 in audio2llm.py # can't remember why tho, I don't recall if there's a specific reason for this other than I was testing different values at the time
    - arguments supported by audio2llm.py but not by md2notes.py:
        tf (transcript file. it's practically a mandatory argument in md2notes.py, as it doesn't support processing audio file)
        lang (to pick the language that Whisper will use)
        output_dir (perhaps worth implementing in md2notes.py but I don't want to deal with filepath with the different modes of accepting --tf in file/dir/glob for now)
    - oh, but at this point there is practically NO distinct feature that md2notes.py have but audio2llm.py doesn't...
        audio2llm.py is a superset of md2notes.py. so I can actually just delete md2notes.py
        well, unless I want to keep the conceptual interface input-oriented and separate
        (e.g. url2md, youtube2llm, podcast2llm, md2llm) so I/O of each script is clear and each does a specific functionality
        in that case I need to refactor them into something more modular

√ 1. iterate on 'segments' instead of just getting the wall of "text"
    and then output each segment into separate lines, with timestamps...
    should be more useful (but more expensive in terms of tokens and longer chunks)
√ 2. can provide additional context (used by this insta video analysis I am doing)

md2notes.py's TODO (have been incorporated here as well):
√ 1. handle all .mds or .txt in a folder (usually the output of url2markdown.py or other transcriber/text generator like vision-* scripts)
√ 2. tweak the prompt, seems too verbose at this point. e.g. summary-12-favorite-problems.md and note-12-favorite-problems.md
√ 3. specify the prompt to use to summarise/take notes
√ 4. use sentence tokenizing / chunking (NLTK)
√ 5. handle a folder, of presumably markdown/text files sharing the same topic
   e.g. output/mds/ul-sora
    or just do a bash loop over the. kalo cm mau batch the processing. for example:
        for f in output/mds/ul-sora/itemid*.md; python3 md2notes.py --mode summary "$f" --lmodel='gpt-3.5-turbo-16k'
    kasus kyk apa ya yg butuh aggregating the MDs?
        - dimana gw bs kumpulin "everything I have gathered about this subject" trus generate embeddings for
        - when I need to create a KB... like I want on Liz's corpus
    kalo hackernews kebetulan gw udah manually cluster the subthreads with specific topics,
        karena use casenya adalah get a sense of the different angles and points people have about this topic. hmm.
        so discussions... KB.... podcasts transcripts, interview (editorialised), essays...
            each with their best mechanisms and prompts to help us manage, synthesise, and create things out of data->information->knowledge->skill->wisdom

'''