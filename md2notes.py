#!/usr/bin/env python3

'''
Usage:
    md2notes.py output/some.md # defaults to QnAs
    
    # pass in a path, will process all .txt and .md in that directory
    md2notes.py --mode summary output/mds/www.newyorker.com/ 
    
    # pass in a glob
    md2notes.py --mode note 'output/mds/www.newyorker.com/*.md' # if the glob mode that will return more than 1 file, need to pass it as a string
    md2notes.py --mode note 'output/mds/www.newyorker.com/*.md' --prompt "what are some underlying themes between these articles?"
    md2notes.py --mode note 'output/summary-the-*' --prompt "what are some underlying themes between these articles?" # I had two .md summaries of two articles I clipped from newyorker. not sure if there's any shared theme or not, thought I'd ask
    
    # can also use ollama's models e.g. 'mistral'
    md2notes.py --mode QnAs output/mds/www.freethink.com/eastern-philosophy-neuroscience-no-self.md --lmodel gpt-4
    
    # the other modes (not exhaustive, see md2notes.py --help for full list)
    md2notes.py --mode note output/some.md
    md2notes.py --mode summary output/some.md
    md2notes.py --mode cbb output/mds/bigthink.com/everyone-is-wrong-about-love-languages-heres-why.md
    md2notes.py --mode topix --lmodel gpt-3.5-turbo-16k output/mds/qntm.org/responsibility.md # use larger context window to handle tagging (where a zoomed-out view is sufficient)
    md2notes.py --mode thread --lmodel mistral output/mds/terribleminds.com/a-i-and-the-fetishization-of-ideas.md
    
    # zoom / go more granular into some section of the snippet (do a 2nd pass on a transcript): only process that specific section index from the main transcript, as well as using smaller chunk size. only supports 2 passes at the moment
    md2notes.py output/PkXELH6Y2lM-transcript.txt --lmodel mistral --cm word --specific_snippet 3
    
    # pass a prompt to drill even further on that section
    md2notes.py output/hGxZOTobATM-transcript.txt --lmodel mistral --cm word --specific_snippet 1 --prompt "quotes that reflect the speaker's insight"
'''

import os
import sys
import re
import argparse
import datetime
import glob
import json
import torch
import openai
import tqdm
import whisper
import inflect

from pathlib import Path
from nltk import sent_tokenize # had to run nltk.download('punkt') first

# NOT implemented here because I'm going to deprecate this script anyway
from prompts import prompts # we define the generic prompt for each mode here

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
        nltk_n = int(8000/n_div)
        # BUT this number can perhaps be further increased as the original code here even allowed for sending previous context to the next GPT call
        if(model_name.endswith('-16k') or model_name.endswith('-1106')):
            nltk_n = int(32000/n_div)
        print(f"nltk_n used: {nltk_n}")
        # Assuming at least 4 hyphens separate sections, if any.
        #   this helps keep semantically related/relevant pieces of text together when there are separators in the original article itself
        sections = re.split('-{4,}', transcript)
        for i, section in enumerate(sections):
            snippet.extend(split_into_chunks(section, max_length=nltk_n)) # put the chunks into one big list to be processed further down
        print("number of snippets produced by sentence chunking method: "+str(len(snippet)))

    ### word chunking method
    # when working with youtube transcripts, this mode is preferred
    else: # if(chunking_method == 'words'):
        # n = 700 # so the response can be longer and more detailed
        n = int(1000/n_div) # idk, should this be divide or multiply? kalo divide, jadi lebih granular sih mestinya
        if(model_name.endswith('-16k') or model_name.endswith('-1106')):
            n = int(5000/n_div)
        print(f"n used: {n}")
        # 1300 used to work fine for turbo (or any other models that supports 4096 tokens of context window)
        #   but when I summarised some HN threads, it had some hiccups (context limit hit) (can perhaps use the 16k for HN threads?)
        # gpt-3.5-turbo-1106 supports 16,385 tokens (sweet spot for n is 5300 or 5000) just like *-16k. otherwise assume it's 4,096 tokens (sweet spot for n: 1300 or 1000),
        #   which is probably where the original 1300 figure in the original code came from (~33% of the max token?) -- I got the initial code from some github project
        #   if I changed this to 3300 for gpt-3.5-turbo-0613, the response would get truncated. ok2 kewl, got it
        # but the result note is not as good when we have more chunks (it's less detailed). hmm. what's the sweet spot?
        #   depends on the use case sih kyknya
        split_transcript = transcript.split()
        print("word count of transcript (split_transcript): "+str(len(split_transcript))) # this is basically the number of space-separated words found in the text kan?
        snippet = [' '.join(split_transcript[i:i+n]) for i in range(0,len(split_transcript),n)] # create x list of words that when concatenated with ' ' will be < n?
        print("number of snippets produced by words chunking method: "+str(len(snippet)))
        
    return snippet

def llm_process(transcript, transcript_file, mode='QnAs', model='gpt-3.5-turbo', context_filename='', prompt='', output_filepath='', specific_snippet=0, chunking_method='word', be_thorough = False): # comment_filename = ''
    do_note = False
    do_summarise = False
    do_translate = False
    do_topix = False
    do_define = False
    do_thread = False
    do_tp = False
    do_cbb = False
    do_tagging = False
    if mode in['note']:
        do_note = True
    if mode in['summary', 'kp']:
        do_summarise = True
    if mode == 'translation':
        do_translate = True
    if mode == 'topix':
        do_topix = True
    if mode == 'definition':
        do_define = True
    if mode == 'thread':
        do_thread = True
    if mode == 'tp':
        do_tp = True
    if mode == 'cbb':
        do_cbb = True
        # model = 'gpt-3.5-turbo-16k' # zoom out, go high-level, as less chunking as possible # but this doesn't perform as well as vanilla turbo??
    if mode == 'tag':
        do_tagging = True
        model = 'gpt-3.5-turbo-16k'
    
    ## we define the generic prompt for each mode here
    
    result = ""
    # previous = ""
    
    if('[concatenated text]' in transcript):
        be_thorough = True
    
    # default prompt, for extracting Questions in the text and generate one-sentence answers from
    # system_content = "1) state all the key arguments made, then 2) list all the questions asked and a three-sentence answer to each (include all examples of concrete situations and stories shared in the answer)"
    system_content = "list all the questions asked and a three-sentence answer to each (include all examples of concrete situations and stories shared in the answer)"
    
    if(do_note):
        system_content = "Please make a succinct but detailed notes from the text. " \
                       "Pay attention to all headings, sections, and table of content that exist in the HTML / rich text / markdown " \
                        "as they could be pointers for the different points and arguments. " \
                        "Do not summarize and keep every information."
    elif (do_summarise):
        system_content = "Summarise the text from the essay or article provided in first-person as if the author has produced a short version of the original text. " \
                         "Include the anecdotes, key points, arguments, and actionable takeaways. " \
                         "Inject some narrative and bullet points as appropriate into your summary so the summary can be easily read. " \
                         "Please use simple language and don't repeat points you have previously made."
    
    elif (do_translate):
        system_content = "You are a translator who handles English to Indonesian and vice versa. " \
                        "Please produce an accurate translation of the transcribed text. "\
                        "If the text is in English, then translate to all languages you know. " \
                        "If the text is non English, please translate it to English"
    
    elif (do_tagging):
        system_content = "what are some hashtags appropriate for this?"
    
    elif (do_topix): # perhaps same as tagging. this works but not meaningful enough
        # system_content = "Please list several topics and keywords you think best represent the content of the text. List each of them in hashtags." # the wikipedia taxonomy thing
        system_content = "Extract the 3 topics / themes / concepts that you see. Please use Wikipedia's concept taxonomy for it."
        # if I am working with one file, I'd like it to be distilled to just three main themes. but when I'm aggregating several text from disparate pieces of texts (as is the case in IG collections analysis, I'd like it to be as thorough as possible)
        if(be_thorough): # rather than be succinct, aka, for an atomic piece of text
            system_content = "Aggregate information from the provided texts and extract a comprehensive list of main topics, themes, or concepts. "\
            "Utilize Wikipedia's concept taxonomy to identify and categorize the diverse range of subjects covered. " \
            "Aim for a detailed and nuanced representation of the underlying ideas present in the texts, providing a more exhaustive exploration of the content. "\
            "Then at the end, provide:" \
            "1. the distilled version of the deeper underlying theme of all the ideas." \
            "2. what intersections of topics are being dicussed"
    
    elif (do_thread):
        # still very.... robotic? idk how to describe it. it's nice, but not... engaging? like first grader. lack flare, personality, hooks?
        system_content = "Rewrite as a Twitter thread of 6 tweets or less. " \
                         "Speak in first person, use informal, conversational, and witty style. Write for the ears rather than for the eyes. " \
                         "Introduce the essay with the surprising actionable takeaway and then go over the main arguments made in the essay. " \
                         "Be as granular as possible."
    
    elif (do_tp): # for Instagram post mostly, if not specifically
        system_content = "Please create 2 talking points I can use for a video script through the lens of human nature based on how different arguments made in comments relate to the content and argument of the main post." \
                         "Speak in first person narrative. Use informal, conversational, and witty style. Write for the ears rather than for the eyes. " \
                         "Use active voice. Avoid passive voice as much as possible."
    
    elif (do_cbb):
        system_content = "here's a news article. please turn the title of the article into a question and find the answer to the question in the text provided"
    
    if (do_define):
        system_content = "list of all definitions made in the discussion"
    
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
        # - "list of concepts, aphorisms, and proverbial assertions made"

    # stitch the final output filename
    output_filename = Path(transcript_file).stem
    output_fileext = '.md' # Path(transcript_file).suffix # let's hard code it to .md so it shows up larger in quicklook, lol
    # output_filename = Path(transcript_file).name
    if(not output_filepath):
        output_filepath = str(Path(transcript_file).parent)
    
    if(specific_snippet > 0):
        output_filename = f"{output_filename}-{convert_to_ordinal(specific_snippet)}_snippet-{mode}-{model}{output_fileext}"
    else:
        output_filename = f"{output_filename}-{mode}-{model}{output_fileext}"
    if(not output_filename.endswith('.txt') and not output_filename.endswith('.md')): # if no extension (generated through directory-processing)
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
        print("Processing snippet {} of {} with length {} with model {}".format(i+1,len(snippet), len(snippet[i]), model))
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
            # if('gpt' not in model): # use_api is True but it might still be GPT rather than ollama
            #     result += "use ollama API\n----\n"
            try:
                gpt_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        # {"role": "user", "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the text. For additional context here is the previous result  \n " + previous}
                        {"role": "user", "content": "\"" + snippet[i] + "\"\n Do not include anything that is not in the text.\n "}
                    ],
                    temperature = 0.3,
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
    with open(output_filepath+"/"+output_filename, "w",encoding='utf-8') as text_file:
        text_file.write(result)
        print("Saved result to "+output_filepath+'/'+output_filename)

if __name__ == '__main__':
    text = ''
    output_filename_path = ''
    transcript_file = ''

    parser = argparse.ArgumentParser(description='summarizing articles (usually markdowns but should work for text)')
    parser.add_argument('md', type=str, help='the md to process. can also pass a directory with glob')
    # parser.add_argument('images', metavar='image', type=str, nargs='+', help='the image file(s) to analyse, e.g. image1.jpg image2.png image3.jpeg')
    parser.add_argument('--mode', type=str, default='QnAs', help='QnAs, cbb, thread, tp, note, summary/kp, translation, topix, definition') # note and summary produce quite similar output
    parser.add_argument('--prompt', type=str, help='the prompt to use in lieu of the default ones')
    parser.add_argument('--be_thorough', action='store_true', help='(currently only) for topix mode, longer prompt, more elaborate analysis')
    parser.add_argument('--lmodel', type=str, default="gpt-3.5-turbo", help='the GPT model to use for summarization (default: gpt-3.5-turbo)')
    parser.add_argument('--cf', type=str, help='the additional context file to use (JSON)')
    parser.add_argument('--save', action='store_true', help='if working with multiple files, should it save the concatenated text?')
    parser.add_argument('--specific_snippet', type=int, help='just process this part of the snippet (usually used together with --prompt, after seeing a summary/note and want to zoom in on specific point in the transcript)')
    parser.add_argument('--cm', type=str, default='word', help='chunking method to use (sentence or words)') # set to be the default, as Whisper starts to generate unpunctuated transcript in the middle of the conversation on April 6th. observed for the Cal Fussman Frank Blake episode, and the WCDHT - why Liz disappeared episode. but the Whisper transcription invoked by utub.py at around the same time, works as expected, punctuated and all. strange
    
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

    if(os.path.isfile(args.md)):
        print(f"it's a file, {args.md}")
        text = get_transcript(args.md, convert_md=False, ignore_links=True)
        output_filename_path = str(Path(args.md).parent)
        transcript_file = args.md
    elif(os.path.isdir(args.md)):
        print(f"it's a dir, {args.md}")
        # but how can I exclude the GPT-generated files? aka "QnAs-","note-","topix-","summary-","definition-","prompt-","concat_transcript-"
        file_list = glob.glob(args.md+'/*.md') + glob.glob(args.md+'/*.txt')
        for dmf in file_list:
            print(f"  dmf: {dmf}")
            text += get_transcript(dmf, convert_md=False, ignore_links=True)
            text += "\n\n\n[concatenated text]\n\n\n"
        parent_dir = Path(file_list[0]).parent
        output_filename_path = str(parent_dir) # assign the path to the parent directory of the first file in the list
        print(f"main :: output_filename_path: {output_filename_path}")
        # output_filename_path = os.path.dirname(args.md+'/') # assign the path to dirname. the last '/' is to ensure the dir is recognised as a dir in case I didn't pass the last slash. it doesn't hurt if we have double slash there either
        if(save_transcript):
            # for the sake of verbosity
            with(open(output_filename_path+'/concat_transcript-'+parent_dir.name+'.txt', 'w') as f):
                f.write(text)
        transcript_file = output_filename_path # these will be the same when we're processing a folder rather than a file
    elif(len(glob.glob(args.md)) > 0):
        matching_files = glob.glob(args.md)
        # print(f"matching_files: {matching_files}")
        if(len(matching_files) == 1 and args.md == matching_files[0]): # glob returns the folder itself (which exists)
            print("no match jg") # tp ini gak pernah keliatan nge-hit?
        else: # more than 1 file matched the glob pattern
            for gmf in matching_files:
                print(f"  gmf: {gmf}")
                text += get_transcript(gmf, convert_md=False, ignore_links=True)
                text += "\n\n\n[concatenated text]\n\n\n"
            parent_dir = Path(matching_files[0]).parent
            output_filename_path = str(parent_dir) # assign the path to the parent directory of the first file in the list
            print(f"main :: output_filename_path: {output_filename_path}")
            if(save_transcript):
                # for audit and posterity
                with(open(output_filename_path+'/concat_transcript-'+parent_dir.name+'.txt', 'w') as f):
                    f.write(text)
            transcript_file = output_filename_path # these will be the same when we're processing a folder rather than a file
            
    print("transcript acquired of length: "+str(len(text)))

    if(text and output_filename_path):
        llm_process(
            text, transcript_file, be_thorough=be_thorough,
            mode=args.mode, model=args.lmodel, context_filename=args.cf, prompt=args.prompt,
            output_filepath=output_filename_path, specific_snippet=specific_snippet, chunking_method=args.cm
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