#!/usr/bin/env python3

'''
Goal: minimise number of browser tabs I need to open when researching, get LLM's help to reduce some cognitive load and do preliminary skimming.

Usage:
    python3 url2md.py --url https://thereader.mitpress.mit.edu/noam-chomsky-and-andrea-moro-on-the-limits-of-our-comprehension/
    python3 url2md.py --ul url_list.txt
    python3 url2md.py --ul ul-sora.txt --redo

Behavior:
    If both ul and url args are provided, all URLs are processed. if url is not provided, only ul is processed.
    This script get a JS check on twitter: https://x.com/dwarkesh_sp/status/1741972913501405435 ("JavaScript is not available.")
    A Facebook URL (perhaps private as well) also returns some blank / redirection thing

Sample output:
    python3 url2md.py --url https://en.wikipedia.org/wiki/The_Revolt_of_the_Masses
    retrieving url: https://en.wikipedia.org/wiki/The_Revolt_of_the_Masses

TODO
√1. make these into the same folder, can specify "research_codename" as such.
    oh, via ul aja. jadi pake nama txt-nya. so all urls are assumed to be of that theme. simple!
2. support providing some DOM selector, e.g. extract top comments only from a HN thread.
    or to not get the navigations from some wikipedia or blog pages
    mirip shot-scraper sebenernya, so let's see if that is sufficient

'''
import os
import argparse
import html2text
import requests_html

from time import sleep
from pathlib import Path
from urllib3.util.url import parse_url
from urllib.parse import parse_qs
from base64 import b64decode

zyte_api_key = os.getenv("ZYTE_API_KEY")

if __name__ == '__main__':
    session = requests_html.HTMLSession()
    md_converter = html2text.HTML2Text()
    md_converter.body_width = 0  # Disable line wrapping

    url_list = []
    url = ''
    output_dir = 'output/mds/'

    parser = argparse.ArgumentParser(description='get markdown for the url')
    parser.add_argument('--url', type=str, default=url, help='url you\'d like to markdown')
    parser.add_argument('--ul', type=str, help='path to url list, one url per line')
    parser.add_argument('--output_dir', type=str, help='dir to use to save the output file')
    parser.add_argument('--redo', action='store_true', help='if specified, will overwrite existing md files')
    args = parser.parse_args()

    ul = ''
    if args.ul:
        ul = args.ul
    else:
        url = args.url
    
    if url:
        url_list.append(url)
    
    # if ul is provided then the output folder is theme-based
    if(ul and os.path.exists(ul)):
        with open(ul, 'r') as ful:
            urls = ful.read().split("\n")
            for url in urls:
                if url:
                    url_list.append(url)
        output_dir = output_dir+Path(ul).stem # os.path.basename(ul)

    for url in url_list:
        pu = parse_url(url)
        netloc = pu.host
        slug = netloc # if some mere netloc / mere homepage was given
        if pu.path:
            # if I wanna keep the netloc in the file and not rely on the subdirectories to infer source.
            # hmm. better include the source URL in the MD? done
            # slug += '-'+[segment for segment in pu.path.split('/') if segment][-1]
            # replace the slug (filled with netloc) with just the path onwards
            slug = [segment for segment in pu.path.split('/') if segment][-1]
        if pu.query:
            # bisa jelek kalo ada query params kyk gini nih
            # url = 'https://www.mas.gov.sg/publications?date=2021-01-01T00%3A00%3A00Z%2C2021-12-31T23%3A59%3A59Z&content_type=Monographs%2FInformation%20Papers&page=1&q=ubin'
            sanitised_qs = {param: parse_qs(pu.query).get(param, [None])[0] for param in parse_qs(pu.query).keys()}
            # not efficient and kinda dumb lah, just to a string replace on the final slug
            # sanitised_qs = {param: parse_qs(pu.query).get(param, [None])[0] for param in parse_qs(pu.query).keys() if '/' not in parse_qs(pu.query).get(param, [None])[0]}
            for k in sanitised_qs.keys():
                # slug += k+sanitised_qs[k].replace('/', '')
                slug += k+sanitised_qs[k].replace('/', '').replace(':', '').replace(',', '').replace(' ', '')
        
        if not ul:
            output_dir = output_dir+netloc
        
        # override output_dir if output_dir is specified as an arg
        if(args.output_dir and os.path.exists(args.output_dir)):
            output_dir = args.output_dir

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if(not args.redo and os.path.exists(output_dir+'/'+slug+'.md')):
            print(output_dir+'/'+slug+'.md' + " file exists")
            continue

        headers = {
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # print(f"retrieving url: {url} with headers: {headers}")
        
        if('medium.com' in url):
            # with Zyte API
            response = session.post("https://api.zyte.com/v1/extract",
                auth=(zyte_api_key, ""),
                json={
                    "url": url,
                    "browserHtml": True,
                    "screenshot": True,
                },
            )
            browser_html: str = response.json()["browserHtml"]
            markdown_content = "Downloaded from: "+url+"\n\n---\n\n"
            markdown_content += md_converter.handle(browser_html)

            screenshot: bytes = b64decode(response.json()["screenshot"])
            with open(output_dir+'/'+slug+'.jpg', "wb") as fp:
                fp.write(screenshot)
        else:
            # without Zyte API
            response = session.get(url, headers=headers)
            response.html.render()
            markdown_content = "Downloaded from: "+url+"\n\n---\n\n"
            markdown_content += md_converter.handle(response.html.html)

        with open(output_dir+'/'+slug+'.md', 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            print(output_dir+'/'+slug+'.md') # so I can pipe it to pbcopy and md2notes.py it easily
    
        if(len(url_list) > 1):
            sleep(1) # dumb throttling
