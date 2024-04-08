./url2md.py --url https://thereader.mitpress.mit.edu/noam-chomsky-and-andrea-moro-on-the-limits-of-our-comprehension/

./md2notes.py output/mds/thereader.mitpress.mit.edu/noam-chomsky-and-andrea-moro-on-the-limits-of-our-comprehension.md
./md2notes.py output/mds/thereader.mitpress.mit.edu/noam-chomsky-and-andrea-moro-on-the-limits-of-our-comprehension.md --lmodel mistral

./audio2llm.py --af sample/test_sound.m4a
./audio2llm.py --tf output/test_sound.txt --lmodel mistral

./youtube2llm.py summarise --vid=8Lzo3nhcTSk --nc
./youtube2llm.py summarise --vid=8Lzo3nhcTSk --nc --lmodel mistral
./youtube2llm.py embed --tf output/8Lzo3nhcTSk-transcript.txt
./youtube2llm.py ask --ef output/embeddings/8Lzo3nhcTSk-transcript-transcript_embedding.csv