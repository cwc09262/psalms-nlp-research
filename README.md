January 2026 - Present

# NLP Research on the Psalms

### Focusing on : *Computational Semantics / Semantic Text Analysis*

This repository is dedicated to research on NLP analysis of the Book of Psalms, built on the foundation of the original project, [*st\_david-s-beacon*](https://github.com/cwc09262/st_david-s-beacon/tree/main). It is intended to separate ongoing and future research from the original work doine that started the entire project. All Psalm data originates from the original repository, but the experiments and models here are for research and exploration purposes only. All of the files and data was gathered at the begining of this repoistory's history, comes from the directory "[*fall 2025*](https://github.com/cwc09262/st_david-s-beacon/tree/main/website/scripts/fall%202025)" directory. This can be found by navigating to `website/scripts/fall 2025`, from the main page of **st\_david-s-beacon**.

## Data

This study is only focusing on Psalms from the Christian Orthodox Church. There are two different souces of Psalms being used. One of them comes from The Book of Psalms within the Orthodox Study Bible. The Other source of Psalms comes from The Psalter According to the Seventy.

All of the data, was scraped, organized and cleaned from the original repository as well. The or5iginal commit history from the previous repository has been coppie3d over to this current repository as well.

## Methods Applied

This repository explores four different approaches for analyzing and retrieving text from the Book of Psalms using embedding-based similarity, vector space models, and semantic search techniques.

<b>`TF-IDF`</b> \- A statistical measure used to evaluate the importance of a word in a document relative to a collection of documents \(the corpus\)\.

<b>`TF-IDF scaled by GLoVe Vector Weights`</b> \- Taking the precomputed vectors produced by **TF-IDF**, and weighting each term to it's semantic meaning according to a specific pre-trained GLoVe embedding space.

Using this [github repository](https://github.com/stanfordnlp/GloVe) to generate the GLoVe embeddings

**`BERT`** *Bidirectional Encoder Representations and Transformers* - Using a deep learning model contextual meanigs of the text can be produced. This approahed used the weords and context around it to generate meaning and going beyond the surface level of just word frequency. 

**`SBERT`** *Sentence BERT* - By a modifcation of **BERT** this approach aims to generate fixed-size sentence embeddings, this creates the ability to study semantic similarity and clustering. 
<br>
<br>
Using a mix of contextual transformers and traditional statistical text‑embedding methods - TFIDF, and TFIDF scaled by GLoVe Vectors, BERT, SBERT - evidenced may offer guidance as to what techniques can be applied for getting intended results.

## Blind Scoring Evaluation

An importatn aspect of the research conduycted stems from the evaluation of each of the different [methods](#methods-applied). Being able to get fair results, without bias, was vital. Using code, written by AI, I was able to get results to score blindly for my own scoring. There are 8 different queries and the top 5 results were colected. This was preformed for each of the four different methods mentioned above.

The algorithm works by randomly picking and results that was not scored yet. Once picked, the query is given with the specific result. A promopt was given to rate the specifc relaut based on the query on a scale of 0 to 10, with ten being the most accurate or meaningful result. This was all done via the terminal.

After that, the same algorithms were used to feed a flask website for blind scoring could be done so that I could collect blind scoring from other people. The Flask website is used to ensure ease of the use for the end user and priority of security.

By gathering both scores from my personal perspective as well as from various other Orthodox Christians, Scholars and Seminarians, specific purposes have the possibility of being displayed from the results and therefore can help label different techniques with specific use cases.