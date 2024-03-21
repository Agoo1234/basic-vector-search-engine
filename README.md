# basic-vector-search-engine

A slow vector search engine that uses KNN and cosine similarity to find similar sentences to the query. 

As of now, it only checks sentences against sentences.txt which contains 3200 unique sentences. 

It takes about 2 seconds to return results but may vary from computer to computer. This is nowhere near as fast as google, duckduckgo, bing, or other search engines which utilize better algorithms, higher dimensional vectors, and more. This code also does not use vector embedding to find synonyms and similar phrases, unlike most modern search engines.

Made for a math presentation.