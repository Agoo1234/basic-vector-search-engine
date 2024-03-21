import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import time

class TextVectorSearchEngine:
    def __init__(self, documents=None):
        self.index = {}
        self.pipeline = self._build_pipeline()
        if documents:
            self.fit_documents(documents)

    def fit_documents(self, documents):
        self.pipeline.fit(documents)
        self.index = {i: doc for i, doc in enumerate(documents)}

    def search(self, query_text, k=5):
        query_vector = self.pipeline.transform([query_text])
        results = []
        for doc_id, text in self.index.items():
            doc_vector = self.pipeline.transform([text])
            similarity = self.calculate_similarity(query_vector, doc_vector)
            results.append((doc_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def calculate_similarity(self, query_vector, doc_vector):
        return cosine_similarity(query_vector, doc_vector)[0][0]

    def _build_pipeline(self):
        pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer()),
            ('tfidf_transformer', TfidfTransformer())
        ])
        return pipeline


# Example usage:
if __name__ == "__main__":
    # Initialize the text search engine with training documents
    with open("sentences.txt") as f:
        documents = [line.rstrip('\n') for line in f]
    text_search_engine = TextVectorSearchEngine(documents)

    query_text = input("What is your search query? ")
    n = int(input("How many results? "))
    start_time = time.time()
    results = text_search_engine.search(query_text, k=n)

    # Print the search results
    print("Search Results:")
    print(f"Displayed in: {round(time.time() - start_time, 3)}s")
    for doc_id, similarity in results:
        print(f"{documents[doc_id]}, Similarity: {similarity}")
