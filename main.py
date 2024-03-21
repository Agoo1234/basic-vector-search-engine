import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


class TextVectorSearchEngine:
    def __init__(self):
        self.index = {}
        self.pipeline = self._build_pipeline()
        self.fitted = False

    def index_document(self, doc_id, text):
        self.index[doc_id] = text
        self.fitted = False

    def search(self, query_text, k=5):
        if not self.fitted:
            self.pipeline.fit(list(self.index.values()))
            self.fitted = True

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
    # Initialize the text search engine
    text_search_engine = TextVectorSearchEngine()

    # Index some sample documents
    text_search_engine.index_document("doc1", "This is a sentence about math class.")
    text_search_engine.index_document("doc2", "I like dogs")
    text_search_engine.index_document("doc3", "Math class is amazing.")
    text_search_engine.index_document("doc4", "I hate math class.")

    # Perform a search
    # query_text = "I love math class!"
    query_text = input("Enter a query: ")
    results = text_search_engine.search(query_text)

    # Print the search results
    print("Search Results:")
    for doc_id, similarity in results:
        print(f"Document ID: {doc_id}, Similarity: {similarity}")
