import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import time
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

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

# @app.route("/")
# def main():
#     return render_template("./index.html")
#
# @app.route("/search", methods=["GET"])
# def search():
#     query_text = request.form.get("q")
#     if query_text is None:
#         query_text = "empty"
#     start_time = time.time()
#     results = text_search_engine.search(query_text)
#     time_elapsed = round(time.time() - start_time, 3)
#     search_results = []
#     for doc_id, similarity in results:
#         search_results.append(documents[doc_id])
#     final = {"results": search_results, "time_elapsed": time_elapsed}
#     return jsonify(final)
#
# app.run(host="0.0.0.0", port=3000)