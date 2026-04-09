"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re


def _tokenize(text):
    """
    Lowercase the text and split it into alphanumeric word tokens.
    Punctuation is dropped, so "/api/users" becomes ["api", "users"]
    and a query for "users" will still match it.
    """
    return re.findall(r"\w+", text.lower())


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase tokens to the
        filenames they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }
        """
        index = {}
        for filename, text in documents:
            # Use a set so each filename is recorded once per token,
            # even if the token appears many times in the document.
            for token in set(_tokenize(text)):
                index.setdefault(token, []).append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.

        Each occurrence of a query token in the text adds 1 to the score, so
        documents that mention the query terms more often rank higher.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0

        # Count token frequencies in the document once, then look up
        # each query token. This is O(len(text) + len(query)).
        text_counts = {}
        for token in _tokenize(text):
            text_counts[token] = text_counts.get(token, 0) + 1

        return sum(text_counts.get(qt, 0) for qt in query_tokens)

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document
        snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Step 1: use the inverted index to find candidate documents
        # that contain at least one query token. This skips docs that
        # cannot possibly match.
        candidates = set()
        for token in query_tokens:
            for filename in self.index.get(token, []):
                candidates.add(filename)

        if not candidates:
            return []

        # Step 2: score each candidate by counting query-term occurrences.
        doc_lookup = {filename: text for filename, text in self.documents}
        scored = []
        for filename in candidates:
            text = doc_lookup[filename]
            score = self.score_document(query, text)
            scored.append((score, filename, text))

        # Step 3: sort highest score first and return top_k.
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(filename, text) for _, filename, text in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
