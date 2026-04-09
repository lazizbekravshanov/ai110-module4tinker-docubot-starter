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
import math
import re


# Common English filler words. Filtering these from queries prevents
# documents from ranking highly just because they share "the" or "is"
# with the question, which is the main failure mode of bag-of-words
# scoring on short docs.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can",
    "this", "that", "these", "those",
    "i", "me", "my", "you", "your", "he", "she", "it", "its",
    "we", "us", "our", "they", "them", "their",
    "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "about", "as", "into", "than", "over", "under",
    "any", "all", "some", "no", "not", "every",
    "there", "here", "out", "up", "down", "off",
})


def _tokenize(text):
    """
    Lowercase the text and split it into alphanumeric word tokens.
    Splitting on [a-z0-9]+ (instead of \\w+) drops underscores too, so
    identifiers like "auth_utils.py" become ["auth", "utils", "py"]
    and "AUTH_SECRET_KEY" becomes ["auth", "secret", "key"]. This lets
    a query for "auth" match snake_case identifiers in the docs.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text):
    """Tokenize and drop English stopwords. Used for query processing."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _split_into_chunks(text):
    """
    Split a document into paragraph-sized chunks separated by blank
    lines. Returns a list of stripped, non-empty chunks. Markdown lists
    and code blocks (which don't contain blank lines) stay intact.
    """
    raw = re.split(r"\n\s*\n", text)
    return [chunk.strip() for chunk in raw if chunk.strip()]


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

        # Flatten documents into paragraph-sized chunks. Retrieval works
        # at the chunk level so answers stay focused instead of dumping
        # an entire file as a "snippet".
        self.chunks = self._build_chunks(self.documents)  # [(filename, text)]

        # Build a retrieval index over the chunks.
        self.index = self.build_index(self.chunks)

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
    # Chunking (Phase 3)
    # -----------------------------------------------------------

    def _build_chunks(self, documents):
        """
        Flatten (filename, full_text) docs into a list of paragraph-sized
        (filename, chunk_text) pairs. Each chunk keeps its filename so
        answers can still cite their source.
        """
        chunks = []
        for filename, text in documents:
            for chunk_text in _split_into_chunks(text):
                chunks.append((filename, chunk_text))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1, refined in Phase 3)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build an inverted index mapping lowercase tokens to the chunk
        indices they appear in. Indexing at chunk level means retrieve()
        can return small, focused passages instead of whole documents.

        Example structure:
        {
            "token": [3, 7, 12],
            "database": [21, 22],
        }
        """
        index = {}
        for chunk_idx, (_, chunk_text) in enumerate(chunks):
            # Use a set so each chunk is recorded once per distinct token,
            # not once per occurrence.
            for token in set(_tokenize(chunk_text)):
                index.setdefault(token, []).append(chunk_idx)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1, refined in Phase 3)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a relevance score for how well the text matches the query.

        Stopwords are stripped from the query before scoring so common
        filler words ("the", "is", "of") cannot inflate the score. Each
        occurrence of a remaining query token in the text adds 1.
        """
        query_tokens = _content_tokens(query)
        if not query_tokens:
            return 0

        # Count token frequencies in the text once, then look up each
        # query token. This is O(len(text) + len(query)).
        text_counts = {}
        for token in _tokenize(text):
            text_counts[token] = text_counts.get(token, 0) + 1

        return sum(text_counts.get(qt, 0) for qt in query_tokens)

    def retrieve(self, query, top_k=3):
        """
        Find the top_k most relevant chunks for a query.

        Two explicit guardrails refuse to return anything when there is
        not enough evidence to answer:

        1. The query has no content tokens (all stopwords or empty).
        2. No single chunk contains enough distinct query tokens. When
           the query has more than one content token, the top-ranked
           chunk must contain at least ceil(N/2) of those distinct
           tokens, where N is the number of distinct content tokens in
           the query. This catches the "one generic word match" failure
           mode -- e.g. a question about GraphQL should not be answered
           just because "app" matches "app.db" somewhere.

        Returning [] here lets answer_retrieval_only() and answer_rag()
        emit a clean "I do not know" instead of guessing.
        """
        query_tokens = _content_tokens(query)
        if not query_tokens:
            # Guardrail 1: nothing meaningful to search for.
            return []

        distinct_query = set(query_tokens)

        # Gather candidate chunks: anything containing at least one
        # query content token. Skips chunks with zero overlap.
        candidates = set()
        for token in distinct_query:
            for chunk_idx in self.index.get(token, []):
                candidates.add(chunk_idx)

        if not candidates:
            return []

        # Score candidates. Track BOTH the raw frequency score and the
        # number of distinct query tokens present in the chunk; the
        # latter is a much better signal for "this passage is actually
        # on topic" vs "this passage happens to contain one filler word".
        scored = []
        for chunk_idx in candidates:
            filename, chunk_text = self.chunks[chunk_idx]
            chunk_token_set = set(_tokenize(chunk_text))
            distinct_hits = len(distinct_query & chunk_token_set)
            score = self.score_document(query, chunk_text)
            if score > 0:
                scored.append((distinct_hits, score, filename, chunk_text))

        if not scored:
            return []

        # Sort by distinct-token coverage first, then by raw frequency
        # as a tiebreaker. A chunk that hits 3 different query terms
        # ranks above one that hits the same term 5 times.
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        # Guardrail 2: refuse if even the best chunk does not contain
        # enough distinct query tokens. Single-token queries skip this
        # rule so short valid lookups (e.g. "users") still succeed.
        if len(distinct_query) > 1:
            required = math.ceil(len(distinct_query) / 2)
            if scored[0][0] < required:
                return []

        return [(filename, text) for _, _, filename, text in scored[:top_k]]

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
