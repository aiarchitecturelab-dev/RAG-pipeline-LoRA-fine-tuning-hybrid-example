"""A small, readable RAG (retrieval-augmented generation) reference implementation.

The package follows the pipeline from the video, one module per step:

    loaders.py      read documents (markdown / txt / optional PDF) plus metadata
    chunking.py     cut documents into overlapping, heading-aware chunks
    embedders.py    turn text into vectors (offline hashing by default)
    vectorstore.py  store the vectors + metadata, search with metadata filters
    retrieve.py     embed the question, apply the PERMISSION FILTER, take top-k
    rerank.py       re-score the candidates and keep the best few
    prompting.py    build the prompt with numbered sources, validate citations
    llm.py          call the Anthropic API (the only module that needs a key)
    pipeline.py     glue that wires the steps together

Nothing here needs a network connection except llm.py.
"""

__version__ = "0.1.0"
