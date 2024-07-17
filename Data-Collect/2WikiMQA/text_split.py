import json
from typing import List, Any

from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, SpacyTextSplitter, TextSplitter
from text_process import sentence_tokenize_extract


class TableAwareExtractJsonSplitter(TextSplitter):
    """Implementation of splitting json text into sentences. Can be only used in combination with a ExtractJsonLoader."""

    def __init__(self, separator: str = "\n\n", **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = [
            sentence
            for ix, sentence in enumerate(sentence_tokenize_extract(json.loads(text)))
        ]
        return self._merge_splits(splits, self._separator)


def get_text_splitter(splitter, chunk_size, separator, chunk_overlap):
    if splitter == "character":
        text_splitter = CharacterTextSplitter(chunk_size = chunk_size, separator=separator,
                                              chunk_overlap=chunk_overlap)
    elif splitter == "token":
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter == "spacy":
        # Multi sentence splitter with overlap
        text_splitter = SpacyTextSplitter(chunk_size=chunk_size, separator=separator,
                                          chunk_overlap=chunk_overlap)
    elif splitter == "table_aware_extract_json":
        # Single sentence splitter without overlap
        text_splitter = TableAwareExtractJsonSplitter(chunk_size=chunk_size, separator=separator,
                                                      chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Invalid text splitter")

    return text_splitter