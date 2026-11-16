"""Create the 'policies' index in Azure AI Search and fill it.

Each '## ' section of each markdown file in docs/policies becomes one document:
the text is stored for keyword search and its embedding for vector search.

Run from the project root:  python src/rag/build_index.py
"""
import glob
import os
import re

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from rag_search import INDEX, embed, search_credential, search_endpoint

EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown document into (section title, section text) pairs.

    The document title ('# ...') is kept at the top of every chunk so each
    chunk still says which policy it belongs to.
    """
    parts = re.split(r"\n(?=## )", text.strip())
    title = parts[0].strip().splitlines()[0].lstrip("# ").strip()
    sections = []
    for part in parts:
        if not part.startswith("## "):
            continue
        heading = part.splitlines()[0].lstrip("# ").strip()
        sections.append((heading, f"{title}\n{part.strip()}"))
    return sections


def main() -> None:
    index = SearchIndex(
        name=INDEX,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="section", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMENSIONS,
                vector_search_profile_name="vp",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="vp", algorithm_configuration_name="hnsw")],
        ),
    )
    SearchIndexClient(search_endpoint, search_credential).create_or_update_index(index)

    docs = []
    for path in sorted(glob.glob("docs/policies/*.md")):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            sections = split_sections(f.read())
        for i, (heading, chunk) in enumerate(sections):
            docs.append({
                "id": f"{name[:-3]}-{i}",
                "source": name,
                "section": heading,
                "content": chunk,
                "vector": embed(chunk),
            })

    results = SearchClient(search_endpoint, INDEX, search_credential).upload_documents(docs)
    ok = sum(1 for r in results if r.succeeded)
    print(f"Indexed {ok}/{len(docs)} sections from {len(glob.glob('docs/policies/*.md'))} documents")


if __name__ == "__main__":
    main()
