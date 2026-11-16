"""Shared clients and the policy search function (hybrid: keywords + vectors)."""
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

INDEX = "policies"
EMBED_DEPLOYMENT = "embed"   # deployment names in the Azure AI resource
CHAT_DEPLOYMENT = "chat"

aoai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
search_endpoint = os.environ["SEARCH_ENDPOINT"]
search_credential = AzureKeyCredential(os.environ["SEARCH_KEY"])


def embed(text: str) -> list[float]:
    return aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=text).data[0].embedding


def search_policies(query: str, k: int = 4) -> list[dict]:
    """Return the k policy sections that best match the query.

    Hybrid search: `search_text` matches exact words, the vector query matches
    meaning, and Azure AI Search merges the two rankings.
    """
    client = SearchClient(search_endpoint, INDEX, search_credential)
    hits = client.search(
        search_text=query,
        top=k,
        vector_queries=[VectorizedQuery(vector=embed(query), k_nearest_neighbors=k, fields="vector")],
        select=["source", "section", "content"],
    )
    return [{"source": h["source"], "section": h["section"], "content": h["content"]} for h in hits]


def answer_from_policies(question: str) -> str:
    """Policy Q&A: answer only from retrieved sections, with citations."""
    hits = search_policies(question)
    context = "\n\n".join(f"[{h['source']}, {h['section']}]\n{h['content']}" for h in hits)
    resp = aoai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Answer using ONLY the policy sections provided. Cite each fact as "
                "[source, section]. If the sections do not contain the answer, say so.")},
            {"role": "user", "content": f"Policy sections:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What discount can a month-to-month customer with 8 months tenure get?"
    for h in search_policies(q):
        print(f"- {h['source']} | {h['section']}")
    print()
    print(answer_from_policies(q))
