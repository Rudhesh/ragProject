import os
import chromadb

from google import genai
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


# 1. CONFIGURATION

MODEL_NAME = "gemini-3.8-flash"
DATABASE_PATH = "./company_db"
COLLECTION_NAME = "company_policies"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Please configure your environment variable."
    )

gemini_client = genai.Client(api_key=api_key)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


# 2. COMPANY POLICIES

POLICIES = [
    "The resignation notice period is 30 days.",
    "Working hours are from 9 AM to 5 PM.",
    "Employees receive 25 vacation days per year.",
    "Employees can work remotely two days per week.",
]


def load_policies():
    """Add initial policies if the collection is empty."""

    if collection.count() == 0:
        collection.add(
            ids=[
                "policy_1",
                "policy_2",
                "policy_3",
                "policy_4",
            ],
            documents=POLICIES,
        )


# 3. RETRIEVE RELEVANT POLICIES

def retrieve_policies(question):
    """Find the two most relevant policies."""

    results = collection.query(
        query_texts=[question],
        n_results=2,
    )

    return results["documents"][0]


# 4. GENERATE AN ANSWER

def generate_answer(question, documents):
    """Generate an answer using the retrieved policies."""

    if not documents:
        return "I could not find a relevant company policy."

    context = "\n".join(documents)

    prompt = f"""
    Answer the question using only the company policies below.

    If the answer is not available in these policies,
    say that you could not find the answer.

    Company policies:
    {context}

    Question:
    {question}
    """

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        return response.text or "Sorry, Gemini returned an empty response."

    except Exception as error:
        print(f"Gemini API error: {error}")
        return (
            "Sorry, I couldn't generate an answer right now. "
            "Please try again later."
        )
# 5. RUN THE APPLICATION

def main():
    load_policies()

    question = input(
        "Ask a question about company policies: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    documents = retrieve_policies(question)

    print("\nRelevant company policies:")

    for document in documents:
        print("-", document)

    answer = generate_answer(question, documents)

    print("\nGemini's answer:")
    print(answer)


if __name__ == "__main__":
    main()