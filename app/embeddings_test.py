import ollama
import math


MODEL = "embeddinggemma"


texts = [
    "The train was delayed by forty minutes this morning.",
    "My commute took much longer than usual today.",
    "I need to finish my data structures assignment.",
    "I want to get better at Python programming."
]


# Generate embeddings
response = ollama.embed(
    model=MODEL,
    input=texts
)

embeddings = response["embeddings"]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


print("\nSimilarity between sentences:\n")


for i in range(len(texts)):
    for j in range(i + 1, len(texts)):

        similarity = cosine_similarity(
            embeddings[i],
            embeddings[j]
        )

        print(
            f"Text {i + 1} ↔ Text {j + 1}: "
            f"{similarity:.4f}"
        )
