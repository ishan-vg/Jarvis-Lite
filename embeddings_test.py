import ollama
import math


MODEL = "embeddinggemma"


texts = [
    "The berry flavor is being overwhelmed by the milky whey taste.",
    "I am having problems with the flavor of my protein powder.",
    "My cybersecurity course is progressing slowly.",
    "I want to improve my Python programming skills."
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