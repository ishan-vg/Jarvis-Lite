import math
import os

import chromadb
import ollama

from knowledge import load_markdown_files


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "embeddinggemma"

CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "jarvis_memory"


# --------------------------------------------------
# Split a note into smaller chunks
# --------------------------------------------------

import re


def chunk_text(text, max_words=350):
    """
    Split a Markdown note into chunks that respect its heading structure.

    Returns a list of dictionaries:

        {
            "text":        text that actually gets embedded,
            "section":     heading path, e.g. "Daily Floor System > Monday",
            "chunk_index": position of this chunk inside the note
        }

    The heading path used to exist only as a prefix inside the text.
    It is now also returned separately so retrieval can reason about
    document structure, not just semantic similarity.
    """

    lines = text.splitlines()

    chunks = []
    heading_stack = []
    current_lines = []

    def flush_section():
        nonlocal current_lines

        body = "\n".join(current_lines).strip()

        if not body:
            current_lines = []
            return

        # Preserve the Markdown heading hierarchy
        heading_context = " > ".join(
            title for _, title in heading_stack
        )

        words = body.split()

        # If the section is small enough, keep it together.
        # If it is too large, split it into parts.
        if len(words) <= max_words:
            parts = [body]

        else:
            parts = [
                " ".join(words[i:i + max_words])
                for i in range(0, len(words), max_words)
            ]

        for part in parts:

            # The heading context stays inside the embedded text,
            # because it genuinely helps the embedding model.
            if heading_context:
                embedded_text = f"SECTION: {heading_context}\n\n{part}"
            else:
                embedded_text = part

            chunks.append({
                "text": embedded_text,
                "section": heading_context,
                "chunk_index": len(chunks)
            })

        current_lines = []

    for line in lines:

        # Detect Markdown headings
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)

        if match:
            # Save the previous section
            flush_section()

            level = len(match.group(1))
            title = match.group(2).strip()

            # Remove headings at the same or deeper level
            heading_stack = [
                (old_level, old_title)
                for old_level, old_title in heading_stack
                if old_level < level
            ]

            # Add the new heading
            heading_stack.append((level, title))

        else:
            current_lines.append(line)

    # Save the final section
    flush_section()

    return chunks


# --------------------------------------------------
# Calculate cosine similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:

        return 0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# --------------------------------------------------
# Create / load persistent ChromaDB
# --------------------------------------------------

def get_collection():

    os.makedirs(
        CHROMA_PATH,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

    return collection


# --------------------------------------------------
# Build persistent knowledge index
# --------------------------------------------------

def build_index(vault_path):

    collection = get_collection()

    notes = load_markdown_files(vault_path)

    chunks = []

    for note in notes:

        note_chunks = chunk_text(
            note["content"]
        )

        for chunk in note_chunks:

            chunks.append({

                "id": f"{note['path']}::{chunk['chunk_index']}",

                "file": note["file"],

                "path": note["path"],

                "section": chunk["section"],

                "chunk_index": chunk["chunk_index"],

                "content": chunk["text"]
            })


    print(
        f"Found {len(chunks)} chunks in vault."
    )


    if not chunks:

        print("No Markdown files found.")

        return collection


    # --------------------------------------------------
    # Remove existing versions of the current notes
    # --------------------------------------------------

    current_paths = set(
        chunk["path"]
        for chunk in chunks
    )

    existing = collection.get(
        include=["metadatas"]
    )

    existing_ids = existing.get(
        "ids",
        []
    )

    existing_metadata = existing.get(
        "metadatas",
        []
    )


    ids_to_delete = []

    for existing_id, metadata in zip(
        existing_ids,
        existing_metadata
    ):

        if metadata:

            existing_path = metadata.get(
                "path"
            )

            if existing_path not in current_paths:

                ids_to_delete.append(
                    existing_id
                )


    if ids_to_delete:

        collection.delete(
            ids=ids_to_delete
        )

        print(
            f"Removed {len(ids_to_delete)} "
            "deleted chunks."
        )


    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = response["embeddings"]


    # --------------------------------------------------
    # Store chunks in ChromaDB
    # --------------------------------------------------

    collection.upsert(

        ids=[
            chunk["id"]
            for chunk in chunks
        ],

        embeddings=embeddings,

        documents=[
            chunk["content"]
            for chunk in chunks
        ],

        metadatas=[
            {
                "file": chunk["file"],
                "path": chunk["path"],

                # Structural metadata: this is what lets retrieval
                # ask "which sections are siblings of this one?"
                "section": chunk["section"] or "(no heading)",
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
        ]
    )


    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    print(
        "Knowledge database ready."
    )

    return collection


# --------------------------------------------------
# Search persistent knowledge index
# --------------------------------------------------

def search_index(
    query,
    collection,
    top_k=5,
    min_similarity=0.35
):

    if collection.count() == 0:

        return []


    # --------------------------------------------------
    # Convert question into an embedding
    # --------------------------------------------------

    query_response = ollama.embed(

        model=EMBEDDING_MODEL,

        input=query
    )

    query_embedding = (
        query_response["embeddings"][0]
    )


    # --------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------

    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=min(
            top_k,
            collection.count()
        ),

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    if not results["documents"]:

        return []


    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]


    formatted_results = []


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # Chroma's default distance is cosine distance.
        # Similarity = 1 - distance.

        similarity = 1 - distance


        formatted_results.append({

            "similarity": similarity,

            "file": metadata.get(
                "file",
                "Unknown"
            ),

            "path": metadata.get(
                "path",
                ""
            ),

            "section": metadata.get(
                "section",
                "(no heading)"
            ),

            "chunk_index": metadata.get(
                "chunk_index",
                -1
            ),

            "content": document
        })


    # --------------------------------------------------
    # Apply relevance threshold
    # --------------------------------------------------

    filtered_results = [

        result

        for result in formatted_results

        if result["similarity"] >= min_similarity
    ]


    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not filtered_results:

        return formatted_results[:1]


    return filtered_results[:top_k]


# --------------------------------------------------
# Section expansion
# --------------------------------------------------
#
# Semantic similarity answers "which chunks look most like
# the question?".
#
# It does NOT answer "give me all the parts of this system".
#
# Seven sibling sections like:
#
#     Daily Floor System > Monday
#     Daily Floor System > Tuesday
#     ...
#
# are almost identical to an embedding model, so top-k will
# return an arbitrary one or two of them and silently drop
# the rest.
#
# The rule below is deliberately small and explainable:
#
#   For each retrieved chunk, look at its sibling sections.
#     - If the question names one of those siblings,
#       the user asked about a specific one -> do not expand.
#     - If the question names none of them,
#       the user asked about the whole group -> add them all.
#
# This is string matching on heading names. It is not a
# classifier, and it can be read and predicted by a human.


def split_section(section):
    """
    Split "Daily Floor System > Tuesday" into
    ("Daily Floor System", "Tuesday").

    A top-level section has no parent, so it has no siblings
    and returns (None, section).
    """

    if not section or ">" not in section:
        return None, section

    parts = [
        part.strip()
        for part in section.split(">")
    ]

    parent = " > ".join(parts[:-1])

    leaf = parts[-1]

    return parent, leaf


def query_mentions(query, phrase):
    """
    True if the question names this heading, as a whole word.

    "what time is college on Wednesday?" mentions "Wednesday".
    "what is my schedule" mentions no day at all.
    """

    if not phrase:
        return False

    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"

    return re.search(pattern, query.lower()) is not None


def get_file_chunks(collection, path, cache):
    """
    Fetch every chunk belonging to one note, once per search.
    """

    if path in cache:
        return cache[path]

    records = collection.get(

        where={"path": path},

        include=[
            "documents",
            "metadatas"
        ]
    )

    chunks = []

    for document, metadata in zip(
        records["documents"],
        records["metadatas"]
    ):

        chunks.append({

            "similarity": None,

            "expanded": True,

            "file": metadata.get("file", "Unknown"),

            "path": metadata.get("path", ""),

            "section": metadata.get("section", "(no heading)"),

            "chunk_index": metadata.get("chunk_index", -1),

            "content": document
        })

    cache[path] = chunks

    return chunks


def find_siblings(collection, path, parent, cache):
    """
    Direct children of `parent` inside one note.

    "Daily Floor System" -> Monday, Tuesday, ... Sunday
    but NOT anything nested deeper than one level.
    """

    prefix = parent + " > "

    siblings = []

    for chunk in get_file_chunks(collection, path, cache):

        section = chunk["section"]

        if not section.startswith(prefix):
            continue

        remainder = section[len(prefix):]

        # Only direct children, not grandchildren
        if ">" in remainder:
            continue

        siblings.append(chunk)

    return siblings


def expand_sections(
    query,
    results,
    collection,
    max_expanded=15
):
    """
    Add missing sibling sections to a set of search results.

    The original semantic results are kept in rank order.
    Expanded chunks are appended afterwards in document order,
    so Monday..Sunday read in the order they appear in the note.

    Nothing is removed. This layer only adds evidence.
    """

    if not results:
        return results

    cache = {}

    # Chunks we already have
    existing = set(
        (result["path"], result["chunk_index"])
        for result in results
    )

    handled_groups = set()

    added = []

    for result in results:

        parent, leaf = split_section(
            result["section"]
        )

        # Top-level section: no siblings to expand into
        if parent is None:
            continue

        group = (result["path"], parent)

        if group in handled_groups:
            continue

        handled_groups.add(group)

        siblings = find_siblings(
            collection,
            result["path"],
            parent,
            cache
        )

        # A group of one is not a group
        if len(siblings) <= 1:
            continue

        # Did the question name specific siblings?
        named = [
            sibling
            for sibling in siblings
            if query_mentions(
                query,
                split_section(sibling["section"])[1]
            )
        ]

        if named:
            # The question was specific. Do not pull in the
            # whole group just because one member matched.
            chosen = named

        else:
            # The question was general. Completeness matters.
            chosen = siblings

        for sibling in chosen:

            key = (
                sibling["path"],
                sibling["chunk_index"]
            )

            if key in existing:
                continue

            existing.add(key)

            added.append(sibling)

    # Document order, so sections read naturally
    added.sort(
        key=lambda chunk: (
            chunk["path"],
            chunk["chunk_index"]
        )
    )

    if len(added) > max_expanded:

        print(
            f"[expansion] {len(added)} sibling sections found, "
            f"keeping the first {max_expanded}."
        )

        added = added[:max_expanded]

    return results + added


# --------------------------------------------------
# Test retrieval system
# --------------------------------------------------

if __name__ == "__main__":

    import json


    with open(

        "config/config.json",

        "r",

        encoding="utf-8"

    ) as file:

        config = json.load(file)


    vault_path = config["vault_path"]


    print(
        "\nBuilding / loading knowledge database..."
    )


    index = build_index(
        vault_path
    )


    print(
        "\nKnowledge database ready."
    )


    while True:

        query = input(
            "\nSearch your knowledge base: "
        )


        if query.lower() == "exit":

            break


        results = search_index(

            query,

            index,

            top_k=5
        )

        results = expand_sections(
            query,
            results,
            index
        )


        print(
            "\nMost relevant knowledge:\n"
        )


        for result in results:

            if result.get("expanded"):
                score = "expanded"
            else:
                score = f"{result['similarity']:.4f}"

            print(
                f"[{score}] {result['file']}"
            )

            print(
                f"   SECTION: {result['section']}"
            )

            print(
                result["content"][:500]
            )

            print(
                "\n" + "-" * 60
            )