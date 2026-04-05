import chromadb
import logging
import os
import uuid

from backend.utils import normalize_topic

logger = logging.getLogger(__name__)

# CONFIGURATION
DB_PATH = "./chroma_db"  # Folder where data will be stored on disk
COLLECTION_NAME = "tabla_nueva"
TOPIC_COLLECTION_PREFIX = "topic_"


def obtain_client():
    """Establish a connection to the persistent ChromaDB database."""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        return client
    except Exception as e:
        logger.error("Error connecting to ChromaDB: %s", e)
        return None


def topic_to_collection_name(topic: str) -> str:
    return f"{TOPIC_COLLECTION_PREFIX}{normalize_topic(topic)}"


def _infer_collection_name(collection_name=None, topic=None, metadata=None):
    if collection_name:
        return collection_name
    if topic:
        return topic_to_collection_name(topic)
    if isinstance(metadata, dict):
        meta_topic = metadata.get("topic")
        if isinstance(meta_topic, str) and meta_topic.strip():
            return topic_to_collection_name(meta_topic)
        source = metadata.get("source")
        if isinstance(source, str) and source.strip():
            source_base = os.path.splitext(source)[0]
            return topic_to_collection_name(source_base)
    return COLLECTION_NAME


def prepare_collection(client, collection_name=None, topic=None):
    """Create the collection if it does not exist."""
    target_collection = _infer_collection_name(collection_name=collection_name, topic=topic)
    logger.debug("Preparing collection '%s'", target_collection)
    collection = client.get_or_create_collection(name=target_collection)
    return collection


def delete_collection(collection_name=None, topic=None):
    """Delete one topic collection or all topic collections when unspecified."""
    client = obtain_client()
    if not client:
        return

    target_collection = _infer_collection_name(collection_name=collection_name, topic=topic)

    try:
        if collection_name or topic:
            client.delete_collection(name=target_collection)
            logger.info("Collection '%s' deleted.", target_collection)
            return

        collections = client.list_collections()
        deleted = 0
        for item in collections:
            name = item.name if hasattr(item, "name") else str(item)
            if name.startswith(TOPIC_COLLECTION_PREFIX) or name == COLLECTION_NAME:
                client.delete_collection(name=name)
                deleted += 1
        logger.info("Deleted %d collection(s).", deleted)
    except Exception as e:
        logger.warning("Could not delete collection (may not exist): %s", e)


def save_text_chroma(texto, id=None, metadata=None, collection_name=None, topic=None):
    """Save a text chunk with its embedding in the Chroma collection."""
    client = obtain_client()
    if not client:
        return None

    collection = prepare_collection(client, collection_name=collection_name, topic=topic)
    if id is None:
        id = str(uuid.uuid4())

    try:
        add_kwargs = {"ids": [id], "documents": [texto]}
        if isinstance(metadata, dict) and len(metadata) > 0:
            add_kwargs["metadatas"] = [metadata]
        collection.add(**add_kwargs)
        logger.debug("Document saved in Chroma id=%s", id)
        return id
    except Exception as e:
        logger.error("Error saving to Chroma: %s", e)
        return None


def search_similars(texto, n=3, collection_name=None, topic=None):
    """Query the Chroma collection by similarity."""
    client = obtain_client()
    if not client:
        return None

    collection = prepare_collection(client, collection_name=collection_name, topic=topic)
    try:
        return collection.query(
            query_texts=[texto],
            n_results=n,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        logger.error("Error querying Chroma: %s", e)
        return None


def search_by_source(source: str, collection_name=None, topic=None) -> list[str]:
    """Retrieve all chunks from ChromaDB that belong to a specific source file.

    Args:
        source: filename as stored in metadata (e.g. 'neural_networks.txt')

    Returns:
        List of chunk strings ordered by chunk index, or [] on error.
    """
    client = obtain_client()
    if not client:
        return []

    collection = prepare_collection(client, collection_name=collection_name, topic=topic)
    try:
        results = collection.get(
            where={"source": source},
            include=["metadatas", "documents"],
        )
        if not results or not results.get("documents"):
            return []
        pairs = list(zip(results["metadatas"], results["documents"]))
        pairs.sort(key=lambda p: p[0].get("chunk", 0))
        return [doc for _, doc in pairs]
    except Exception as e:
        logger.error("Error retrieving documents by source: %s", e)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cliente = obtain_client()
    if cliente:
        mi_collection = prepare_collection(cliente)
        ejemplo = "This is a test text to store in ChromaDB."
        doc_id = save_text_chroma(ejemplo)
        if doc_id:
            res = search_similars("test text")
            print(res)
