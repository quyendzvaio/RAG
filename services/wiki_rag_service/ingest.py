from services.wiki_rag_service.app import _wiki_store


if __name__ == "__main__":
    result = _wiki_store().ingest()
    print("Wiki ingest complete:")
    for key, value in result.items():
        print(f"- {key}: {value}")
