from services.wiki_rag_service.store import WikiStore
from shared.config import get_settings


if __name__ == "__main__":
    paths = get_settings().paths
    result = WikiStore(db_path=paths.wiki_db, raw_sources_dir=paths.raw_sources_dir).ingest()
    print("Wiki ingest complete:")
    for key, value in result.items():
        print(f"- {key}: {value}")
