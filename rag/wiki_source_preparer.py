import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.documents import Document

from rag.data_loader import parse_legal_html


def prepare_wiki_sources(input_dir: str, output_dir: str, include_jsonl_content: bool = False) -> Dict[str, int]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    sources_dir = output_path / "sources"
    records_path = output_path / "articles.jsonl"
    manifest_path = output_path / "manifest.json"
    index_path = output_path / "index.md"
    schema_path = output_path / "RAW_SOURCE_SCHEMA.md"

    sources_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "version": 1,
        "generated_at": generated_at,
        "input_dir": str(input_path),
        "record_file": "articles.jsonl",
        "source_count": 0,
        "article_count": 0,
        "sources": [],
    }

    index_lines = [
        "---",
        "type: raw_sources_index",
        f"generated_at: {json.dumps(generated_at, ensure_ascii=False)}",
        "---",
        "",
        "# Raw Legal Sources Index",
        "",
        "Danh muc cac source Markdown da duoc chuan hoa tu `data_raw/*.html`.",
        "",
    ]

    with records_path.open("w", encoding="utf-8") as records_file:
        for html_file in sorted(input_path.glob("*.html")):
            html = html_file.read_text(encoding="utf-8")
            documents = parse_legal_html(html, html_file.name)
            source_hash = _sha256_bytes(html.encode("utf-8"))
            source_slug = html_file.stem
            source_md_name = f"{source_slug}.md"
            source_md_path = sources_dir / source_md_name

            _write_source_markdown(
                path=source_md_path,
                source_file=html_file.name,
                source_hash=source_hash,
                documents=documents,
                generated_at=generated_at,
            )

            for document in documents:
                record = _document_record(document, source_hash, source_md_name, include_content=include_jsonl_content)
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            doc_types = sorted({doc.metadata.get("doc_type", "") for doc in documents if doc.metadata.get("doc_type")})
            effective_dates = sorted(
                {doc.metadata.get("effective_date", "") for doc in documents if doc.metadata.get("effective_date")}
            )
            source_summary = {
                "source_file": html_file.name,
                "source_hash": source_hash,
                "source_markdown": f"sources/{source_md_name}",
                "article_count": len(documents),
                "doc_types": doc_types,
                "effective_dates": effective_dates,
            }
            manifest["sources"].append(source_summary)
            manifest["source_count"] += 1
            manifest["article_count"] += len(documents)

            index_lines.extend(
                [
                    f"## [[sources/{source_md_name}|{html_file.name}]]",
                    "",
                    f"- Hash: `{source_hash}`",
                    f"- So dieu khoan: {len(documents)}",
                    f"- Loai van ban: {', '.join(doc_types) if doc_types else 'unknown'}",
                    f"- Ngay hieu luc: {', '.join(effective_dates[:8]) if effective_dates else 'unknown'}",
                    "",
                ]
            )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    schema_path.write_text(_raw_source_schema(), encoding="utf-8")

    return {
        "source_count": manifest["source_count"],
        "article_count": manifest["article_count"],
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "index_path": str(index_path),
        "schema_path": str(schema_path),
    }


def _write_source_markdown(
    *,
    path: Path,
    source_file: str,
    source_hash: str,
    documents: List[Document],
    generated_at: str,
) -> None:
    doc_type_counts = Counter(doc.metadata.get("doc_type", "") for doc in documents if doc.metadata.get("doc_type"))
    frontmatter = {
        "type": "raw_legal_source",
        "source_file": source_file,
        "source_hash": source_hash,
        "article_count": len(documents),
        "doc_types": sorted(doc_type_counts),
        "generated_at": generated_at,
        "status": "immutable_normalized_source",
    }
    lines = [
        _frontmatter(frontmatter),
        f"# {source_file}",
        "",
        "> Normalized raw source generated from immutable HTML. Wiki-service may read this file but should not edit it.",
        "",
    ]

    for document in documents:
        metadata = document.metadata
        title = " ".join(
            value
            for value in (
                metadata.get("article_code", ""),
                metadata.get("article_title", ""),
            )
            if value
        ).strip()
        lines.extend(
            [
                f"## {title or 'Untitled Article'}",
                "",
                f"- Source file: `{metadata.get('source_file', source_file)}`",
                f"- Article index: {metadata.get('article_index', '')}",
                f"- Chapter: {metadata.get('chapter', '')} {metadata.get('chapter_title', '')}".rstrip(),
                f"- Document type: {metadata.get('doc_type', '') or 'unknown'}",
                f"- Effective date: {metadata.get('effective_date', '') or 'unknown'}",
                "",
            ]
        )

        legal_note = metadata.get("legal_note", "")
        if legal_note:
            lines.extend(["### Legal Note", "", legal_note, ""])

        urls = metadata.get("urls") or []
        if urls:
            lines.extend(["### Source URLs", ""])
            lines.extend(f"- {url}" for url in urls)
            lines.append("")

        lines.extend(["### Content", "", document.page_content, ""])

        related_text = metadata.get("related_text", "")
        related_urls = metadata.get("related_urls") or []
        if related_text or related_urls:
            lines.extend(["### Related Legal References", ""])
            if related_text:
                lines.extend([related_text, ""])
            if related_urls:
                lines.extend(f"- {url}" for url in related_urls)
                lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _document_record(document: Document, source_hash: str, source_markdown: str, include_content: bool) -> Dict:
    metadata = dict(document.metadata)
    article_heading = " ".join(
        value
        for value in (
            metadata.get("article_code", ""),
            metadata.get("article_title", ""),
        )
        if value
    ).strip()
    record = {
        "id": f"{Path(metadata.get('source_file', metadata.get('source', 'source'))).stem}:{metadata.get('article_index', '')}",
        "source_hash": source_hash,
        "source_markdown": f"sources/{source_markdown}",
        "article_heading": article_heading,
        "metadata": metadata,
        "content_hash": _sha256_bytes(document.page_content.encode("utf-8")),
        "content_preview": document.page_content[:500],
    }
    if include_content:
        record["content"] = document.page_content
    return record


def _frontmatter(values: Dict) -> str:
    lines = ["---"]
    for key, value in values.items():
        lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    return "\n".join(lines)


def _yaml_value(value):
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw_source_schema() -> str:
    return """# Raw Source Schema

This directory is the immutable normalized source layer for wiki-service.

## Files

- `manifest.json`: generation metadata, source checksums, source/article counts.
- `index.md`: Obsidian-friendly catalog of normalized source Markdown files.
- `articles.jsonl`: one article/reference record per line.
- `sources/*.md`: full normalized Markdown content for each original HTML source.

## Rules

- Wiki-service may read files in this directory but must not edit them.
- LLM-generated wiki pages must be written to a separate layer, e.g. `wiki_data/wiki/`.
- `data_raw/*.html` remains the original source of truth.
- If `data_raw` changes, regenerate this directory and compare `manifest.json`.

## articles.jsonl record

```json
{
  "id": "source-stem:article-index",
  "source_hash": "sha256(original html)",
  "source_markdown": "sources/source-stem.md",
  "article_heading": "Điều ...",
  "metadata": {
    "source_file": "...html",
    "chapter": "Chương ...",
    "chapter_title": "...",
    "article_code": "Điều ...",
    "article_title": "...",
    "legal_note": "...",
    "doc_type": "Luật/Nghị định/Thông tư/Quyết định",
    "effective_date": "dd/mm/yyyy",
    "urls": [],
    "related_text": "...",
    "related_urls": [],
    "article_index": 1
  },
  "content_hash": "sha256(normalized article content)",
  "content_preview": "first 500 characters"
}
```

By default `articles.jsonl` does not duplicate full content. Read `source_markdown` for the complete text.
"""


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare immutable normalized raw sources for wiki-service.")
    parser.add_argument("--input", default="data_raw", help="Directory containing raw HTML files.")
    parser.add_argument("--output", default="wiki_data/raw_sources", help="Output directory for normalized wiki sources.")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before generating.")
    parser.add_argument(
        "--include-jsonl-content",
        action="store_true",
        help="Also include full article content in articles.jsonl. Off by default to avoid duplicated storage.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.clean and os.path.exists(args.output):
        shutil.rmtree(args.output)

    result = prepare_wiki_sources(args.input, args.output, include_jsonl_content=args.include_jsonl_content)
    print("Prepared wiki raw sources:")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
