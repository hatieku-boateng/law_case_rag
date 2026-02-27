import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_VECTOR_STORE_NAME = "supreme court cases"


def _find_vector_store_id(client: OpenAI, *, vector_store_id: Optional[str], vector_store_name: str) -> Optional[str]:
    if vector_store_id:
        return vector_store_id

    try:
        page = client.vector_stores.list(limit=100)
    except Exception as exc:
        raise RuntimeError(f"Failed to list vector stores: {exc!r}") from exc

    for vs in getattr(page, "data", []) or []:
        if (getattr(vs, "name", "") or "") == vector_store_name:
            return getattr(vs, "id", None)

    return None


def _iter_vector_store_files(client: OpenAI, *, vector_store_id: str, limit: int = 100):
    after = None
    while True:
        kwargs = {"vector_store_id": vector_store_id, "limit": limit}
        if after:
            kwargs["after"] = after

        page = client.vector_stores.files.list(**kwargs)
        items = getattr(page, "data", []) or []
        if not items:
            return

        for item in items:
            yield item

        has_more = bool(getattr(page, "has_more", False))
        if not has_more:
            return

        after = getattr(items[-1], "id", None)
        if not after:
            return


def clear_vector_store(
    *,
    vector_store_id: Optional[str],
    vector_store_name: str,
    delete_store: bool,
    delete_uploaded_files: bool,
    dry_run: bool,
) -> int:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)

    vs_id = _find_vector_store_id(
        client,
        vector_store_id=vector_store_id,
        vector_store_name=vector_store_name,
    )

    if not vs_id:
        print(f"Vector store not found (name={vector_store_name!r}). Nothing to clear.")
        return 0

    print(f"Target vector store: {vector_store_name!r} | id={vs_id}")

    if delete_store:
        if dry_run:
            print("DRY RUN: would delete the entire vector store.")
            return 0

        client.vector_stores.delete(vector_store_id=vs_id)
        print("Deleted vector store.")
        return 0

    removed = 0
    deleted_files = 0

    for vs_file in _iter_vector_store_files(client, vector_store_id=vs_id):
        # In list results, `file_id` is the uploaded file id; `id` may be the vector-store-file id.
        # The delete endpoint expects the uploaded file id.
        file_id = getattr(vs_file, "file_id", None) or getattr(vs_file, "id", None)
        if not file_id:
            continue

        if dry_run:
            print(f"DRY RUN: would remove file from vector store: {file_id}")
            removed += 1
            continue

        client.vector_stores.files.delete(vector_store_id=vs_id, file_id=file_id)
        removed += 1

        if delete_uploaded_files:
            try:
                client.files.delete(file_id)
                deleted_files += 1
            except Exception as exc:
                print(f"Warning: failed to delete uploaded file {file_id}: {exc!r}")

    print(f"Removed {removed} file(s) from vector store.")
    if delete_uploaded_files:
        print(f"Deleted {deleted_files} uploaded file(s) from your account.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clears an OpenAI Vector Store so you can re-ingest documents from scratch.\n\n"
            "Default behavior: remove ALL files from the vector store but keep the store itself."
        )
    )

    parser.add_argument(
        "--vector-store-name",
        default=DEFAULT_VECTOR_STORE_NAME,
        help=f"Vector store name to clear (default: {DEFAULT_VECTOR_STORE_NAME!r})",
    )
    parser.add_argument(
        "--vector-store-id",
        default=None,
        help="Vector store id to clear (overrides --vector-store-name)",
    )
    parser.add_argument(
        "--delete-store",
        action="store_true",
        help="Delete the entire vector store (instead of removing files).",
    )
    parser.add_argument(
        "--delete-uploaded-files",
        action="store_true",
        help=(
            "Also delete the uploaded File objects from your OpenAI account after removing them from the vector store. "
            "Only use this if those files are not needed elsewhere."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted, without making changes.",
    )

    args = parser.parse_args()

    return clear_vector_store(
        vector_store_id=args.vector_store_id,
        vector_store_name=args.vector_store_name,
        delete_store=args.delete_store,
        delete_uploaded_files=args.delete_uploaded_files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
