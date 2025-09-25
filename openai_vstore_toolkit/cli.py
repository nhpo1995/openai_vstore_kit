"""
CLI for managing Vector Stores and Vector Store Files using your existing package
(`openai_vstore_toolkit`). This CLI only wraps the classes/methods already present
in your codebase; no new services or business logic are introduced.
"""

from __future__ import annotations

import os
from typing import List

import typer
from rich import print as rprint
from rich.table import Table
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

from .rag_services import StoreService
from .rag_services import FileService
from .rag_services import ResponseRAGService
from .rag_services import ConversationService

from openai_vstore_toolkit.utils import Helper

# Top-level app
app = typer.Typer(
    help="CLI to manage Vector Stores & Vector Store Files (based on your repo code)",
    context_settings={"max_content_width": 120},
)


# ----------------------------
# Helpers
# ----------------------------
def get_client() -> OpenAI:
    """Build an OpenAI client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _print_stores(stores: List[dict]):
    """Pretty-print a list of vector stores."""
    table = Table(title="Vector Stores")
    table.add_column("id", style="cyan")
    table.add_column("name", style="magenta")
    table.add_column("created_at", style="green")
    for s in stores:
        table.add_row(
            str(s.get("id", "")), str(s.get("name", "")), str(s.get("created_at", ""))
        )
    rprint(table)


def _print_files(files: List[dict]):
    """Pretty-print a list of vector store files."""
    table = Table(title="Vector Store Files")
    table.add_column("id", style="cyan")
    table.add_column("file_id", style="magenta")
    table.add_column("status", style="green")
    table.add_column("attributes", style="white")
    for f in files:
        table.add_row(
            str(f.get("id", "")),
            str(f.get("file_id", "")),
            str(f.get("status", "")),
            str(f.get("attributes", "")),
        )
    rprint(table)


def _parse_kv(kvs: List[str]) -> dict:
    """Parse repeated '--attr key=value' flags into a dictionary."""
    out = {}
    for kv in kvs or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ----------------------------
# store group
# ----------------------------
store_app = typer.Typer(
    help="Manage Vector Stores - Typing <vstore store --help> for the detail cli",
    context_settings={"max_content_width": 120},
)
app.add_typer(store_app, name="store")


@store_app.command(
    "get-or-create",
    help="vstore store get-or-create 'Your-Store-Name' - Create new if store name doesn't exist or get store_id",
)
def store_get_or_create(name: str = typer.Argument(..., help="Vector store name")):
    """Call StoreService.get_or_create(name)."""
    client = get_client()
    svc = StoreService(client)
    store_id = svc.get_or_create(name)
    rprint(f"[bold green]store_id:[/bold green] {store_id}")


@store_app.command(
    "get-id-by-name",
    help="vstore store get-id-by-name 'Your-Store-Name' - Get the store_id by its name",
)
def store_get_id_by_name(
    store_name: str = typer.Argument(..., help="Vector store name")
):
    """Call StoreService.get_or_create(name)."""
    client = get_client()
    svc = StoreService(client)
    store_id = svc.find_id_by_name(store_name=store_name)
    rprint(f"[bold green]store_id:[/bold green] {store_id}")


@store_app.command(
    "get",
    help="vstore store get <store_id>",
)
def store_get(store_id: str = typer.Argument(...)):
    """Call StoreService.get(store_id)."""
    client = get_client()
    svc = StoreService(client)
    data = svc.get(store_id)
    rprint(data)


@store_app.command(
    "list",
    help="vstore store list",
)
def store_list():
    """Call StoreService.list_store()."""
    client = get_client()
    svc = StoreService(client)
    stores = svc.list_store()
    _print_stores(stores)


@store_app.command(
    "delete",
    help="vstore store delete <store_id>",
)
def store_delete(store_id: str = typer.Argument(...)):
    """Call StoreService.delete(store_id)."""
    client = get_client()
    svc = StoreService(client)
    ok = svc.delete(store_id)
    rprint("[green]OK[/green]" if ok else "[yellow]Failed[/yellow]")


# ----------------------------
# file group
# ----------------------------
file_app = typer.Typer(
    help="Manage Files within a Vector Store - Typing <vstore file --help> for the detail cli",
    context_settings={"max_content_width": 120},
)
app.add_typer(file_app, name="file")


@file_app.command(
    "list",
    help="vstore file list <store_id> --limit 100 - List all file file in vector store, the --limit is optional",
)
def file_list(
    store_id: str = typer.Argument(..., help="Vector store ID"),
    limit: int = typer.Option(100, help="Max items per page (service may paginate)"),
):
    """Call FileService(store_id).list(limit)."""
    client = get_client()
    fsvc = FileService(client, store_id)
    files = fsvc.list(limit=limit)
    _print_files(files)


@file_app.command(
    "find-id-by-name",
    help="vstore file find-id-by-name <store_id> handbook.pdf",
)
def file_find_id_by_name(
    store_id: str = typer.Argument(...),
    file_name: str = typer.Argument(
        ..., help="File name (matched via attributes.file_name)"
    ),
):
    """Call FileService(store_id).find_id_by_name(file_name)."""
    client = get_client()
    fsvc = FileService(client, store_id)
    fid = fsvc.find_id_by_name(file_name)
    rprint(fid or "")


@file_app.command(
    "upload",
    help='vstore file upload <store_id> "./docs/handbook.pdf" or <url> --attr[optional] source=internal --attr[optional] lang=vi --max-chunk-size[optional] 800 --chunk-overlap[optional] 400',
)
def file_upload(
    store_id: str = typer.Argument(...),
    path_or_url: str = typer.Argument(..., help="Local file path or URL"),
    use_url: bool = typer.Option(False, "--url", help="Set if path_or_url is a URL"),
    max_chunk_size: int = typer.Option(
        800, help="custom_chunk_strategy.max_chunk_size"
    ),
    chunk_overlap: int = typer.Option(400, help="custom_chunk_strategy.chunk_overlap"),
    attr: List[str] = typer.Option(
        None,
        "--attr",
        help="Attach attributes, e.g. --attr source=policy --attr lang=vi",
    ),
):
    """
    Steps:
    1) Create FileObject via FileService.create_file_object(path_or_url).
        The underlying implementation handles local path/URL per your code.
    2) Create chunking strategy via FileService.custom_chunk_strategy(...).
    3) Attach the file to the vector store via FileService.add(...).
    Returns the vector_store_file_id.
    """
    client = get_client()
    fsvc = FileService(client, store_id)

    file_obj = fsvc.create_file_object(path_or_url)
    strategy = fsvc.custom_chunk_strategy(
        max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap
    )
    attributes = _parse_kv(attr or [])
    vs_file_id = fsvc.add(
        file_object=file_obj, chunking_strategy=strategy, attributes=attributes
    )
    rprint(vs_file_id or "")


@file_app.command(
    "update-attrs",
    help="vstore file update-attrs <store_id> <vector_store_file_id> --attr tag=hr --attr owner=data-team",
)
def file_update_attrs(
    store_id: str = typer.Argument(...),
    vector_store_file_id: str = typer.Argument(
        ..., help="Vector store file ID (not the raw file_id)"
    ),
    attr: List[str] = typer.Option(
        None, "--attr", help="--attr key=value (repeatable)"
    ),
):
    """Call FileService.update_attributes(attribute=dict, file_id=vector_store_file_id)."""
    client = get_client()
    fsvc = FileService(client, store_id)
    attributes = _parse_kv(attr or [])
    ok = fsvc.update_attributes(attribute=attributes, file_id=vector_store_file_id)
    rprint("[green]OK[/green]" if ok else "[yellow]Failed[/yellow]")


@file_app.command(
    "delete",
    help="vstore file delete <store_id> <vector_store_file_id>",
)
def file_delete(
    store_id: str = typer.Argument(...),
    vector_store_file_id: str = typer.Argument(..., help="Vector store file ID"),
):
    """Call FileService.delete(vector_store_file_id)."""
    client = get_client()
    fsvc = FileService(client, store_id)
    ok = fsvc.delete(vector_store_file_id)
    rprint("[green]OK[/green]" if ok else "[yellow]Failed[/yellow]")


@file_app.command(
    "semantic-retrieve",
    help='vstore file semantic-retrieve <store_id> "Summarize vacation policy" --model[optional] gpt-4o-mini --top-k[optional] 10',
)
def file_semantic_retrieve(
    store_id: str = typer.Argument(...),
    query: str = typer.Argument(..., help="User query / search text"),
    model: str = typer.Option(
        "gpt-4o-mini", help="Default model as used by your FileService"
    ),
    top_k: int = typer.Option(10, help="Limit for file_search results"),
):
    """Call FileService.semantic_retrieve(query, model, top_k) and print the formatted output."""
    client = get_client()
    fsvc = FileService(client, store_id)
    out = fsvc.semantic_retrieve(query=query, model=model, top_k=top_k)
    rprint(out)


@file_app.command(
    "get_detail",
    help='vstore file get_detail "./docs/handbook.pdf" or <url>',
)
def get_file_detail(
    file_path: str = typer.Argument(...),
):
    """Call FileService.semantic_retrieve(query, model, top_k) and print the formatted output."""
    helper = Helper()
    file_detail = helper.get_file_detail([file_path])
    rprint(file_detail[0] if file_detail else None)


if __name__ == "__main__":
    app()
