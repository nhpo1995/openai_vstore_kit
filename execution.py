# execution.py
# Simple terminal runner to exercise vector-store & chat flows.
# Usage examples:
#   python execution.py stores create --name "Docs"
#   python execution.py files upload --store-id vs_123 --path ./readme.pdf
#   python execution.py chat start --model gpt-4o-mini --input "Hi" [--vector-store-id vs_123]
#   python execution.py chat continue --model gpt-4o-mini --conversation-id conv_123 --input "Next?"
#   python execution.py chat cancel --response-id resp_123
#   python execution.py files list --store-id vs_123
#
# Requirements:
#   - OPENAI_API_KEY in environment.
#   - Your package layout:
#       openai_vstore_kit/managers/{store_manager.py,file_manager.py}
#       services/{conversation_service.py,response_service.py}
#
# Notes:
#   - This script does not change any library code; it only calls into your managers/services.
#   - All errors are raised up (not swallowed); run from terminal to see trace for debugging.

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

# managers (your existing code)
from openai_vstore_toolkit.rag_services.store_service
from openai_vstore_toolkit import FileService

# services (the thin wrappers we added)
from openai_vstore_toolkit import ConversationService
from openai_vstore_toolkit import ResponseService


def _init_openai_client() -> OpenAI:
    """Create an OpenAI client from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    # If you need to customize base_url or organization, add here.
    return OpenAI(api_key=api_key)


def _pp(obj: Any) -> None:
    """Pretty-print Python objects as JSON when possible."""
    try:
        if hasattr(obj, "model_dump"):
            print(json.dumps(obj.model_dump(), indent=2, ensure_ascii=False))
        elif isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, ensure_ascii=False))
        else:
            print(obj)
    except Exception:
        print(obj)


# ---------------------------
# Stores commands
# ---------------------------
def cmd_stores_create(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    sm = StoreService(client)
    store_id = sm.create(args.name)
    logger.info(f"Store ID: {store_id}")
    print(store_id)


def cmd_stores_get(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    sm = StoreService(client)
    info = sm.get(args.store_id)
    _pp(info)


def cmd_stores_list(_: argparse.Namespace) -> None:
    client = _init_openai_client()
    sm = StoreService(client)
    stores = sm.list_store()
    _pp(stores)


def cmd_stores_delete(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    sm = StoreService(client)
    ok = sm.delete(args.store_id)
    print(json.dumps({"deleted": ok}))


# ---------------------------
# Files commands
# ---------------------------
def cmd_files_upload(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    fm = FileService(client, store_id=args.store_id)
    file_obj = fm.create_file_object(file_path=args.path, purpose="assistants")
    vs_file_id = fm.add(
        file_object=file_obj,
        chunking_strategy=fm.custom_chunk_strategy(
            max_chunk_size=args.max_chunk_size, chunk_overlap=args.chunk_overlap
        ),
        attributes={"uploader": args.uploader} if args.uploader else {},
    )
    print(vs_file_id)


def cmd_files_list(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    fm = FileService(client, store_id=args.store_id)
    files = fm.list(limit=args.limit)
    _pp(files)


def cmd_files_delete(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    fm = FileService(client, store_id=args.store_id)
    ok = fm.delete(file_id=args.file_id)
    print(json.dumps({"deleted": ok}))


def cmd_files_update_attrs(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    fm = FileService(client, store_id=args.store_id)
    # Parse key=value pairs into a dict
    attrs: Dict[str, str] = {}
    for kv in args.attr or []:
        if "=" not in kv:
            raise ValueError(f"Invalid attribute '{kv}', expected key=value.")
        k, v = kv.split("=", 1)
        attrs[k.strip()] = v.strip()
    ok = fm.update_attributes(attribute=attrs, file_id=args.file_id)
    print(json.dumps({"updated": ok, "attributes": attrs}))


def cmd_files_semantic_retrieve(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    fm = FileService(client, store_id=args.store_id)
    res = fm.semantic_retrieve(query=args.query, model=args.model, top_k=args.top_k)
    _pp(res)


# ---------------------------
# Chat (Conversations + Responses)
# ---------------------------
def _build_response_kwargs(args: argparse.Namespace) -> Dict:
    kwargs: Dict[str, Any] = {
        "model": args.model,
        "input_text": args.input,
        "stream": args.stream,
    }
    if args.instructions:
        kwargs["instructions"] = args.instructions
    if args.metadata:
        # read metadata JSON from CLI
        try:
            kwargs["metadata"] = json.loads(args.metadata)
        except Exception:
            raise ValueError("metadata must be a valid JSON object string.")
    if args.vector_store_id:
        kwargs["vector_store_ids"] = [args.vector_store_id]
        kwargs["include"] = ["file_search_call.results"]
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
    return kwargs


def _consume_stream(stream_obj) -> None:
    """Consume a streamed response iterator and print content as it arrives."""
    try:
        for event in stream_obj:
            # Depending on SDK, this may yield chunks; print raw for debugging
            _pp(event)
    except Exception as e:
        logger.error(f"Error while streaming: {e}")
        raise


def cmd_chat_start(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    conv = ConversationService(client)
    resp = ResponseService(client)

    # 1) Create conversation
    conv_id = conv.create()

    # 2) Create first response
    kwargs = _build_response_kwargs(args)
    kwargs["conversation_id"] = conv_id
    created = resp.create(**kwargs)

    if args.stream:
        _consume_stream(created)
        print(json.dumps({"conversation_id": conv_id}))
    else:
        _pp(created)
        print(json.dumps({"conversation_id": conv_id}))


def cmd_chat_continue(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    resp = ResponseService(client)

    kwargs = _build_response_kwargs(args)
    kwargs["conversation_id"] = args.conversation_id
    created = resp.create(**kwargs)

    if args.stream:
        _consume_stream(created)
    else:
        _pp(created)


def cmd_chat_cancel(args: argparse.Namespace) -> None:
    client = _init_openai_client()
    resp = ResponseService(client)
    ok = resp.cancel(args.response_id)
    print(json.dumps({"cancelled": ok}))


# ---------------------------
# CLI wiring
# ---------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Terminal runner for vector-store and chat flows."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # stores
    p_stores = sub.add_parser("stores", help="Vector store operations")
    sub_stores = p_stores.add_subparsers(dest="op", required=True)

    ps_create = sub_stores.add_parser("create", help="Create a vector store")
    ps_create.add_argument("--name", required=True, help="Store name")
    ps_create.set_defaults(func=cmd_stores_create)

    ps_get = sub_stores.add_parser("get", help="Get a vector store")
    ps_get.add_argument("--store-id", required=True)
    ps_get.set_defaults(func=cmd_stores_get)

    ps_list = sub_stores.add_parser("list", help="List vector stores")
    ps_list.set_defaults(func=cmd_stores_list)

    ps_delete = sub_stores.add_parser("delete", help="Delete a vector store")
    ps_delete.add_argument("--store-id", required=True)
    ps_delete.set_defaults(func=cmd_stores_delete)

    # files
    p_files = sub.add_parser("files", help="Vector store file operations")
    sub_files = p_files.add_subparsers(dest="op", required=True)

    pf_upload = sub_files.add_parser("upload", help="Upload & attach a file to a store")
    pf_upload.add_argument("--store-id", required=True)
    pf_upload.add_argument("--path", required=True, help="Local path or URL")
    pf_upload.add_argument("--uploader", required=False, help="Attribute 'uploader'")
    pf_upload.add_argument("--max-chunk-size", type=int, default=800)
    pf_upload.add_argument("--chunk-overlap", type=int, default=400)
    pf_upload.set_defaults(func=cmd_files_upload)

    pf_list = sub_files.add_parser("list", help="List files of a store")
    pf_list.add_argument("--store-id", required=True)
    pf_list.add_argument("--limit", type=int, default=100)
    pf_list.set_defaults(func=cmd_files_list)

    pf_delete = sub_files.add_parser(
        "delete", help="Delete (detach) a file from a store"
    )
    pf_delete.add_argument("--store-id", required=True)
    pf_delete.add_argument("--file-id", required=True)
    pf_delete.set_defaults(func=cmd_files_delete)

    pf_update = sub_files.add_parser(
        "update-attrs", help="Update attributes of a vector store file"
    )
    pf_update.add_argument("--store-id", required=True)
    pf_update.add_argument("--file-id", required=True)
    pf_update.add_argument(
        "--attr",
        nargs="*",
        help="Attributes in key=value format (e.g., owner=jane lang=vi)",
    )
    pf_update.set_defaults(func=cmd_files_update_attrs)

    pf_semret = sub_files.add_parser(
        "semantic-retrieve", help="Run retrieval over a store (debug)"
    )
    pf_semret.add_argument("--store-id", required=True)
    pf_semret.add_argument("--query", required=True)
    pf_semret.add_argument("--model", default="gpt-4o-mini")
    pf_semret.add_argument("--top-k", type=int, default=10)
    pf_semret.set_defaults(func=cmd_files_semantic_retrieve)

    # chat
    p_chat = sub.add_parser("chat", help="Chat flows (Responses + Conversations)")
    sub_chat = p_chat.add_subparsers(dest="op", required=True)

    pc_start = sub_chat.add_parser(
        "start", help="Start a conversation and create the first response"
    )
    pc_start.add_argument("--model", required=True)
    pc_start.add_argument("--input", required=True)
    pc_start.add_argument("--vector-store-id", required=False)
    pc_start.add_argument("--top-k", type=int, required=False)
    pc_start.add_argument("--instructions", required=False)
    pc_start.add_argument(
        "--metadata", required=False, help='JSON string, e.g. \'{"app":"cli"}\''
    )
    pc_start.add_argument("--stream", action="store_true")
    pc_start.set_defaults(func=cmd_chat_start)

    pc_continue = sub_chat.add_parser(
        "continue", help="Create a response on an existing conversation"
    )
    pc_continue.add_argument("--conversation-id", required=True)
    pc_continue.add_argument("--model", required=True)
    pc_continue.add_argument("--input", required=True)
    pc_continue.add_argument("--vector-store-id", required=False)
    pc_continue.add_argument("--top-k", type=int, required=False)
    pc_continue.add_argument("--instructions", required=False)
    pc_continue.add_argument(
        "--metadata", required=False, help='JSON string, e.g. \'{"app":"cli"}\''
    )
    pc_continue.add_argument("--stream", action="store_true")
    pc_continue.set_defaults(func=cmd_chat_continue)

    pc_cancel = sub_chat.add_parser("cancel", help="Cancel a running response")
    pc_cancel.add_argument("--response-id", required=True)
    pc_cancel.set_defaults(func=cmd_chat_cancel)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
