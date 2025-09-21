import os
from openai import OpenAI
from openai_vstore_toolkit import FileService, StoreService
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

sm = StoreService(client=client)

list_store_id = sm._list_store_id()[0]


def add_file(file_path: str, store_id: str) -> str | None:
    try:
        fm = FileService(client=client, store_id=store_id)
        file_object = fm.create_file_object(file_path=file_path)
        file_id = fm.add(file_object=file_object)
        return file_id
    except Exception as e:
        logger.error(f"Failed to add file to store {store_id}: {e}")
