import os
from openai import OpenAI
from dotenv import load_dotenv
from openai_vstore_kit import StoreManager, FileManager
from pprint import pprint


load_dotenv()
client = OpenAI()

store_manager = StoreManager(client=client)
store_id = store_manager.get_or_create(store_name="My Store")
# if store_id:
#     file_manager = FileManager(client=client, store_id=store_id)
#     file_response = file_manager.create_file_object(
#         file_path="https://cdn.openai.com/API/docs/deep_research_blog.pdf"
#     )
#     chunking_strategy = file_manager.custom_chunk_strategy(
#         max_chunk_size=500, chunk_overlap=128
#     )
#     file_manager.add(
#         file_object=file_response,
#         chunking_strategy=chunking_strategy,
#     )
# file_manager = FileManager(client=client, store_id=store_id)
# list_file = file_manager.list()
# update_file = list_file

print(store_manager._list_store_id())
