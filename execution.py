import os
from openai import OpenAI
from dotenv import load_dotenv
from openai_vstore_kit import StoreManager, FileManager

load_dotenv()
client = OpenAI()

store_manager = StoreManager(client=client)
store_id = store_manager.create(store_name="My Store")
if store_id:
    file_manager = FileManager(client=client, store_id=store_id)
    file_response = file_manager.create_file_object(file_path="https://cdn.openai.com/API/docs/deep_research_blog.pdf")
    file_manager.add(file_object=file_response, chunking_strategy=)
