import os
from openai import OpenAI
from dotenv import load_dotenv
from openai_vstore_kit import StoreManager, FileManager
from pprint import pprint


load_dotenv()
client = OpenAI()

store_manager = StoreManager(client=client)
# store_id = store_manager.get_or_create(store_name="My Store")
store_id = store_manager.find_id_by_name("My Store")
file_manager = FileManager(client=client, store_id=store_id)


# file_object = file_manager.create_file_object(file_path="https://cdn.openai.com/API/docs/deep_research_blog.pdf")
# file_id = file_manager.add(file_object=file_object)
query = ""
while query != "exit":
    query = input("ask me something!\n>: ")
    response = file_manager.semantic_retrieve(query=query)
    print(response)
# file_id = file_manager.find_id_by_name("deep_research_blog.pdf")
# print(file_id)

# file_manager.update_attributes(
#     attribute={"role": "manager"}, file_id="file-S5aGLJVhvNZAW3XcSJKX1X"
# )
# print(f"fileid: {file_id}")
# list_file = file_manager.list()
# print(list_file)
