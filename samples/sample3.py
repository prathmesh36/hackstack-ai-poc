from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import convert_files_to_dicts

document_store = InMemoryDocumentStore()

# Assuming you have saved the model and tokenizer locally
model_dir = "./model"

reader = FARMReader(model_name_or_path=model_dir)
# Alternatively, use TransformersReader
# reader = TransformersReader(model_name_or_path=model_dir, tokenizer=model_dir)

# Setup the pipeline
pipeline = ExtractiveQAPipeline(reader, document_store)

# Path to your local documents
document_dir = ".document/docs1.txt"

# Convert and index documents
docs = convert_files_to_dicts(dir_path=document_dir, split_paragraphs=True)
document_store.write_documents(docs)


# Example query
prediction = pipeline.run(query="What is Haystack?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
print(prediction)

