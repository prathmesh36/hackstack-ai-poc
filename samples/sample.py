from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore


document_dir = "./document/docs1.txt"

with open(document_dir, 'r', encoding='utf-8') as file:
    file_content = file.read()

docstore = InMemoryDocumentStore()
#docstore.write_documents([Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")])
docstore.write_documents([Document(content=file_content)])

generator = HuggingFaceLocalGenerator(model="google/flan-t5-large",
                                      task="text2text-generation",
                                      generation_kwargs={
                                        "max_new_tokens": 100,
                                        "temperature": 0.9,
                                        })

query = "what was the final result of the Indian elections??"

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""
pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

res=pipe.run({
    "prompt_builder": {
        "query": query
    },
    "retriever": {
        "query": query
    }
})

print(res)