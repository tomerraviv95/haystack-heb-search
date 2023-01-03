import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# In-Memory Document Store
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(embedding_dim=512)

from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http


# Let's first get some documents that we want to query
# Here: 517 Wikipedia articles for Game of Thrones
# doc_dir = "data/tutorial3"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
doc_dir = "1k"
# convert files to dicts containing documents that can be indexed to our datastore
# You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
# It must take a str as input, and return a str.
docs = convert_files_to_docs(dir_path=doc_dir,  split_paragraphs=True) # clean_func=clean_wiki_text,

# We now have a list of dictionaries that we can write to our document store.
# If your texts come from a different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries yourself.
# The default format here is: {"name": "", "content": ""}

# Let's have a look at the first 3 entries:
print(docs[:3])

# Now, let's write the docs to our DB.
document_store.write_documents(docs)

from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
    model_format="sentence_transformers",
)
# Important:
# Now that we initialized the Retriever, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on the corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it to the existing document embeddings, which is very fast.
document_store.update_embeddings(retriever)

from haystack.nodes import TransformersReader,FARMReader
my_model = "timpal0l/mdeberta-v3-base-squad2"
reader = FARMReader(model_name_or_path=my_model, use_gpu=True, return_no_answer=True)

from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)

# You can configure how many candidates the reader and retriever shall return
# The higher top_k for retriever, the better (but also the slower) your answers.
prediction = pipe.run(
    query="האם החקירה כנגד בנק לאומי נגמרה?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

from haystack.utils import print_answers

print_answers(prediction, details="all")