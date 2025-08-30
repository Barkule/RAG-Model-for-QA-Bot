# RAG Model for QA Bot with Pinecone

A Retrieval Augmented Generation (RAG) model for a business QA bot that leverages OpenAI API and Pinecone vector database.

## Features

- **Document Processing**: Load and process business documents
- **Vector Embeddings**: Convert text to embeddings using OpenAI
- **Vector Storage**: Store and retrieve embeddings using Pinecone
- **Question Answering**: Natural language querying with context-aware responses
- **Business Focus**: Specialized for technology and AI/ML business content

## Installation

```bash
pip install langchain openai pinecone-client chromadb tiktoken
```

## Setup

1. Set up your environment variables:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"
os.environ["PINECONE_ENVIRONMENT"] = "your-pinecone-environment"
```

2. Initialize Pinecone:

```python
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
```

## Usage

### Basic Document Processing

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load and process documents
documents = ["Your business document text here..."]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents([Document(page_content=doc) for doc in documents])

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query the system
response = qa({"query": "What is the main theme of the business?"})
print(response["result"])
```

### With Pinecone Integration

```python
# Initialize Pinecone index
index_name = "business-qa-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine"
    )

index = pc.Index(index_name)

# Store embeddings in Pinecone
vectorstore = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Query with Pinecone
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
```

## Example Business Document

The system is pre-loaded with sample business content:
- "XYZ Inc. is a technology company that specializes in developing artificial intelligence and machine learning solutions. Our team of experts has years of experience in the field and is dedicated to delivering high-quality products and services to our clients."

## Query Examples

```python
# Example queries
queries = [
    "What does XYZ Inc. specialize in?",
    "What kind of solutions does the company develop?",
    "What is the expertise of the team?",
    "What is the company's commitment to clients?"
]

for query in queries:
    response = qa({"query": query})
    print(f"Q: {query}")
    print(f"A: {response['result']}\n")
```

## Configuration

### Text Splitting Options
```python
text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # Size of each text chunk
    chunk_overlap=200,    # Overlap between chunks
    separator="\n"        # Separator for splitting
)
```

### Retrieval Options
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for maximum marginal relevance
    search_kwargs={"k": 3}     # Number of documents to retrieve
)
```

## Dependencies

- **langchain**: Framework for building LLM applications
- **openai**: OpenAI API client for embeddings and completions
- **pinecone-client**: Pinecone vector database client
- **chromadb**: Local vector database for development
- **tiktoken**: Token counting for OpenAI models

## Business Applications

This RAG model can be used for:
- Customer support automation
- Employee knowledge base
- Technical documentation querying
- Business intelligence questioning
- Product information retrieval

## Customization

To customize for your specific business:

1. Replace the sample documents with your business content
2. Adjust the chunk size and overlap based on your document structure
3. Modify the retrieval parameters for optimal performance
4. Add domain-specific preprocessing if needed

## Performance Tips

- Use appropriate chunk sizes for your content type
- Experiment with different search types (similarity vs MMR)
- Consider adding metadata filtering for complex document sets
- Monitor token usage with tiktoken for cost management

## Limitations

- Performance depends on document quality and structure
- Large documents may require optimized chunking strategies
- Complex queries might need additional context handling
- API costs should be monitored for production use

## Support

For issues and questions:
1. Check the LangChain documentation
2. Review Pinecone API documentation
3. Ensure proper API key configuration
4. Verify network connectivity to external services

## License

This project is intended for educational and business use. Please ensure compliance with OpenAI and Pinecone terms of service.
