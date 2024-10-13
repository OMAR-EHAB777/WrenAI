
![Provider Activity Diagram](/docs/activity-diagram-provider.png)
# Providers Module

The `providers` folder in this project encapsulates all the necessary components and configurations for interacting with various services such as document stores, language models, and embedding providers. Each provider offers an interface for integrating third-party services like Qdrant, OpenAI, and Ollama into the larger system.

## Providers Folder Overview

This module handles the dynamic registration of different services required by the core pipeline. Here’s how the flow works:

1. **Load Providers** – Initiates the loading of all submodules under the `providers` folder.
2. **Import Modules** – Dynamically imports all necessary submodules.
3. **Get Provider** – Depending on the type of service requested, the provider is fetched from the available options:
    - **Document Store Providers:** 
      - Registers services like `AsyncQdrantDocumentStore` and its retrievers.
    - **Embedder Providers:** 
      - Registers text and document embedders from sources like OpenAI, Azure, and Ollama.
    - **LLM Providers:** 
      - Handles services such as `AzureOpenAI`, `Ollama`, and `OpenAI` for text generation.
4. **Register Providers** – Once fetched, each provider is registered for use within the system.
5. **Process Requests** – The registered providers are now available to serve requests from the pipelines.
6. **Provide Data to Core Pipeline** – Data from the providers integrates into the core pipeline for further processing.

## Structure Overview

- **documentstore/**: Contains modules to interact with various document stores, such as Qdrant, asynchronously.
- **embedder/**: Contains text and document embedders for Azure OpenAI, OpenAI, and Ollama, which are used to convert text or documents into embeddings.
- **llm/**: Manages large language models (LLM) interactions, including generation, prompt handling, and streaming. Includes providers for Azure OpenAI, OpenAI, and Ollama.
- **engine/**: Provides custom query execution engines like Wren, Wren UI, Wren Ibis, and Wren Engine, enabling SQL execution with or without dry runs.
- **loader.py**: Handles dynamic loading and registration of providers, offering utilities for fetching and initializing specific provider instances.

---

## Provider Modules

### Document Store Providers

- **QdrantDocumentStore**: An asynchronous document store based on Qdrant for embedding and querying documents. Supports operations like filtering, indexing, querying by embeddings, and batched document writes.
- **AsyncQdrantEmbeddingRetriever**: Handles asynchronous retrieval of documents using embeddings from the Qdrant store.

### Embedder Providers

- **Azure OpenAI Embedder**: Provides text and document embeddings using Azure's OpenAI service.
- **OpenAI Embedder**: Embeds text and documents via OpenAI's API, supporting models like `text-embedding-ada-002`.
- **Ollama Embedder**: Interfaces with the Ollama service for embedding text and documents asynchronously.

### LLM Providers

- **Azure OpenAI LLM**: Handles text generation tasks using Azure's OpenAI models, such as `gpt-4-turbo`. It supports streaming and non-streaming completions.
- **Ollama LLM**: Uses Ollama models like `gemma2:9b` for language generation tasks.
- **OpenAI LLM**: Manages OpenAI’s GPT-4 models for text generation with configurations for temperature, tokens, and responses.

### Engine Providers

- **Wren Engines**: Provides SQL query execution engines including:
  - **Wren UI**: Executes SQL queries and handles dry runs.
  - **Wren Ibis**: Executes SQL queries through the Ibis engine with dry run support.
  - **Wren Engine**: Executes queries for the MDL pipeline using provided manifests.

### Utilities

- **loader.py**: Manages the dynamic loading of providers and registering them to the system. Includes functions like:
  - `import_mods`: Dynamically imports all provider submodules.
  - `provider`: A decorator to register and manage provider instances.
  - `get_provider`: Fetches registered providers by name.

---

## How to Use

1. **Registering Providers**: 
   Use the `provider()` decorator to register a new provider, allowing easy retrieval via `get_provider()`. 
   
   ```python
   @provider("custom_provider")
   class CustomProvider:
       # Provider implementation
   ```

2. **Embedding Models**: 
   Providers for embedding text and documents are initialized using environment variables or directly passing the necessary configuration. The `get_text_embedder()` and `get_document_embedder()` methods can then be used to obtain the embedder.

3. **Executing SQL with Wren Engines**:
   The `WrenUI`, `WrenIbis`, and `WrenEngine` classes allow for running SQL queries asynchronously, including dry runs, through various endpoints.

4. **Pulling Ollama Models**: 
   The `pull_ollama_model()` function allows for pulling models from an Ollama instance if they do not already exist locally.

