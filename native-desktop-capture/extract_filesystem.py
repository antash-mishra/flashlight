import os
import stat
from pathlib import Path
from datetime import datetime
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

ef = BGEM3EmbeddingFunction(use_fp16=True,device="cuda")
dense_dim = ef.dim["dense"]

# Connect to Milvus given URI
connections.connect(uri="./filesystem.db")

# Specify the data schema for the new Collection
fields = [
    # Use auto generated id as primary key
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    # For now, use only dense vectors to avoid sparse vector format issues
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields)

# Create collection (drop the old one if exists)
collection_name = "filesystem_metadata"
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
col = Collection(collection_name, schema, consistency_level="Bounded")

# To make vector search efficient, we need to create indices for the vector fields
dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

def extract_file_metadata(file_path):
    """
    Extract filesystem metadata and filename from a file path.

    Args:
        file_path (str or Path): Path to the file

    Returns:
        dict: Dictionary containing file metadata
    """
    try:
        path = Path(file_path)
        file_stat = path.stat()

        metadata = {
            'filename': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'parent_dir': path.parent.name,
            'full_path': str(path.absolute()),
            'created_date': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'is_file': path.is_file(),
            'is_dir': path.is_dir()
        }

        return metadata

    except (OSError, PermissionError) as e:
        return {
            'filename': Path(file_path).name,
            'error': str(e),
            'full_path': str(file_path)
        }

def scan_home_directory():
    """
    Recursively scan the home directory for filesystem metadata.

    Yields:
        dict: File metadata for each file found
    """

    home_dir = Path.home()
    try:
        for root, dirs, files in os.walk(home_dir):
            # process files
            for file in files:
                file_path = Path(root)/file
                try:
                    metadata = extract_file_metadata(file_path)
                    yield metadata
                except (OSError, PermissionError) as e:
                    yield {
                        'filename': file,
                        'full_path': str(file_path),
                        'error': f"Access denied: {str(e)}",
                    }
                    continue
            # skip hidden directories and common large files
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]

    except PermissionError as e:
        print(f"Permission denied access to home directory: {home_dir}")

def dense_search(col, query_dense_embedding, limit=10):
    """Search using only dense vectors."""
    import numpy as np
    # Convert query embedding to numpy array with float32 dtype
    query_dense = np.array(query_dense_embedding, dtype=np.float32)
    
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=10):
    """Search using only sparse vectors."""
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    """Search using both dense and sparse vectors with weighted ranking."""
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

def ask_question(query_text, search_type="dense", limit=10):
    """
    Ask a question and search through filesystem metadata.
    
    Args:
        query_text (str): The search query
        search_type (str): Type of search - currently only "dense" is supported
        limit (int): Number of results to return
    
    Returns:
        list: Search results
    """
    # Generate embeddings for the query
    try:
        query_embeddings = ef([query_text])
    except Exception as e:
        print(f"Error generating query embeddings: {e}")
        return []
    
    # Perform search using dense vectors only
    return dense_search(col, query_embeddings["dense"][0], limit)

def process_all_files():
    """Process all files in home directory and insert into collection."""
    file_count = 0
    batch_size = 50
    batch_texts = []
    
    for file_metadata in scan_home_directory():
        if 'error' not in file_metadata:
            # Create text representation for search
            text_content = f"File: {file_metadata['filename']} in {file_metadata['parent_dir']} directory. Path: {file_metadata['full_path']}"
            
            batch_texts.append(text_content)
            
            if len(batch_texts) >= batch_size:
                # Generate embeddings for the batch
                try:
                    embeddings = ef(batch_texts)
                    
                    # Insert each record individually
                    for i, text in enumerate(batch_texts):
                        # Convert dense vector to numpy array with float32 dtype
                        import numpy as np
                        dense_vec = np.array(embeddings["dense"][i], dtype=np.float32)
                        
                        record_data = {
                            "text": text,
                            "dense_vector": dense_vec
                        }
                        col.insert([record_data])
                    
                    print(f"Inserted batch of {len(batch_texts)} files with embeddings")
                    batch_texts = []
                    
                except Exception as e:
                    print(f"Error generating embeddings for batch: {e}")
                    print(f"Debug - sparse vector type: {type(embeddings['sparse'][0])}")
                    print(f"Debug - sparse vector content: {embeddings['sparse'][0]}")
                    batch_texts = []
                    continue
            
            file_count += 1
        else:
            print(f"Skipped: {file_metadata.get('filename', 'unknown')} - {file_metadata['error']}")
    
    # Insert remaining batch
    if batch_texts:
        try:
            embeddings = ef(batch_texts)
            
            # Insert each record individually
            for i, text in enumerate(batch_texts):
                # Convert dense vector to numpy array with float32 dtype
                import numpy as np
                dense_vec = np.array(embeddings["dense"][i], dtype=np.float32)
                
                record_data = {
                    "text": text,
                    "dense_vector": dense_vec
                }
                col.insert([record_data])
            
            print(f"Inserted final batch of {len(batch_texts)} files with embeddings")
        except Exception as e:
            print(f"Error generating embeddings for final batch: {e}")
    
    print(f"Total Files Processed: {file_count}")

# Example usage
if __name__ == "__main__":
    # Process files
    process_all_files()
    
    # Example search
    query = "data-capture"
    results = ask_question(query, search_type="dense", limit=5)
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")