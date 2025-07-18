#!/usr/bin/env python3
"""
Test script to verify FAISS functionality
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def test_faiss():
    """Test FAISS functionality"""
    print("Testing FAISS functionality...")
    
    try:
        # Initialize model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Create FAISS index
        dimension = 768  # all-mpnet-base-v2 dimension
        index = faiss.IndexFlatIP(dimension)
        
        # Test data
        texts = [
            "Google search engine",
            "GitHub code repository", 
            "Stack Overflow programming questions"
        ]
        
        # Create embeddings
        embeddings = model.encode(texts)
        print(f"✅ Created embeddings: {embeddings.shape}")
        
        # Add to FAISS index
        index.add(embeddings.astype('float32'))
        print(f"✅ Added to FAISS index: {index.ntotal} vectors")
        
        # Test search
        query = "github"
        query_embedding = model.encode([query])[0]
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        distances, indices = index.search(query_vector, 3)
        print(f"✅ Search successful: {distances[0]}")
        print(f"✅ Indices: {indices[0]}")
        
        print("✅ FAISS test passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        return False

if __name__ == "__main__":
    test_faiss() 