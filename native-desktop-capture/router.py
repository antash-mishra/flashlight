from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import re
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import threading
import numpy as np
import faiss
import json
import hashlib
import pickle
import os
from datetime import datetime
from rank_bm25 import BM25Okapi

class Router:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')

        # FAISS index for vector storage
        self.dimension = 768  # all-mpnet-base-v2 dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        
        # Tab data storage
        self.current_tabs = {}
        self.tab_embeddings = {}
        self.tab_index_map = {}  # Maps tab_id to FAISS index position
        self.last_tab_index_hash = None  # Hash of tabIndex for change detection
        
        # Memory persistence
        self.memory_file = 'tab_memory.pkl'
        self.index_file = 'faiss_index.bin'
        
        # Load existing data from memory
        self.load_from_memory()
        
        # Start Flask server for browser extension communication
        self.start_server()

    def save_to_memory(self):
        """Save tab data and FAISS index to disk"""
        try:
            # Save tab embeddings and metadata
            memory_data = {
                'tab_embeddings': self.tab_embeddings,
                'tab_index_map': self.tab_index_map,
                'last_tab_index_hash': self.last_tab_index_hash,
                'current_tabs': self.current_tabs,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            print(f"✅ Saved {len(self.tab_embeddings)} tabs to memory")
            
        except Exception as e:
            print(f"Error saving to memory: {e}")
    
    def load_from_memory(self):
        """Load tab data and FAISS index from disk"""
        try:
            if os.path.exists(self.memory_file) and os.path.exists(self.index_file):
                # Load tab embeddings and metadata
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                
                self.tab_embeddings = memory_data.get('tab_embeddings', {})
                self.tab_index_map = memory_data.get('tab_index_map', {})
                self.last_tab_index_hash = memory_data.get('last_tab_index_hash')
                self.current_tabs = memory_data.get('current_tabs', {})
                
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                print(f"✅ Loaded {len(self.tab_embeddings)} tabs from memory")
                print(f"✅ FAISS index loaded with {self.index.ntotal} vectors")
                
            else:
                print("No existing memory found, starting fresh")
                
        except Exception as e:
            print(f"Error loading from memory: {e}")
            # Start fresh if loading fails
            self.tab_embeddings = {}
            self.tab_index_map = {}
            self.last_tab_index_hash = None
            self.current_tabs = {}
    
    def clear_memory(self):
        """Clear all stored memory"""
        try:
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            
            self.tab_embeddings = {}
            self.tab_index_map = {}
            self.last_tab_index_hash = None
            self.current_tabs = {}
            self.index = faiss.IndexFlatIP(self.dimension)
            
            print("✅ Memory cleared")
            
        except Exception as e:
            print(f"Error clearing memory: {e}")
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        try:
            memory_size = os.path.getsize(self.memory_file) if os.path.exists(self.memory_file) else 0
            index_size = os.path.getsize(self.index_file) if os.path.exists(self.index_file) else 0
            
            return {
                'tab_count': len(self.tab_embeddings),
                'memory_size_mb': memory_size / (1024 * 1024),
                'index_size_mb': index_size / (1024 * 1024),
                'total_size_mb': (memory_size + index_size) / (1024 * 1024)
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}

    def start_server(self):
        """Start Flask server in background thread"""
        self.app = Flask(__name__)
        self.app.add_url_rule('/update-tabs', 'update_tabs', self.update_tabs, methods=['POST'])
        self.app.add_url_rule('/search', 'search_tabs', self.search_tabs_api, methods=['POST'])
        self.app.add_url_rule('/open-tab', 'open_tab', self.open_tab_api, methods=['POST'])
        self.app.add_url_rule('/memory/stats', 'memory_stats', self.memory_stats_api, methods=['GET'])
        self.app.add_url_rule('/memory/clear', 'memory_clear', self.memory_clear_api, methods=['POST'])
        
        # Start server in daemon thread
        server_thread = threading.Thread(target=self.run_server, daemon=True)
        server_thread.start()
        
        print("Server started on http://localhost:8080")

    def run_server(self):
        """Run Flask server"""
        self.app.run(host='localhost', port=8080, debug=False)

    def update_tabs(self):
        """Receive tab data from browser extension"""
        try:
            data = request.get_json()
            if data is None:
                raise ValueError("No JSON data received in request")
            
            self.current_tabs = data.get('openTabs', [])
            tab_index = data.get('tabIndex', {})
            
            # Check if tabs have changed using tabIndex hash
            current_hash = self._hash_tab_index(tab_index)
            
            if current_hash != self.last_tab_index_hash:
                print(f"Tab changes detected. Updating FAISS index...")
                self._update_faiss_index(self.current_tabs, tab_index)
                self.last_tab_index_hash = current_hash
            else:
                print("No tab changes detected. Skipping re-indexing.")
            
            return jsonify({'status': 'success', 'received': len(self.current_tabs)})
        except Exception as e:
            print(f"Error updating tabs: {e}")
            return jsonify({'status': 'error', 'message': str(e)})
    
    def _hash_tab_index(self, tab_index):
        """Create hash of tabIndex for change detection"""
        return hashlib.md5(json.dumps(tab_index, sort_keys=True).encode()).hexdigest()

    def _update_faiss_index(self, tabs: List[Dict], tab_index: Any) -> None:
        """
        Update FAISS index with optimized tab embeddings for better search quality
        Ensures index and tab_embeddings are always in sync, even after retriggering.
        """
        try:
            print("Processing embeddings and updating FAISS index...")
            
            # Create a new index instead of clearing the old one
            new_index = faiss.IndexFlatIP(self.dimension)
            new_tab_embeddings = {}
            new_tab_index_map = {}

            if not tabs:
                print("No tabs to process")
                return

            searchable_texts = []
            tab_data_list = []
            for tab in tabs:
                optimized_text = self._create_optimized_embedding_text(tab)
                searchable_texts.append(optimized_text)
                tab_data_list.append(tab)

            embeddings = self._encode_with_optimization(searchable_texts)
            embeddings = self._normalize_embeddings(embeddings)

            # Add to new FAISS index (one vector per tab)
            new_index.add(x=embeddings.astype('float32'))
            print(f"[DEBUG] After adding, new FAISS index size: {new_index.ntotal}")
            print(f"[DEBUG] After adding, new FAISS index size: {new_index.ntotal}")

            for i, (tab, embedding) in enumerate(zip(tab_data_list, embeddings)):
                tab_id = tab.get('id')
                window_id = tab.get('windowId')
                tab_key = f"{tab_id}_{window_id}"
                
                new_tab_embeddings[tab_key] = {
                    'embedding': embedding,
                    'tab_data': tab,
                    'searchable_text': searchable_texts[i],
                    'faiss_index': i,
                    'text_hash': self._get_text_hash(searchable_texts[i])
                }
                new_tab_index_map[tab_id] = i

            # Atomic update: replace old data with new data
            self.index = new_index
            self.tab_embeddings = new_tab_embeddings
            self.tab_index_map = new_tab_index_map

            print(f"✅ FAISS index updated with {len(tabs)} tabs (index size: {self.index.ntotal})")
            self.save_to_memory()
        except Exception as e:
            print(f"Error updating FAISS index: {e}")
            import traceback
            traceback.print_exc()

    def _create_optimized_embedding_text(self, tab: Dict) -> str:
        """
        Create optimized text for embedding with better structure and weighting
        """
        # Extract and clean components
        title = self._clean_text(tab.get('title', ''))
        url = tab.get('url', '')
        description = self._clean_text(tab.get('description', ''))
        keywords = self._clean_text(tab.get('keywords', ''))
        og_description = self._clean_text(tab.get('ogDescription', ''))
        
        # Extract domain for context
        domain = self._extract_domain(url)
        
        # Extract URL path keywords
        url_keywords = self._extract_url_keywords(url)
        
        # Create weighted text structure for better embeddings
        # Title gets highest weight by repetition and position
        components = []
        
        # Primary content (most important)
        if title:
            components.append(f"Title: {title}")
            components.append(title)  # Repeat for emphasis
        
        # Secondary content
        if description:
            components.append(f"Description: {description}")
        
        if og_description and og_description != description:
            components.append(f"Content: {og_description}")
        
        # Contextual information
        if domain:
            components.append(f"Site: {domain}")
        
        if url_keywords:
            components.append(f"Page: {' '.join(url_keywords)}")
        
        if keywords:
            components.append(f"Keywords: {keywords}")
        
        # Join with periods for better sentence structure
        return ". ".join(filter(None, components))

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better embeddings"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that don't add semantic meaning
        text = re.sub(r'[^\w\s\-.,!?():]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[-]{2,}', '-', text)
        
        # Limit length to prevent embedding quality degradation
        if len(text) > 500:
            text = text[:500].rsplit(' ', 1)[0] + '...'
        
        return text

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www and common prefixes
            domain = re.sub(r'^www\.', '', domain)
            return domain
        except:
            return ""

    def _extract_url_keywords(self, url: str) -> List[str]:
        """Extract meaningful keywords from URL path"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # Split path and clean
            keywords = []
            for segment in path.split('/'):
                if segment and len(segment) > 1:
                    # Replace hyphens/underscores with spaces
                    segment = re.sub(r'[-_]', ' ', segment)
                    # Remove file extensions
                    segment = re.sub(r'\.[a-zA-Z0-9]+$', '', segment)
                    # Filter out numeric-only segments and common words
                    if not segment.isdigit() and segment not in ['www', 'com', 'html', 'index']:
                        keywords.append(segment)
            
            return keywords[:5]  # Limit to 5 most relevant keywords
        except:
            return []

    def _encode_with_optimization(self, texts: List[str]) -> np.ndarray:
        """Encode texts with optimized settings for better embeddings"""
        try:
            # Use batch processing for efficiency
            batch_size = min(32, len(texts))
            
            # Encode with specific parameters for better quality
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=False,  # We'll normalize manually
                show_progress_bar=len(texts) > 50
            )
            
            return embeddings
        except Exception as e:
            print(f"Error in encoding: {e}")
            # Fallback to simple encoding
            return self.model.encode(texts)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for better cosine similarity"""
        # L2 normalization for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to help with deduplication"""
        return hashlib.md5(text.encode()).hexdigest()

    def _should_use_hnsw_index(self, num_tabs: int) -> bool:
        """Determine if HNSW index should be used based on tab count"""
        return num_tabs > 100  # Use HNSW for large collections

    def _create_optimized_index(self, num_tabs: int) -> faiss.Index:
        """Create the most appropriate FAISS index based on collection size"""
        if self._should_use_hnsw_index(num_tabs):
            # Use HNSW for large collections
            return faiss.IndexFlatIP(self.dimension)
        else:
            # Use flat index for smaller collections
            return faiss.IndexFlatIP(self.dimension)

    
    def trigger_embedding_update(self):
        """Trigger embedding processing for UI updates"""
        if self.current_tabs:
            threading.Thread(target=self._update_faiss_index, 
                           args=(self.current_tabs, {}), daemon=True).start()
    
    def search_tabs_api(self):
        """API endpoint for searching tabs"""
        try:
            if not request.is_json:
                return jsonify({'error': 'Invalid or missing JSON in request'}), 400
            data = request.get_json(silent=True) or {}
            query = data.get('query', '')
            results = self.search_tabs(query)
            return jsonify({'results': results})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        

    def tokenize(self, text):
        # Simple tokenizer: lowercase and split on non-alphabetic characters
        return re.findall(r'\w+', text.lower())

    def build_bm25_corpus(self):
        """Build BM25 corpus and model from tab searchable text"""
        corpus = []
        tab_keys = []

        for key, data in self.tab_embeddings.items():
            tokens = self.tokenize(data['searchable_text'])
            corpus.append(tokens)
            tab_keys.append(key)
        
        bm25 = BM25Okapi(corpus)
        return bm25, corpus, tab_keys
    
    def search_tabs(self, query):
        """Search tabs using FAISS semantic similarity with fallback to direct embedding comparison"""
        if not query or not self.tab_embeddings:
            print("[DEBUG] Empty query or no tab embeddings.")
            return []
        
        try:
            print(f"[DEBUG] search_tabs called with query: {query}")
            print(f"[DEBUG] Number of tab_embeddings: {len(self.tab_embeddings)}")
            print(f"[DEBUG] FAISS index size: {self.index.ntotal}")
            
            # Create query embedding
            query_embedding = self.model.encode([query])[0]
            # normalize query vector
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            else:
                query_embedding = np.zeros_like(query_embedding)

            # Try FAISS search first
            results = []
            try:
                query_vector = query_embedding.reshape(1, -1).astype('float32')
                k = min(10, len(self.tab_embeddings))
                print(f"[DEBUG] k (top results to fetch): {k}")
                # Search using FAISS with explicit parameter names
                distances, indices = self.index.search(x=query_vector, k=k)
                print(f"[DEBUG] FAISS search distances: {distances}")
                print(f"[DEBUG] FAISS search indices: {indices}")

                # Process FAISS results
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    print(f"[DEBUG] Result {i}: distance={distance}, idx={idx}")
                    if idx < 0 or idx >= len(self.tab_embeddings):  # FAISS returns -1 for empty slots
                        print(f"[DEBUG] Skipping idx {idx} (invalid index)")
                        continue
                        
                    # Find tab data by FAISS index
                    tab_key = None
                    for key, data in self.tab_embeddings.items():
                        if data.get('faiss_index') == idx:
                            tab_key = key
                            break
                    
                    if tab_key:
                        tab_data = self.tab_embeddings[tab_key]
                        faiss_score = 1 - distance
                        results.append({
                            'tab_id': tab_data['tab_data'].get('id'),
                            'window_id': tab_data['tab_data'].get('windowId'),
                            'title': tab_data['tab_data'].get('title', ''),
                            'url': tab_data['tab_data'].get('url', ''),
                            'similarity': float(distance),
                            'searchable_text': tab_data['searchable_text'][:200] + '...' if len(tab_data['searchable_text']) > 200 else tab_data['searchable_text'],
                            'faiss_score': faiss_score
                        })
                        
            except Exception as faiss_error:
                print(f"[DEBUG] FAISS search failed: {faiss_error}")
                # Fallback to direct embedding comparison
                results = self._search_with_direct_comparison(query_embedding)
            
            # Add BM25 lexical search
            if results:
                bm25_model, corpus, tab_keys = self.build_bm25_corpus()
                tokenized_query = self.tokenize(query)
                bm25_scores = bm25_model.get_scores(tokenized_query)
                
                # Combine FAISS and BM25 scores
                for result in results:
                    tab_key = f"{result['tab_id']}_{result['window_id']}"
                    if tab_key in tab_keys:
                        bm25_score = bm25_scores[tab_keys.index(tab_key)]
                        faiss_score = result.get('faiss_score', 0)
                        hybrid_score = 0.7 * faiss_score + 0.4 * bm25_score
                        result['hybrid_score'] = hybrid_score
                    else:
                        result['hybrid_score'] = result.get('faiss_score', 0)
                
                # Sort by hybrid score
                results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            print(f"[DEBUG] Final results count: {len(results)}")
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _search_with_direct_comparison(self, query_embedding):
        """Fallback search using direct embedding similarity when FAISS fails"""
        results = []
        
        for tab_key, tab_data in self.tab_embeddings.items():
            try:
                # Calculate cosine similarity directly
                embedding = tab_data['embedding']
                similarity = np.dot(query_embedding, embedding)
                
                results.append({
                    'tab_id': tab_data['tab_data'].get('id'),
                    'window_id': tab_data['tab_data'].get('windowId'),
                    'title': tab_data['tab_data'].get('title', ''),
                    'url': tab_data['tab_data'].get('url', ''),
                    'similarity': float(1 - similarity),  # Convert to distance
                    'searchable_text': tab_data['searchable_text'][:200] + '...' if len(tab_data['searchable_text']) > 200 else tab_data['searchable_text'],
                    'faiss_score': similarity
                })
            except Exception as e:
                print(f"Error calculating similarity for tab {tab_key}: {e}")
                continue
        
        # Sort by similarity score
        results.sort(key=lambda x: x['faiss_score'], reverse=True)
        return results[:10]

    def open_tab_api(self):
        """API endpoint to open a tab"""
        try:
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'No JSON data received'}), 400
                
            tab_id = data.get('tab_id')
            window_id = data.get('window_id')
            
            # Here you would send a message back to the browser extension
            # For now, we'll just return success
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def memory_stats_api(self):
        """API endpoint to get memory statistics"""
        try:
            stats = self.get_memory_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def memory_clear_api(self):
        """API endpoint to clear memory"""
        try:
            self.clear_memory()
            return jsonify({'status': 'success', 'message': 'Memory cleared'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500