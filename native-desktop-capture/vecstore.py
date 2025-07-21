import re
import hashlib
import pickle
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import threading

class VectorStore:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.dimension = 768  # all-mpnet-base-v2 dimension
        self.index = faiss.IndexFlatIP(self.dimension)

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

    def save_to_memory(self):
        """Save tab data and FAISS index to disk"""
        try:
            memory_data = {
                'tab_embeddings': self.tab_embeddings,
                'tab_index_map': self.tab_index_map,
                'last_tab_index_hash': self.last_tab_index_hash,
                'current_tabs': self.current_tabs,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            faiss.write_index(self.index, self.index_file)
            print(f"✅ Saved {len(self.tab_embeddings)} tabs to memory")
        except Exception as e:
            print(f"Error saving to memory: {e}")

    def load_from_memory(self):
        """Load tab data and FAISS index from disk"""
        try:
            if os.path.exists(self.memory_file) and os.path.exists(self.index_file):
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                self.tab_embeddings = memory_data.get('tab_embeddings', {})
                self.tab_index_map = memory_data.get('tab_index_map', {})
                self.last_tab_index_hash = memory_data.get('last_tab_index_hash')
                self.current_tabs = memory_data.get('current_tabs', {})
                self.index = faiss.read_index(self.index_file)
                print(f"✅ Loaded {len(self.tab_embeddings)} tabs from memory")
                print(f"✅ FAISS index loaded with {self.index.ntotal} vectors")
            else:
                print("No existing memory found, starting fresh")
        except Exception as e:
            print(f"Error loading from memory: {e}")
            self.tab_embeddings = {}
            self.tab_index_map = {}
            self.last_tab_index_hash = None
            self.current_tabs = {}

    def clear_memory(self):
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

    def _hash_tab_index(self, tab_index):
        return hashlib.md5(json.dumps(tab_index, sort_keys=True).encode()).hexdigest()

    def update_index(self, tabs: List[Dict], tab_index: Any) -> None:
        try:
            print("Processing embeddings and updating FAISS index...")
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
            new_index.add(x=embeddings.astype('float32'))
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
        title = self._clean_text(tab.get('title', ''))
        url = tab.get('url', '')
        description = self._clean_text(tab.get('description', ''))
        keywords = self._clean_text(tab.get('keywords', ''))
        og_description = self._clean_text(tab.get('ogDescription', ''))
        domain = self._extract_domain(url)
        url_keywords = self._extract_url_keywords(url)
        components = []
        if title:
            components.append(f"Title: {title}")
            components.append(title)
        if description:
            components.append(f"Description: {description}")
        if og_description and og_description != description:
            components.append(f"Content: {og_description}")
        if domain:
            components.append(f"Site: {domain}")
        if url_keywords:
            components.append(f"Page: {' '.join(url_keywords)}")
        if keywords:
            components.append(f"Keywords: {keywords}")
        return ". ".join(filter(None, components))

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\-.,!?():]', ' ', text)
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[-]{2,}', '-', text)
        if len(text) > 500:
            text = text[:500].rsplit(' ', 1)[0] + '...'
        return text

    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            domain = re.sub(r'^www\.', '', domain)
            return domain
        except:
            return ""

    def _extract_url_keywords(self, url: str) -> List[str]:
        try:
            parsed = urlparse(url)
            path = parsed.path
            keywords = []
            for segment in path.split('/'):
                if segment and len(segment) > 1:
                    segment = re.sub(r'[-_]', ' ', segment)
                    segment = re.sub(r'\.[a-zA-Z0-9]+$', '', segment)
                    if not segment.isdigit() and segment not in ['www', 'com', 'html', 'index']:
                        keywords.append(segment)
            return keywords[:5]
        except:
            return []

    def _encode_with_optimization(self, texts: List[str]) -> np.ndarray:
        try:
            batch_size = min(32, len(texts))
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=False,
                show_progress_bar=len(texts) > 50
            )
            return embeddings
        except Exception as e:
            print(f"Error in encoding: {e}")
            return self.model.encode(texts)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
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

    def search(self, query: str):
        if not query or not self.tab_embeddings:
            print("[DEBUG] Empty query or no tab embeddings.")
            return []
        try:
            print(f"[DEBUG] search called with query: {query}")
            print(f"[DEBUG] Number of tab_embeddings: {len(self.tab_embeddings)}")
            print(f"[DEBUG] FAISS index size: {self.index.ntotal}")
            query_embedding = self.model.encode([query])[0]
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            else:
                query_embedding = np.zeros_like(query_embedding)
            results = []
            try:
                query_vector = query_embedding.reshape(1, -1).astype('float32')
                k = min(10, len(self.tab_embeddings))
                distances, indices = self.index.search(x=query_vector, k=k)
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < 0 or idx >= len(self.tab_embeddings):
                        continue
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
        results = []
        for tab_key, tab_data in self.tab_embeddings.items():
            try:
                embedding = tab_data['embedding']
                similarity = np.dot(query_embedding, embedding)
                results.append({
                    'tab_id': tab_data['tab_data'].get('id'),
                    'window_id': tab_data['tab_data'].get('windowId'),
                    'title': tab_data['tab_data'].get('title', ''),
                    'url': tab_data['tab_data'].get('url', ''),
                    'similarity': float(1 - similarity),
                    'searchable_text': tab_data['searchable_text'][:200] + '...' if len(tab_data['searchable_text']) > 200 else tab_data['searchable_text'],
                    'faiss_score': similarity
                })
            except Exception as e:
                print(f"Error calculating similarity for tab {tab_key}: {e}")
                continue
        results.sort(key=lambda x: x['faiss_score'], reverse=True)
        return results[:10]

    def trigger_embedding_update(self):
        """Trigger embedding processing for UI updates"""
        if self.current_tabs:
            threading.Thread(target=self.update_index, 
                           args=(self.current_tabs, {}), daemon=True).start()
