#!/usr/bin/env python3
"""
Test script to verify browser extension sync with desktop app
"""

import requests
import json
import time

def test_sync():
    """Test the sync functionality"""
    print("Testing browser extension sync with desktop app...")
    
    # Simulate tab data from browser extension
    test_data = {
        "openTabs": [
            {
                "id": 1,
                "windowId": 1,
                "title": "Google - Search Engine",
                "url": "https://www.google.com",
                "description": "Search the world's information, including webpages, images, videos and more.",
                "lastAccessed": "2024-01-15T10:30:00Z"
            },
            {
                "id": 2,
                "windowId": 1,
                "title": "GitHub - Build software better, together",
                "url": "https://github.com",
                "description": "GitHub is where over 100 million developers shape the future of software.",
                "lastAccessed": "2024-01-15T10:35:00Z"
            },
            {
                "id": 3,
                "windowId": 1,
                "title": "Stack Overflow - Where Developers Learn, Share, & Build Careers",
                "url": "https://stackoverflow.com",
                "description": "Stack Overflow is the largest, most trusted online community for developers.",
                "lastAccessed": "2024-01-15T10:40:00Z"
            }
        ],
        "lastUpdated": "2024-01-15T10:45:00Z"
    }
    
    try:
        # Send data to desktop app
        response = requests.post(
            'http://localhost:8080/update-tabs',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sync successful! Received {result.get('received', 0)} tabs")
            
            # Test search functionality
            time.sleep(2)  # Wait for embeddings to process
            
            search_response = requests.post(
                'http://localhost:8080/search',
                json={'query': 'github'},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if search_response.status_code == 200:
                search_results = search_response.json()
                print(f"✅ Search successful! Found {len(search_results.get('results', []))} results")
                
                for result in search_results.get('results', []):
                    print(f"  - {result['title']} (similarity: {result['similarity']:.2%})")
            else:
                print(f"❌ Search failed: {search_response.status_code}")
                
        else:
            print(f"❌ Sync failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to desktop app. Make sure it's running on localhost:8080")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_sync() 