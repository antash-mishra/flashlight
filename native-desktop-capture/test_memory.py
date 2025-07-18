#!/usr/bin/env python3
"""
Test script to verify memory persistence functionality
"""

import requests
import json
import time

def test_memory_persistence():
    """Test memory persistence functionality"""
    print("Testing memory persistence...")
    
    # Test data
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
            }
        ],
        "tabIndex": {"1": 0, "2": 1},
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
            print("✅ Data sent successfully")
            
            # Wait for processing
            time.sleep(2)
            
            # Test memory stats
            stats_response = requests.get('http://localhost:8080/memory/stats')
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"✅ Memory stats: {stats}")
            else:
                print(f"❌ Memory stats failed: {stats_response.status_code}")
            
            # Test search to verify persistence
            search_response = requests.post(
                'http://localhost:8080/search',
                json={'query': 'github'},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if search_response.status_code == 200:
                search_results = search_response.json()
                print(f"✅ Search successful: {len(search_results.get('results', []))} results")
            else:
                print(f"❌ Search failed: {search_response.status_code}")
                
        else:
            print(f"❌ Data send failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to desktop app. Make sure it's running on localhost:8080")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_memory_clear():
    """Test memory clearing functionality"""
    print("\nTesting memory clearing...")
    
    try:
        # Clear memory
        clear_response = requests.post('http://localhost:8080/memory/clear')
        if clear_response.status_code == 200:
            print("✅ Memory cleared successfully")
            
            # Check stats after clearing
            stats_response = requests.get('http://localhost:8080/memory/stats')
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"✅ Memory stats after clearing: {stats}")
        else:
            print(f"❌ Memory clear failed: {clear_response.status_code}")
            
    except Exception as e:
        print(f"❌ Error clearing memory: {e}")

if __name__ == "__main__":
    test_memory_persistence()
    test_memory_clear() 