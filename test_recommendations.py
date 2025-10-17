#!/usr/bin/env python3
"""
Test recommendations API
"""

import requests
import json

def test_recommendations():
    base_url = "http://localhost:5000"
    
    print("Testing Movie Recommendation System...")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats API working: {stats['total_users']} users, {stats['total_movies']} movies")
        else:
            print(f"❌ Stats API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats API error: {e}")
    
    # Test recommendations for a valid user
    test_users = [1, 10, 50, 100, 500]
    
    for user_id in test_users:
        try:
            response = requests.get(f"{base_url}/api/recommendations/{user_id}")
            if response.status_code == 200:
                recs = response.json()
                print(f"\n✅ User {user_id} recommendations:")
                for model_name, movies in recs.items():
                    print(f"  {model_name}: {len(movies)} movies")
                    if movies:
                        print(f"    First movie: {movies[0]['title']} ({movies[0]['year']})")
            else:
                print(f"❌ User {user_id} failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ User {user_id} error: {e}")
    
    # Test invalid user
    try:
        response = requests.get(f"{base_url}/api/recommendations/9999")
        if response.status_code == 404:
            print(f"✅ Invalid user correctly rejected: {response.json()['error']}")
        else:
            print(f"❌ Invalid user should return 404, got {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid user test error: {e}")

if __name__ == "__main__":
    test_recommendations()
