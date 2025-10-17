#!/usr/bin/env python3
"""
Test the working app
"""

import requests
import json
import time

def test_working_app():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Working Movie Recommendation System...")
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats API working: {stats['total_users']} users, {stats['total_movies']} movies")
        else:
            print(f"❌ Stats API failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Stats API error: {e}")
        return
    
    # Test recommendations for user 790 (the one that was failing)
    print(f"\n🎯 Testing User 790 (the one that was failing)...")
    try:
        response = requests.get(f"{base_url}/api/recommendations/790", timeout=30)
        if response.status_code == 200:
            recs = response.json()
            print(f"✅ User 790 recommendations SUCCESS!")
            for model_name, movies in recs.items():
                print(f"  {model_name}: {len(movies)} movies")
                if movies:
                    print(f"    First movie: {movies[0]['title']} ({movies[0]['year']})")
        else:
            print(f"❌ User 790 failed: {response.status_code}")
            print(f"    Error: {response.text}")
    except Exception as e:
        print(f"❌ User 790 error: {e}")
    
    # Test a few more users
    test_users = [1, 10, 50, 100, 500, 900, 943]
    
    for user_id in test_users:
        print(f"\n🎯 Testing User {user_id}...")
        try:
            response = requests.get(f"{base_url}/api/recommendations/{user_id}", timeout=15)
            if response.status_code == 200:
                recs = response.json()
                total_movies = sum(len(movies) for movies in recs.values())
                print(f"✅ User {user_id}: {total_movies} total recommendations")
            else:
                print(f"❌ User {user_id} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ User {user_id} error: {e}")
    
    # Test invalid user
    print(f"\n🎯 Testing Invalid User 9999...")
    try:
        response = requests.get(f"{base_url}/api/recommendations/9999", timeout=10)
        if response.status_code == 404:
            print(f"✅ Invalid user correctly rejected: {response.json()['error']}")
        else:
            print(f"❌ Invalid user should return 404, got {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid user test error: {e}")

if __name__ == "__main__":
    test_working_app()
