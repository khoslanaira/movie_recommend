#!/usr/bin/env python3
"""
Debug user validation issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataLoader

def debug_user_validation():
    print("🔍 Debugging User Validation...")
    
    # Load data exactly like the app does
    loader = MovieLensDataLoader()
    loader.load_data()
    loader.clean_data(min_ratings=5)
    
    user_to_idx = loader.user_to_idx
    item_to_idx = loader.item_to_idx
    
    print(f"Total users in dataset: {len(user_to_idx)}")
    print(f"User ID range: {min(user_to_idx.keys())} to {max(user_to_idx.keys())}")
    print(f"User 790 exists: {790 in user_to_idx}")
    print(f"User 790 index: {user_to_idx.get(790, 'NOT FOUND')}")
    
    # Test some users
    test_users = [1, 10, 50, 100, 500, 790, 900, 943]
    print(f"\nTesting user validation:")
    for user_id in test_users:
        exists = user_id in user_to_idx
        print(f"  User {user_id}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    # Check if there are any issues with the data
    print(f"\nData integrity check:")
    print(f"  user_to_idx type: {type(user_to_idx)}")
    print(f"  user_to_idx is None: {user_to_idx is None}")
    print(f"  First few users: {list(user_to_idx.keys())[:5]}")
    print(f"  Last few users: {list(user_to_idx.keys())[-5:]}")

if __name__ == "__main__":
    debug_user_validation()
