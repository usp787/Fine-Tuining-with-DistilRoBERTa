#!/usr/bin/env python3
"""
Test script for LoRA Emotion Classifier API
Tests all endpoints locally before deployment
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8080"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed")

def test_emotions_list():
    """Test emotions list endpoint"""
    print("\n" + "="*60)
    print("Testing /emotions endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/emotions")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of emotions: {data['count']}")
    print(f"Emotions: {', '.join(data['emotions'][:5])}...")
    
    assert response.status_code == 200
    assert data['count'] == 28
    print("✓ Emotions list test passed")

def test_single_prediction():
    """Test single text prediction"""
    print("\n" + "="*60)
    print("Testing /predict endpoint")
    print("="*60)
    
    payload = {
        "text": "I'm so happy and excited about this!",
        "threshold": 0.3,
        "top_k": 5
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    elapsed = time.time() - start
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response time: {elapsed:.3f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nText: {data['text']}")
        print(f"\nTop Predicted Emotions:")
        for emotion in data['predicted_emotions']:
            print(f"  • {emotion['emotion']:15s} ({emotion['probability']:.2%})")
        
        assert len(data['predicted_emotions']) > 0
        print("\n✓ Single prediction test passed")
    else:
        print(f"Error: {response.text}")
        raise Exception("Single prediction failed")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("Testing /predict/batch endpoint")
    print("="*60)
    
    payload = {
        "texts": [
            "I'm so happy today!",
            "This is terrible and frustrating.",
            "I'm not sure what to think about this.",
            "Wow, this is absolutely amazing!",
            "I'm feeling a bit nervous about tomorrow."
        ],
        "threshold": 0.5,
        "top_k": 3
    }
    
    print(f"Sending {len(payload['texts'])} texts...")
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    elapsed = time.time() - start
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response time: {elapsed:.3f}s")
    print(f"Average time per text: {elapsed/len(payload['texts']):.3f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nProcessed {data['count']} texts\n")
        
        for i, pred in enumerate(data['predictions'], 1):
            print(f"Text {i}: {pred['text']}")
            print(f"Top emotions: ", end="")
            print(", ".join(
                f"{e['emotion']}({e['probability']:.2f})" 
                for e in pred['predicted_emotions']
            ))
            print()
        
        assert data['count'] == len(payload['texts'])
        print("✓ Batch prediction test passed")
    else:
        print(f"Error: {response.text}")
        raise Exception("Batch prediction failed")

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("Testing error handling")
    print("="*60)
    
    # Test empty text
    print("\n1. Testing empty text...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": ""}
    )
    print(f"Status Code: {response.status_code}")
    print("✓ Empty text handled")
    
    # Test invalid threshold
    print("\n2. Testing invalid threshold...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "test", "threshold": 1.5}
    )
    print(f"Status Code: {response.status_code}")
    print("✓ Invalid threshold handled")
    
    # Test large batch
    print("\n3. Testing oversized batch...")
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": ["test"] * 101}
    )
    print(f"Status Code: {response.status_code}")
    print("✓ Oversized batch handled")
    
    print("\n✓ Error handling tests passed")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LoRA Emotion Classifier API - Test Suite")
    print("="*60)
    print(f"Testing against: {BASE_URL}")
    
    try:
        # Test if server is running
        print("\nChecking if server is running...")
        requests.get(BASE_URL, timeout=2)
        print("✓ Server is running\n")
        
        # Run tests
        test_health()
        test_emotions_list()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYour API is ready for deployment!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print(f"Make sure the API is running at {BASE_URL}")
        print("\nTo start the server, run:")
        print("  python app.py")
        print("or")
        print("  uvicorn app:app --host 0.0.0.0 --port 8080")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
