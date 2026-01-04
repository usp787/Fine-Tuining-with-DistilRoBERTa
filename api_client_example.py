#!/usr/bin/env python3
"""
Example client for LoRA Emotion Classifier API
Shows how to call your deployed API from Python
"""

import requests
import json
from typing import List, Dict, Optional

class EmotionClassifierClient:
    """Python client for emotion classification API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize client
        
        Args:
            base_url: Base URL of your API (change after deployment)
                     Examples:
                     - Local: "http://localhost:8080"
                     - EC2: "http://your-ec2-ip"
                     - ALB: "https://your-alb-dns"
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """Check if API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_emotions(self) -> List[str]:
        """Get list of all supported emotions"""
        response = requests.get(f"{self.base_url}/emotions")
        response.raise_for_status()
        return response.json()['emotions']
    
    def predict(
        self, 
        text: str, 
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Predict emotions for a single text
        
        Args:
            text: Text to classify
            threshold: Minimum probability threshold (0.0-1.0)
            top_k: Return only top K emotions (overrides threshold)
        
        Returns:
            Prediction dictionary with emotions and probabilities
        """
        payload = {
            "text": text,
            "threshold": threshold
        }
        if top_k is not None:
            payload["top_k"] = top_k
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Predict emotions for multiple texts
        
        Args:
            texts: List of texts to classify
            threshold: Minimum probability threshold
            top_k: Return only top K emotions per text
        
        Returns:
            List of prediction dictionaries
        """
        payload = {
            "texts": texts,
            "threshold": threshold
        }
        if top_k is not None:
            payload["top_k"] = top_k
        
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()['predictions']


def example_1_basic_usage():
    """Example 1: Basic single prediction"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Initialize client
    client = EmotionClassifierClient("http://localhost:8080")
    
    # Check if API is healthy
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Make a prediction
    text = "I'm so excited about this new opportunity!"
    result = client.predict(text, top_k=3)
    
    print(f"\nText: {result['text']}")
    print("\nTop 3 Emotions:")
    for emotion in result['predicted_emotions']:
        print(f"  • {emotion['emotion']:15s} {emotion['probability']:.1%}")


def example_2_batch_processing():
    """Example 2: Process multiple texts at once"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    client = EmotionClassifierClient("http://localhost:8080")
    
    # Multiple texts
    texts = [
        "This movie was absolutely terrible!",
        "I can't believe I won the lottery!",
        "I'm not sure what to think about this situation.",
        "The sunset was breathtakingly beautiful.",
        "I'm really worried about the exam tomorrow."
    ]
    
    # Process all at once
    results = client.predict_batch(texts, threshold=0.3)
    
    print(f"\nProcessed {len(results)} texts:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text'][:50]}...")
        emotions = [e['emotion'] for e in result['predicted_emotions'][:3]]
        print(f"   Emotions: {', '.join(emotions)}\n")


def example_3_threshold_tuning():
    """Example 3: Experiment with different thresholds"""
    print("\n" + "="*60)
    print("Example 3: Threshold Tuning")
    print("="*60)
    
    client = EmotionClassifierClient("http://localhost:8080")
    
    text = "I have mixed feelings about this decision."
    
    print(f"Text: {text}\n")
    
    # Try different thresholds
    for threshold in [0.2, 0.4, 0.6]:
        result = client.predict(text, threshold=threshold)
        print(f"Threshold {threshold:.1f}: {len(result['predicted_emotions'])} emotions")
        for emotion in result['predicted_emotions']:
            print(f"  • {emotion['emotion']:15s} {emotion['probability']:.2%}")
        print()


def example_4_sentiment_analysis():
    """Example 4: Use for sentiment analysis"""
    print("\n" + "="*60)
    print("Example 4: Sentiment Analysis")
    print("="*60)
    
    client = EmotionClassifierClient("http://localhost:8080")
    
    # Analyze customer reviews
    reviews = [
        "This product exceeded all my expectations! Highly recommend.",
        "Terrible quality. Broke after one use. Waste of money.",
        "It's okay, nothing special but does the job.",
        "I love everything about this! Best purchase ever!",
        "Very disappointed. Not as described."
    ]
    
    results = client.predict_batch(reviews, top_k=2)
    
    print("Customer Review Analysis:\n")
    for i, (review, result) in enumerate(zip(reviews, results), 1):
        emotions = result['predicted_emotions']
        top_emotion = emotions[0]['emotion']
        confidence = emotions[0]['probability']
        
        # Simple sentiment mapping
        positive = {'joy', 'love', 'excitement', 'admiration', 'gratitude'}
        negative = {'anger', 'sadness', 'disappointment', 'disgust', 'disapproval'}
        
        if top_emotion in positive:
            sentiment = "POSITIVE"
        elif top_emotion in negative:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        print(f"{i}. [{sentiment}] {review[:50]}...")
        print(f"   Emotion: {top_emotion} ({confidence:.1%})\n")


def example_5_content_moderation():
    """Example 5: Content moderation use case"""
    print("\n" + "="*60)
    print("Example 5: Content Moderation")
    print("="*60)
    
    client = EmotionClassifierClient("http://localhost:8080")
    
    # Comments to moderate
    comments = [
        "This is a thoughtful and well-reasoned argument.",
        "You're an idiot and nobody cares what you think!",
        "I disagree, but I respect your perspective.",
        "Everyone who believes this is stupid.",
        "Thanks for sharing! I learned something new."
    ]
    
    results = client.predict_batch(comments, threshold=0.4)
    
    print("Content Moderation Results:\n")
    
    # Flag aggressive emotions
    aggressive_emotions = {'anger', 'disgust', 'disapproval', 'annoyance'}
    
    for comment, result in zip(comments, results):
        detected_emotions = {e['emotion'] for e in result['predicted_emotions']}
        is_aggressive = bool(detected_emotions & aggressive_emotions)
        
        flag = "⚠️  FLAG" if is_aggressive else "✓ OK"
        print(f"{flag}: {comment[:50]}...")
        
        if is_aggressive:
            flagged = detected_emotions & aggressive_emotions
            print(f"       Detected: {', '.join(flagged)}")
        print()


def example_6_api_endpoint_change():
    """Example 6: Connect to deployed API"""
    print("\n" + "="*60)
    print("Example 6: Connect to Deployed API")
    print("="*60)
    
    # After deployment, change the URL
    # Local: http://localhost:8080
    # EC2: http://your-ec2-public-ip
    # ALB: https://your-load-balancer-dns
    
    # Example with deployed EC2 instance
    deployed_url = "http://your-ec2-ip"  # Change this!
    
    print(f"Connecting to: {deployed_url}")
    print("(Change 'deployed_url' variable to your actual endpoint)")
    
    # Same code works with deployed API
    # client = EmotionClassifierClient(deployed_url)
    # result = client.predict("Hello from the cloud!")
    # print(result)


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("LoRA Emotion Classifier API - Usage Examples")
    print("="*60)
    print("\nMake sure your API is running at http://localhost:8080")
    print("(Run: python app.py  OR  uvicorn app:app --host 0.0.0.0 --port 8080)")
    
    try:
        # Check if server is running
        requests.get("http://localhost:8080/health", timeout=2)
        print("\n✓ API is running!\n")
        
        # Run examples
        example_1_basic_usage()
        example_2_batch_processing()
        example_3_threshold_tuning()
        example_4_sentiment_analysis()
        example_5_content_moderation()
        example_6_api_endpoint_change()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  python app.py")
        print("\nOr change the base_url in EmotionClassifierClient")


if __name__ == "__main__":
    main()
