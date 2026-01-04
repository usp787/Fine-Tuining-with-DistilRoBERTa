#!/usr/bin/env python3
"""
LoRA Emotion Classifier - Inference Script
Multi-label emotion classification using DistilRoBERTa with LoRA adapters
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import json
import sys
from typing import List, Dict, Union
import warnings
warnings.filterwarnings('ignore')


# GoEmotions 28 simplified labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class EmotionClassifier(nn.Module):
    """DistilRoBERTa with LoRA adapters + classification head"""
    
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(pooled)
        return logits


def build_model(model_path: str, rank: int = 16, device: str = 'cuda') -> EmotionClassifier:
    """
    Reconstruct the LoRA model architecture and load weights
    
    Args:
        model_path: Path to the saved .pt file
        rank: LoRA rank (must match training config)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model ready for inference
    """
    print(f"Loading model from: {model_path}")
    
    # 1. Load base backbone
    backbone = AutoModel.from_pretrained('distilroberta-base')
    
    # 2. Configure LoRA (must match training setup)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['key', 'query', 'value']
    )
    
    # 3. Inject LoRA adapters
    backbone = get_peft_model(backbone, peft_config)
    
    # 4. Build classifier head (must match training architecture)
    hidden_size = backbone.config.hidden_size  # 768 for DistilRoBERTa
    classifier_head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 28)  # 28 emotions
    )
    
    # 5. Combine into full model
    model = EmotionClassifier(backbone, classifier_head)
    
    # 6. Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    return model


def preprocess_text(
    texts: Union[str, List[str]], 
    tokenizer, 
    max_length: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Tokenize input text(s)
    
    Args:
        texts: Single string or list of strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with input_ids and attention_mask tensors
    """
    if isinstance(texts, str):
        texts = [texts]
    
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }


def predict(
    model: EmotionClassifier,
    texts: Union[str, List[str]],
    tokenizer,
    device: str = 'cuda',
    threshold: float = 0.5,
    top_k: int = None
) -> List[Dict]:
    """
    Run inference on input text(s)
    
    Args:
        model: Trained EmotionClassifier
        texts: Single text or batch of texts
        tokenizer: HuggingFace tokenizer
        device: 'cuda' or 'cpu'
        threshold: Probability threshold for positive predictions
        top_k: If set, return only top K emotions (overrides threshold)
    
    Returns:
        List of prediction dictionaries, one per input text
    """
    # Ensure batch format
    if isinstance(texts, str):
        texts = [texts]
    
    # Preprocess
    inputs = preprocess_text(texts, tokenizer)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Format results
    results = []
    for i, text in enumerate(texts):
        pred_probs = probs[i]
        
        # Create emotion-probability pairs
        emotion_scores = [
            {'emotion': EMOTION_LABELS[j], 'probability': float(pred_probs[j])}
            for j in range(len(EMOTION_LABELS))
        ]
        
        # Sort by probability
        emotion_scores.sort(key=lambda x: x['probability'], reverse=True)
        
        # Filter predictions
        if top_k:
            predicted_emotions = emotion_scores[:top_k]
        else:
            predicted_emotions = [
                e for e in emotion_scores if e['probability'] >= threshold
            ]
        
        results.append({
            'text': text,
            'predicted_emotions': predicted_emotions,
            'all_scores': emotion_scores  # Keep all for reference
        })
    
    return results


def format_output(results: List[Dict], output_format: str = 'pretty') -> str:
    """
    Format prediction results for display
    
    Args:
        results: List of prediction dictionaries
        output_format: 'pretty', 'json', or 'compact'
    
    Returns:
        Formatted string
    """
    if output_format == 'json':
        return json.dumps(results, indent=2)
    
    output = []
    for i, result in enumerate(results, 1):
        if output_format == 'pretty':
            output.append(f"\n{'='*60}")
            output.append(f"Prediction {i}/{len(results)}")
            output.append(f"{'='*60}")
            output.append(f"Text: {result['text']}")
            output.append(f"\nPredicted Emotions:")
            
            if result['predicted_emotions']:
                for emotion in result['predicted_emotions']:
                    output.append(
                        f"  • {emotion['emotion']:15s} "
                        f"({emotion['probability']:.2%})"
                    )
            else:
                output.append("  (No emotions above threshold)")
        
        elif output_format == 'compact':
            emotions_str = ', '.join(
                f"{e['emotion']}({e['probability']:.2f})"
                for e in result['predicted_emotions']
            )
            output.append(f"{result['text']} -> {emotions_str}")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='LoRA Emotion Classifier - Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python lora_inference.py --text "I'm so happy today!"
  
  # Batch prediction from file
  python lora_inference.py --input texts.txt
  
  # Get top 3 emotions only
  python lora_inference.py --text "This is amazing!" --top-k 3

  # JSON output
  python lora_inference.py --text "Hello world" --format json
  
  # Use CPU
  python lora_inference.py --text "Test" --device cpu
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='best_LoRA_model.pt',
        help='Path to the trained model .pt file'
    )
    parser.add_argument(
        '--rank', 
        type=int, 
        default=16,
        help='LoRA rank (must match training config)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text', 
        type=str,
        help='Single text to classify'
    )
    input_group.add_argument(
        '--input', 
        type=str,
        help='Path to file with texts (one per line)'
    )
    input_group.add_argument(
        '--interactive', 
        action='store_true',
        help='Interactive mode (enter texts one by one)'
    )
    
    # Prediction arguments
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Probability threshold for predictions (0.0-1.0)'
    )
    parser.add_argument(
        '--top-k', 
        type=int,
        help='Return top K emotions (overrides threshold)'
    )
    parser.add_argument(
        '--max-length', 
        type=int, 
        default=128,
        help='Maximum sequence length'
    )
    
    # Output arguments
    parser.add_argument(
        '--format', 
        type=str, 
        default='pretty',
        choices=['pretty', 'json', 'compact'],
        help='Output format'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Save results to file (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model and tokenizer
    print(f"\n{'='*60}")
    print("LoRA Emotion Classifier - Inference")
    print(f"{'='*60}\n")
    
    model = build_model(args.model_path, args.rank, args.device)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    
    # Get input texts
    if args.text:
        texts = [args.text]
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} texts from {args.input}")
    elif args.interactive:
        print("\nInteractive Mode (Ctrl+C or 'quit' to exit)")
        print("-" * 60)
        texts = []
        try:
            while True:
                text = input("\nEnter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if text:
                    texts.append(text)
                    # Process immediately in interactive mode
                    result = predict(
                        model, text, tokenizer, args.device, 
                        args.threshold, args.top_k
                    )
                    print(format_output(result, args.format))
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
        sys.exit(0)
    
    # Run predictions
    print(f"\nRunning inference on {len(texts)} text(s)...")
    results = predict(
        model, texts, tokenizer, args.device,
        args.threshold, args.top_k
    )
    
    # Format and output
    output_str = format_output(results, args.format)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"\n✓ Results saved to {args.output}")
    else:
        print(output_str)
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
