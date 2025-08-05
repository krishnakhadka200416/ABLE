"""
Example script demonstrating programmatic usage of ABLE
"""

import sys
import os

# Add the current directory to the path to import able module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from able import (
    load_dataset, 
    create_and_train_model, 
    generate_able_explanation
)
import torch

def run_able_example():
    """Run a simple ABLE example programmatically."""
    
    # Configuration
    dataset_name = 'adult'
    model_name = 'MLP'
    test_index = 0
    attack_method = 'PGD'
    top_k = 5
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    X_train, X_test, y_train, y_test, feature_names, scaler, num_classes = load_dataset(dataset_name)
    input_dim = X_train.shape[1]
    
    # Create and train model
    print(f"Training {model_name} model...")
    classifier = create_and_train_model(
        model_name, X_train, y_train, X_test, y_test, 
        input_dim, num_classes, device
    )
    
    # Get test instance
    x_test = X_test[test_index]
    
    # Generate ABLE explanation
    print(f"Generating ABLE explanation for test instance {test_index}...")
    result = generate_able_explanation(
        classifier=classifier,
        x_test=x_test,
        X_train=X_train,
        feature_names=feature_names,
        attack_method=attack_method,
        num_adversarial_pairs=30,  # Smaller number for faster execution
        neighbor_radius=0.5,
        num_neighbors=50,  # Smaller number for faster execution
        top_k=top_k
    )
    
    if result:
        print("\n" + "="*60)
        print("ABLE EXPLANATION RESULTS")
        print("="*60)
        print(f"Original Prediction: Class {result['original_label']}")
        print(f"Confidence: {result['original_confidence']:.3f}")
        print(f"Explanation Fidelity: {result['fidelity']:.3f}")
        print(f"Generation Time: {result['explanation_time']:.2f}s")
        print(f"Success Rate: {result['success_rate']:.1%}")
        print(f"\nTop-{top_k} Important Features:")
        for i, (feat_name, score) in enumerate(result['feature_scores']):
            print(f"  {i+1}. {feat_name}: {score:.4f}")
        print("="*60)
        
        return result
    else:
        print("Failed to generate ABLE explanation")
        return None

def compare_attack_methods():
    """Compare different attack methods on the same instance."""
    
    dataset_name = 'adult'
    model_name = 'MLP'
    test_index = 0
    attack_methods = ['PGD', 'ETDEEPFOOL']  # Reduced for faster execution
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and train model once
    X_train, X_test, y_train, y_test, feature_names, scaler, num_classes = load_dataset(dataset_name)
    input_dim = X_train.shape[1]
    
    classifier = create_and_train_model(
        model_name, X_train, y_train, X_test, y_test, 
        input_dim, num_classes, device
    )
    
    x_test = X_test[test_index]
    results = {}
    
    print("\n" + "="*80)
    print("COMPARING ATTACK METHODS")
    print("="*80)
    
    for attack_method in attack_methods:
        print(f"\nTesting {attack_method} attack...")
        
        result = generate_able_explanation(
            classifier=classifier,
            x_test=x_test,
            X_train=X_train,
            feature_names=feature_names,
            attack_method=attack_method,
            num_adversarial_pairs=20,  # Smaller for faster execution
            neighbor_radius=0.5,
            num_neighbors=50,
            top_k=5
        )
        
        if result:
            results[attack_method] = result
            print(f"{attack_method} - Fidelity: {result['fidelity']:.3f}, Time: {result['explanation_time']:.2f}s")
    
    # Summary comparison
    print("\n" + "="*80)
    print("ATTACK METHOD COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<12} {'Fidelity':<10} {'Time (s)':<10} {'Success Rate':<15}")
    print("-" * 80)
    
    for method, result in results.items():
        print(f"{method:<12} {result['fidelity']:<10.3f} {result['explanation_time']:<10.2f} {result['success_rate']:<15.1%}")
    
    return results

if __name__ == "__main__":
    print("ABLE Example Script")
    print("="*50)
    
    # Run basic example
    print("\n1. Running basic ABLE example...")
    basic_result = run_able_example()
    
    # Compare attack methods
    print("\n2. Comparing attack methods...")
    comparison_results = compare_attack_methods()
    
    print("\nExample script completed!") 