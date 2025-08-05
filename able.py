"""
Adversarially Bracketed Local Explanation (ABLE)

A novel approach for generating stable and high-fidelity local explanations 
by using adversarial attacks to bracket the local decision boundary.

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import copy
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression

# ART (Adversarial Robustness Toolbox)
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod, HopSkipJump

# Local modules
from datasets import load_dataset
from models import create_and_train_model

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)





###############################################################################
# ET DEEPFOOL IMPLEMENTATION
###############################################################################
def deepfool_et_tabular(data, net, target_class, overshoot=0.02, min_confidence=60, max_iter=50, device='cpu', debug=False):
    """
    Custom ET DeepFool implementation for tabular data with targeted adversarial attacks.
    
    This is the ET (Enhanced Targeted) DeepFool implementation specifically designed for tabular data,
    not the standard ART DeepFool. It focuses on generating minimal perturbations while maintaining
    high confidence in the target class.
    
    :param data: Input tabular data of size (1, num_features)
    :param net: network (input: tabular data, output: values of activation **BEFORE** softmax).
    :param target_class: target class that the data should be misclassified as
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param min_confidence: minimum amount of confidence for ET Deepfool (default = 60)
    :param max_iter: maximum number of iterations
    :param device: device to run on ('cpu' or 'cuda')
    :return: minimal perturbation, number of iterations, original label, final label, perturbed data, confidence scores
    """
    
    # Ensure data is on the correct device
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data).to(device)
    else:
        data = data.to(device)
    
    # Get original prediction (always use eval mode for initial prediction)
    with torch.no_grad():
        net.eval()
        f_output = net.forward(data)
        # Handle different output formats (tensor, list, tuple)
        if isinstance(f_output, (list, tuple)):
            f_output = f_output[0]
        f_data = f_output.data.cpu().numpy().flatten()
    I = (np.array(f_data)).flatten().argsort()[::-1]
    original_label = I[0]
    
    # If already classified as target, return original data
    if original_label == target_class:
        if debug:
            print(f"Debug ET DeepFool: Already classified as target (orig={original_label}, target={target_class})")
        return np.zeros_like(data.cpu().numpy()), 0, original_label, target_class, data.cpu().numpy(), f_data[target_class]
    
    input_shape = data.cpu().numpy().shape
    pert_data = copy.deepcopy(data)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    
    loop_i = 0
    
    x = Variable(pert_data, requires_grad=True)
    
    # Check if model is TabNet (has BatchNorm issues with batch_size=1)
    is_tabnet = (hasattr(net, 'encoder') or 
                 'TabNet' in str(type(net)) or 
                 hasattr(net, 'cat_emb_dim') or
                 'tabnet' in str(type(net)).lower())
    if is_tabnet:
        # Keep TabNet in eval mode to avoid BatchNorm issues with single samples
        net.eval()
        if debug:
            print(f"Debug ET DeepFool: TabNet detected - using eval mode to avoid BatchNorm issues")
    else:
        net.train()  # Set to train mode for gradient computation
    
    fs_raw = net.forward(x)
    # Handle different output formats (tensor, list, tuple)
    if isinstance(fs_raw, (list, tuple)):
        fs = fs_raw[0]
    else:
        fs = fs_raw
    k_i = original_label
    confidence = 0
    
    if debug:
        mode_str = "eval (TabNet)" if is_tabnet else "train"
        print(f"Debug ET DeepFool: Starting optimization loop - orig={original_label}, target={target_class}, min_conf={min_confidence}, mode={mode_str}")
    
    while (k_i != target_class or confidence < min_confidence) and loop_i < max_iter:
        
        # Clear gradients
        if x.grad is not None:
            x.grad.zero_()
        
        # Compute gradient w.r.t. CURRENT predicted class (not original)
        fs[0, k_i].backward(retain_graph=True)
        grad_current = x.grad.data.cpu().numpy().copy()
        
        # Clear gradients manually (modern PyTorch approach)
        x.grad.data.zero_()
        
        # Compute gradient w.r.t. target class  
        fs[0, target_class].backward(retain_graph=True)
        grad_target = x.grad.data.cpu().numpy().copy()
        
        # Debug: Check if gradients are computed properly
        if debug and loop_i == 0:
            grad_norm_current = np.linalg.norm(grad_current.flatten())
            grad_norm_target = np.linalg.norm(grad_target.flatten())
            print(f"Debug ET DeepFool: Gradient norms - current: {grad_norm_current:.6f}, target: {grad_norm_target:.6f}")
        
        # Compute optimal perturbation direction (following original ET DeepFool)
        w_k = grad_target - grad_current
        f_k = (fs[0, target_class] - fs[0, k_i]).data.cpu().numpy()
        
        # Avoid division by zero
        w_k_norm = np.linalg.norm(w_k.flatten())
        if w_k_norm == 0:
            if debug:
                print(f"Debug ET DeepFool: Zero gradient norm at iteration {loop_i} (current={k_i}, target={target_class})")
            break
            
        pert_k = abs(f_k) / w_k_norm
        
        # Compute minimal perturbation (following original)
        w = (pert_k + 1e-4) * w_k / w_k_norm
        
        r_i = (1 + overshoot) * w
        r_tot = np.float32(r_tot + r_i)
        
        # Apply total perturbation to original data (following original)
        pert_data = data + (1 + overshoot) * torch.from_numpy(r_tot).to(device)
        
        # Create new Variable for next iteration (following original)
        x = Variable(pert_data, requires_grad=True)
        
        # Ensure TabNet stays in eval mode throughout optimization
        if is_tabnet:
            net.eval()
        
        fs_raw = net.forward(x)
        # Handle different output formats (tensor, list, tuple)
        if isinstance(fs_raw, (list, tuple)):
            fs = fs_raw[0]
        else:
            fs = fs_raw
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        
        # Calculate confidence for target class
        confidence = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item() * 100
        
        if debug and (loop_i % 5 == 0 or loop_i < 3):  # Print progress occasionally
            print(f"Debug ET DeepFool: Iter {loop_i}: k_i={k_i}, target={target_class}, conf={confidence:.1f}%")
        
        loop_i += 1
    
    if debug:
        print(f"Debug ET DeepFool: Optimization finished - iterations={loop_i}, final_label={k_i}, confidence={confidence:.1f}%")
    
    # Final perturbation (following original ET DeepFool)
    r_tot = (1 + overshoot) * r_tot
    
    # Calculate final confidences
    with torch.no_grad():
        net.eval()  # Always use eval mode for final evaluation
        final_output_raw = net.forward(pert_data)
        # Handle different output formats (tensor, list, tuple)
        if isinstance(final_output_raw, (list, tuple)):
            final_output = final_output_raw[0]
        else:
            final_output = final_output_raw
        confidence_target = torch.nn.functional.softmax(final_output, dim=1)[0, target_class].item()
        confidence_orig = torch.nn.functional.softmax(final_output, dim=1)[0, original_label].item()
    
    return r_tot, loop_i, original_label, k_i, pert_data.cpu().numpy(), confidence_target

def et_deepfool_attack(classifier, x_in, target_class=None, max_iter=50, device='auto', min_confidence=60):
    """
    Convenience wrapper for custom ET DeepFool tabular attack.
    
    :param classifier: ART classifier
    :param x_in: Input data
    :param target_class: Target class (if None, uses untargeted attack)
    :param max_iter: Maximum iterations
    :param device: Device to use ('auto', 'cuda', or 'cpu')
    :return: Adversarial example or None if attack fails
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Extract the PyTorch model from ART classifier
        if hasattr(classifier, '_model'):
            torch_model = classifier._model
        elif hasattr(classifier, 'model'):
            torch_model = classifier.model
        else:
            print("ET DeepFool: Cannot extract PyTorch model from classifier")
            return None
        
        # Ensure model is on correct device
        torch_model = torch_model.to(device)
        
        # Convert input to tensor
        if isinstance(x_in, np.ndarray):
            x_tensor = torch.FloatTensor(x_in).to(device)
        else:
            x_tensor = x_in.to(device)
        
        # If no target class specified, use a random different class
        if target_class is None:
            with torch.no_grad():
                torch_model.eval()
                output = torch_model(x_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                orig_label = torch.argmax(output, dim=1).item()
                num_classes = output.shape[1]
                available_targets = [i for i in range(num_classes) if i != orig_label]
                if len(available_targets) == 0:
                    return None
                target_class = np.random.choice(available_targets)
        
        # Run ET DeepFool
        r_tot, iterations, orig_label_ret, final_label, x_adv, confidence = deepfool_et_tabular(
            data=x_tensor,
            net=torch_model,
            target_class=target_class,
            overshoot=0.02,
            min_confidence=min_confidence,
            max_iter=max_iter,
            device=device,
            debug=False
        )
        
        return x_adv
        
    except Exception as e:
        print(f"ET DeepFool attack failed: {e}")
        return None

###############################################################################
# ADVERSARIAL ATTACK METHODS
###############################################################################
def create_adversarial_attack(classifier, attack_method='PGD', eps=0.3, max_iter=20, 
                             targeted=False, norm=np.inf, num_random_init=0, **kwargs):
    """Create an adversarial attack instance based on the specified method."""
    attack_method = attack_method.upper()
    
    if attack_method == 'PGD':
        return ProjectedGradientDescent(
            estimator=classifier,
            norm=norm,
            eps=eps,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            **kwargs
        )
    elif attack_method == 'FGSM':
        return FastGradientMethod(
            estimator=classifier,
            norm=norm,
            eps=eps,
            targeted=targeted,
            **kwargs
        )

    elif attack_method == 'ETDEEPFOOL':
        # Custom ET DeepFool - handled in generate_dual_adversary
        return None
    elif attack_method == 'HOPSKIPJUMP':
        return HopSkipJump(
            classifier=classifier,
            targeted=targeted,
            max_iter=max_iter,
            max_eval=1000,
            init_eval=100,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")

def generate_dual_adversary(classifier, x_in, orig_label, attack_method='PGD', eps=0.6):
    """Generate dual adversarial examples that bracket the decision boundary."""
    attack_method = attack_method.upper()
    
    # First adversarial attack: untargeted to flip away from original class
    if attack_method in ['FGSM', 'PGD']:
        # For FGSM, implement retry mechanism with increasing epsilon
        if attack_method == 'FGSM':
            current_eps = 0.6  # Start with 0.6
            max_eps = 2.5
            eps_increment = 0.15
            
            while current_eps <= max_eps:
                attack1 = FastGradientMethod(
                    estimator=classifier,
                    norm=np.inf,
                    eps=current_eps,
                    targeted=False
                )
                
                x_adv1 = attack1.generate(x=x_in)
                adv_label1 = np.argmax(classifier.predict(x_adv1), axis=1)[0]
                
                if adv_label1 != orig_label:
                    break
                    
                current_eps += eps_increment
                
            if adv_label1 == orig_label:
                return (False, None)
                
            eps2 = current_eps + 0.2
            attack2 = FastGradientMethod(
                estimator=classifier,
                norm=np.inf,
                eps=eps2,
                targeted=True
            )
            
        else:  # PGD
            attack1 = ProjectedGradientDescent(
                estimator=classifier,
                norm=np.inf,
                eps=eps,
                max_iter=20,
                targeted=False
            )
            
            x_adv1 = attack1.generate(x=x_in)
            adv_label1 = np.argmax(classifier.predict(x_adv1), axis=1)[0]
            
            if adv_label1 == orig_label:
                return (False, None)
                
            eps2 = eps * 1.2 
            attack2 = ProjectedGradientDescent(
                estimator=classifier,
                norm=np.inf,
                eps=eps2,
                max_iter=20,
                targeted=True
            )
        
        target_labels = np.array([orig_label])
        x_adv2 = attack2.generate(x=x_adv1, y=target_labels)

    elif attack_method == 'ETDEEPFOOL':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        pred = classifier.predict(x_in)
        num_classes = pred.shape[1]
        available_targets = [i for i in range(num_classes) if i != orig_label]
        
        if len(available_targets) == 0:
            return (False, None)
        
        target1 = np.random.choice(available_targets)
        x_adv1 = et_deepfool_attack(
            classifier=classifier,
            x_in=x_in,
            target_class=target1,
            max_iter=50,
            device=device,
            min_confidence=60
        )
        
        if x_adv1 is None:
            return (False, None)
        
        adv_label1 = np.argmax(classifier.predict(x_adv1), axis=1)[0]
        if adv_label1 == orig_label:
            return (False, None)
        
        x_adv2 = et_deepfool_attack(
            classifier=classifier,
            x_in=x_adv1,
            target_class=orig_label,
            max_iter=50,
            device=device,
            min_confidence=60
        )
        
        if x_adv2 is None:
            return (False, None)
            
    elif attack_method == 'HOPSKIPJUMP':
        attack1 = HopSkipJump(
            classifier=classifier,
            targeted=False,
            max_iter=20,
            max_eval=1000,
            init_eval=100
        )
        x_adv1 = attack1.generate(x=x_in)
        adv_label1 = np.argmax(classifier.predict(x_adv1), axis=1)[0]
        
        if adv_label1 == orig_label:
            return (False, None)

        attack2 = HopSkipJump(
            classifier=classifier,
            targeted=True,
            max_iter=20,
            max_eval=1000,
            init_eval=100
        )
        target_labels = np.array([orig_label])
        x_adv2 = attack2.generate(x=x_adv1, y=target_labels)
        
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")
    
    adv_label2 = np.argmax(classifier.predict(x_adv2), axis=1)[0]
    if adv_label2 == orig_label:
        return (True, [x_adv1[0], x_adv2[0]])
    return (False, None)

def generate_neighborhood(x0, radius=0.5, n_samples=50, clip_min=None, clip_max=None, random_state=42):
    """Generate neighborhood points around x0 by adding Gaussian noise."""
    rng = np.random.RandomState(random_state)
    n_feats = x0.shape[1]
    X_perturbed = np.repeat(x0, n_samples, axis=0)
    noise = rng.normal(0, radius, (n_samples, n_feats))
    X_perturbed += noise
    if clip_min is not None and clip_max is not None:
        X_perturbed = np.clip(X_perturbed, clip_min, clip_max)
    return X_perturbed.astype(np.float32)

def get_topk_features_from_lr(lr_model, k=5):
    """Get top-k features from logistic regression coefficients."""
    coefs = lr_model.coef_
    if coefs.ndim == 1:
        # Binary classification case
        coef_abs = np.abs(coefs)
    else:
        # Multi-class case: take the maximum absolute coefficient across all classes
        coef_abs = np.max(np.abs(coefs), axis=0)
    
    top_inds = np.argsort(coef_abs)[::-1][:k]
    return set(top_inds)



###############################################################################
# ABLE CORE IMPLEMENTATION
###############################################################################
def generate_able_explanation(classifier, x_test, X_train, feature_names, 
                             scaler, ht, attack_method='PGD', num_adversarial_pairs=50,
                             neighbor_radius=0.5, num_neighbors=100, top_k=5):
    """
    Generate ABLE explanation for a test instance.
    
    Args:
        classifier: Trained ART classifier
        x_test: Test instance (1D array)
        X_train: Training data for clip bounds
        feature_names: List of feature names
        scaler: StandardScaler used for feature scaling
        ht: RDT HyperTransformer used for feature transformation
        attack_method: Attack method ('PGD', 'FGSM', 'DEEPFOOL', 'HOPSKIPJUMP')
        num_adversarial_pairs: Number of adversarial pairs to generate
        neighbor_radius: Radius for neighborhood generation
        num_neighbors: Number of neighbors to generate
        top_k: Number of top features to return
        
    Returns:
        dict: Results containing explanation and metrics
    """
    # Ensure x_test is 2D
    if len(x_test.shape) == 1:
        x_test = x_test.reshape(1, -1)
    
    x_test = x_test.astype(np.float32)
    
    # Get original prediction
    orig_pred = classifier.predict(x_test)
    orig_label = np.argmax(orig_pred, axis=1)[0]
    
    # Generate neighborhood
    X_neighbors = generate_neighborhood(
        x_test, radius=neighbor_radius, n_samples=num_neighbors,
        clip_min=X_train.min(), clip_max=X_train.max()
    )
    
    # Generate adversarial pairs
    X_adv_all = []
    count_success = 0
    count_trials = 0
    
    for i in range(num_adversarial_pairs):
        count_trials += 1
        success, X_adv_pair = generate_dual_adversary(
            classifier, x_test, orig_label, attack_method=attack_method
        )
        if success:
            X_adv_all.extend(X_adv_pair)
            count_success += 1
        
        if count_success >= num_adversarial_pairs:
            break
    
    if len(X_adv_all) == 0:
        return None
    
    X_adv_all = np.array(X_adv_all).astype(np.float32)
    
    # Combine neighbors and adversarial examples
    X_combined = np.vstack([X_neighbors, X_adv_all])
    
    # Get predictions for combined data
    y_combined = np.argmax(classifier.predict(X_combined), axis=1)
    
    # Train logistic regression surrogate
    lr_model = LogisticRegression(
        C=1.0, 
        solver='liblinear', 
        random_state=42, 
        max_iter=1000
    )
    lr_model.fit(X_combined, y_combined)
    
    y_pred_lr = lr_model.predict_proba(X_combined)
    y_pred_bb = classifier.predict(X_combined)
    
    # Handle shape mismatch between black-box and surrogate predictions
    if y_pred_bb.shape[1] != y_pred_lr.shape[1]:
        # If surrogate has fewer classes, pad with zeros
        if y_pred_lr.shape[1] < y_pred_bb.shape[1]:
            padding = np.zeros((y_pred_lr.shape[0], y_pred_bb.shape[1] - y_pred_lr.shape[1]))
            y_pred_lr = np.hstack([y_pred_lr, padding])
        # If surrogate has more classes, truncate
        elif y_pred_lr.shape[1] > y_pred_bb.shape[1]:
            y_pred_lr = y_pred_lr[:, :y_pred_bb.shape[1]]
    
    # Get top-k features
    top_features = get_topk_features_from_lr(lr_model, k=top_k)
    top_feature_names = [feature_names[i] for i in sorted(top_features)]
    
    # Get feature importance scores
    coefs = lr_model.coef_
    if coefs.ndim == 1:
        feature_importance = np.abs(coefs)
    else:
        feature_importance = np.max(np.abs(coefs), axis=0)
    
    top_feature_scores = [(feature_names[i], feature_importance[i]) for i in sorted(top_features)]
    top_feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get original values for top-k features (reverse transform to get meaningful values)
    try:
        x_test_unscaled = scaler.inverse_transform(x_test)
        
        import pandas as pd
        x_test_df = pd.DataFrame(x_test_unscaled, columns=feature_names)
        x_test_original = ht.reverse_transform(x_test_df)
        
        top_feature_values = []
        for i in sorted(top_features):
            feature_name = feature_names[i]
            
            original_value = x_test_unscaled[0, i]
            
            if '_' in feature_name and feature_name.split('_')[0] in x_test_original.columns:
                base_feature = feature_name.split('_')[0]
                original_value = x_test_original[base_feature].iloc[0]
            elif feature_name in x_test_original.columns:
                original_value = x_test_original[feature_name].iloc[0]
            
            top_feature_values.append((feature_name, original_value))
            
    except Exception as e:
        print(f"Warning: Could not fully reverse transform values, using unscaled values. Error: {e}")
        x_test_unscaled = scaler.inverse_transform(x_test)
        top_feature_values = []
        for i in sorted(top_features):
            feature_name = feature_names[i]
            original_value = x_test_unscaled[0, i]
            top_feature_values.append((feature_name, original_value))
    
    importance_order = {name: idx for idx, (name, _) in enumerate(top_feature_scores)}
    top_feature_values.sort(key=lambda x: importance_order[x[0]])
    
    return {
        'top_features': top_feature_names,
        'feature_scores': top_feature_scores,
        'feature_values': top_feature_values,
    }

###############################################################################
# COMMAND LINE INTERFACE
###############################################################################
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adversarially Bracketed Local Explanation (ABLE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python able.py --model MLP --dataset adult --test-index 20 --top-k 5
  python able.py --model TabNet --dataset car --test-index 5 --attack PGD
  python able.py --model TabTransformer --dataset covertype --test-index 10 --attack FGSM --top-k 3
  python able.py --model MLP --dataset adult --test-index 20 --attack ETDEEPFOOL --adversarial-pairs 30
  python able.py --model TabNet --dataset mushroom --test-index 2 --attack ETDEEPFOOL --radius 0.8
  python able.py --model MLP --dataset car --test-index 21 --attack HOPSKIPJUMP --adversarial-pairs 20
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['MLP', 'TabNet', 'TabTransformer'],
        help='Black-box model to use'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['credit', 'adult', 'breast_cancer', 'mushroom', 'car', 'covertype'],
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--test-index',
        type=int,
        required=True,
        help='Index of the test instance to explain'
    )
    
    parser.add_argument(
        '--attack',
        type=str,
        default='PGD',
        choices=['PGD', 'FGSM', 'ETDEEPFOOL', 'HOPSKIPJUMP'],
        help='Adversarial attack method (default: PGD)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top features to return (default: 5)'
    )
    
    parser.add_argument(
        '--adversarial-pairs',
        type=int,
        default=50,
        help='Number of adversarial pairs to generate (default: 50)'
    )
    
    parser.add_argument(
        '--radius',
        type=float,
        default=0.5,
        help='Neighborhood radius (default: 0.5)'
    )
    
    parser.add_argument(
        '--neighbors',
        type=int,
        default=100,
        help='Number of neighbors to generate (default: 100)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    
    return parser.parse_args()

def main():
    """Main function to run ABLE."""
    args = parse_arguments()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load dataset
    X_train, X_test, y_train, y_test, feature_names, scaler, ht, num_classes = load_dataset(args.dataset)
    input_dim = X_train.shape[1]
    
    # Check test index validity
    if args.test_index >= len(X_test):
        print(f"❌ Error: Test index {args.test_index} is out of range. Max index: {len(X_test)-1}")
        return
    
    # Create and train model
    classifier = create_and_train_model(
        args.model, X_train, y_train, X_test, y_test, 
        input_dim, num_classes, device
    )
    
    # Get test instance
    x_test = X_test[args.test_index]
    
    # Generate ABLE explanation
    result = generate_able_explanation(
        classifier=classifier,
        x_test=x_test,
        X_train=X_train,
        feature_names=feature_names,
        scaler=scaler,
        ht=ht,
        attack_method=args.attack,
        num_adversarial_pairs=args.adversarial_pairs,
        neighbor_radius=args.radius,
        num_neighbors=args.neighbors,
        top_k=args.top_k
    )
    
    if result:
        print(f"\nDataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Attack Method: ABLE_{args.attack}")
        print(f"Test Instance: {args.test_index}")
        print(f"Top-{args.top_k} Features:")
        for i, (feature_name, feature_value) in enumerate(result['feature_values'], 1):
            if isinstance(feature_value, (int, float)) and not isinstance(feature_value, bool):
                formatted_value = f"{feature_value:.4f}"
            else:
                formatted_value = str(feature_value)
            print(f"  {i}. {feature_name}: {formatted_value}")
    else:
        print("❌ Failed to generate ABLE explanation")

if __name__ == "__main__":
    main() 