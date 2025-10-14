"""
ABLE Metrics Module 

This module provides classes for calculating fidelity and Jaccard index metrics


Classes:
    - FidelityCalculator: Calculates fidelity metrics for ABLE models
    - JaccardCalculator: Calculates Jaccard index for feature importance comparison
"""

import numpy as np
from typing import Set, Union
from sklearn.linear_model import LogisticRegression


class FidelityCalculator:
    """
    Calculates fidelity metrics for ABLE models.
    
    Based on the adversarial_r2_score function from main_code.ipynb.
    """
    
    def __init__(self, blackbox_model, surrogate_model):
        """
        Initialize the fidelity calculator.
        
        Args:
            blackbox_model: The original black-box model
            surrogate_model: The ABLE model (trained on adversarial examples)
        """
        self.blackbox_model = blackbox_model
        self.surrogate_model = surrogate_model
    
    def adversarial_r2_score(self, X_data: np.ndarray) -> float:
        """
        Calculate R² score between black-box and ABLE predictions.
        
        
        Args:
            X_data: Data points to evaluate (adversarial examples or neighbors)
            
        Returns:
            R² score (higher is better, max = 1.0)
        """
        try:
            # Get black-box predictions
            if hasattr(self.blackbox_model, 'predict_proba'):
                bb_probs = self.blackbox_model.predict_proba(X_data)
            else:
                bb_probs = self.blackbox_model.predict(X_data)
            
            # Get ABLE predictions
            lr_probs = self.surrogate_model.predict_proba(X_data)
            
        
            ss_res = np.sum((bb_probs - lr_probs)**2)
            ss_tot = np.sum((bb_probs - np.mean(bb_probs))**2)
            
            return 1 - (ss_res / (ss_tot + 1e-9))
            
        except Exception as e:
            print(f"Error calculating adversarial R² score: {e}")
            return 0.0
    
    def calculate_able_fidelity(self, X_neighbors: np.ndarray) -> float:
        """
        Calculate ABLE fidelity on neighborhood data.
        
        
        Args:
            X_neighbors: Neighborhood samples around the test instance
            
        Returns:
            ABLE fidelity score
        """
        return self.adversarial_r2_score(X_neighbors)


class JaccardCalculator:
    """
    Calculates Jaccard index for comparing feature importance between different methods.
 
    """
    
    @staticmethod
    def jaccard_index(set_a: Set[int], set_b: Set[int]) -> float:
        """
        Calculate Jaccard index between two sets of feature indices.
        
        
        Args:
            set_a: First set of feature indices
            set_b: Second set of feature indices
            
        Returns:
            Jaccard index (0 = no overlap, 1 = identical sets)
        """
        inter = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(inter)/len(union) if len(union) > 0 else 0.
    
    @staticmethod
    def get_topk_features_from_lr(lr_model: LogisticRegression, k: int = 5) -> Set[int]:
        """
        Extract top-k features from logistic regression model.
        

        Args:
            lr_model: Trained logistic regression model
            k: Number of top features to extract
            
        Returns:
            Set of top-k feature indices
        """
        try:
            coefs = lr_model.coef_[0]
            top_inds = np.argsort(np.abs(coefs))[::-1][:k]
            return set(top_inds)
        except Exception as e:
            print(f"Error extracting top-k features from LR: {e}")
            return set()
    
    @staticmethod
    def get_topk_features_from_lime(exp, predicted_label: int, k: int = 5) -> Set[int]:
        """
        Extract top-k features from LIME explanation.
        
        
        Args:
            exp: LIME explanation object
            predicted_label: Predicted label for the explanation
            k: Number of top features to extract
            
        Returns:
            Set of top-k feature indices
        """
        try:
            local_map = exp.as_map()[predicted_label]
            local_map_sorted = sorted(local_map, key=lambda x: abs(x[1]), reverse=True)[:k]
            feat_inds = [fm[0] for fm in local_map_sorted]
            return set(feat_inds)
        except Exception as e:
            print(f"Error extracting top-k features from LIME: {e}")
            return set()
    
    def compare_lime_vs_able(self, lime_explanation, predicted_label: int, 
                                 able_model: LogisticRegression, k: int = 5) -> float:
        """
        Compare feature importance between LIME and ABLE explanations.
        

        
        Args:
            lime_explanation: LIME explanation object
            predicted_label: Predicted label
            able_model: ABLE logistic regression model
            k: Number of top features to compare
            
        Returns:
            Jaccard index between LIME and ABLE feature sets
        """
        # Get LIME features
        lime_features = self.get_topk_features_from_lime(lime_explanation, predicted_label, k)
        
        # Get ABLE features
        able_features = self.get_topk_features_from_lr(able_model, k)
        
        # Calculate Jaccard index
        return self.jaccard_index(lime_features, able_features)



if __name__ == "__main__":
    print("Metrics Module")
    print("=" * 50)
    print("Classes available:")
    print("- FidelityCalculator: Calculate fidelity metrics")
    print("- JaccardCalculator: Calculate Jaccard index for feature comparison")
    print("\nThese classes replicate the exact functionality:")
    print("- adversarial_r2_score() -> FidelityCalculator.adversarial_r2_score()")
    print("- jaccard_index() -> JaccardCalculator.jaccard_index()")
    print("- get_topk_features_from_lr() -> JaccardCalculator.get_topk_features_from_lr()")
    print("- get_topk_features_from_lime() -> JaccardCalculator.get_topk_features_from_lime()")
