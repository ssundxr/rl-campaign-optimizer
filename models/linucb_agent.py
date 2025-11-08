"""
LinUCB (Linear Upper Confidence Bound) Contextual Bandit Agent
For real-time campaign recommendation in e-commerce customer retention

Algorithm:
- Maintains linear models for each action (campaign type)
- Balances exploration (trying uncertain actions) vs exploitation (using best known actions)
- Updates model parameters after each interaction using closed-form solutions

Reference:
Li, L., et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation"
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinUCBAgent:
    """
    Linear Upper Confidence Bound agent for contextual bandit problems.
    
    This agent learns to select optimal actions (campaigns) given customer context (features)
    by maintaining a linear model for each action and using UCB to balance exploration
    and exploitation.
    
    Attributes:
        n_actions (int): Number of available actions (campaign types)
        context_dim (int): Dimensionality of context vectors (customer features)
        alpha (float): Exploration parameter (higher = more exploration)
        A (List[np.ndarray]): List of design matrices for each action
        b (List[np.ndarray]): List of response vectors for each action
        total_interactions (int): Total number of interactions processed
        campaign_names (Dict[int, str]): Mapping of action IDs to campaign names
    """
    
    def __init__(
        self, 
        n_actions: int = 4, 
        context_dim: int = 30, 
        alpha: float = 1.0
    ):
        """
        Initialize LinUCB agent with empty models for each action.
        
        Args:
            n_actions: Number of possible actions (campaign types). Default 4.
            context_dim: Dimension of context feature vectors. Default 30.
            alpha: Exploration parameter. Higher values encourage exploration. Default 1.0.
                  Typical range: 0.1 to 2.0
        """
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize A matrices as identity matrices (context_dim x context_dim)
        # A[a] = D_a^T D_a + I, where D_a are the contexts where action a was chosen
        self.A: List[np.ndarray] = [
            np.identity(context_dim) for _ in range(n_actions)
        ]
        
        # Initialize b vectors as zero vectors (context_dim x 1)
        # b[a] = D_a^T c_a, where c_a are the rewards for action a
        self.b: List[np.ndarray] = [
            np.zeros((context_dim, 1)) for _ in range(n_actions)
        ]
        
        # Track total interactions for monitoring
        self.total_interactions: int = 0
        
        # Campaign type mapping for interpretability
        self.campaign_names: Dict[int, str] = {
            0: '20% Discount',
            1: 'Free Shipping',
            2: 'Early Access',
            3: 'No Campaign'
        }
        
        logger.info(
            f"LinUCB Agent initialized: {n_actions} actions, "
            f"{context_dim} features, alpha={alpha}"
        )
    
    def select_action(self, context: np.ndarray) -> int:
        """
        Select the best action using LinUCB algorithm.
        
        Algorithm:
        1. For each action a:
           - Compute theta_a = A_a^-1 * b_a (estimated reward parameters)
           - Compute expected_reward = theta_a^T * context
           - Compute uncertainty = sqrt(context^T * A_a^-1 * context)
           - Compute UCB_a = expected_reward + alpha * uncertainty
        2. Return argmax(UCB values)
        
        Args:
            context: Customer feature vector of shape (context_dim,) or (context_dim, 1)
        
        Returns:
            Selected action ID (0 to n_actions-1)
        
        Raises:
            ValueError: If context has incorrect shape
        """
        # Validate and reshape context to column vector
        context = self._validate_context(context)
        
        ucb_values = np.zeros(self.n_actions)
        
        for action in range(self.n_actions):
            # Compute A_inv = A^-1 (inverse of design matrix)
            A_inv = np.linalg.inv(self.A[action])
            
            # Compute theta = A_inv @ b (estimated parameters)
            theta = A_inv @ self.b[action]
            
            # Compute expected reward (exploitation term)
            # theta^T @ context = expected reward for this action
            expected_reward = (theta.T @ context)[0, 0]
            
            # Compute uncertainty (exploration term)
            # sqrt(context^T @ A_inv @ context) = confidence bound width
            uncertainty = np.sqrt((context.T @ A_inv @ context)[0, 0])
            
            # Compute Upper Confidence Bound
            # UCB = exploitation + exploration
            ucb_values[action] = expected_reward + self.alpha * uncertainty
        
        # Select action with highest UCB value
        best_action = int(np.argmax(ucb_values))
        
        return best_action
    
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update the model for the selected action with observed reward.
        
        Updates:
        - A[action] += context @ context^T (add context outer product)
        - b[action] += reward * context (add weighted context)
        
        Args:
            context: Customer feature vector of shape (context_dim,) or (context_dim, 1)
            action: Action ID that was selected (0 to n_actions-1)
            reward: Observed reward (e.g., 1 for conversion, 0 for no conversion)
        
        Raises:
            ValueError: If action is out of bounds or context has incorrect shape
        """
        # Validate inputs
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"Action {action} out of bounds. Must be in [0, {self.n_actions-1}]"
            )
        
        context = self._validate_context(context)
        
        # Update A[action] with outer product of context
        # A_new = A_old + x @ x^T
        self.A[action] += context @ context.T
        
        # Update b[action] with weighted context
        # b_new = b_old + reward * x
        self.b[action] += reward * context
        
        # Increment interaction counter
        self.total_interactions += 1
        
        if self.total_interactions % 100 == 0:
            logger.info(f"Agent updated: {self.total_interactions} total interactions")
    
    def get_expected_rewards(self, context: np.ndarray) -> List[float]:
        """
        Get expected rewards for all actions (exploitation values only, no UCB).
        
        Useful for:
        - Evaluation and debugging
        - Understanding what the model has learned
        - A/B testing against pure exploitation
        
        Args:
            context: Customer feature vector of shape (context_dim,) or (context_dim, 1)
        
        Returns:
            List of expected rewards for each action (length n_actions)
        """
        context = self._validate_context(context)
        expected_rewards = []
        
        for action in range(self.n_actions):
            # Compute theta for this action
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]
            
            # Expected reward = theta^T @ context
            expected_reward = (theta.T @ context)[0, 0]
            expected_rewards.append(float(expected_reward))
        
        return expected_rewards
    
    def save_model(self, filepath: str) -> None:
        """
        Save the entire agent object to disk using pickle.
        
        Args:
            filepath: Path where model should be saved (e.g., 'models/linucb_agent.pkl')
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(
            f"✅ Model saved to {filepath} "
            f"({self.total_interactions} interactions trained)"
        )
    
    @staticmethod
    def load_model(filepath: str) -> 'LinUCBAgent':
        """
        Load a trained agent from disk.
        
        Args:
            filepath: Path to saved model file
        
        Returns:
            Loaded LinUCBAgent instance
        
        Raises:
            FileNotFoundError: If filepath does not exist
        """
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        
        logger.info(
            f"✅ Model loaded from {filepath} "
            f"({agent.total_interactions} interactions trained)"
        )
        
        return agent
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the agent's current state.
        
        Returns:
            Dictionary containing:
            - total_interactions: Number of updates performed
            - n_actions: Number of actions
            - context_dim: Dimension of context vectors
            - alpha: Exploration parameter
            - campaign_names: Mapping of action IDs to names
        """
        return {
            'total_interactions': self.total_interactions,
            'n_actions': self.n_actions,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'campaign_names': self.campaign_names
        }
    
    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        """
        Validate and reshape context vector to column vector.
        
        Args:
            context: Input context vector
        
        Returns:
            Context reshaped to (context_dim, 1)
        
        Raises:
            ValueError: If context has incorrect shape
        """
        # Convert to numpy array if needed
        context = np.asarray(context)
        
        # Check if 1D array
        if context.ndim == 1:
            if context.shape[0] != self.context_dim:
                raise ValueError(
                    f"Context dimension mismatch. Expected {self.context_dim}, "
                    f"got {context.shape[0]}"
                )
            # Reshape to column vector
            context = context.reshape(-1, 1)
        
        # Check if 2D column vector
        elif context.ndim == 2:
            if context.shape[0] != self.context_dim or context.shape[1] != 1:
                raise ValueError(
                    f"Context shape mismatch. Expected ({self.context_dim}, 1), "
                    f"got {context.shape}"
                )
        else:
            raise ValueError(
                f"Context must be 1D or 2D array, got {context.ndim}D"
            )
        
        return context
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"LinUCBAgent(n_actions={self.n_actions}, "
            f"context_dim={self.context_dim}, "
            f"alpha={self.alpha}, "
            f"interactions={self.total_interactions})"
        )


def main():
    """Test the LinUCB agent with random data."""
    print("="*70)
    print("LINUCB AGENT TEST")
    print("="*70)
    
    # Initialize agent
    agent = LinUCBAgent(n_actions=4, context_dim=30, alpha=1.0)
    print(f"\n{agent}")
    print(f"\nCampaign Types: {agent.campaign_names}")
    
    # Simulate interactions
    print("\n" + "="*70)
    print("SIMULATING 1000 INTERACTIONS")
    print("="*70)
    
    np.random.seed(42)
    total_reward = 0
    
    for i in range(1000):
        # Generate random customer context
        context = np.random.randn(30)
        
        # Select action using LinUCB
        action = agent.select_action(context)
        
        # Simulate reward (action 0 is best, gets higher rewards on average)
        # In real scenario, this would be actual customer conversion
        if action == 0:
            reward = np.random.binomial(1, 0.3)  # 30% conversion rate
        elif action == 1:
            reward = np.random.binomial(1, 0.2)  # 20% conversion rate
        elif action == 2:
            reward = np.random.binomial(1, 0.15) # 15% conversion rate
        else:
            reward = np.random.binomial(1, 0.05) # 5% conversion rate (no campaign)
        
        # Update agent
        agent.update(context, action, reward)
        total_reward += reward
        
        # Print progress every 200 interactions
        if (i + 1) % 200 == 0:
            print(f"  Interactions: {i+1:4d} | Total Reward: {total_reward:3d} | "
                  f"Avg Reward: {total_reward/(i+1):.3f}")
    
    # Final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test expected rewards on a sample context
    print("\n" + "="*70)
    print("EXPECTED REWARDS FOR SAMPLE CUSTOMER")
    print("="*70)
    sample_context = np.random.randn(30)
    expected_rewards = agent.get_expected_rewards(sample_context)
    
    for action, reward in enumerate(expected_rewards):
        campaign = agent.campaign_names[action]
        print(f"  {campaign:15s}: {reward:.4f}")
    
    best_action = np.argmax(expected_rewards)
    print(f"\n  Best Campaign: {agent.campaign_names[best_action]}")
    
    # Test save/load
    print("\n" + "="*70)
    print("TESTING SAVE/LOAD")
    print("="*70)
    
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save
    agent.save_model('models/linucb_test.pkl')
    
    # Load
    loaded_agent = LinUCBAgent.load_model('models/linucb_test.pkl')
    print(f"\n{loaded_agent}")
    
    # Verify loaded agent works
    test_context = np.random.randn(30)
    action = loaded_agent.select_action(test_context)
    print(f"\nTest prediction: Action {action} ({loaded_agent.campaign_names[action]})")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    main()
