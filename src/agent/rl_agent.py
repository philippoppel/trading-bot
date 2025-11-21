"""
Reinforcement Learning agent for cryptocurrency trading.
Uses Stable-Baselines3 for training PPO and other algorithms.
"""

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config.settings import get_settings
from src.environment.trading_env import CryptoTradingEnv, create_env
from src.utils.logger import get_logger

logger = get_logger()


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading metrics during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_returns = []

    def _on_step(self) -> bool:
        # Log custom metrics when episode ends
        for info in self.locals.get("infos", []):
            if "final_portfolio_value" in info:
                self.episode_returns.append(info["total_return"])

                if self.verbose > 0 and len(self.episode_returns) % 10 == 0:
                    avg_return = np.mean(self.episode_returns[-10:])
                    logger.info(
                        f"Episode {len(self.episode_returns)}: "
                        f"Avg Return={avg_return*100:.2f}%, "
                        f"Trades={info.get('total_trades', 0)}, "
                        f"Win Rate={info.get('win_rate', 0)*100:.1f}%"
                    )

        return True

    def _on_training_end(self) -> None:
        if self.episode_returns:
            logger.info(
                f"Training complete. Episodes: {len(self.episode_returns)}, "
                f"Avg Return: {np.mean(self.episode_returns)*100:.2f}%, "
                f"Best: {max(self.episode_returns)*100:.2f}%, "
                f"Worst: {min(self.episode_returns)*100:.2f}%"
            )


class TradingAgent:
    """
    RL trading agent using Stable-Baselines3.

    Supports PPO, DQN, and A2C algorithms.
    """

    ALGORITHMS = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C
    }

    def __init__(
        self,
        env: CryptoTradingEnv | None = None,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        **kwargs
    ):
        """
        Initialize the trading agent.

        Args:
            env: Trading environment
            algorithm: RL algorithm to use
            policy: Policy network type
            **kwargs: Additional arguments for the algorithm
        """
        self.settings = get_settings()
        self.algorithm_name = algorithm
        self.policy = policy

        # Get algorithm class
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.ALGORITHMS.keys())}")

        self.algorithm_class = self.ALGORITHMS[algorithm]

        # Set up environment
        self.env = env
        self.model = None

        # Store kwargs for model creation
        self.model_kwargs = kwargs

        logger.info(f"Initialized TradingAgent with {algorithm}")

    def create_model(self, env: CryptoTradingEnv | None = None) -> None:
        """
        Create the RL model.

        Args:
            env: Environment to use (uses self.env if None)
        """
        if env is not None:
            self.env = env

        if self.env is None:
            raise ValueError("No environment provided")

        # Wrap environment
        vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        # Get hyperparameters from config
        if self.algorithm_name == "PPO":
            ppo_config = self.settings.agent.ppo
            default_kwargs = {
                "learning_rate": ppo_config.learning_rate,
                "n_steps": ppo_config.n_steps,
                "batch_size": ppo_config.batch_size,
                "n_epochs": ppo_config.n_epochs,
                "gamma": ppo_config.gamma,
                "gae_lambda": ppo_config.gae_lambda,
                "clip_range": ppo_config.clip_range,
                "ent_coef": ppo_config.ent_coef,
                "vf_coef": ppo_config.vf_coef,
                "max_grad_norm": ppo_config.max_grad_norm,
                "verbose": 1,
                "tensorboard_log": str(self.settings.get_log_dir() / "tensorboard"),
                # Larger network architecture for better feature extraction
                "policy_kwargs": {
                    "net_arch": dict(pi=[256, 256], vf=[256, 256])
                }
            }
        else:
            default_kwargs = {
                "verbose": 1,
                "tensorboard_log": str(self.settings.get_log_dir() / "tensorboard")
            }

        # Override with custom kwargs
        default_kwargs.update(self.model_kwargs)

        # Create model
        self.model = self.algorithm_class(
            self.policy,
            vec_env,
            **default_kwargs
        )

        logger.info(f"Created {self.algorithm_name} model with policy {self.policy}")

    def train(
        self,
        total_timesteps: int | None = None,
        eval_env: CryptoTradingEnv | None = None,
        callbacks: list[BaseCallback] | None = None
    ) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total timesteps to train
            eval_env: Environment for evaluation
            callbacks: Additional callbacks
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        if total_timesteps is None:
            total_timesteps = self.settings.training.total_timesteps

        # Set up callbacks
        callback_list = [TradingCallback(verbose=1)]

        # Checkpoint callback
        model_dir = self.settings.get_model_dir()
        checkpoint_callback = CheckpointCallback(
            save_freq=self.settings.training.save_freq,
            save_path=str(model_dir / "checkpoints"),
            name_prefix="trading_model"
        )
        callback_list.append(checkpoint_callback)

        # Evaluation callback
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=str(model_dir / "best"),
                log_path=str(self.settings.get_log_dir() / "eval"),
                eval_freq=self.settings.training.eval_freq,
                n_eval_episodes=self.settings.training.n_eval_episodes,
                deterministic=True
            )
            callback_list.append(eval_callback)

        # Add custom callbacks
        if callbacks:
            callback_list.extend(callbacks)

        logger.info(f"Starting training for {total_timesteps} timesteps")

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=self.settings.training.log_interval
        )

        logger.info("Training complete")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[int, Any]:
        """
        Predict action for given observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, states)
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")

        action, states = self.model.predict(observation, deterministic=deterministic)
        return int(action), states

    def evaluate(
        self,
        env: CryptoTradingEnv,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> dict:
        """
        Evaluate the agent on an environment.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")

        # Evaluate
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            return_episode_rewards=True
        )

        # Run episodes to get detailed stats
        all_stats = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            stats = env.get_episode_stats()
            all_stats.append(stats)

        # Aggregate stats
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_return": np.mean([s["total_return"] for s in all_stats]),
            "mean_sharpe": np.mean([s["sharpe_ratio"] for s in all_stats]),
            "mean_max_drawdown": np.mean([s["max_drawdown"] for s in all_stats]),
            "mean_trades": np.mean([s["total_trades"] for s in all_stats]),
            "mean_win_rate": np.mean([s["win_rate"] for s in all_stats])
        }

        logger.info(
            f"Evaluation ({n_episodes} episodes): "
            f"Return={results['mean_return']*100:.2f}%, "
            f"Sharpe={results['mean_sharpe']:.2f}, "
            f"MaxDD={results['mean_max_drawdown']*100:.2f}%"
        )

        return results

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load a model from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = self.algorithm_class.load(str(path))
        logger.info(f"Model loaded from {path}")


def train_agent(
    train_df,
    val_df,
    feature_columns: list[str],
    save_path: str | Path | None = None
) -> TradingAgent:
    """
    Convenience function to train an agent.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        feature_columns: Feature columns to use
        save_path: Path to save the trained model

    Returns:
        Trained agent
    """
    settings = get_settings()

    # Create environments
    train_env = create_env(train_df, feature_columns)
    val_env = create_env(val_df, feature_columns)

    # Create agent
    agent = TradingAgent(
        algorithm=settings.agent.algorithm,
        policy=settings.agent.policy
    )

    # Create model
    agent.create_model(train_env)

    # Train
    agent.train(eval_env=val_env)

    # Save model
    if save_path is None:
        save_path = settings.get_model_dir() / "final_model"

    agent.save(save_path)

    # Final evaluation
    logger.info("Final evaluation on validation set:")
    agent.evaluate(val_env)

    return agent
