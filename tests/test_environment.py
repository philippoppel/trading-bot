"""
Unit Tests für Trading Environments.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestAdvancedTradingEnv:
    """Tests für AdvancedTradingEnv."""

    @pytest.fixture
    def sample_df(self):
        """Erstellt Sample DataFrame mit Features."""
        np.random.seed(42)
        n = 500

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'close': close,
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n)
        })

        return df

    @pytest.fixture
    def env(self, sample_df):
        """Erstellt Trading Environment."""
        from src.environment.advanced_trading_env import AdvancedTradingEnv

        feature_cols = ['feature_1', 'feature_2', 'feature_3']

        return AdvancedTradingEnv(
            df=sample_df,
            feature_columns=feature_cols,
            initial_balance=10000.0,
            observation_window=20
        )

    def test_env_creation(self, env):
        """Test Environment-Erstellung."""
        assert env is not None
        assert env.initial_balance == 10000.0

    def test_reset(self, env):
        """Test Reset."""
        obs, info = env.reset()

        assert obs is not None
        assert len(obs.shape) == 1
        assert info['portfolio_value'] == 10000.0
        assert info['position'] == 0.0

    def test_action_space(self, env):
        """Test Action Space."""
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_step(self, env):
        """Test Step."""
        env.reset()

        # Buy action
        action = np.array([0.5])  # 50% long
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert info['position'] == 0.5

    def test_multiple_steps(self, env):
        """Test mehrere Steps."""
        env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        assert info['num_trades'] > 0

    def test_position_changes(self, env):
        """Test Position-Änderungen."""
        env.reset()

        # Open long
        env.step(np.array([0.5]))
        assert env.position == 0.5

        # Close and go short
        env.step(np.array([-0.5]))
        assert env.position == -0.5

        # Close
        env.step(np.array([0.0]))
        assert abs(env.position) < 0.01

    def test_fees_applied(self, env):
        """Test dass Gebühren angewendet werden."""
        env.reset()

        initial_fees = env.total_fees

        # Execute trade
        env.step(np.array([1.0]))

        assert env.total_fees > initial_fees


class TestAgentCreation:
    """Tests für Agent-Erstellung."""

    @pytest.fixture
    def env(self):
        """Erstellt einfaches Environment."""
        from src.environment.advanced_trading_env import AdvancedTradingEnv

        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n)
        })

        return AdvancedTradingEnv(
            df=df,
            feature_columns=['feature_1', 'feature_2'],
            observation_window=10
        )

    def test_create_sac_agent(self, env):
        """Test SAC Agent Erstellung."""
        from src.environment.advanced_trading_env import create_sac_agent

        agent = create_sac_agent(env, verbose=0)
        assert agent is not None

    def test_create_ddpg_agent(self, env):
        """Test DDPG Agent Erstellung."""
        from src.environment.advanced_trading_env import create_ddpg_agent

        agent = create_ddpg_agent(env, verbose=0)
        assert agent is not None

    def test_create_td3_agent(self, env):
        """Test TD3 Agent Erstellung."""
        from src.environment.advanced_trading_env import create_td3_agent

        agent = create_td3_agent(env, verbose=0)
        assert agent is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
