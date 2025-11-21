"""
Gymnasium environment for cryptocurrency trading with RL.
"""

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger()


class Actions(IntEnum):
    """Trading actions for discrete action space."""
    HOLD = 0
    BUY = 1
    SELL = 2


class CryptoTradingEnv(gym.Env):
    """
    Gymnasium environment for cryptocurrency trading.

    Observation Space:
        - Historical price features (normalized)
        - Portfolio state (balance, position, unrealized PnL)

    Action Space (discrete):
        - 0: Hold
        - 1: Buy
        - 2: Sell

    Reward:
        - Based on portfolio value change (profit/loss)
        - Can be adjusted for risk (Sharpe ratio, etc.)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
        slippage: float = 0.0005,
        window_size: int = 48,
        max_position: float = 1.0,
        render_mode: str | None = None
    ):
        """
        Initialize the trading environment.

        Args:
            df: Processed DataFrame with features and prices
            feature_columns: List of feature column names to use
            initial_balance: Starting balance in quote currency (USDT)
            trading_fee: Trading fee as decimal (0.001 = 0.1%)
            slippage: Slippage as decimal (0.0005 = 0.05%)
            window_size: Number of time steps in observation window
            max_position: Maximum position size (1.0 = 100% of portfolio)
            render_mode: Rendering mode
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.window_size = window_size
        self.max_position = max_position
        self.render_mode = render_mode

        # Validate data
        if len(self.df) < self.window_size + 1:
            raise ValueError(f"DataFrame must have at least {self.window_size + 1} rows")

        for col in self.feature_columns:
            if col not in self.df.columns:
                raise ValueError(f"Feature column '{col}' not found in DataFrame")

        # Number of features per time step
        self.n_features = len(self.feature_columns)

        # Define action space (discrete: hold, buy, sell)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Features: (window_size, n_features) flattened + portfolio state (3,)
        obs_shape = self.window_size * self.n_features + 3  # 3 = balance_ratio, position, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float32
        )

        # Episode state
        self._reset_state()

        logger.info(
            f"Initialized CryptoTradingEnv: {len(self.df)} steps, "
            f"{self.n_features} features, window={self.window_size}"
        )

    def _reset_state(self) -> None:
        """Reset the episode state."""
        self.balance = self.initial_balance
        self.position = 0.0  # Amount of crypto held
        self.position_value = 0.0  # Value when position was opened
        self.current_step = self.window_size
        self.done = False

        # Tracking metrics
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.returns = []  # Track returns for Sharpe calculation
        self.total_trades = 0
        self.winning_trades = 0

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            Observation array
        """
        # Get feature window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        features = self.df[self.feature_columns].iloc[start_idx:end_idx].values
        features = features.flatten().astype(np.float32)

        # Get portfolio state
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()

        # Normalize portfolio state
        balance_ratio = self.balance / self.initial_balance
        position_ratio = (self.position * current_price) / self.initial_balance if self.initial_balance > 0 else 0

        # Unrealized PnL ratio
        if self.position > 0:
            unrealized_pnl = (current_price * self.position - self.position_value) / self.initial_balance
        else:
            unrealized_pnl = 0.0

        portfolio_state = np.array([balance_ratio, position_ratio, unrealized_pnl], dtype=np.float32)

        # Combine features and portfolio state
        observation = np.concatenate([features, portfolio_state])

        return observation

    def _get_current_price(self) -> float:
        """Get the current close price."""
        return float(self.df["close"].iloc[self.current_step])

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_price = self._get_current_price()
        return self.balance + self.position * current_price

    def _execute_trade(self, action: int) -> tuple[float, dict]:
        """
        Execute a trading action.

        Args:
            action: Trading action (0=hold, 1=buy, 2=sell)

        Returns:
            Tuple of (reward, info_dict)
        """
        current_price = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value()

        info = {
            "action": Actions(action).name,
            "price": current_price,
            "trade_executed": False
        }

        if action == Actions.BUY and self.position == 0:
            # Buy: use all available balance
            amount_to_spend = self.balance * self.max_position

            # Apply slippage (buy at slightly higher price)
            execution_price = current_price * (1 + self.slippage)

            # Calculate amount of crypto to buy (after fee)
            fee = amount_to_spend * self.trading_fee
            amount_to_buy = (amount_to_spend - fee) / execution_price

            # Execute trade
            self.position = amount_to_buy
            self.position_value = amount_to_spend - fee
            self.balance -= amount_to_spend

            info["trade_executed"] = True
            info["trade_type"] = "BUY"
            info["amount"] = amount_to_buy
            info["execution_price"] = execution_price
            info["fee"] = fee

            self.total_trades += 1
            logger.debug(f"BUY: {amount_to_buy:.6f} @ {execution_price:.2f}")

        elif action == Actions.SELL and self.position > 0:
            # Sell: sell entire position
            # Apply slippage (sell at slightly lower price)
            execution_price = current_price * (1 - self.slippage)

            # Calculate proceeds
            gross_proceeds = self.position * execution_price
            fee = gross_proceeds * self.trading_fee
            net_proceeds = gross_proceeds - fee

            # Calculate trade profit
            trade_profit = net_proceeds - self.position_value

            # Execute trade
            self.balance += net_proceeds

            info["trade_executed"] = True
            info["trade_type"] = "SELL"
            info["amount"] = self.position
            info["execution_price"] = execution_price
            info["fee"] = fee
            info["profit"] = trade_profit

            if trade_profit > 0:
                self.winning_trades += 1

            self.trades.append({
                "step": self.current_step,
                "profit": trade_profit,
                "entry_value": self.position_value,
                "exit_value": net_proceeds
            })

            self.total_trades += 1
            self.position = 0.0
            self.position_value = 0.0

            logger.debug(f"SELL: @ {execution_price:.2f}, profit: {trade_profit:.2f}")

        # Calculate reward based on portfolio value change
        new_portfolio_value = self._get_portfolio_value()

        # Calculate return for this step
        step_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns.append(step_return)

        # Sharpe-ratio inspired reward
        # Use rolling window for risk-adjusted return
        settings = get_settings()

        if len(self.returns) >= 20:
            # Calculate rolling Sharpe-like reward
            recent_returns = np.array(self.returns[-20:])
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns) + 1e-8  # Avoid division by zero

            # Sharpe-like reward: mean / std
            sharpe_reward = mean_return / std_return

            # Combine immediate return with risk-adjusted component
            reward = 0.5 * step_return + 0.5 * sharpe_reward * 0.1
        else:
            # Not enough history, use simple return
            reward = step_return

        # Penalize excessive trading (to reduce overtrading)
        if info["trade_executed"]:
            reward -= 0.0001  # Small penalty for each trade

        # Apply reward scaling from settings
        reward *= settings.environment.reward_scaling

        info["portfolio_value"] = new_portfolio_value
        info["balance"] = self.balance
        info["position"] = self.position

        return reward, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Trading action

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Execute the trade
        reward, info = self._execute_trade(action)

        # Move to next time step
        self.current_step += 1

        # Track portfolio value
        self.portfolio_values.append(self._get_portfolio_value())

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Additional termination: if portfolio value drops too low
        if self._get_portfolio_value() < self.initial_balance * 0.1:  # Lost 90%
            terminated = True
            info["early_termination"] = "portfolio_depleted"

        if terminated:
            self.done = True
            info["final_portfolio_value"] = self._get_portfolio_value()
            info["total_return"] = (self._get_portfolio_value() - self.initial_balance) / self.initial_balance
            info["total_trades"] = self.total_trades
            info["win_rate"] = self.winning_trades / max(self.total_trades, 1)
            info["sharpe_ratio"] = self._calculate_sharpe_ratio()

        # Get new observation
        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self._reset_state()

        observation = self._get_observation()
        info = {
            "initial_balance": self.initial_balance,
            "starting_step": self.current_step
        }

        return observation, info

    def render(self) -> None:
        """Render the environment state."""
        if self.render_mode == "human":
            current_price = self._get_current_price()
            portfolio_value = self._get_portfolio_value()
            print(
                f"Step {self.current_step}: "
                f"Price={current_price:.2f}, "
                f"Balance={self.balance:.2f}, "
                f"Position={self.position:.6f}, "
                f"Portfolio={portfolio_value:.2f}"
            )

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the Sharpe ratio for the episode.

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(self.portfolio_values) < 2:
            return 0.0

        # Calculate returns
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize (assuming hourly data, ~8760 hours/year)
        periods_per_year = 8760
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return
        sharpe_annualized = sharpe * np.sqrt(periods_per_year)

        return float(sharpe_annualized)

    def get_episode_stats(self) -> dict:
        """
        Get comprehensive statistics for the episode.

        Returns:
            Dictionary with episode statistics
        """
        portfolio_values = np.array(self.portfolio_values)

        # Calculate max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        return {
            "final_value": self._get_portfolio_value(),
            "total_return": (self._get_portfolio_value() - self.initial_balance) / self.initial_balance,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "avg_trade_profit": np.mean([t["profit"] for t in self.trades]) if self.trades else 0,
            "total_steps": self.current_step - self.window_size
        }


def create_env(
    df: pd.DataFrame,
    feature_columns: list[str],
    **kwargs
) -> CryptoTradingEnv:
    """
    Factory function to create a trading environment.

    Args:
        df: Processed DataFrame with features
        feature_columns: Feature columns to use
        **kwargs: Additional arguments for CryptoTradingEnv

    Returns:
        Configured trading environment
    """
    settings = get_settings()

    # Set defaults from config
    defaults = {
        "initial_balance": settings.trading.initial_balance,
        "trading_fee": settings.trading.trading_fee,
        "slippage": settings.trading.slippage,
        "window_size": settings.environment.observation_window,
        "max_position": settings.environment.max_position
    }

    # Override with provided kwargs
    defaults.update(kwargs)

    return CryptoTradingEnv(df, feature_columns, **defaults)
