"""
Erweiterte Trading-Umgebung für Reinforcement Learning.

Features:
- Kontinuierlicher Action Space mit Position Sizing
- Sharpe-basierte Reward-Funktion
- Support für DDPG, SAC, TD3
- Marktregime-Awareness
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any
from collections import deque
from loguru import logger


class AdvancedTradingEnv(gym.Env):
    """
    Erweiterte Trading-Umgebung mit kontinuierlichem Action Space.

    Action Space:
        Box(-1, 1, shape=(1,))
        -1 = Full Short, 0 = No Position, 1 = Full Long

    Observation Space:
        Features + Position Info + Portfolio Info + Market Regime

    Reward:
        Sharpe Ratio basierte risikoadjustierte Rendite
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df,
        feature_columns: list,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        observation_window: int = 60,
        reward_scaling: float = 1.0,
        sharpe_window: int = 20,
        max_position: float = 1.0,
        leverage: float = 1.0,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            df: DataFrame mit Preis- und Feature-Daten
            feature_columns: Liste der Feature-Spalten
            initial_balance: Startkapital
            fee_rate: Handelsgebühren
            slippage_rate: Slippage-Rate
            observation_window: Anzahl historischer Schritte
            reward_scaling: Skalierungsfaktor für Rewards
            sharpe_window: Fenster für Sharpe-Berechnung
            max_position: Maximale Positionsgröße (als Anteil)
            leverage: Hebel für Trading
            render_mode: Render-Modus
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.observation_window = observation_window
        self.reward_scaling = reward_scaling
        self.sharpe_window = sharpe_window
        self.max_position = max_position
        self.leverage = leverage
        self.render_mode = render_mode

        # Preise extrahieren
        self.prices = self.df['close'].values
        self.features = self.df[feature_columns].values

        # Action Space: Kontinuierlich von -1 (short) bis 1 (long)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation Space
        # Features + [position, unrealized_pnl, portfolio_value, sharpe_estimate]
        n_features = len(feature_columns)
        obs_dim = (observation_window * n_features) + 4

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State Variablen
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset der Umgebung."""
        super().reset(seed=seed)

        self.current_step = self.observation_window
        self.balance = self.initial_balance
        self.position = 0.0  # -1 bis 1 (normalisiert)
        self.position_value = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0

        # Tracking
        self.portfolio_values = deque([self.initial_balance], maxlen=self.sharpe_window + 1)
        self.returns_history = deque([0.0], maxlen=self.sharpe_window)
        self.trades = []
        self.total_fees = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Führt einen Schritt aus.

        Args:
            action: Zielposition [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Extrahiere Action
        target_position = float(np.clip(action[0], -1.0, 1.0))

        # Aktueller Preis
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self._calculate_portfolio_value(current_price)

        # Position anpassen
        position_change = target_position - self.position

        if abs(position_change) > 0.01:  # Mindest-Änderung
            self._execute_trade(position_change, current_price)

        # Update unrealized PnL
        self._update_unrealized_pnl(current_price)

        # Neuer Portfolio-Wert
        new_portfolio_value = self._calculate_portfolio_value(current_price)

        # Return berechnen
        if prev_portfolio_value > 0:
            step_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            step_return = 0.0

        self.returns_history.append(step_return)
        self.portfolio_values.append(new_portfolio_value)

        # Reward berechnen (Sharpe-basiert)
        reward = self._calculate_reward(step_return)

        # Nächster Schritt
        self.current_step += 1

        # Episode beendet?
        terminated = self.current_step >= len(self.prices) - 1
        truncated = new_portfolio_value <= self.initial_balance * 0.5  # -50% Drawdown

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_trade(self, position_change: float, price: float):
        """Führt Trade aus."""
        # Trade-Wert
        trade_value = abs(position_change) * self.initial_balance * self.leverage

        # Slippage
        if position_change > 0:  # Buying
            exec_price = price * (1 + self.slippage_rate)
        else:  # Selling
            exec_price = price * (1 - self.slippage_rate)

        # Gebühren
        fees = trade_value * self.fee_rate
        self.total_fees += fees
        self.balance -= fees

        # Position aktualisieren
        old_position = self.position
        self.position += position_change

        # Entry Price aktualisieren (gewichteter Durchschnitt)
        if abs(self.position) > 0.01:
            if old_position * position_change > 0:  # Gleiche Richtung
                total_value = abs(old_position) * self.entry_price + abs(position_change) * exec_price
                self.entry_price = total_value / abs(self.position)
            else:  # Richtungswechsel oder Reduzierung
                if abs(self.position) > abs(old_position):
                    self.entry_price = exec_price
        else:
            self.entry_price = 0.0

        # Position-Wert
        self.position_value = abs(self.position) * self.initial_balance * self.leverage

        # Trade aufzeichnen
        self.trades.append({
            'step': self.current_step,
            'price': exec_price,
            'position_change': position_change,
            'new_position': self.position,
            'fees': fees
        })

    def _update_unrealized_pnl(self, current_price: float):
        """Aktualisiert unrealized PnL."""
        if abs(self.position) > 0.01 and self.entry_price > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * self.position_value * price_change
        else:
            self.unrealized_pnl = 0.0

    def _calculate_portfolio_value(self, price: float) -> float:
        """Berechnet aktuellen Portfolio-Wert."""
        self._update_unrealized_pnl(price)
        return self.balance + self.position_value + self.unrealized_pnl

    def _calculate_reward(self, step_return: float) -> float:
        """
        Berechnet Sharpe-basierten Reward.

        Kombiniert:
        - Risikoadjustierte Rendite (Sharpe)
        - Drawdown-Penalty
        - Trade-Effizienz
        """
        # Basis: Step Return
        reward = step_return * 100  # Skalieren

        # Sharpe-Komponente
        if len(self.returns_history) >= self.sharpe_window:
            returns = np.array(self.returns_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252 * 24)  # Annualisiert
                reward += sharpe * 0.1

        # Drawdown-Penalty
        if len(self.portfolio_values) > 1:
            peak = max(self.portfolio_values)
            current = self.portfolio_values[-1]
            drawdown = (peak - current) / peak

            if drawdown > 0.1:  # > 10% Drawdown
                reward -= drawdown * 10

        # Inaktivitäts-Penalty
        if abs(self.position) < 0.01:
            reward -= 0.001

        return float(reward * self.reward_scaling)

    def _get_observation(self) -> np.ndarray:
        """Erstellt Observation."""
        # Feature-Fenster
        start_idx = self.current_step - self.observation_window
        feature_window = self.features[start_idx:self.current_step].flatten()

        # Portfolio-Info
        current_price = self.prices[self.current_step]
        portfolio_value = self._calculate_portfolio_value(current_price)

        # Sharpe-Estimate
        if len(self.returns_history) >= 5:
            returns = np.array(self.returns_history)
            sharpe_estimate = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe_estimate = 0.0

        # Normalisierte Portfolio-Features
        portfolio_features = np.array([
            self.position,  # [-1, 1]
            self.unrealized_pnl / self.initial_balance,  # Normalisiert
            portfolio_value / self.initial_balance - 1,  # Return
            sharpe_estimate
        ])

        observation = np.concatenate([feature_window, portfolio_features])

        return observation.astype(np.float32)

    def _get_info(self) -> dict:
        """Erstellt Info-Dictionary."""
        current_price = self.prices[self.current_step]
        portfolio_value = self._calculate_portfolio_value(current_price)

        return {
            'step': self.current_step,
            'price': current_price,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'return': (portfolio_value / self.initial_balance - 1) * 100,
            'total_fees': self.total_fees,
            'num_trades': len(self.trades),
            'unrealized_pnl': self.unrealized_pnl
        }

    def render(self):
        """Rendert die Umgebung."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']}: "
                  f"Price={info['price']:.2f}, "
                  f"Position={info['position']:.2f}, "
                  f"Portfolio={info['portfolio_value']:.2f}, "
                  f"Return={info['return']:.2f}%")


def create_ddpg_agent(env: AdvancedTradingEnv, **kwargs):
    """
    Erstellt DDPG Agent für kontinuierlichen Action Space.

    Args:
        env: Trading Environment
        **kwargs: Zusätzliche Parameter für DDPG

    Returns:
        DDPG Agent
    """
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    default_params = {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'action_noise': action_noise,
        'verbose': 1
    }

    default_params.update(kwargs)

    return DDPG(env=env, **default_params)


def create_sac_agent(env: AdvancedTradingEnv, **kwargs):
    """
    Erstellt SAC Agent für kontinuierlichen Action Space.

    Args:
        env: Trading Environment
        **kwargs: Zusätzliche Parameter für SAC

    Returns:
        SAC Agent
    """
    from stable_baselines3 import SAC

    default_params = {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'ent_coef': 'auto',
        'verbose': 1
    }

    default_params.update(kwargs)

    return SAC(env=env, **default_params)


def create_td3_agent(env: AdvancedTradingEnv, **kwargs):
    """
    Erstellt TD3 Agent für kontinuierlichen Action Space.

    Args:
        env: Trading Environment
        **kwargs: Zusätzliche Parameter für TD3

    Returns:
        TD3 Agent
    """
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    default_params = {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'action_noise': action_noise,
        'policy_delay': 2,
        'verbose': 1
    }

    default_params.update(kwargs)

    return TD3(env=env, **default_params)
