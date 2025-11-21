"""
Configuration management for the Trading Bot.
Loads settings from YAML config and environment variables.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()


class TradingConfig(BaseModel):
    """Trading parameters configuration."""
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    slippage: float = 0.0005


class DataConfig(BaseModel):
    """Data collection configuration."""
    history_days: int = 365
    update_interval: int = 3600
    cache_dir: str = "data"


class IndicatorConfig(BaseModel):
    """Technical indicator configuration."""
    rsi: dict = {"period": 14}
    macd: dict = {"fast": 12, "slow": 26, "signal": 9}
    bollinger: dict = {"period": 20, "std": 2}
    atr: dict = {"period": 14}
    volume_sma: dict = {"period": 20}


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    lookback_periods: list[int] = [7, 14, 30, 60]
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)


class EnvironmentConfig(BaseModel):
    """RL Environment configuration."""
    observation_window: int = 48
    action_space: str = "discrete"
    reward_scaling: float = 1.0
    max_position: float = 1.0


class PPOConfig(BaseModel):
    """PPO algorithm hyperparameters."""
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class AgentConfig(BaseModel):
    """RL Agent configuration."""
    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    ppo: PPOConfig = Field(default_factory=PPOConfig)


class TrainingConfig(BaseModel):
    """Training configuration."""
    total_timesteps: int = 100000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    save_freq: int = 25000
    log_interval: int = 1
    seed: int = 42
    model_dir: str = "models"
    log_dir: str = "logs"
    tensorboard_dir: str = "logs/tensorboard"


class RiskManagementConfig(BaseModel):
    """Risk management configuration."""
    max_position_pct: float = 0.5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    daily_loss_limit: float = 0.10


class PaperTradingConfig(BaseModel):
    """Paper trading configuration."""
    enabled: bool = True
    update_interval: int = 3600
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    rotation: str = "1 day"
    retention: str = "30 days"


class Settings(BaseModel):
    """Main settings class containing all configuration."""

    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # API credentials from environment
    binance_api_key: str = Field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = Field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    binance_testnet: bool = Field(default_factory=lambda: os.getenv("BINANCE_TESTNET", "true").lower() == "true")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Load settings from a YAML configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    def get_data_dir(self) -> Path:
        """Get the data directory path."""
        data_dir = self.get_project_root() / self.data.cache_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def get_model_dir(self) -> Path:
        """Get the models directory path."""
        model_dir = self.get_project_root() / self.training.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_log_dir(self) -> Path:
        """Get the logs directory path."""
        log_dir = self.get_project_root() / self.training.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir


# Global settings instance
_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None) -> Settings:
    """
    Get the global settings instance.

    Args:
        config_path: Optional path to config file. If not provided,
                    looks for config/config.yaml in project root.

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None:
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        if Path(config_path).exists():
            _settings = Settings.from_yaml(config_path)
        else:
            # Use defaults if no config file
            _settings = Settings()

    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
