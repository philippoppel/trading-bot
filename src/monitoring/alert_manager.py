"""
Alert Manager für Benachrichtigungen.

Unterstützt:
- Telegram
- Discord
- E-Mail (optional)
"""

import asyncio
import aiohttp
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from datetime import datetime
import json


class AlertLevel(Enum):
    """Alert-Priorität."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Art des Alerts."""
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    DRAWDOWN = "drawdown"
    DAILY_LIMIT = "daily_limit"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class Alert:
    """Alert-Nachricht."""
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Optional[Dict] = None


class TelegramNotifier:
    """Telegram Benachrichtigungen."""

    def __init__(self, bot_token: str, chat_id: str):
        """
        Args:
            bot_token: Telegram Bot Token
            chat_id: Chat ID für Nachrichten
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Sendet Nachricht über Telegram.

        Args:
            message: Nachrichtentext
            parse_mode: HTML oder Markdown

        Returns:
            True wenn erfolgreich
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Telegram error: {await response.text()}")
                        return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class DiscordNotifier:
    """Discord Webhook Benachrichtigungen."""

    def __init__(self, webhook_url: str):
        """
        Args:
            webhook_url: Discord Webhook URL
        """
        self.webhook_url = webhook_url

    async def send(self, message: str, embed: Optional[Dict] = None) -> bool:
        """
        Sendet Nachricht über Discord.

        Args:
            message: Nachrichtentext
            embed: Optional Embed-Objekt

        Returns:
            True wenn erfolgreich
        """
        payload = {"content": message}

        if embed:
            payload["embeds"] = [embed]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status in [200, 204]:
                        return True
                    else:
                        logger.error(f"Discord error: {await response.text()}")
                        return False
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False


class AlertManager:
    """
    Verwaltet Alerts und Benachrichtigungen.

    Features:
    - Multi-Channel Support (Telegram, Discord)
    - Alert-Filterung nach Level
    - Rate-Limiting
    - Alert-History
    """

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        min_alert_level: AlertLevel = AlertLevel.INFO
    ):
        """
        Args:
            telegram_token: Telegram Bot Token
            telegram_chat_id: Telegram Chat ID
            discord_webhook: Discord Webhook URL
            min_alert_level: Minimales Level für Alerts
        """
        self.min_level = min_alert_level
        self.notifiers: List = []
        self.alert_history: List[Alert] = []

        # Telegram Setup
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
            self.notifiers.append(('telegram', self.telegram))
            logger.info("Telegram notifier configured")

        # Discord Setup
        if discord_webhook:
            self.discord = DiscordNotifier(discord_webhook)
            self.notifiers.append(('discord', self.discord))
            logger.info("Discord notifier configured")

        if not self.notifiers:
            logger.warning("No notifiers configured")

    def _format_telegram_message(self, alert: Alert) -> str:
        """Formatiert Alert für Telegram."""
        # Emoji basierend auf Level
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }

        emoji = emoji_map.get(alert.level, "📢")

        message = f"{emoji} <b>{alert.title}</b>\n\n"
        message += f"{alert.message}\n\n"

        if alert.data:
            message += "<pre>"
            for key, value in alert.data.items():
                message += f"{key}: {value}\n"
            message += "</pre>\n"

        message += f"<i>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>"

        return message

    def _format_discord_message(self, alert: Alert) -> tuple:
        """Formatiert Alert für Discord."""
        # Farbe basierend auf Level
        color_map = {
            AlertLevel.INFO: 0x3498db,  # Blau
            AlertLevel.WARNING: 0xf1c40f,  # Gelb
            AlertLevel.ERROR: 0xe74c3c,  # Rot
            AlertLevel.CRITICAL: 0x9b59b6  # Lila
        }

        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color_map.get(alert.level, 0x95a5a6),
            "timestamp": alert.timestamp.isoformat(),
            "fields": []
        }

        if alert.data:
            for key, value in alert.data.items():
                embed["fields"].append({
                    "name": key,
                    "value": str(value),
                    "inline": True
                })

        return "", embed

    async def send_alert(self, alert: Alert) -> bool:
        """
        Sendet Alert über alle konfigurierten Kanäle.

        Args:
            alert: Alert-Objekt

        Returns:
            True wenn mindestens ein Kanal erfolgreich
        """
        # Prüfe Level
        level_order = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        if level_order.index(alert.level) < level_order.index(self.min_level):
            return False

        # Speichere in History
        self.alert_history.append(alert)

        success = False

        for name, notifier in self.notifiers:
            try:
                if name == 'telegram':
                    message = self._format_telegram_message(alert)
                    if await notifier.send(message):
                        success = True

                elif name == 'discord':
                    message, embed = self._format_discord_message(alert)
                    if await notifier.send(message, embed):
                        success = True

            except Exception as e:
                logger.error(f"Error sending to {name}: {e}")

        return success

    def alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        Erstellt und sendet Alert (synchron).

        Args:
            alert_type: Art des Alerts
            level: Priorität
            title: Titel
            message: Nachricht
            data: Zusätzliche Daten
        """
        alert = Alert(
            type=alert_type,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data=data
        )

        # Asynchron senden
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_alert(alert))
            else:
                loop.run_until_complete(self.send_alert(alert))
        except RuntimeError:
            # Kein Event Loop
            asyncio.run(self.send_alert(alert))

    # Convenience Methods

    def trade_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float
    ):
        """Alert für geöffneten Trade."""
        self.alert(
            AlertType.TRADE_OPEN,
            AlertLevel.INFO,
            f"Trade Opened: {symbol}",
            f"Opened {side.upper()} position",
            {
                "Symbol": symbol,
                "Side": side,
                "Price": f"${price:,.2f}",
                "Amount": f"{amount:.4f}"
            }
        )

    def trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float
    ):
        """Alert für geschlossenen Trade."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING

        self.alert(
            AlertType.TRADE_CLOSE,
            level,
            f"Trade Closed: {symbol}",
            f"Closed {side.upper()} position with {'profit' if pnl >= 0 else 'loss'}",
            {
                "Symbol": symbol,
                "PnL": f"${pnl:+,.2f}",
                "Return": f"{pnl_pct:+.2f}%"
            }
        )

    def stop_loss_triggered(self, symbol: str, loss: float):
        """Alert für Stop-Loss."""
        self.alert(
            AlertType.STOP_LOSS,
            AlertLevel.WARNING,
            f"Stop-Loss Triggered: {symbol}",
            "Position closed due to stop-loss",
            {
                "Symbol": symbol,
                "Loss": f"${loss:,.2f}"
            }
        )

    def drawdown_warning(self, current_dd: float, max_dd: float):
        """Alert für Drawdown-Warnung."""
        self.alert(
            AlertType.DRAWDOWN,
            AlertLevel.WARNING,
            "Drawdown Warning",
            f"Current drawdown approaching limit",
            {
                "Current": f"{current_dd:.1f}%",
                "Limit": f"{max_dd:.1f}%"
            }
        )

    def daily_limit_reached(self, loss: float, limit: float):
        """Alert für Tageslimit."""
        self.alert(
            AlertType.DAILY_LIMIT,
            AlertLevel.ERROR,
            "Daily Loss Limit Reached",
            "Trading paused for today",
            {
                "Daily Loss": f"{loss:.1f}%",
                "Limit": f"{limit:.1f}%"
            }
        )

    def system_error(self, error: str):
        """Alert für Systemfehler."""
        self.alert(
            AlertType.ERROR,
            AlertLevel.ERROR,
            "System Error",
            error
        )

    def get_history(self, limit: int = 50) -> List[Alert]:
        """Gibt Alert-History zurück."""
        return self.alert_history[-limit:]
