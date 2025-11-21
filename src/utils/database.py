"""
Datenbank-Integration für persistentes Logging.

Speichert:
- Trade History
- Portfolio Values
- Alerts
- Model Predictions
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger
import os

Base = declarative_base()


class Trade(Base):
    """Trade-Tabelle."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    order_type = Column(String(20), default='market')
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    value = Column(Float)
    fee = Column(Float, default=0.0)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    position_after = Column(Float)
    model_signal = Column(String(50))
    confidence = Column(Float)
    notes = Column(Text)


class PortfolioSnapshot(Base):
    """Portfolio-Wert Snapshots."""
    __tablename__ = 'portfolio_snapshots'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float)
    position_value = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    drawdown = Column(Float)
    num_positions = Column(Integer)
    positions_json = Column(JSON)


class Alert(Base):
    """Alert-Tabelle."""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String(50), nullable=False)
    level = Column(String(20), nullable=False)
    title = Column(String(200))
    message = Column(Text)
    data_json = Column(JSON)
    acknowledged = Column(Boolean, default=False)


class ModelPrediction(Base):
    """Model-Vorhersagen."""
    __tablename__ = 'model_predictions'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    model_name = Column(String(100), nullable=False)
    prediction = Column(String(50))  # 'up', 'down', 'sideways'
    confidence = Column(Float)
    probabilities_json = Column(JSON)
    actual_outcome = Column(String(50))
    was_correct = Column(Boolean)


class PerformanceMetric(Base):
    """Performance-Metriken."""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    period = Column(String(20))  # 'daily', 'weekly', 'monthly'
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    num_trades = Column(Integer)
    avg_trade = Column(Float)


class DatabaseManager:
    """
    Manager für Datenbank-Operationen.

    Unterstützt SQLite und PostgreSQL.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Args:
            database_url: SQLAlchemy Database URL
                         Default: SQLite in ./data/trading.db
        """
        if database_url is None:
            # Default: SQLite
            db_path = os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'data', 'trading.db'
            )
            database_url = f"sqlite:///{db_path}"

        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Tabellen erstellen
        Base.metadata.create_all(self.engine)

        logger.info(f"Database initialized: {database_url}")

    def get_session(self) -> Session:
        """Gibt neue Session zurück."""
        return self.SessionLocal()

    # ==================== Trade Operations ====================

    def log_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        **kwargs
    ) -> int:
        """
        Speichert Trade.

        Returns:
            Trade ID
        """
        with self.get_session() as session:
            trade = Trade(
                symbol=symbol,
                side=side,
                price=price,
                amount=amount,
                value=price * amount,
                **kwargs
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id

        logger.debug(f"Logged trade {trade_id}: {side} {amount} {symbol} @ {price}")
        return trade_id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Holt Trades aus der Datenbank."""
        with self.get_session() as session:
            query = session.query(Trade)

            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)

            trades = query.order_by(Trade.timestamp.desc()).limit(limit).all()

        return trades

    # ==================== Portfolio Operations ====================

    def log_portfolio_snapshot(
        self,
        total_value: float,
        cash_balance: float = 0.0,
        position_value: float = 0.0,
        unrealized_pnl: float = 0.0,
        positions: Optional[Dict] = None
    ) -> int:
        """Speichert Portfolio-Snapshot."""
        with self.get_session() as session:
            # Berechne Drawdown
            last_peak = session.query(
                PortfolioSnapshot.total_value
            ).order_by(PortfolioSnapshot.total_value.desc()).first()

            peak = last_peak[0] if last_peak else total_value
            drawdown = (peak - total_value) / peak if peak > 0 else 0

            snapshot = PortfolioSnapshot(
                total_value=total_value,
                cash_balance=cash_balance,
                position_value=position_value,
                unrealized_pnl=unrealized_pnl,
                drawdown=drawdown,
                num_positions=len(positions) if positions else 0,
                positions_json=positions
            )
            session.add(snapshot)
            session.commit()
            snapshot_id = snapshot.id

        return snapshot_id

    def get_portfolio_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PortfolioSnapshot]:
        """Holt Portfolio-History."""
        with self.get_session() as session:
            query = session.query(PortfolioSnapshot)

            if start_date:
                query = query.filter(PortfolioSnapshot.timestamp >= start_date)
            if end_date:
                query = query.filter(PortfolioSnapshot.timestamp <= end_date)

            snapshots = query.order_by(PortfolioSnapshot.timestamp).limit(limit).all()

        return snapshots

    # ==================== Alert Operations ====================

    def log_alert(
        self,
        alert_type: str,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ) -> int:
        """Speichert Alert."""
        with self.get_session() as session:
            alert = Alert(
                alert_type=alert_type,
                level=level,
                title=title,
                message=message,
                data_json=data
            )
            session.add(alert)
            session.commit()
            alert_id = alert.id

        return alert_id

    def get_alerts(
        self,
        level: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50
    ) -> List[Alert]:
        """Holt Alerts."""
        with self.get_session() as session:
            query = session.query(Alert)

            if level:
                query = query.filter(Alert.level == level)
            if unacknowledged_only:
                query = query.filter(Alert.acknowledged == False)

            alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()

        return alerts

    def acknowledge_alert(self, alert_id: int):
        """Markiert Alert als bestätigt."""
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                session.commit()

    # ==================== Prediction Operations ====================

    def log_prediction(
        self,
        symbol: str,
        model_name: str,
        prediction: str,
        confidence: float,
        probabilities: Optional[Dict] = None
    ) -> int:
        """Speichert Model-Vorhersage."""
        with self.get_session() as session:
            pred = ModelPrediction(
                symbol=symbol,
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                probabilities_json=probabilities
            )
            session.add(pred)
            session.commit()
            pred_id = pred.id

        return pred_id

    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_outcome: str
    ):
        """Aktualisiert Vorhersage mit tatsächlichem Ergebnis."""
        with self.get_session() as session:
            pred = session.query(ModelPrediction).filter(
                ModelPrediction.id == prediction_id
            ).first()

            if pred:
                pred.actual_outcome = actual_outcome
                pred.was_correct = (pred.prediction == actual_outcome)
                session.commit()

    def get_model_accuracy(
        self,
        model_name: str,
        days: int = 30
    ) -> Dict[str, float]:
        """Berechnet Model-Accuracy."""
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as session:
            predictions = session.query(ModelPrediction).filter(
                ModelPrediction.model_name == model_name,
                ModelPrediction.timestamp >= start_date,
                ModelPrediction.was_correct.isnot(None)
            ).all()

            if not predictions:
                return {'accuracy': 0.0, 'total': 0}

            correct = sum(1 for p in predictions if p.was_correct)
            total = len(predictions)

            return {
                'accuracy': correct / total * 100,
                'correct': correct,
                'total': total
            }

    # ==================== Performance Metrics ====================

    def log_performance_metrics(
        self,
        period: str,
        metrics: Dict[str, float]
    ) -> int:
        """Speichert Performance-Metriken."""
        with self.get_session() as session:
            perf = PerformanceMetric(
                period=period,
                total_return=metrics.get('total_return'),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                sortino_ratio=metrics.get('sortino_ratio'),
                max_drawdown=metrics.get('max_drawdown'),
                win_rate=metrics.get('win_rate'),
                profit_factor=metrics.get('profit_factor'),
                num_trades=metrics.get('num_trades'),
                avg_trade=metrics.get('avg_trade')
            )
            session.add(perf)
            session.commit()
            perf_id = perf.id

        return perf_id

    def get_latest_metrics(self, period: str = 'daily') -> Optional[PerformanceMetric]:
        """Holt neueste Metriken."""
        with self.get_session() as session:
            metric = session.query(PerformanceMetric).filter(
                PerformanceMetric.period == period
            ).order_by(PerformanceMetric.timestamp.desc()).first()

        return metric


# Globale Instanz
_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Gibt globale DatabaseManager Instanz zurück."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
