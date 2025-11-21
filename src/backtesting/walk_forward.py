"""
Walk-Forward-Optimierung für robuste Strategieentwicklung.

Vermeidet Overfitting durch:
- Rollierende Train/Test-Splits
- Out-of-Sample Validierung
- Parameter-Stabilität-Analyse
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

from .engine import BacktestEngine, BacktestConfig
from .metrics import BacktestMetrics


@dataclass
class WalkForwardWindow:
    """Ein Walk-Forward Fenster."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_idx: int


@dataclass
class OptimizationResult:
    """Ergebnis einer Parameter-Optimierung."""
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    window_idx: int


class WalkForwardOptimizer:
    """
    Walk-Forward-Optimierung für Handelsstrategien.

    Teilt Daten in überlappende Train/Test-Fenster und
    optimiert Parameter auf Train-Daten, validiert auf Test-Daten.
    """

    def __init__(
        self,
        train_pct: float = 0.7,
        n_windows: int = 5,
        overlap: float = 0.5,
        optimization_metric: str = 'sharpe_ratio'
    ):
        """
        Args:
            train_pct: Anteil der Trainingsdaten pro Fenster
            n_windows: Anzahl der Walk-Forward Fenster
            overlap: Überlappung zwischen Fenstern
            optimization_metric: Metrik für Optimierung
        """
        self.train_pct = train_pct
        self.n_windows = n_windows
        self.overlap = overlap
        self.optimization_metric = optimization_metric

    def create_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Erstellt Walk-Forward Fenster.

        Args:
            data: Vollständiger Datensatz

        Returns:
            Liste von WalkForwardWindow
        """
        n = len(data)
        windows = []

        # Berechne Fenstergrößen
        total_window = n / (1 + (self.n_windows - 1) * (1 - self.overlap))
        train_size = int(total_window * self.train_pct)
        test_size = int(total_window * (1 - self.train_pct))
        step_size = int(total_window * (1 - self.overlap))

        for i in range(self.n_windows):
            start = i * step_size
            train_end = start + train_size
            test_end = train_end + test_size

            if test_end > n:
                break

            window = WalkForwardWindow(
                train_start=data.index[start],
                train_end=data.index[train_end - 1],
                test_start=data.index[train_end],
                test_end=data.index[min(test_end - 1, n - 1)],
                window_idx=i
            )
            windows.append(window)

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable[[Dict[str, Any]], Callable],
        param_grid: Dict[str, List[Any]],
        backtest_config: Optional[BacktestConfig] = None,
        n_jobs: int = 1
    ) -> List[OptimizationResult]:
        """
        Führt Walk-Forward-Optimierung durch.

        Args:
            data: Vollständiger Datensatz
            strategy_factory: Funktion die Strategie aus Parametern erstellt
            param_grid: Dictionary mit Parameter-Listen
            backtest_config: Backtest-Konfiguration
            n_jobs: Anzahl paralleler Jobs

        Returns:
            Liste von OptimizationResult für jedes Fenster
        """
        windows = self.create_windows(data)
        config = backtest_config or BacktestConfig()
        results = []

        # Generiere alle Parameter-Kombinationen
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        for window in windows:
            logger.info(f"\nWindow {window.window_idx + 1}/{len(windows)}")
            logger.info(f"Train: {window.train_start} to {window.train_end}")
            logger.info(f"Test: {window.test_start} to {window.test_end}")

            # Hole Train/Test Daten
            train_data = data[window.train_start:window.train_end]
            test_data = data[window.test_start:window.test_end]

            # Optimiere auf Train-Daten
            best_params = None
            best_train_score = float('-inf')
            best_train_metrics = None

            for combo in param_combinations:
                params = dict(zip(param_names, combo))

                # Erstelle Strategie
                strategy = strategy_factory(params)

                # Backtest auf Train-Daten
                engine = BacktestEngine(config)
                result = engine.run(train_data, strategy, verbose=False)

                score = result['metrics'].get(self.optimization_metric, 0)

                if score > best_train_score:
                    best_train_score = score
                    best_params = params
                    best_train_metrics = result['metrics']

            # Validiere auf Test-Daten
            if best_params:
                strategy = strategy_factory(best_params)
                engine = BacktestEngine(config)
                test_result = engine.run(test_data, strategy, verbose=False)

                opt_result = OptimizationResult(
                    params=best_params,
                    train_metrics=best_train_metrics,
                    test_metrics=test_result['metrics'],
                    window_idx=window.window_idx
                )
                results.append(opt_result)

                logger.info(f"Best params: {best_params}")
                logger.info(f"Train {self.optimization_metric}: {best_train_score:.4f}")
                logger.info(f"Test {self.optimization_metric}: {test_result['metrics'].get(self.optimization_metric, 0):.4f}")

        return results

    def analyze_results(self, results: List[OptimizationResult]) -> Dict:
        """
        Analysiert Walk-Forward Ergebnisse.

        Args:
            results: Liste von OptimizationResult

        Returns:
            Dictionary mit Analyse
        """
        if not results:
            return {}

        # Sammle Metriken
        train_metrics = [r.train_metrics.get(self.optimization_metric, 0) for r in results]
        test_metrics = [r.test_metrics.get(self.optimization_metric, 0) for r in results]

        # Parameter-Stabilität
        all_params = [r.params for r in results]
        param_stability = {}

        for key in all_params[0].keys():
            values = [p[key] for p in all_params]
            if isinstance(values[0], (int, float)):
                param_stability[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            else:
                # Kategorische Parameter
                from collections import Counter
                counter = Counter(values)
                param_stability[key] = {
                    'most_common': counter.most_common(1)[0][0],
                    'distribution': dict(counter),
                    'values': values
                }

        # Efficiency Ratio
        avg_train = np.mean(train_metrics)
        avg_test = np.mean(test_metrics)
        efficiency = avg_test / avg_train if avg_train != 0 else 0

        analysis = {
            'n_windows': len(results),
            'optimization_metric': self.optimization_metric,

            'train_metrics': {
                'mean': avg_train,
                'std': np.std(train_metrics),
                'min': np.min(train_metrics),
                'max': np.max(train_metrics)
            },

            'test_metrics': {
                'mean': avg_test,
                'std': np.std(test_metrics),
                'min': np.min(test_metrics),
                'max': np.max(test_metrics)
            },

            'efficiency_ratio': efficiency,
            'param_stability': param_stability,

            # Overfitting-Indikator
            'overfitting_score': 1 - efficiency if efficiency < 1 else 0
        }

        return analysis

    def format_analysis(self, analysis: Dict) -> str:
        """Formatiert Analyse für Ausgabe."""
        output = []
        output.append("=" * 60)
        output.append("WALK-FORWARD OPTIMIZATION RESULTS")
        output.append("=" * 60)

        output.append(f"\nWindows: {analysis['n_windows']}")
        output.append(f"Optimization Metric: {analysis['optimization_metric']}")

        output.append("\n📈 TRAIN PERFORMANCE")
        train = analysis['train_metrics']
        output.append(f"  Mean:  {train['mean']:>10.4f}")
        output.append(f"  Std:   {train['std']:>10.4f}")
        output.append(f"  Range: [{train['min']:.4f}, {train['max']:.4f}]")

        output.append("\n📊 TEST PERFORMANCE (Out-of-Sample)")
        test = analysis['test_metrics']
        output.append(f"  Mean:  {test['mean']:>10.4f}")
        output.append(f"  Std:   {test['std']:>10.4f}")
        output.append(f"  Range: [{test['min']:.4f}, {test['max']:.4f}]")

        output.append("\n⚙️  ROBUSTNESS")
        output.append(f"  Efficiency Ratio: {analysis['efficiency_ratio']:>10.4f}")
        output.append(f"  Overfitting Score: {analysis['overfitting_score']:>9.4f}")

        output.append("\n🔧 PARAMETER STABILITY")
        for param, stats in analysis['param_stability'].items():
            if 'mean' in stats:
                output.append(f"  {param}: {stats['mean']:.4f} (±{stats['std']:.4f})")
            else:
                output.append(f"  {param}: {stats['most_common']}")

        output.append("=" * 60)

        return "\n".join(output)


def anchored_walk_forward(
    data: pd.DataFrame,
    strategy_factory: Callable,
    param_grid: Dict[str, List],
    n_windows: int = 10,
    initial_train_pct: float = 0.3
) -> List[OptimizationResult]:
    """
    Anchored Walk-Forward: Training beginnt immer am Anfang.

    Args:
        data: Vollständiger Datensatz
        strategy_factory: Strategie-Factory
        param_grid: Parameter-Grid
        n_windows: Anzahl der Fenster
        initial_train_pct: Initialer Trainingsanteil

    Returns:
        Optimierungsergebnisse
    """
    optimizer = WalkForwardOptimizer(
        train_pct=0.8,  # Wird überschrieben
        n_windows=n_windows
    )

    n = len(data)
    results = []
    config = BacktestConfig()

    # Parameter-Kombinationen
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    # Anchored Windows
    test_size = int(n * (1 - initial_train_pct) / n_windows)

    for i in range(n_windows):
        train_end = int(n * initial_train_pct) + i * test_size
        test_end = train_end + test_size

        if test_end > n:
            break

        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:test_end]

        logger.info(f"\nAnchored Window {i + 1}/{n_windows}")
        logger.info(f"Train: 0 to {train_end} ({len(train_data)} bars)")
        logger.info(f"Test: {train_end} to {test_end} ({len(test_data)} bars)")

        # Optimiere
        best_params = None
        best_score = float('-inf')
        best_train_metrics = None

        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            strategy = strategy_factory(params)

            engine = BacktestEngine(config)
            result = engine.run(train_data, strategy, verbose=False)

            score = result['metrics'].get('sharpe_ratio', 0)

            if score > best_score:
                best_score = score
                best_params = params
                best_train_metrics = result['metrics']

        # Test
        if best_params:
            strategy = strategy_factory(best_params)
            engine = BacktestEngine(config)
            test_result = engine.run(test_data, strategy, verbose=False)

            results.append(OptimizationResult(
                params=best_params,
                train_metrics=best_train_metrics,
                test_metrics=test_result['metrics'],
                window_idx=i
            ))

    return results
