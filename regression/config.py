"""Validated resource limits and reproducible experiment configuration."""
from dataclasses import dataclass, field


class AnalysisError(ValueError):
    pass


@dataclass
class AnalysisConfig:
    target: str
    features: list[str] | None = None
    test_size: float = .2
    seed: int = 0
    inner_folds: int = 3
    outer_folds: int = 3
    feature_counts: list[int] | None = None
    max_encoded_features: int = 60
    max_comparison_features: int = 12
    max_comparisons: int = 8
    max_rows: int = 5000
    max_workload: int = 15000
    max_seconds: int = 180
    ordered_observations: bool = False
    title: str = "Uploaded dataset"
    unit: str = "target units"

    def validate(self):
        if not self.target or not isinstance(self.target, str):
            raise AnalysisError("Select a numeric target column.")
        if not isinstance(self.seed,int) or not 0 <= self.seed <= 4294967295:
            raise AnalysisError("Random seed must be an integer between 0 and 4294967295.")
        if len(self.title)>120 or len(self.unit)>40:
            raise AnalysisError("Dataset title/unit must be at most 120/40 characters.")
        if not .1 <= self.test_size <= .4:
            raise AnalysisError("Test fraction must be between 0.1 and 0.4.")
        if not 2 <= self.inner_folds <= 5 or not 2 <= self.outer_folds <= 5:
            raise AnalysisError("Inner and outer CV folds must be between 2 and 5.")
        if not 1 <= self.max_encoded_features <= 100:
            raise AnalysisError("Maximum encoded features must be between 1 and 100.")
        if not 1 <= self.max_comparison_features <= self.max_encoded_features:
            raise AnalysisError("Invalid feature comparison limit.")
        if not 1 <= self.max_comparisons <= 16:
            raise AnalysisError("At most 16 feature-count comparisons are allowed.")
        if not 20 <= self.max_rows <= 20000 or not 1 <= self.max_seconds <= 300:
            raise AnalysisError("Invalid row/time limits (20–20000 rows; 1–300 seconds).")
        if not 1 <= self.max_workload <= 100000:
            raise AnalysisError("Workload limit must be between 1 and 100000.")
        if self.feature_counts is not None:
            if not self.feature_counts or any(type(k) is not int or k < 1 for k in self.feature_counts):
                raise AnalysisError("Feature counts must be positive integers.")
            if len(set(self.feature_counts)) > self.max_comparisons:
                raise AnalysisError("Too many feature-count comparisons.")
        return self
