import json
import os
import csv
import math
from typing import Dict, Any, Optional, Tuple, Callable


DEFAULT_THRESHOLDS_PATH = os.path.join("assets", "thresholds.json")
CSV_THRESHOLDS_PATH = os.path.join("assets", "abn_squat", "thresholds_table.csv")
CSV_STABILITY_PATH = os.path.join("assets", "abn_squat", "stability_thresholds_table.csv")


def _load_thresholds(config_path: Optional[str] = None) -> Dict[str, Any]:
	path = config_path or DEFAULT_THRESHOLDS_PATH
	if not os.path.exists(path):
		return {}
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
			return data if isinstance(data, dict) else {}
	except Exception:
		return {}


def _score_direction(value: float, rule: Dict[str, Any]) -> str:
	"""Return 'Good' | 'Partial' | 'Poor' according to rule.

	Rule format examples:
	- {"direction": ">=", "good": 10, "partial": 7}
	- {"direction": "<=", "good": 10, "partial": 13}
	- {"direction": "range", "good": [5, 15], "partial": [3, 18]}
	"""
	if value is None:
		return "N/A"
	# Treat NaN/Inf as N/A (do not penalize)
	try:
		if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
			return "N/A"
	except Exception:
		pass
	try:
		direction = rule.get("direction")
		if direction == ">=":
			good = float(rule.get("good"))
			partial = float(rule.get("partial", good))
			if value >= good:
				return "Good"
			elif value >= partial:
				return "Partial"
			else:
				return "Poor"
		elif direction == "<=":
			good = float(rule.get("good"))
			partial = float(rule.get("partial", good))
			if value <= good:
				return "Good"
			elif value <= partial:
				return "Partial"
			else:
				return "Poor"
		elif direction == "range":
			g = rule.get("good", [])
			p = rule.get("partial", [])
			if len(g) == 2 and g[0] <= value <= g[1]:
				return "Good"
			if len(p) == 2 and p[0] <= value <= p[1]:
				return "Partial"
			return "Poor"
		else:
			return "N/A"
	except Exception:
		return "N/A"


def filter_and_score_metrics(
	metrics: Dict[str, Any],
	config_path: Optional[str] = None,
    sex: Optional[str] = None,
    age: Optional[int] = None,
) -> Dict[str, Any]:
    """Filter metrik agar hanya yang ada di CSV thresholds, sekaligus beri status.

    - Membaca assets/abn_squat/thresholds_table.csv dan stability_thresholds_table.csv
    - Memetakan setiap baris CSV ke key internal/terturun dari metrics
    - Mengembalikan hanya metrik yang ada di CSV, plus suffix __status
    """

    # Map nama CSV → key internal (atau aggregator) dan fungsi ambil nilai dari metrics
    # Tiap entry: csv_name: (output_key, getter)
    def _nanmax(values):
        vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return max(vals) if vals else float("nan")

    def _nanmin(values):
        vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return min(vals) if vals else float("nan")

    mapping: Dict[str, Tuple[str, Callable[[Dict[str, Any]], float]]] = {
        "Depth: Thigh angle to horizontal (deg)": (
            "squat_depth_thigh_deg",
            lambda m: float(m.get("squat_depth_thigh_deg", float("nan")))
        ),
        "Foot pronation (calcaneal eversion, deg)": (
            "foot_pronation_proxy_deg_worst_at_depth",
            lambda m: _nanmax([
                m.get("foot_pronation_proxy_deg_L_at_depth"),
                m.get("foot_pronation_proxy_deg_R_at_depth")
            ])
        ),
        "Heel lift height at bottom (% foot length)": (
            "heel_lift_percent_worst_at_depth",
            lambda m: float(m.get("heel_lift_percent_worst_at_depth", float("nan")))
        ),
        "Knee tracking deviation from 2nd toe (deg)": (
            "knee_tracking_dev_worst_deg_at_depth",
            lambda m: float(m.get("knee_tracking_dev_worst_deg_at_depth", float("nan")))
        ),
        "Knee valgus (worst side) (deg)": (
            "knee_valgus_deg_worst_at_depth",
            lambda m: float(m.get("knee_valgus_deg_worst_at_depth", float("nan")))
        ),
        "L/R symmetry (knee/tibia diff at bottom, deg)": (
            "lr_symmetry_worst_deg_at_depth",
            lambda m: _nanmax([
                m.get("lr_tibia_symmetry_deg_at_depth"),
                m.get("lr_knee_tracking_symmetry_deg_at_depth")
            ])
        ),
        "Pelvic lateral shift at bottom (% pelvis width)": (
            "pelvic_lateral_shift_percent_at_depth",
            lambda m: float(m.get("pelvic_lateral_shift_percent_at_depth", float("nan")))
        ),
        "Pelvic tilt change (Butt wink flexion from neutral, deg)": (
            "butt_wink_deg",
            lambda m: float(m.get("butt_wink_deg", float("nan")))
        ),
        "Stance width deviation from target band (% pelvis width)": (
            "stance_width_deviation_percent_at_depth",
            lambda m: float(m.get("stance_width_deviation_percent_at_depth", float("nan")))
        ),
        "Tempo control deviation (avg per rep, seconds)": (
            "tempo_control_deviation_sec",
            lambda m: float(m.get("tempo_control_deviation_sec", float("nan")))
        ),
        "Thoracic/Head forward deviation (deg)": (
            "trunk_lean_max_deg",
            lambda m: float(m.get("trunk_lean_max_deg", float("nan")))
        ),
        "Tibia forward angle relative to vertical (deg)": (
            "tibia_forward_deg_min_at_depth",
            lambda m: _nanmin([
                m.get("tibia_forward_deg_L_at_depth"),
                m.get("tibia_forward_deg_R_at_depth")
            ])
        ),
        "Toe-out angle deviation from target band (deg)": (
            "foot_ER_dev_deg_worst_at_depth",
            lambda m: _nanmax([
                m.get("foot_ER_dev_deg_L_at_depth"),
                m.get("foot_ER_dev_deg_R_at_depth")
            ])
        ),
        "Trunk–Tibia angle difference at bottom (deg)": (
            "trunk_minus_tibia_deg_at_depth",
            lambda m: float(m.get("trunk_minus_tibia_deg_at_depth", float("nan")))
        )
    }

    # Utility parse for AgeBand in CSV (supports ≤, ≥, en dash 41–55)
    def _ageband_label(a: int) -> str:
        if a <= 40: return "≤40"
        if 41 <= a <= 55: return "41–55"
        if 56 <= a <= 70: return "56–70"
        return "≥71"

    sex_key = (sex or "").strip().upper()
    age_label = _ageband_label(int(age)) if age is not None else None

    # Load thresholds from CSVs
    csv_rules: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(CSV_THRESHOLDS_PATH):
        with open(CSV_THRESHOLDS_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("Metric", "").strip()
                direction = row.get("Direction", "").strip()
                sx = (row.get("Sex", "").strip().upper())
                band = row.get("AgeBand", "").strip()
                good = row.get("Good", "").strip()
                partial = row.get("Partial", "").strip()
                if not metric:
                    continue
                m = csv_rules.setdefault(metric, {"direction": direction, "bands": {}})
                bands = m.setdefault("bands", {})
                bands.setdefault(sx, {})[band] = {"good": float(good), "partial": float(partial)}

    # Stability thresholds (CoV%) → rep_duration_cov_percent
    stability_rule = None
    if os.path.exists(CSV_STABILITY_PATH):
        with open(CSV_STABILITY_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sx = (row.get("Sex", "").strip().upper())
                band = row.get("AgeBand", "").strip()
                good = row.get("Good (CoV% ≤)", "").strip()
                partial = row.get("Partial (CoV% ≤)", "").strip()
                if not sx or not band:
                    continue
                if stability_rule is None:
                    stability_rule = {"direction": "<=", "bands": {}}
                stability_rule.setdefault("bands", {}).setdefault(sx, {})[band] = {
                    "good": float(good),
                    "partial": float(partial)
                }

    # Build filtered output only for metrics present in CSVs
    filtered: Dict[str, Any] = {}
    for csv_name, (out_key, getter) in mapping.items():
        rule = csv_rules.get(csv_name)
        if rule is None:
            continue  # not requested in CSV → drop
        value = getter(metrics)
        filtered[out_key] = value
        # Pick banded rule
        picked = rule
        if sex_key and age_label and isinstance(rule, dict) and "bands" in rule:
            picked_band = rule.get("bands", {}).get(sex_key, {}).get(age_label)
            if isinstance(picked_band, dict):
                picked = {**picked_band, "direction": rule.get("direction")}
        # attach status
        try:
            filtered[f"{out_key}__status"] = _score_direction(float(value), picked)
        except Exception:
            filtered[f"{out_key}__status"] = "N/A"

    # Add stability scoring if available
    if stability_rule is not None and ("rep_duration_cov_percent" in metrics):
        value = float(metrics.get("rep_duration_cov_percent"))
        picked = stability_rule
        if sex_key and age_label:
            b = stability_rule.get("bands", {}).get(sex_key, {}).get(age_label)
            if isinstance(b, dict):
                picked = {**b, "direction": stability_rule.get("direction")}
        filtered["rep_duration_cov_percent"] = value
        try:
            filtered["rep_duration_cov_percent__status"] = _score_direction(value, picked)
        except Exception:
            filtered["rep_duration_cov_percent__status"] = "N/A"

    return filtered


