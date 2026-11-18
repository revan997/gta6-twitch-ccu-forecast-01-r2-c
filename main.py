"""Family-slope blend forecaster for launch-day Twitch viewers (Lane C, round 2).

Model (written fresh in this session):
  For a case with hour-1 print p1, forecast log-ratios y_h = ln(viewers_h / p1)
  for hours 2..24 as the blend of two per-hour quantile forecasts, in log space:

  (A) Pooled local-linear (Model B's machinery, re-implemented here): per hour,
      weighted local-linear regression of the log-ratio on ln(hour-1 viewers)
      over the tier-"comparable" donors, Gaussian kernel bandwidth 1.5 log units,
      trim beyond 3 bandwidths, slope clamped to +/-1.5; weighted empirical
      quantiles of the residuals, shrunk 0.5 toward a normal with the same
      scale, narrowed by 0.85; median curve lightly smoothed across hours.

  (B) PS-exclusive rolling-midnight family slope: the six PS-exclusive
      rolling-midnight launches in the kit (The Last of Us Part II,
      Ghost of Tsushima, FINAL FANTASY VII REBIRTH, Stellar Blade,
      Ghost of Yotei, Saros) -- the closest family to a PS5-only launch like
      GTA VI. Per hour, OLS of the log-ratio on ln(hour-1) *within the
      family only* (evening slope about -0.36: huge hour-1 -> flat curve),
      centre = mean + slope * (ln p1 - mean ln hour-1 of the family);
      residual quantiles shrunk 0.5 toward normal, spread 1.0.

  Blend: q = 0.25 * (B) + 0.75 * (A) per quantile level, then deviations from
  the blended median narrowed by 0.85. Weights and spread were chosen on
  practice + leave-one-out validation (see REPORT.md); the blend beats Model B
  slightly on every split measured, but does NOT reach the 0.135 bar, so this
  is an honest improvement, not a breakthrough.

  Excluded from shape fitting: target-history (GTA VI's own pre-launch
  category curve -- not a launch curve), early-access-contaminated and
  Tier-weak comparables (dropped from part A; part B uses the family only).

  Stdlib only, no network, no randomness, deterministic.

  Robustness: hour-1 is validated explicitly (must be a finite positive
  number; a missing or garbage print fails loudly instead of silently
  producing a degenerate forecast), and every emitted hour is self-checked
  for finite, positive, monotone non-decreasing quantiles before it is
  returned. Neither check changes any forecast on valid inputs.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

FAMILY = {"The Last of Us Part II", "Ghost of Tsushima",
          "FINAL FANTASY VII REBIRTH", "Stellar Blade",
          "Ghost of Yōtei", "Saros"}

# Part A (pooled local-linear) constants.
BW = 1.5
TRIM = 3.0
SLOPE_CLAMP = 1.5
SHRINK = 0.5
SPREAD_A = 0.85
MIN_POINTS = 5
# Part B (family slope) constants.
FAM_SLOPE_CLAMP = 1.5
FAM_SHRINK = 0.5
# Blend.
W_FAMILY = 0.25
BLEND_SPREAD = 0.85
FALLBACK_SIGMA = 0.35


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hour_key(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tier(unlock):
    text = unlock or ""
    if "Tier: weak" in text:
        return "weak"
    if "early-access-contaminated" in text:
        return "ea"
    if "target-history" in text:
        return "target"
    return "comp"


def _release_day_curve(comp):
    try:
        start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is not None and viewers > 0:
            try:
                key = _hour_key(_parse_time(point.get("from", "")))
            except (TypeError, ValueError):
                key = point.get("from")
            by_hour[key] = float(viewers)
    values = []
    for offset in range(24):
        value = by_hour.get(_hour_key(start + timedelta(hours=offset)))
        if value is None:
            return None
        values.append(value)
    return values if values[0] > 0 else None


def _collect(case):
    """Return (tier-comp curves, family curves) as lists of 24-value lists."""
    comp_curves, fam_curves = [], []
    for comp in case.get("comparables") or []:
        values = _release_day_curve(comp)
        if values is None:
            continue
        if _tier(comp.get("unlock", "")) == "comp":
            comp_curves.append(values)
            if comp.get("name") in FAMILY:
                fam_curves.append(values)
    if len(comp_curves) < 8:  # filter too aggressive: use everything usable
        comp_curves = []
        for comp in case.get("comparables") or []:
            values = _release_day_curve(comp)
            if values is not None and _tier(comp.get("unlock", "")) != "target":
                comp_curves.append(values)
    return comp_curves, fam_curves


def _weighted_quantile(pairs, tau):
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for index, (value, weight) in enumerate(ordered):
        before = acc
        acc += weight
        if acc >= target:
            if index == 0 or weight <= 0:
                return value
            frac = (target - before) / weight
            return ordered[index - 1][0] + frac * (value - ordered[index - 1][0])
    return ordered[-1][0]


def _emp_quantile(values, tau):
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = tau * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def _pooled_quantiles(curves, p1, hour):
    """Part A: local-linear log-ratio quantiles for one hour (index 1..23)."""
    x0 = math.log(p1)
    points = []
    for values in curves:
        ratio = values[hour] / values[0]
        if ratio <= 0:
            continue
        dx = (math.log(values[0]) - x0) / BW
        if abs(dx) > TRIM:
            continue
        weight = math.exp(-0.5 * dx * dx)
        if weight < 1e-6:
            continue
        points.append((math.log(values[0]), math.log(ratio), weight))
    if len(points) < MIN_POINTS:
        points = [(math.log(v[0]), math.log(v[hour] / v[0]), 1.0)
                  for v in curves if v[hour] > 0]
    if not points:
        return None
    sw = sum(w for _, _, w in points)
    sx = sum(w * (x - x0) for x, _, w in points)
    sy = sum(w * y for _, y, w in points)
    sxx = sum(w * (x - x0) ** 2 for x, _, w in points)
    sxy = sum(w * (x - x0) * y for x, y, w in points)
    denom = sw * sxx - sx * sx
    if denom > 1e-12:
        slope = (sw * sxy - sx * sy) / denom
        centre = (sy - slope * sx) / sw
    else:
        slope, centre = 0.0, sy / sw
    slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, slope))
    residuals = [(y - (centre + slope * (x - x0)), w) for x, y, w in points]
    variance = sum(w * e * e for e, w in residuals) / sw
    scale = math.sqrt(max(variance, 0.0))
    return [centre + SPREAD_A * ((1.0 - SHRINK) * _weighted_quantile(residuals, tau)
                                  + SHRINK * z * scale)
            for tau, z in zip(LEVELS, Z)]


def _family_quantiles(fam_curves, p1, hour):
    """Part B: within-family OLS slope log-ratio quantiles for one hour."""
    x0 = math.log(p1)
    xs = [math.log(v[0]) for v in fam_curves]
    ys = [math.log(v[hour] / v[0]) for v in fam_curves]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = 0.0
    if den > 1e-9:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    slope = max(-FAM_SLOPE_CLAMP, min(FAM_SLOPE_CLAMP, slope))
    centre = my + slope * (x0 - mx)
    res = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    sd = statistics.pstdev(res) if len(res) > 1 else FALLBACK_SIGMA
    return [centre + (1.0 - FAM_SHRINK) * _emp_quantile(res, tau)
            + FAM_SHRINK * z * sd
            for tau, z in zip(LEVELS, Z)]


def _round2(value):
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100.0
    return int(rounded) if float(rounded).is_integer() else rounded


def forecast_case(case):
    known = next((row for row in case.get("known") or []
                  if row.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    try:
        raw_p1 = float(known["viewers"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("case has no usable hour-1 viewers print")
    if not math.isfinite(raw_p1) or raw_p1 <= 0:
        raise ValueError("hour-1 viewers must be a finite positive number")
    p1 = max(1.0, raw_p1)
    comp_curves, fam_curves = _collect(case)
    hours = sorted(row["hour"] for row in case.get("forecast") or [])
    if not hours:
        hours = list(range(2, 25))

    raw_a, raw_b = {}, {}
    for hour in hours:
        qa = _pooled_quantiles(comp_curves, p1, hour - 1) if comp_curves else None
        if qa is None:
            sigma = FALLBACK_SIGMA * math.sqrt(hour - 1)
            qa = [math.log(q / p1) for q in
                  [p1 * math.exp(z * sigma) for z in Z]]
        raw_a[hour] = qa
        raw_b[hour] = (_family_quantiles(fam_curves, p1, hour - 1)
                       if len(fam_curves) >= 4 else None)

    # Smooth the pooled median across neighbouring hours (as in Model B);
    # apply the same shift to all five quantiles of that hour.
    medians = [raw_a[h][2] for h in hours]
    smoothed = []
    for i in range(len(hours)):
        if len(hours) == 1:
            smoothed.append(medians[0])
        elif i == 0:
            smoothed.append(0.75 * medians[0] + 0.25 * medians[1])
        elif i == len(hours) - 1:
            smoothed.append(0.25 * medians[-2] + 0.75 * medians[-1])
        else:
            smoothed.append(0.25 * medians[i - 1] + 0.5 * medians[i]
                            + 0.25 * medians[i + 1])
    shifts = [smoothed[i] - medians[i] for i in range(len(hours))]

    rows = []
    for index, hour in enumerate(hours):
        qa = [q + shifts[index] for q in raw_a[hour]]
        if raw_b[hour] is None:
            blended = qa
        else:
            blended = [W_FAMILY * b + (1.0 - W_FAMILY) * a
                       for a, b in zip(qa, raw_b[hour])]
        med = blended[2]
        blended = [med + BLEND_SPREAD * (q - med) for q in blended]
        quantiles = [math.exp(q) * p1 for q in blended]
        cleaned = []
        previous = 0.0
        for value in quantiles:
            value = min(max(value, previous), 1e9)
            value = max(value, 0.01)
            cleaned.append(_round2(value))
            previous = cleaned[-1]
        for j in range(1, 5):
            if cleaned[j] < cleaned[j - 1]:
                cleaned[j] = cleaned[j - 1]
        rows.append({"hour": hour, "q": cleaned})
    # Runtime self-check: every hour's quantiles must be finite, positive and
    # monotone non-decreasing in q. The cleaning above enforces this, so a
    # violation here is an internal error -- fail loudly instead of emitting
    # a malformed forecast.
    for row in rows:
        qs = row["q"]
        if len(qs) != len(LEVELS) or any(
                not math.isfinite(q) or q <= 0 for q in qs):
            raise ValueError(
                f"non-finite/non-positive quantiles at hour {row['hour']}")
        if any(qs[j] < qs[j - 1] for j in range(1, len(qs))):
            raise ValueError(
                f"quantiles not monotone at hour {row['hour']}")
    return {"hours": rows}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
