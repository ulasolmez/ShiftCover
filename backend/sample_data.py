"""
ShiftCover – Sample demand-curve generator for testing.
Creates realistic weekly demand patterns and exports to CSV / Excel.
"""

import numpy as np
import pandas as pd

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
INTERVALS_PER_DAY = 288
TOTAL_INTERVALS = 2016


def _bell(x, centre, width, height):
    """Gaussian-ish bump."""
    return height * np.exp(-0.5 * ((x - centre) / width) ** 2)


def generate_sample_demand(
    peak_agents: int = 25,
    base_agents: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Return a (2016,) int array of required personnel every 5 min for a week.

    Weekdays get two peaks (morning + afternoon), weekends get one lower peak.
    """
    rng = np.random.default_rng(seed)
    demand = np.zeros(TOTAL_INTERVALS, dtype=float)

    for day in range(7):
        offset = day * INTERVALS_PER_DAY
        x = np.arange(INTERVALS_PER_DAY, dtype=float)

        # Per-day randomised peak height and timing jitter
        day_scale = rng.uniform(0.7, 1.3)
        jitter = rng.integers(-18, 19)  # ±90 min shift of peaks

        if day < 5:  # weekday
            # Randomise number of peaks: occasionally add a third
            n_peaks = rng.choice([2, 2, 3], p=[0.4, 0.4, 0.2])
            curve = (
                _bell(x, 108 + jitter, rng.uniform(20, 40), peak_agents * rng.uniform(0.5, 0.9)) +
                _bell(x, 168 + jitter, rng.uniform(18, 32), peak_agents * day_scale) +
                base_agents
            )
            if n_peaks == 3:
                curve += _bell(x, rng.integers(216, 252), rng.uniform(15, 28),
                               peak_agents * rng.uniform(0.3, 0.6))
        else:  # weekend
            curve = (
                _bell(x, 144 + jitter, rng.uniform(28, 52), peak_agents * rng.uniform(0.3, 0.65)) +
                base_agents * rng.uniform(0.4, 0.9)
            )

        # Multi-scale noise: slow drift + fast spikes
        slow_noise = np.interp(
            x,
            np.linspace(0, INTERVALS_PER_DAY - 1, 12),
            rng.normal(0, peak_agents * 0.12, 12),
        )
        fast_noise = rng.normal(0, peak_agents * 0.06, INTERVALS_PER_DAY)
        curve += slow_noise + fast_noise
        curve = np.clip(curve, 0, None)
        demand[offset:offset + INTERVALS_PER_DAY] = curve

    return np.round(demand).astype(int)


def demand_to_dataframe(demand: np.ndarray) -> pd.DataFrame:
    """Convert a (2016,) array into a DataFrame with time labels."""
    rows = []
    for t in range(TOTAL_INTERVALS):
        day = t // INTERVALS_PER_DAY
        intra = t % INTERVALS_PER_DAY
        h, m = divmod(intra * 5, 60)
        rows.append({
            "Interval": t,
            "Day": DAY_NAMES[day],
            "Time": f"{h:02d}:{m:02d}",
            "Required": int(demand[t]),
        })
    return pd.DataFrame(rows)


def save_sample_csv(path: str = "sample_demand.csv", **kwargs):
    demand = generate_sample_demand(**kwargs)
    df = demand_to_dataframe(demand)
    df.to_csv(path, index=False)
    return path


def save_sample_excel(path: str = "sample_demand.xlsx", **kwargs):
    demand = generate_sample_demand(**kwargs)
    df = demand_to_dataframe(demand)
    df.to_excel(path, index=False)
    return path


if __name__ == "__main__":
    save_sample_csv()
    print("Saved sample_demand.csv")
