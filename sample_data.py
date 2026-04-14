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

        if day < 5:  # weekday
            curve = (
                _bell(x, 108, 30, peak_agents * 0.7) +       # 09:00 peak
                _bell(x, 168, 25, peak_agents) +              # 14:00 peak
                _bell(x, 204, 35, peak_agents * 0.55) +       # 17:00 tail
                base_agents
            )
        else:  # weekend
            curve = (
                _bell(x, 144, 40, peak_agents * 0.45) +       # noon peak
                base_agents * 0.6
            )

        # add a little noise
        curve += rng.normal(0, 0.8, INTERVALS_PER_DAY)
        curve = np.clip(curve, 0, None)
        demand[offset:offset + INTERVALS_PER_DAY] = curve

    return np.round(demand).astype(int)


def generate_noisy_demand(
    peak_agents: int = 25,
    base_agents: int = 2,
    noise_level: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Return a (2016,) int array with highly irregular demand.

    Each day gets randomised peak counts, positions, and widths.
    High-frequency noise and random spikes are layered on top,
    producing erratic workloads useful for stress-testing the solver.

    Parameters
    ----------
    noise_level : float
        Amplitude multiplier for random noise and spikes (1 = mild, 5 = chaotic).
    """
    rng = np.random.default_rng(seed)
    demand = np.zeros(TOTAL_INTERVALS, dtype=float)

    for day in range(7):
        offset = day * INTERVALS_PER_DAY
        x = np.arange(INTERVALS_PER_DAY, dtype=float)

        is_weekend = day >= 5
        day_scale = rng.uniform(0.4, 1.0) if is_weekend else rng.uniform(0.6, 1.2)

        # fluctuating baseline
        base = base_agents * rng.uniform(0.4, 1.4)
        curve = np.full(INTERVALS_PER_DAY, base, dtype=float)

        # random number of peaks per day (1-3 weekday, 1-2 weekend)
        n_peaks = int(rng.integers(1, 4 if not is_weekend else 3))
        for _ in range(n_peaks):
            centre = rng.uniform(72, 240)          # 06:00 – 20:00
            width  = rng.uniform(12, 50)
            height = rng.uniform(0.25, 1.0) * peak_agents * day_scale
            curve += _bell(x, centre, width, height)

        # high-frequency noise
        sigma = noise_level * peak_agents * 0.07
        curve += rng.normal(0, sigma, INTERVALS_PER_DAY)

        # random short spikes (surges)
        n_spikes = int(rng.integers(0, max(1, int(noise_level * 2))))
        for _ in range(n_spikes):
            t0 = int(rng.integers(60, 260))
            hw = int(rng.integers(2, 10))
            h  = rng.uniform(0.1, 0.35) * peak_agents * noise_level * 0.4
            lo, hi = max(0, t0 - hw), min(INTERVALS_PER_DAY, t0 + hw)
            curve[lo:hi] += h

        # slow-varying intra-day wander
        wander = np.cumsum(rng.normal(0, noise_level * 0.15, INTERVALS_PER_DAY))
        wander -= wander.mean()
        curve += wander * (peak_agents * 0.04)

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
