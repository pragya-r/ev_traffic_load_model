"""
Monte Carlo Simulation — Daily EV Data Usage
=============================================
Every model parameter is editable: personas can be added/removed/edited,
NBN tiers can be added/removed/edited, bandwidth scenarios can be added/
removed/edited, and shared population ranges are fully adjustable.

Outputs:
  • Total upstream GB distribution (histogram + KPIs)
  • Upload time by NBN tier (box plot + percentile table)
  • Population / timing distributions (arrival, busy hour, etc.)
  • CSV export of raw results
"""

import copy
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import truncnorm

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PHYSICAL / UNIT CONSTANTS  (not user-editable)
# ─────────────────────────────────────────────────────────────
BYTES_TO_GB            = 1024 ** -3
GB_TO_MEGA_BITS        = 8192
MB_TO_BYTES            = 1024 ** 2
HOURS_PER_SIMULATION   = 24
SECONDS_PER_HOUR       = 3600
SECONDS_PER_SIMULATION = HOURS_PER_SIMULATION * SECONDS_PER_HOUR

BAR_COLOR    = "#A23B72"
TIER_PALETTE = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#91cf60",
                "#66c2a5", "#3288bd", "#5e4fa2", "#9e0142"]  # extends if >6 tiers

TELEMETRY_GROUPS = [
    "charging", "driving", "powertrain", "location", "safety",
    "vehicle_state", "media", "climate", "service",
    "vehicle_config", "user_preference",
]
CHARGING_ONLY = {"charging"}
ALWAYS_ACTIVE = {"service", "vehicle_config", "user_preference"}

# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG  (the "factory settings" — used by Reset button)
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "user_personas": {
        "minimal": {
            "weight": 0.2,
            "hours_charging": (4.0,  8.0),
            "hours_driving":  (0.5,  3.0),
            "safety_n":       1,
            "safety_p":       0.005,
        },
        "moderate": {
            "weight": 0.5,
            "hours_charging": (6.0, 10.0),
            "hours_driving":  (1.0,  6.0),
            "safety_n":       1,
            "safety_p":       0.010,
        },
        "heavy": {
            "weight": 0.3,
            "hours_charging": (8.0, 12.0),
            "hours_driving":  (3.0, 10.0),
            "safety_n":       1,
            "safety_p":       0.015,
        },
    },
    "shared_ranges": {
        "base_events_per_hour":   (2.0,  15.0),
        "data_sharing_opt_in_p":  0.3,
        "adas_usage_percent":     (0.0,   1.0),
        "event_clip_size_MB":     (60.0, 100.0),
        "event_clip_metadata_MB":(1.0,    5.0),
        "adas_clip_multiplier":   (2.0,   5.0)
    },
    "telemetry_groups": {
        "charging":        (191.0, 291.0, 1.0,    2.0),
        "driving":         (38.0,   70.0, 1.0,    2.0),
        "powertrain":      (140.0, 260.0, 1.0,    2.0),
        "location":        (40.0,  100.0, 0.5,    1.0),
        "safety":          (12.0,   24.0, 0.1,    0.5),
        "vehicle_state":   (100.0, 142.0, 0.1,    0.5),
        "media":           (24.0,  136.0, 0.05,   0.1),
        "climate":         (89.0,  105.0, 0.05,   0.1),
        "service":         (92.0,  144.0, 0.005,  0.05),
        "vehicle_config":  (8.0,   172.0, 0.005,  0.05),
        "user_preference": (0.0,    25.0, 0.0005, 0.005),
    },
    "nbn_tiers": {
        "1 Mbps":   {"upload_speed_mbps": 1.0,   "weight": 0.05},
        "5 Mbps":   {"upload_speed_mbps": 5.0,   "weight": 0.14},
        "10 Mbps":  {"upload_speed_mbps": 10.0,  "weight": 0.07},
        "20 Mbps":  {"upload_speed_mbps": 20.0,  "weight": 0.40},
        "50 Mbps":  {"upload_speed_mbps": 50.0,  "weight": 0.28},
        "100 Mbps": {"upload_speed_mbps": 100.0, "weight": 0.06},
    },
    "bandwidth_utilisation": {
        "baseline":      1.00,
        "10% reduction": 0.90,
    },
}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state["config"] = copy.deepcopy(DEFAULT_CONFIG)

for _k, _v in {
    "mc_results": None, "mc_elapsed": None,
    "mc_scenario": None, "mc_n": None, "mc_run": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def cfg():
    return st.session_state["config"]


# ─────────────────────────────────────────────────────────────
# SIMULATION CORE  — all reads from `config`, nothing hardcoded
# ─────────────────────────────────────────────────────────────

def _norm_dist(rng, low, high, size):
    if low == high:
        return np.full(size, float(low))
    mu, sigma = (low + high) / 2, (high - low) / 4
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)


def _sample_beta_mixture(rng, n, w=0.90, a1=7.5, b1=2.5, a2=0.5, b2=0.5):
    comp = rng.binomial(1, w, size=n).astype(bool)
    samples = np.empty(n)
    samples[ comp] = rng.beta(a1, b1, size=comp.sum())
    samples[~comp] = rng.beta(a2, b2, size=(~comp).sum())
    return 24 * samples


def _weighted_choice(rng, keys, weights, n):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return rng.choice(keys, size=n, p=weights)


def generate_inputs(rng, n, config):
    personas_cfg = config["user_personas"]
    tiers_cfg    = config["nbn_tiers"]
    shared       = config["shared_ranges"]

    if not personas_cfg:
        raise ValueError("At least one persona is required.")
    if not tiers_cfg:
        raise ValueError("At least one NBN tier is required.")

    persona_names = list(personas_cfg.keys())
    persona_w     = [personas_cfg[p]["weight"] for p in persona_names]
    personas      = _weighted_choice(rng, persona_names, persona_w, n)

    tier_names = list(tiers_cfg.keys())
    tier_w     = [tiers_cfg[t]["weight"] for t in tier_names]
    tiers      = _weighted_choice(rng, tier_names, tier_w, n)
    up_speed   = np.array([tiers_cfg[t]["upload_speed_mbps"] for t in tiers], dtype=float)

    s = {
        "persona": personas, "nbn_tier": tiers,
        "upload_speed_mbps": up_speed,
        "home_arrival_time": _sample_beta_mixture(rng, n),
    }

    for param in ("hours_charging", "hours_driving"):
        col = np.zeros(n)
        for name in persona_names:
            mask = personas == name
            lo, hi = personas_cfg[name][param]
            col[mask] = _norm_dist(rng, lo, hi, mask.sum())
        s[param] = col

    safety_col = np.zeros(n)
    for name in persona_names:
        mask = personas == name
        p_cfg = personas_cfg[name]
        safety_col[mask] = rng.binomial(int(p_cfg["safety_n"]), p_cfg["safety_p"], size=mask.sum())
    s["safety_event"] = safety_col

    s["base_events_per_hour"]   = _norm_dist(rng, *shared["base_events_per_hour"],   n)
    s["data_sharing_opt_in"]    = rng.binomial(1, shared["data_sharing_opt_in_p"],   n)
    s["adas_usage_percent"]     = _norm_dist(rng, *shared["adas_usage_percent"],     n)
    s["event_clip_size_MB"]     = _norm_dist(rng, *shared["event_clip_size_MB"],     n)
    s["event_clip_metadata_MB"] = _norm_dist(rng, *shared["event_clip_metadata_MB"], n)
    s["adas_clip_multiplier"]   = _norm_dist(rng, *shared["adas_clip_multiplier"],   n)

    for group in TELEMETRY_GROUPS:
        sz_lo, sz_hi, rt_lo, rt_hi = config["telemetry_groups"][group]
        s[f"{group}_size"] = _norm_dist(rng, sz_lo, sz_hi, n)
        s[f"{group}_rate"] = _norm_dist(rng, rt_lo, rt_hi, n)

    return pd.DataFrame(s)


def calculate_data_usage(df):
    sec_charging = df["hours_charging"].values * SECONDS_PER_HOUR
    sec_driving  = df["hours_driving"].values  * SECONDS_PER_HOUR
    tel = np.zeros(len(df))
    for group in TELEMETRY_GROUPS:
        sz = df[f"{group}_size"].values
        rt = df[f"{group}_rate"].values
        period = (sec_charging if group in CHARGING_ONLY
                  else SECONDS_PER_SIMULATION if group in ALWAYS_ACTIVE
                  else sec_driving)
        tel += sz * rt * period

    safe  = ((df["event_clip_size_MB"].values + df["event_clip_metadata_MB"].values)
             * MB_TO_BYTES * df["safety_event"].values)
    w_hrs = df["hours_driving"].values * (
        (1 - df["adas_usage_percent"].values)
        + df["adas_usage_percent"].values * df["adas_clip_multiplier"].values
    )
    fleet = (df["data_sharing_opt_in"].values * df["base_events_per_hour"].values
             * df["event_clip_size_MB"].values * MB_TO_BYTES * w_hrs)
    up_tot = tel + safe + fleet

    return pd.DataFrame({
        "upstream_bytes":        up_tot,
        "upstream_gb":           up_tot * BYTES_TO_GB,
        "upstream_telemetry_gb": tel   * BYTES_TO_GB,
        "upstream_safety_gb":    safe  * BYTES_TO_GB,
        "upstream_fleet_gb":     fleet * BYTES_TO_GB,
    })


def analyse_by_tier(usage_df, scenario, config):
    up_bits = usage_df["upstream_bytes"].values * BYTES_TO_GB * GB_TO_MEGA_BITS
    util    = config["bandwidth_utilisation"].get(scenario, 1.0)
    return pd.DataFrame({
        f"{tier}_upload_hours": (up_bits / (info["upload_speed_mbps"] * util)) / SECONDS_PER_HOUR
        for tier, info in config["nbn_tiers"].items()
    })


@st.cache_data(show_spinner=False)
def run_simulation(n, scenario, seed, config, _cache_key):
    rng = np.random.default_rng(seed)
    inputs = generate_inputs(rng, n, config)
    usage  = calculate_data_usage(inputs)
    tiers  = analyse_by_tier(usage, scenario, config)
    return pd.concat([inputs, usage, tiers], axis=1)


def _config_cache_key(config, n, scenario, seed):
    """Hashable repr so st.cache_data invalidates whenever config changes."""
    return repr((n, scenario, seed, sorted(
        (k, sorted(v.items()) if isinstance(v, dict) else v)
        for k, v in config.items()
    )))


# ─────────────────────────────────────────────────────────────
# HELPERS — timing / busy hour
# ─────────────────────────────────────────────────────────────

def _stats(series):
    return {k: series.quantile(q) if k not in ("mean",) else series.mean()
            for k, q in [("mean", None), ("median", 0.5),
                         ("p5", 0.05), ("p25", 0.25), ("p75", 0.75), ("p95", 0.95)]}


def _to_twelve_hour(h):
    h  = h % 24
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        mm, hh = 0, (hh + 1) % 24
    period = "AM" if hh < 12 else "PM"
    hh12   = hh % 12 or 12
    return f"{hh12}:{mm:02d} {period}"


def _compute_upload_times(results, scenario, config):
    util  = config["bandwidth_utilisation"].get(scenario, 1.0)
    eff   = results["upload_speed_mbps"].values * util
    bits  = results["upstream_bytes"].values * BYTES_TO_GB * GB_TO_MEGA_BITS
    start = results["home_arrival_time"].values
    dur   = (bits / eff) / SECONDS_PER_HOUR
    return start, start + dur


def _compute_busy_hour_profile(results, scenario, config):
    start, end = _compute_upload_times(results, scenario, config)
    util = config["bandwidth_utilisation"].get(scenario, 1.0)
    mbps = results["upload_speed_mbps"].values * util

    hours     = np.arange(0, 30)
    ev_count  = np.zeros(len(hours))
    bandwidth = np.zeros(len(hours))
    for i, h in enumerate(hours):
        active = (start <= h + 1) & (end >= h)
        ev_count[i]  = active.sum()
        bandwidth[i] = mbps[active].sum()
    return pd.DataFrame({"hour": hours, "ev_count": ev_count, "bandwidth_mbps": bandwidth})


# ─────────────────────────────────────────────────────────────
# CONFIG EDITOR UI
# ─────────────────────────────────────────────────────────────

def render_persona_editor():
    # st.markdown("##### User Personas")
    st.caption(
        "Each persona has a population **weight**, charging/driving hour ranges, "
        "and a safety-event Binomial(n, p)."
    )
    personas = cfg()["user_personas"]
    to_delete = None

    for name in list(personas.keys()):
        p = personas[name]
        with st.container(border=True):
            top = st.columns([3, 1])
            new_name = top[0].text_input("Persona name", value=name, key=f"persona_name_{name}")
            if top[1].button("Remove", key=f"persona_del_{name}", use_container_width=True):
                to_delete = name
                continue

            c1, c2, c3 = st.columns(3)
            with c1:
                p["weight"] = st.number_input(
                    "Population weight", min_value=0.0, value=float(p["weight"]),
                    step=0.05, key=f"persona_weight_{name}",
                )
            with c2:
                p["hours_charging"] = tuple(st.slider(
                    "Charging hours", 0.0, 24.0, tuple(p["hours_charging"]),
                    step=0.5, key=f"persona_chg_{name}",
                ))
            with c3:
                p["hours_driving"] = tuple(st.slider(
                    "Driving hours", 0.0, 24.0, tuple(p["hours_driving"]),
                    step=0.5, key=f"persona_drv_{name}",
                ))

            c4, c5 = st.columns(2)
            with c4:
                p["safety_n"] = st.number_input(
                    "Safety event — n (Binomial trials)", min_value=0, value=int(p["safety_n"]),
                    step=1, key=f"persona_safety_n_{name}",
                )
            with c5:
                p["safety_p"] = st.number_input(
                    "Safety event — p (probability)", min_value=0.0, max_value=1.0,
                    value=float(p["safety_p"]), step=0.001, format="%.4f",
                    key=f"persona_safety_p_{name}",
                )

            if new_name != name and new_name.strip():
                if new_name in personas:
                    st.error(f"A persona named '{new_name}' already exists.")
                else:
                    personas[new_name] = personas.pop(name)

    if to_delete:
        del personas[to_delete]
        st.rerun()

    if st.button("➕ Add persona", use_container_width=True, key="add_persona_btn"):
        base = "new_persona"
        new_name, i = base, 1
        while new_name in personas:
            new_name = f"{base}_{i}"
            i += 1
        personas[new_name] = {
            "weight": 0.1, "hours_charging": (4.0, 8.0), "hours_driving": (1.0, 4.0),
            "safety_n": 1, "safety_p": 0.01,
        }
        st.rerun()

    total_w = sum(p["weight"] for p in personas.values())
    if personas:
        ok = abs(total_w - 1.0) < 1e-6
        (st.caption if ok else st.error)(
            f"Total weight: {total_w:.4f}  "
            + ("✓" if ok else "(weights must sum exactly to 1.0)")
        )


def render_advanced_editor():
    st.caption(
        "These parameters apply across all personas and are not persona-specific. "
        "Adjust only if you understand the underlying model."
    )
    with st.expander("Behavioural ranges", expanded=False):
        sr = cfg()["shared_ranges"]

        c1, c2, c3 = st.columns(3)
        with c1:
            sr["base_events_per_hour"] = tuple(st.slider(
                "Base events / hour", 0.0, 30.0, tuple(sr["base_events_per_hour"]),
                step=0.5, key="sr_base_ev",
            ))
            sr["event_clip_size_MB"] = tuple(st.slider(
                "Event clip size (MB)", 10.0, 200.0, tuple(sr["event_clip_size_MB"]),
                step=5.0, key="sr_clip_sz",
            ))
        with c2:
            sr["adas_usage_percent"] = tuple(st.slider(
                "ADAS usage fraction", 0.0, 1.0, tuple(sr["adas_usage_percent"]),
                step=0.05, key="sr_adas",
            ))
            sr["event_clip_metadata_MB"] = tuple(st.slider(
                "Clip metadata (MB)", 0.5, 20.0, tuple(sr["event_clip_metadata_MB"]),
                step=0.5, key="sr_clip_meta",
            ))
        with c3:
            sr["adas_clip_multiplier"] = tuple(st.slider(
                "ADAS clip multiplier", 1.0, 10.0, tuple(sr["adas_clip_multiplier"]),
                step=0.5, key="sr_adas_mult",
            ))
            sr["data_sharing_opt_in_p"] = st.slider(
                "Data-sharing opt-in probability", 0.0, 1.0,
                float(sr["data_sharing_opt_in_p"]), step=0.05, key="sr_optin_p",
            )
    # (size in bytes / msg, rate in Hz)
    with st.expander("Telemetry group ranges", expanded=False):
        tg = cfg()["telemetry_groups"]
        st.caption(
            "For each telemetry group, message sizes are measured in bytes and message rate in Hz. "
        )
        for group in TELEMETRY_GROUPS:
            sz_lo, sz_hi, rt_lo, rt_hi = tg[group]
            st.markdown(f"**{group.replace('_', ' ').title()}**")
            tc1, tc2 = st.columns(2)
            with tc1:
                new_sz = st.slider(
                    "Size range (bytes)", 0.0, 400.0, (float(sz_lo), float(sz_hi)),
                    step=1.0, key=f"tg_size_{group}",
                )
            with tc2:
                new_rt = st.slider(
                    "Rate range (Hz)", 0.0, 5.0, (float(rt_lo), float(rt_hi)),
                    step=0.005, key=f"tg_rate_{group}", format="%.4f",
                )
            tg[group] = (new_sz[0], new_sz[1], new_rt[0], new_rt[1])


# ─────────────────────────────────────────────────────────────
# MANUAL DICTIONARY INPUT
# ─────────────────────────────────────────────────────────────
_DICT_REQUIRED_TOP_KEYS = ["USER_PERSONAS", "SHARED_RANGES", "NBN_TIERS", "BANDWIDTH_UTILISATION"]

_DEFAULT_DICT_STR = """{
    'USER_PERSONAS': {
        'minimal': {
            'weight': 0.2,
            'params': {
                'hours_charging': (4, 8),
                'hours_driving':  (0.5, 3),
                'safety_event':   (1, 0.005),   # (n, p) for Binomial
            },
        },
        'moderate': {
            'weight': 0.5,
            'params': {
                'hours_charging': (6, 10),
                'hours_driving':  (1, 6),
                'safety_event':   (1, 0.010),
            },
        },
        'heavy': {
            'weight': 0.3,
            'params': {
                'hours_charging': (8, 12),
                'hours_driving':  (3, 10),
                'safety_event':   (1, 0.015),
            },
        },
    },

    'SHARED_RANGES': {
        # Behavioural
        'base_events_per_hour':   (2, 15),
        'data_sharing_opt_in':    (1, 0.3),    # Binomial (n, p)
        'adas_usage_percent':     (0, 1),
        'event_clip_size_MB':     (60, 100),
        'event_clip_metadata_MB': (1, 5),
        'adas_clip_multiplier':   (2, 5),
        # Telemetry groups: (size_min_B, size_max_B, rate_min_Hz, rate_max_Hz)
        'charging':       (191, 291, 1.0,    2.0),
        'driving':        (38,   70, 1.0,    2.0),
        'powertrain':     (140, 260, 1.0,    2.0),
        'location':       (40,  100, 0.5,    1.0),
        'safety':         (12,   24, 0.1,    0.5),
        'vehicle_state':  (100, 142, 0.1,    0.5),
        'media':          (24,  136, 0.05,   0.1),
        'climate':        (89,  105, 0.05,   0.1),
        'service':        (92,  144, 0.005,  0.05),
        'vehicle_config': (8,   172, 0.005,  0.05),
        'user_preference':(0,    25, 0.0005, 0.005),
    },

    'NBN_TIERS': {
        '1 Mbps':   {'upload_speed_mbps': 1,   'weight': 0.05},
        '5 Mbps':   {'upload_speed_mbps': 5,   'weight': 0.14},
        '10 Mbps':  {'upload_speed_mbps': 10,  'weight': 0.07},
        '20 Mbps':  {'upload_speed_mbps': 20,  'weight': 0.40},
        '50 Mbps':  {'upload_speed_mbps': 50,  'weight': 0.28},
        '100 Mbps': {'upload_speed_mbps': 100, 'weight': 0.06},
    },

    'BANDWIDTH_UTILISATION': {
        'baseline':      1.00,
        '10% reduction': 0.90,
    },
}"""


def _parse_dict_to_config(raw_text):
    """Parse the notebook-style dict literal and convert it into the internal
    config schema. Raises ValueError with a human-readable message on any
    problem — never normalises weights."""
    try:
        parsed = eval(raw_text, {"__builtins__": {}})
    except Exception as e:
        raise ValueError(f"Could not parse as a Python dict literal: {e}")

    if not isinstance(parsed, dict):
        raise ValueError("Input must be a single Python dict literal.")

    missing = [k for k in _DICT_REQUIRED_TOP_KEYS if k not in parsed]
    if missing:
        raise ValueError(f"Missing required top-level keys: {', '.join(missing)}")

    # ── USER_PERSONAS ────────────────────────────────────────
    raw_personas = parsed["USER_PERSONAS"]
    if not isinstance(raw_personas, dict) or not raw_personas:
        raise ValueError("USER_PERSONAS must be a non-empty dict.")

    personas = {}
    for name, spec in raw_personas.items():
        if not isinstance(spec, dict) or "weight" not in spec or "params" not in spec:
            raise ValueError(f"Persona '{name}' must have 'weight' and 'params' keys.")
        p = spec["params"]
        for req in ("hours_charging", "hours_driving", "safety_event"):
            if req not in p:
                raise ValueError(f"Persona '{name}' is missing params['{req}'].")
            v = p[req]
            if not (isinstance(v, tuple) and len(v) == 2):
                raise ValueError(f"Persona '{name}' params['{req}'] must be a 2-tuple, got {v!r}.")
        if p["hours_charging"][0] > p["hours_charging"][1]:
            raise ValueError(f"Persona '{name}': hours_charging min > max.")
        if p["hours_driving"][0] > p["hours_driving"][1]:
            raise ValueError(f"Persona '{name}': hours_driving min > max.")

        personas[name] = {
            "weight":         float(spec["weight"]),
            "hours_charging": (float(p["hours_charging"][0]), float(p["hours_charging"][1])),
            "hours_driving":  (float(p["hours_driving"][0]),  float(p["hours_driving"][1])),
            "safety_n":       int(p["safety_event"][0]),
            "safety_p":       float(p["safety_event"][1]),
        }

    persona_w_sum = sum(p["weight"] for p in personas.values())
    if abs(persona_w_sum - 1.0) > 1e-6:
        raise ValueError(f"USER_PERSONAS weights must sum to 1.0 (currently {persona_w_sum:.4f}).")

    # ── SHARED_RANGES ────────────────────────────────────────
    raw_shared = parsed["SHARED_RANGES"]
    if not isinstance(raw_shared, dict):
        raise ValueError("SHARED_RANGES must be a dict.")

    _scalar_2tuple_keys = [
        "base_events_per_hour", "adas_usage_percent", "event_clip_size_MB",
        "event_clip_metadata_MB", "adas_clip_multiplier"
    ]
    _binom_keys = {
        "data_sharing_opt_in": "data_sharing_opt_in_p"
    }

    shared = {}
    for key in _scalar_2tuple_keys:
        if key not in raw_shared:
            raise ValueError(f"SHARED_RANGES is missing '{key}'.")
        v = raw_shared[key]
        if not (isinstance(v, tuple) and len(v) == 2):
            raise ValueError(f"SHARED_RANGES['{key}'] must be a 2-tuple, got {v!r}.")
        if v[0] > v[1]:
            raise ValueError(f"SHARED_RANGES['{key}']: min > max.")
        shared[key] = (float(v[0]), float(v[1]))

    for raw_key, internal_key in _binom_keys.items():
        if raw_key not in raw_shared:
            raise ValueError(f"SHARED_RANGES is missing '{raw_key}'.")
        v = raw_shared[raw_key]
        if not (isinstance(v, tuple) and len(v) == 2):
            raise ValueError(f"SHARED_RANGES['{raw_key}'] must be a (n, p) tuple, got {v!r}.")
        shared[internal_key] = float(v[1])  # n is always 1 in this model; only p is used

    telemetry = {}
    for group in TELEMETRY_GROUPS:
        if group not in raw_shared:
            raise ValueError(f"SHARED_RANGES is missing telemetry group '{group}'.")
        v = raw_shared[group]
        if not (isinstance(v, tuple) and len(v) == 4):
            raise ValueError(
                f"SHARED_RANGES['{group}'] must be a 4-tuple "
                f"(size_min, size_max, rate_min, rate_max), got {v!r}."
            )
        sz_lo, sz_hi, rt_lo, rt_hi = (float(x) for x in v)
        if sz_lo > sz_hi:
            raise ValueError(f"SHARED_RANGES['{group}']: size min > max.")
        if rt_lo > rt_hi:
            raise ValueError(f"SHARED_RANGES['{group}']: rate min > max.")
        telemetry[group] = (sz_lo, sz_hi, rt_lo, rt_hi)

    # ── NBN_TIERS ────────────────────────────────────────────
    raw_tiers = parsed["NBN_TIERS"]
    if not isinstance(raw_tiers, dict) or not raw_tiers:
        raise ValueError("NBN_TIERS must be a non-empty dict.")

    tiers = {}
    for name, spec in raw_tiers.items():
        if not isinstance(spec, dict) or "upload_speed_mbps" not in spec or "weight" not in spec:
            raise ValueError(f"NBN tier '{name}' must have 'upload_speed_mbps' and 'weight'.")
        tiers[name] = {
            "upload_speed_mbps": float(spec["upload_speed_mbps"]),
            "weight":             float(spec["weight"]),
        }

    tier_w_sum = sum(t["weight"] for t in tiers.values())
    if abs(tier_w_sum - 1.0) > 1e-6:
        raise ValueError(f"NBN_TIERS weights must sum to 1.0 (currently {tier_w_sum:.4f}).")

    # ── BANDWIDTH_UTILISATION ─────────────────────────────────
    raw_bw = parsed["BANDWIDTH_UTILISATION"]
    if not isinstance(raw_bw, dict) or not raw_bw:
        raise ValueError("BANDWIDTH_UTILISATION must be a non-empty dict.")
    bandwidth = {str(k): float(v) for k, v in raw_bw.items()}

    return {
        "user_personas":         personas,
        "shared_ranges":          shared,
        "telemetry_groups":       telemetry,
        "nbn_tiers":              tiers,
        "bandwidth_utilisation":  bandwidth,
    }


def render_dict_input_tab():
    # st.markdown("##### Manual Dictionary Input")
    st.caption(
        "Paste a Python dict literal matching the notebook's original structure — "
        "`USER_PERSONAS`, `SHARED_RANGES`, `NBN_TIERS`, `BANDWIDTH_UTILISATION`. "
        "Weights must sum to exactly 1.0."
        "Loading here overwrites the editors in the other tabs."
    )

    if "dict_input_raw" not in st.session_state:
        st.session_state["dict_input_raw"] = _DEFAULT_DICT_STR

    raw = st.text_area(
        "Parameter dictionary",
        height=420,
        key="dict_input_raw",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        validate_clicked = st.button("Validate", use_container_width=True, key="dict_validate_btn")
    with c2:
        load_clicked = st.button("Load into config", type="primary",
                                 use_container_width=True, key="dict_load_btn")

    if validate_clicked or load_clicked:
        try:
            new_config = _parse_dict_to_config(raw)
            st.success("✅ Dictionary is valid.")
            if load_clicked:
                st.session_state["config"] = new_config
                st.success("✅ Loaded into config — switch tabs to see the values, or run the simulation.")
        except ValueError as e:
            st.error(f"❌ {e}")


def render_nbn_tier_editor():
    # st.markdown("##### NBN Tiers")
    st.caption("Each tier has an upload speed (Mbps) and a population weight.")
    tiers = cfg()["nbn_tiers"]
    to_delete = None

    for name in list(tiers.keys()):
        t = tiers[name]
        with st.container(border=True):
            cols = st.columns([3, 2, 2, 1])
            new_name = cols[0].text_input("Tier name", value=name, key=f"tier_name_{name}")
            t["upload_speed_mbps"] = cols[1].number_input(
                "Upload speed (Mbps)", min_value=0.1, value=float(t["upload_speed_mbps"]),
                step=1.0, key=f"tier_speed_{name}",
            )
            t["weight"] = cols[2].number_input(
                "Population weight", min_value=0.0, value=float(t["weight"]),
                step=0.01, key=f"tier_weight_{name}",
            )
            if cols[3].button("Remove", key=f"tier_del_{name}", use_container_width=True):
                to_delete = name

            if new_name != name and new_name.strip():
                if new_name in tiers:
                    st.error(f"A tier named '{new_name}' already exists.")
                else:
                    tiers[new_name] = tiers.pop(name)

    if to_delete:
        del tiers[to_delete]
        st.rerun()

    if st.button("➕ Add NBN tier", use_container_width=True, key="add_tier_btn"):
        base = "New Tier"
        new_name, i = base, 1
        while new_name in tiers:
            new_name = f"{base} {i}"
            i += 1
        tiers[new_name] = {"upload_speed_mbps": 25.0, "weight": 0.1}
        st.rerun()

    total_w = sum(t["weight"] for t in tiers.values())
    if tiers:
        ok = abs(total_w - 1.0) < 1e-6
        (st.caption if ok else st.error)(
            f"Total weight: {total_w:.4f}  "
            + ("✓" if ok else "(weights must sum exactly to 1.0)")
        )


def render_scenario_editor():
    # st.markdown("##### Bandwidth Utilisation Scenarios")
    st.caption(
        "A scenario scales every NBN tier's effective upload speed by a factor "
        "(e.g. 0.90 = 10% reduction from congestion)."
    )
    scenarios = cfg()["bandwidth_utilisation"]
    to_delete = None

    for name in list(scenarios.keys()):
        with st.container(border=True):
            cols = st.columns([3, 2, 1])
            new_name = cols[0].text_input("Scenario name", value=name, key=f"scen_name_{name}")
            new_val = cols[1].number_input(
                "Utilisation factor", min_value=0.01, max_value=2.0,
                value=float(scenarios[name]), step=0.05, key=f"scen_val_{name}",
            )
            if cols[2].button("Remove", key=f"scen_del_{name}", use_container_width=True):
                to_delete = name
                continue
            scenarios[name] = new_val
            if new_name != name and new_name.strip():
                if new_name in scenarios:
                    st.error(f"A scenario named '{new_name}' already exists.")
                else:
                    scenarios[new_name] = scenarios.pop(name)

    if to_delete:
        del scenarios[to_delete]
        st.rerun()

    if st.button("➕ Add scenario", use_container_width=True, key="add_scenario_btn"):
        base = "New Scenario"
        new_name, i = base, 1
        while new_name in scenarios:
            new_name = f"{base} {i}"
            i += 1
        scenarios[new_name] = 1.0
        st.rerun()


# ─────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────

def render_results(results, scenario, n, elapsed, config):
    upstream = results["upstream_gb"]
    up_stats = _stats(upstream)

    st.caption(f"Completed {n:,} simulations in {elapsed:.2f}s  ·  Scenario: **{scenario}**")
    st.markdown("---")

    st.subheader("Upstream Data — Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean (GB)",            f"{up_stats['mean']:.3f}")
    k2.metric("Median (GB)",          f"{up_stats['median']:.3f}")
    k3.metric("P5 (GB)",              f"{up_stats['p5']:.3f}")
    k4.metric("P95 (GB)",             f"{up_stats['p95']:.3f}")
    k5.metric("Total — all EVs (GB)", f"{upstream.sum():,.1f}")

    st.markdown("---")

    st.subheader("Upstream Data Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=upstream, nbinsx=60,
        marker_color=BAR_COLOR, marker_line_color="black",
        marker_line_width=0.4, opacity=0.82,
    ))
    fig_hist.add_vline(x=up_stats["mean"],   line_dash="dash", line_color="red",
                       annotation_text=f"Mean: {up_stats['mean']:.3f}",
                       annotation_position="top left", annotation_font_color="red")
    # fig_hist.add_vline(x=up_stats["median"], line_dash="dot",  line_color="steelblue",
    #                    annotation_text=f"Median: {up_stats['median']:.3f}",
    #                    annotation_position="top left",  annotation_font_color="steelblue")
    fig_hist.update_layout(
        xaxis=dict(title="Upstream GB / day", showgrid=False),
        yaxis=dict(title="Frequency", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=30, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    st.subheader("Upload Time by NBN Tier")
    tier_cols  = [c for c in results.columns if c.endswith("_upload_hours")]
    tier_names = [c.replace("_upload_hours", "") for c in tier_cols]
    palette    = (TIER_PALETTE * ((len(tier_names) // len(TIER_PALETTE)) + 1))[:len(tier_names)]

    fig_box = go.Figure()
    for col, name, color in zip(tier_cols, tier_names, palette):
        fig_box.add_trace(go.Box(
            y=results[col], name=name,
            marker_color=color, boxmean=True, line_width=1.5,
        ))
    fig_box.update_layout(
        yaxis=dict(title="Upload time (hours)", gridcolor="lightgrey"),
        xaxis=dict(title="NBN Tier"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=420,
        margin=dict(t=30, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.subheader("NBN Tier Distribution")
    speed_to_tier = {info["upload_speed_mbps"]: t for t, info in config["nbn_tiers"].items()}
    tier_order    = list(config["nbn_tiers"].keys())
    tier_counts   = (results["upload_speed_mbps"].map(speed_to_tier)
                     .value_counts().reindex(tier_order, fill_value=0))
    pct = 100 * tier_counts / len(results)

    fig_tier_pop = go.Figure()
    fig_tier_pop.add_trace(go.Bar(
        x=tier_order, y=tier_counts.values,
        marker_color=palette, marker_line_color="black", marker_line_width=0.6,
        text=[f"{p:.1f}%" for p in pct.values], textposition="outside",
    ))
    fig_tier_pop.update_layout(
        xaxis=dict(title="NBN Tier"),
        yaxis=dict(title="Number of users", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=40, b=50, l=60, r=30),
    )
    st.plotly_chart(fig_tier_pop, use_container_width=True)

    st.markdown("---")

    st.subheader("Home Arrival Time Distribution")
    arrival = results["home_arrival_time"] % 24
    arr_mean, arr_median = arrival.mean(), arrival.median()

    fig_arrival = go.Figure()
    fig_arrival.add_trace(go.Histogram(
        x=arrival, xbins=dict(start=0, end=24, size=0.5),
        marker_color="steelblue", marker_line_color="black",
        marker_line_width=0.4, opacity=0.78,
    ))
    fig_arrival.add_vline(x=arr_mean,   line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {_to_twelve_hour(arr_mean)}",
                          annotation_position="top left", annotation_font_color="red")
    # fig_arrival.add_vline(x=arr_median, line_dash="dot",  line_color="blue",
    #                       annotation_text=f"Median: {_to_twelve_hour(arr_median)}",
    #                       annotation_position="top left", annotation_font_color="blue")
    tick_h = list(range(0, 25, 2))
    fig_arrival.update_layout(
        xaxis=dict(title="Time of day", tickmode="array", tickvals=tick_h,
                   ticktext=[_to_twelve_hour(h) for h in tick_h], tickangle=-45),
        yaxis=dict(title="Number of users", gridcolor="lightgrey"),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
        margin=dict(t=30, b=70, l=60, r=30),
    )
    st.plotly_chart(fig_arrival, use_container_width=True)

    st.markdown("---")

    st.subheader("Upload Start & End Times")
    start, end = _compute_upload_times(results, scenario, config)
    tick_h2 = list(range(0, 30, 2))
    bin_edges = dict(start=0, end=30, size=0.5)

    col_s, col_e = st.columns(2)
    for col, vals, title, color in [
        (col_s, start, "Upload Start Time", "steelblue"),
        (col_e, end,   "Upload End Time",   BAR_COLOR),
    ]:
        mean_v, median_v = vals.mean(), np.median(vals)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals, xbins=bin_edges,
            marker_color=color, marker_line_color="black",
            marker_line_width=0.4, opacity=0.78,
        ))
        fig.add_vline(x=mean_v,   line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {_to_twelve_hour(mean_v)}",
                     annotation_position="top left", annotation_font_color="red",
                      )
        # fig.add_vline(x=median_v, line_dash="dot",  line_color="blue",
        #              annotation_text=f"Median: {_to_twelve_hour(median_v)}",
        #              annotation_position="top left", annotation_font_color="blue")
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font_size=14),
            xaxis=dict(title="Time of day", tickmode="array", tickvals=tick_h2,
                       ticktext=[_to_twelve_hour(h) for h in tick_h2], tickangle=-45),
            yaxis=dict(title="Number of users", gridcolor="lightgrey"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=380,
            margin=dict(t=50, b=70, l=60, r=30),
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Network Busy Hour Profile")
    profile = _compute_busy_hour_profile(results, scenario, config)
    tick_h3 = list(range(0, 30, 2))

    col_ev, col_bw = st.columns(2)
    for col, y, ylabel, title, color, unit in [
        (col_ev, "ev_count",      "Number of EVs uploading",   "Active EVs per Hour",            "steelblue", ""),
        (col_bw, "bandwidth_mbps","Aggregate bandwidth (Mbps)", "Aggregate Upload Bandwidth per Hour", BAR_COLOR,   " Mbps"),
    ]:
        vals   = profile[y].values
        hrs    = profile["hour"].values
        peak_h = hrs[np.argmax(vals)]
        peak_v = vals.max()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hrs, y=vals, marker_color=color, marker_line_color="black",
            marker_line_width=0.4, opacity=0.82,
        ))
        fig.add_vline(x=peak_h, line_dash="dash", line_color="red",
                     annotation_text=f"Peak: {peak_v:.0f}{unit} at {_to_twelve_hour(peak_h)}",
                     annotation_position="top left", annotation_font_color="red")
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font_size=14),
            xaxis=dict(title="Hour of day", tickmode="array", tickvals=tick_h3,
                       ticktext=[_to_twelve_hour(h) for h in tick_h3], tickangle=-45),
            yaxis=dict(title=ylabel, gridcolor="lightgrey"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=380,
            margin=dict(t=50, b=70, l=60, r=30),
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Export")
    st.download_button(
        "Download simulation results as CSV",
        data=results.to_csv(index=False),
        file_name=f"mc_{scenario.replace(' ', '_')}_n{n}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────
st.title("Monte Carlo Simulation")
st.markdown(
    "Simulate daily per-EV upstream data usage and upload time across the NBN tier mix. "
    "Every parameter below is editable — add or remove personas, NBN tiers, and "
    "bandwidth scenarios as needed."
)

with st.expander("Simulation Input", expanded=True):
    st.markdown("##### Run settings")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        n = st.number_input("Number of simulations", value=10000, min_value=0, step=1000, key="run_n")
    with fc2:
        scenario_options = list(cfg()["bandwidth_utilisation"].keys())
        scenario = st.selectbox("Scenario", scenario_options, key="run_scenario") if scenario_options else None
    with fc3:
        seed = st.number_input("Random seed", value=42, min_value=0, step=1, key="run_seed")

    st.markdown("---")
    st.markdown("##### Choose Input Method")
    input_mode = st.radio(
        label="Available methods",
        options=["Sliders", "Manual Dictionary Input"],
        horizontal=True,
        key="top_input_mode",
    )
    st.markdown("---")

    if input_mode == "Sliders":
        sub_tabs = st.tabs(["User Personas", "NBN Tiers", "Bandwidth Scenarios", "Advanced Parameters"])
        with sub_tabs[0]:
            render_persona_editor()
        with sub_tabs[1]:
            render_nbn_tier_editor()
        with sub_tabs[2]:
            render_scenario_editor()
        with sub_tabs[3]:
            render_advanced_editor()
    else:
        render_dict_input_tab()

    st.markdown("---")
    run_clicked = st.button("▶  Run simulation", type="primary", use_container_width=True, key="run_btn")


if run_clicked:
    errors = []
    if not cfg()["user_personas"]:
        errors.append("At least one persona is required.")
    if not cfg()["nbn_tiers"]:
        errors.append("At least one NBN tier is required.")
    if not scenario:
        errors.append("At least one bandwidth scenario is required.")

    persona_w_sum = sum(p["weight"] for p in cfg()["user_personas"].values())
    if cfg()["user_personas"] and abs(persona_w_sum - 1.0) > 1e-6:
        errors.append(f"Persona weights must sum to 1.0 (currently {persona_w_sum:.4f}).")

    tier_w_sum = sum(t["weight"] for t in cfg()["nbn_tiers"].values())
    if cfg()["nbn_tiers"] and abs(tier_w_sum - 1.0) > 1e-6:
        errors.append(f"NBN tier weights must sum to 1.0 (currently {tier_w_sum:.4f}).")

    for pname, p in cfg()["user_personas"].items():
        if p["hours_charging"][0] > p["hours_charging"][1]:
            errors.append(f"Persona '{pname}': charging hours min > max.")
        if p["hours_driving"][0] > p["hours_driving"][1]:
            errors.append(f"Persona '{pname}': driving hours min > max.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        config_snapshot = copy.deepcopy(cfg())
        cache_key = _config_cache_key(config_snapshot, n, scenario, seed)
        with st.spinner(f"Running {n:,} simulations…"):
            t0 = time.perf_counter()
            results = run_simulation(n, scenario, seed, config_snapshot, cache_key)
            st.session_state["mc_results"]       = results
            st.session_state["mc_elapsed"]       = time.perf_counter() - t0
            st.session_state["mc_scenario"]      = scenario
            st.session_state["mc_n"]             = n
            st.session_state["mc_run"]           = True
            st.session_state["mc_config_used"]   = config_snapshot

if st.session_state["mc_run"] and st.session_state["mc_results"] is not None:
    render_results(
        st.session_state["mc_results"],
        st.session_state["mc_scenario"],
        st.session_state["mc_n"],
        st.session_state["mc_elapsed"],
        st.session_state["mc_config_used"],
    )