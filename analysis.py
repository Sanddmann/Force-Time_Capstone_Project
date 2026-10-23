import io

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfiltfilt


def load_forcedecks_csv(file):
    """
    Load a ForceDecks Raw Data Export CSV.
    """

    if hasattr(file, "read"):
        file.seek(0)
        text = file.read()

        if isinstance(text, bytes):
            text = text.decode("utf-8-sig")

    else:
        with open(file, "r", encoding="utf-8-sig") as f:
            text = f.read()

    lines = text.splitlines()

    metadata = {}

    for line in lines[:9]:

        if "," not in line:
            continue

        key, value = line.split(",", 1)

        metadata[key.strip()] = value.strip().strip('"')

    header_idx = None

    for i, line in enumerate(lines):

        if line.startswith("Time,Left,Right"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not locate ForceDecks Time/Left/Right header."
        )

    data_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(io.StringIO(data_text))

    required_columns = ["Time", "Left", "Right"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required ForceDecks columns: {missing}"
        )

    df = df[required_columns].copy()

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    df["Fz"] = df["Left"] + df["Right"]

    df["Time_Rel"] = (
        df["Time"] - df["Time"].iloc[0]
    )

    return df, metadata


def lowpass_force(
    df,
    fs,
    cutoff_hz=15.0,
    filter_order=4
):
    """
    Apply a zero-phase Butterworth low-pass filter
    to bilateral vertical ground reaction force.
    """

    if "Fz" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'Fz' column."
        )

    if fs <= 0:
        raise ValueError(
            "Sampling frequency must be greater than 0."
        )

    if cutoff_hz <= 0 or cutoff_hz >= (fs / 2):
        raise ValueError(
            "Cutoff frequency must be greater than 0 "
            "and below the Nyquist frequency."
        )

    df = df.copy()

    sos = butter(
        filter_order,
        cutoff_hz,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    df["Fz_Filtered"] = sosfiltfilt(
        sos,
        df["Fz"]
    )

    return df


def lowpass_landing_force(
    df,
    fs,
    cutoff_hz=85.0,
    filter_order=4
):
    """
    Apply a separate zero-phase Butterworth low-pass filter
    to bilateral vertical ground reaction force for landing
    impact analysis.
    """

    if "Fz" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'Fz' column."
        )

    if fs <= 0:
        raise ValueError(
            "Sampling frequency must be greater than 0."
        )

    if cutoff_hz <= 0 or cutoff_hz >= (fs / 2):
        raise ValueError(
            "Landing cutoff frequency must be greater than 0 "
            "and below the Nyquist frequency."
        )

    df = df.copy()

    sos = butter(
        filter_order,
        cutoff_hz,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    df["Fz_Landing_Filtered"] = sosfiltfilt(
        sos,
        df["Fz"]
    )

    return df


def get_sampling_frequency(df):
    """
    Estimate sampling frequency from the Time_Rel column.
    """

    if "Time_Rel" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Time_Rel' column."
        )

    time = df["Time_Rel"].to_numpy()

    if len(time) < 2:
        raise ValueError(
            "At least two time samples are required."
        )

    dt = np.diff(time)

    if np.any(dt <= 0):
        raise ValueError(
            "Time values must be strictly increasing."
        )

    median_dt = np.median(dt)

    fs = 1 / median_dt

    return fs


def estimate_body_weight(
    df,
    weighing_start_s=0.0,
    weighing_end_s=0.75,
    g=9.81
):
    """
    Estimate body weight and body mass from a quiet-standing
    weighing period using filtered vertical ground reaction force.
    """

    required_columns = [
        "Time_Rel",
        "Fz_Filtered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if weighing_end_s <= weighing_start_s:
        raise ValueError(
            "Weighing end time must be after weighing start time."
        )

    weighing_mask = (
        (df["Time_Rel"] >= weighing_start_s) &
        (df["Time_Rel"] <= weighing_end_s)
    )

    weighing_force = df.loc[
        weighing_mask,
        "Fz_Filtered"
    ]

    if weighing_force.empty:
        raise ValueError(
            "No samples found within the selected weighing period."
        )

    body_weight = weighing_force.mean()
    body_weight_sd = weighing_force.std()

    mass = body_weight / g

    return body_weight, body_weight_sd, mass


def calculate_com_mechanics(
    df,
    body_weight,
    mass
):
    """
    Calculate net force, COM acceleration, velocity,
    displacement, and power from filtered vertical force.
    """

    required_columns = [
        "Time_Rel",
        "Fz_Filtered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if mass <= 0:
        raise ValueError(
            "Body mass must be greater than 0."
        )

    df = df.copy()

    df["Net_Force"] = (
        df["Fz_Filtered"] - body_weight
    )

    df["Acceleration"] = (
        df["Net_Force"] / mass
    )

    df["Velocity"] = cumulative_trapezoid(
        df["Acceleration"],
        df["Time_Rel"],
        initial=0
    )

    df["Displacement"] = cumulative_trapezoid(
        df["Velocity"],
        df["Time_Rel"],
        initial=0
    )

    df["Power"] = (
        df["Fz_Filtered"] * df["Velocity"]
    )

    return df


def detect_movement_onset(
    df,
    body_weight,
    body_weight_sd,
    fs,
    threshold_sd=5.0,
    min_duration_ms=30.0,
    backup_ms=30.0
):
    """
    Detect CMJ movement onset from filtered vertical force.
    """

    if "Fz_Filtered" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'Fz_Filtered'."
        )

    lower_threshold = (
        body_weight - threshold_sd * body_weight_sd
    )

    upper_threshold = (
        body_weight + threshold_sd * body_weight_sd
    )

    outside_threshold = (
        (df["Fz_Filtered"] < lower_threshold) |
        (df["Fz_Filtered"] > upper_threshold)
    )

    min_samples = max(
        1,
        int(round((min_duration_ms / 1000) * fs))
    )

    sustained = (
        outside_threshold
        .rolling(window=min_samples)
        .sum()
        >= min_samples
    )

    if not sustained.any():
        raise ValueError(
            "Movement onset could not be detected."
        )

    first_sustained_idx = (
        sustained[sustained].index[0]
    )

    threshold_crossing_idx = (
        first_sustained_idx
        - min_samples
        + 1
    )

    backup_samples = int(
        round((backup_ms / 1000) * fs)
    )

    movement_onset_idx = max(
        df.index[0],
        threshold_crossing_idx - backup_samples
    )

    movement_onset_time = df.loc[
        movement_onset_idx,
        "Time_Rel"
    ]

    return (
        movement_onset_idx,
        movement_onset_time,
        lower_threshold,
        upper_threshold
    )


# Takeoff detection


def detect_takeoff(
    df,
    search_start_idx,
    fs,
    contact_threshold_n=20.0,
    confirmation_ms=30.0
):
    """
    Detect takeoff using raw vertical force.

    This function is retained for squat-jump analysis.

    Takeoff is the first raw-force sample below the contact
    threshold that begins a sustained interval below the
    threshold for the required confirmation duration.
    """

    required_columns = [
        "Fz",
        "Time_Rel"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    confirmation_samples = max(
        1,
        int(round((confirmation_ms / 1000) * fs))
    )

    below_contact = (
        (df["Fz"] < contact_threshold_n) &
        (df.index >= search_start_idx)
    )

    confirmed_flight = (
        below_contact
        .rolling(window=confirmation_samples)
        .sum()
        >= confirmation_samples
    )

    if not confirmed_flight.any():
        raise ValueError(
            "Takeoff could not be detected."
        )

    flight_confirm_idx = (
        confirmed_flight[
            confirmed_flight
        ].index[0]
    )

    takeoff_idx = (
        flight_confirm_idx
        - confirmation_samples
        + 1
    )

    if takeoff_idx <= search_start_idx:
        raise ValueError(
            "Detected takeoff occurs before the "
            "takeoff search-start event."
        )

    takeoff_time = df.loc[
        takeoff_idx,
        "Time_Rel"
    ]

    return takeoff_idx, takeoff_time


def detect_cmj_takeoff(
    df,
    movement_onset_idx,
    fs,
    provisional_threshold_n=20.0,
    threshold_sd=5.0
):
    """
    Detect CMJ takeoff using the true unloaded flight interval.

    Step 1
    ------
    Identify all contiguous raw-force intervals below the
    provisional 20-N threshold after movement onset.

    Step 2
    ------
    Select the longest contiguous unloaded interval as the
    provisional flight phase. This prevents a short low-force
    interval during the countermovement from being mistaken
    for flight.

    Step 3
    ------
    Estimate unloaded flight-force noise from the middle
    portion of the provisional flight phase.

    Step 4
    ------
    Calculate a trial-specific takeoff threshold:

        mean flight force + 5 * SD flight force

    Step 5
    ------
    Working backward from the provisional flight interval,
    identify the beginning of the continuous raw-force region
    below the trial-specific threshold.

    The resulting event is the final CMJ takeoff.
    """

    required_columns = [
        "Fz",
        "Time_Rel"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if movement_onset_idx >= df.index[-1]:
        raise ValueError(
            "No data are available after movement onset."
        )

    # Provisional unloaded intervals

    post_onset = df.loc[
        movement_onset_idx:
    ]

    below_provisional = (
        post_onset["Fz"]
        < provisional_threshold_n
    )

    # Identify contiguous True/False groups
    group_id = (
        below_provisional
        != below_provisional.shift()
    ).cumsum()

    candidate_intervals = []

    for _, group in post_onset.groupby(
        group_id
    ):

        group_start_idx = group.index[0]

        if not below_provisional.loc[
            group_start_idx
        ]:
            continue

        start_idx = group.index[0]
        end_idx = group.index[-1]

        duration_samples = (
            end_idx - start_idx + 1
        )

        candidate_intervals.append(
            (
                start_idx,
                end_idx,
                duration_samples
            )
        )

    if not candidate_intervals:
        raise ValueError(
            "No unloaded CMJ flight interval could be detected."
        )

    # Select longest unloaded interval

    (
        provisional_takeoff_idx,
        provisional_flight_end_idx,
        provisional_flight_samples
    ) = max(
        candidate_intervals,
        key=lambda interval: interval[2]
    )

    if provisional_takeoff_idx <= movement_onset_idx:
        raise ValueError(
            "Provisional CMJ takeoff occurs before "
            "or at movement onset."
        )

    # Flight-noise window
    #
    # Use the middle 50% of the provisional flight
    # interval to avoid the force transitions around
    # takeoff and landing.

    quarter_samples = int(
        round(
            provisional_flight_samples
            * 0.25
        )
    )

    flight_noise_start_idx = (
        provisional_takeoff_idx
        + quarter_samples
    )

    flight_noise_end_idx = (
        provisional_flight_end_idx
        - quarter_samples
    )

    if (
        flight_noise_end_idx
        <= flight_noise_start_idx
    ):
        raise ValueError(
            "Provisional CMJ flight interval is too short "
            "to estimate flight-force noise."
        )

    flight_noise = df.loc[
        flight_noise_start_idx:
        flight_noise_end_idx,
        "Fz"
    ]

    if flight_noise.empty:
        raise ValueError(
            "CMJ flight-noise window contains no samples."
        )

    flight_noise_mean = (
        flight_noise.mean()
    )

    flight_noise_sd = (
        flight_noise.std()
    )

    takeoff_threshold = (
        flight_noise_mean
        + threshold_sd * flight_noise_sd
    )

    # Final takeoff
    #
    # Start inside the known provisional flight
    # interval and walk backward until force is
    # no longer below the trial-specific threshold.

    takeoff_idx = provisional_takeoff_idx

    while (
        takeoff_idx > movement_onset_idx
        and
        df.loc[
            takeoff_idx - 1,
            "Fz"
        ] < takeoff_threshold
    ):
        takeoff_idx -= 1

    if takeoff_idx <= movement_onset_idx:
        raise ValueError(
            "Final CMJ takeoff could not be separated "
            "from movement onset."
        )

    takeoff_time = df.loc[
        takeoff_idx,
        "Time_Rel"
    ]

    return (
        takeoff_idx,
        takeoff_time,
        takeoff_threshold,
        flight_noise_mean,
        flight_noise_sd,
        flight_noise_start_idx,
        flight_noise_end_idx
    )


# Landing analysis


def detect_landing(
    df,
    takeoff_idx,
    fs,
    provisional_threshold_n=20.0,
    flight_window_start_ms=150.0,
    flight_window_end_ms=50.0,
    threshold_sd=5.0
):
    """
    Detect initial landing contact using a trial-specific
    flight-noise threshold.
    """

    required_columns = [
        "Fz",
        "Time_Rel"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if takeoff_idx >= df.index[-1]:
        raise ValueError(
            "No data are available after takeoff."
        )

    post_takeoff = df.loc[
        takeoff_idx + 1:
    ]

    provisional_contact = (
        post_takeoff["Fz"]
        > provisional_threshold_n
    )

    if not provisional_contact.any():
        raise ValueError(
            "A provisional landing contact could not be detected."
        )

    provisional_landing_idx = (
        provisional_contact[
            provisional_contact
        ].index[0]
    )

    window_start_samples = int(
        round(
            (flight_window_start_ms / 1000)
            * fs
        )
    )

    window_end_samples = int(
        round(
            (flight_window_end_ms / 1000)
            * fs
        )
    )

    flight_window_start_idx = (
        provisional_landing_idx
        - window_start_samples
    )

    flight_window_end_idx = (
        provisional_landing_idx
        - window_end_samples
    )

    if flight_window_start_idx <= takeoff_idx:
        raise ValueError(
            "Flight duration is too short for the selected "
            "landing flight-noise window."
        )

    flight_noise = df.loc[
        flight_window_start_idx:
        flight_window_end_idx,
        "Fz"
    ]

    if flight_noise.empty:
        raise ValueError(
            "Flight-noise window contains no samples."
        )

    flight_noise_mean = (
        flight_noise.mean()
    )

    flight_noise_sd = (
        flight_noise.std()
    )

    landing_threshold = (
        flight_noise_mean
        + threshold_sd * flight_noise_sd
    )

    landing_search = df.loc[
        flight_window_end_idx + 1:
    ]

    above_landing_threshold = (
        landing_search["Fz"]
        > landing_threshold
    )

    if not above_landing_threshold.any():
        raise ValueError(
            "Landing contact could not be detected "
            "using the flight-noise threshold."
        )

    landing_idx = (
        above_landing_threshold[
            above_landing_threshold
        ].index[0]
    )

    if landing_idx <= takeoff_idx:
        raise ValueError(
            "Detected landing occurs before or at takeoff."
        )

    landing_time = df.loc[
        landing_idx,
        "Time_Rel"
    ]

    return (
        landing_idx,
        landing_time,
        landing_threshold,
        flight_noise_mean,
        flight_noise_sd,
        flight_window_start_idx,
        flight_window_end_idx
    )


def calculate_landing_metrics(
    df,
    mass,
    landing_idx,
    fs,
    landing_window_ms=300.0
):
    """
    Calculate peak landing force during the initial landing
    period following ground contact.
    """

    required_columns = [
        "Fz_Landing_Filtered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    landing_window_samples = int(
        round(
            (landing_window_ms / 1000)
            * fs
        )
    )

    landing_end_idx = min(
        landing_idx + landing_window_samples,
        df.index[-1]
    )

    landing_phase = df.loc[
        landing_idx:landing_end_idx
    ]

    if landing_phase.empty:
        raise ValueError(
            "Landing phase contains no samples."
        )

    peak_landing_force = (
        landing_phase[
            "Fz_Landing_Filtered"
        ].max()
    )

    peak_landing_force_idx = (
        landing_phase[
            "Fz_Landing_Filtered"
        ].idxmax()
    )

    peak_landing_force_time = df.loc[
        peak_landing_force_idx,
        "Time_Rel"
    ]

    relative_peak_landing_force = (
        peak_landing_force / mass
    )

    metrics = {
        "peak_landing_force_n": (
            peak_landing_force
        ),

        "relative_peak_landing_force_n_kg": (
            relative_peak_landing_force
        ),

        "peak_landing_force_time_s": (
            peak_landing_force_time
        ),

        "landing_window_ms": (
            landing_window_ms
        )
    }

    return metrics


def calculate_landing_rfd(
    df,
    mass,
    landing_idx,
    fs,
    window_ms=20.0,
    landing_window_ms=300.0
):
    """
    Calculate peak landing loading rate using a moving window.
    """

    required_columns = [
        "Time_Rel",
        "Fz_Landing_Filtered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    window_samples = max(
        1,
        int(round((window_ms / 1000) * fs))
    )

    landing_window_samples = int(
        round(
            (landing_window_ms / 1000)
            * fs
        )
    )

    landing_end_idx = min(
        landing_idx + landing_window_samples,
        df.index[-1]
    )

    last_start_idx = (
        landing_end_idx - window_samples
    )

    if last_start_idx < landing_idx:
        raise ValueError(
            "Landing phase is too short for "
            "the selected loading-rate window."
        )

    phase = df.loc[
        landing_idx:last_start_idx
    ].copy()

    future_force = df[
        "Fz_Landing_Filtered"
    ].shift(-window_samples)

    future_time = df[
        "Time_Rel"
    ].shift(-window_samples)

    delta_force = (
        future_force.loc[phase.index]
        - phase["Fz_Landing_Filtered"]
    )

    delta_time = (
        future_time.loc[phase.index]
        - phase["Time_Rel"]
    )

    loading_rate = (
        delta_force / delta_time
    )

    if loading_rate.dropna().empty:
        raise ValueError(
            "Landing loading rate could not be calculated."
        )

    peak_idx = (
        loading_rate.idxmax()
    )

    peak_loading_rate = (
        loading_rate.loc[peak_idx]
    )

    peak_loading_rate_time = df.loc[
        peak_idx,
        "Time_Rel"
    ]

    metrics = {
        "peak_landing_rfd_n_s": (
            peak_loading_rate
        ),

        "relative_peak_landing_rfd_n_s_kg": (
            peak_loading_rate / mass
        ),

        "peak_landing_rfd_time_s": (
            peak_loading_rate_time
        ),

        "landing_rfd_window_ms": (
            window_ms
        )
    }

    return metrics


# Cmj phases


def detect_cmj_phases(
    df,
    movement_onset_idx,
    takeoff_idx
):
    """
    Detect CMJ phase transition events from COM velocity.

    Velocity analysis is restricted to movement onset through
    the previously detected takeoff event. This prevents flight
    and landing velocity from contaminating CMJ phase detection.
    """

    if "Velocity" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'Velocity'."
        )

    if takeoff_idx <= movement_onset_idx:
        raise ValueError(
            "Takeoff must occur after movement onset."
        )

    jump_window = df.loc[
        movement_onset_idx:takeoff_idx
    ]

    min_velocity_idx = (
        jump_window[
            "Velocity"
        ].idxmin()
    )

    post_min_velocity = df.loc[
        min_velocity_idx:takeoff_idx
    ]

    zero_crossings = (
        (
            post_min_velocity[
                "Velocity"
            ].shift(1) < 0
        )
        &
        (
            post_min_velocity[
                "Velocity"
            ] >= 0
        )
    )

    if not zero_crossings.any():
        raise ValueError(
            "CMJ zero-velocity transition could not be detected."
        )

    zero_velocity_idx = (
        zero_crossings[
            zero_crossings
        ].index[0]
    )

    return (
        min_velocity_idx,
        zero_velocity_idx
    )


def calculate_cmj_metrics(
    df,
    mass,
    movement_onset_idx,
    min_velocity_idx,
    zero_velocity_idx,
    takeoff_idx,
    g=9.81
):
    """
    Calculate core CMJ force-time metrics.
    """

    onset_time = df.loc[
        movement_onset_idx,
        "Time_Rel"
    ]

    min_velocity_time = df.loc[
        min_velocity_idx,
        "Time_Rel"
    ]

    zero_velocity_time = df.loc[
        zero_velocity_idx,
        "Time_Rel"
    ]

    takeoff_time = df.loc[
        takeoff_idx,
        "Time_Rel"
    ]

    unweighting_duration = (
        min_velocity_time - onset_time
    )

    braking_duration = (
        zero_velocity_time
        - min_velocity_time
    )

    propulsive_duration = (
        takeoff_time
        - zero_velocity_time
    )

    contraction_time = (
        takeoff_time - onset_time
    )

    peak_negative_velocity = df.loc[
        min_velocity_idx,
        "Velocity"
    ]

    takeoff_velocity = df.loc[
        takeoff_idx,
        "Velocity"
    ]

    jump_height_m = (
        takeoff_velocity ** 2
    ) / (2 * g)

    onset_displacement = df.loc[
        movement_onset_idx,
        "Displacement"
    ]

    bottom_displacement = (
        df.loc[
            zero_velocity_idx,
            "Displacement"
        ]
        - onset_displacement
    )

    countermovement_depth_m = abs(
        bottom_displacement
    )

    movement_phase = df.loc[
        movement_onset_idx:takeoff_idx
    ]

    peak_force = (
        movement_phase[
            "Fz_Filtered"
        ].max()
    )

    relative_peak_force = (
        peak_force / mass
    )

    propulsive_phase = df.loc[
        zero_velocity_idx:takeoff_idx
    ]

    peak_propulsive_power = (
        propulsive_phase[
            "Power"
        ].max()
    )

    mean_propulsive_power = (
        propulsive_phase[
            "Power"
        ].mean()
    )

    relative_peak_power = (
        peak_propulsive_power / mass
    )

    relative_mean_power = (
        mean_propulsive_power / mass
    )

    braking_phase = df.loc[
        min_velocity_idx:zero_velocity_idx
    ]

    braking_impulse = np.trapezoid(
        braking_phase["Net_Force"],
        braking_phase["Time_Rel"]
    )

    propulsive_impulse = np.trapezoid(
        propulsive_phase["Net_Force"],
        propulsive_phase["Time_Rel"]
    )

    relative_braking_impulse = (
        braking_impulse / mass
    )

    relative_propulsive_impulse = (
        propulsive_impulse / mass
    )

    metrics = {
        "movement_onset_s": onset_time,
        "min_velocity_time_s": min_velocity_time,
        "zero_velocity_time_s": zero_velocity_time,
        "takeoff_time_s": takeoff_time,

        "unweighting_duration_s": unweighting_duration,
        "braking_duration_s": braking_duration,
        "propulsive_duration_s": propulsive_duration,
        "contraction_time_s": contraction_time,

        "peak_negative_velocity_m_s": peak_negative_velocity,
        "takeoff_velocity_m_s": takeoff_velocity,

        "jump_height_m": jump_height_m,
        "countermovement_depth_m": countermovement_depth_m,

        "peak_force_n": peak_force,
        "relative_peak_force_n_kg": relative_peak_force,

        "peak_propulsive_power_w": peak_propulsive_power,
        "relative_peak_power_w_kg": relative_peak_power,

        "mean_propulsive_power_w": mean_propulsive_power,
        "relative_mean_power_w_kg": relative_mean_power,

        "braking_impulse_ns": braking_impulse,
        "propulsive_impulse_ns": propulsive_impulse,

        "relative_braking_impulse_ns_kg": relative_braking_impulse,
        "relative_propulsive_impulse_ns_kg": relative_propulsive_impulse,
    }

    return metrics


def calculate_cmj_rfd(
    df,
    mass,
    min_velocity_idx,
    zero_velocity_idx,
    takeoff_idx,
    fs,
    window_ms=20.0
):
    """
    Calculate phase-specific peak CMJ RFD using a moving window.
    """

    required_columns = [
        "Time_Rel",
        "Fz_Filtered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    window_samples = max(
        1,
        int(round((window_ms / 1000) * fs))
    )

    def peak_phase_rfd(
        start_idx,
        end_idx
    ):

        last_start_idx = (
            end_idx - window_samples
        )

        if last_start_idx < start_idx:
            raise ValueError(
                "Phase is too short for the selected RFD window."
            )

        phase = df.loc[
            start_idx:last_start_idx
        ].copy()

        future_force = df[
            "Fz_Filtered"
        ].shift(-window_samples)

        future_time = df[
            "Time_Rel"
        ].shift(-window_samples)

        delta_force = (
            future_force.loc[
                phase.index
            ]
            - phase[
                "Fz_Filtered"
            ]
        )

        delta_time = (
            future_time.loc[
                phase.index
            ]
            - phase[
                "Time_Rel"
            ]
        )

        rfd = (
            delta_force / delta_time
        )

        if rfd.dropna().empty:
            raise ValueError(
                "RFD could not be calculated."
            )

        peak_idx = (
            rfd.idxmax()
        )

        peak_rfd = (
            rfd.loc[peak_idx]
        )

        peak_time = df.loc[
            peak_idx,
            "Time_Rel"
        ]

        return (
            peak_rfd,
            peak_time
        )

    (
        braking_rfd,
        braking_rfd_time
    ) = peak_phase_rfd(
        min_velocity_idx,
        zero_velocity_idx
    )

    (
        propulsive_rfd,
        propulsive_rfd_time
    ) = peak_phase_rfd(
        zero_velocity_idx,
        takeoff_idx
    )

    metrics = {
        "peak_braking_rfd_n_s": braking_rfd,

        "relative_braking_rfd_n_s_kg": (
            braking_rfd / mass
        ),

        "peak_propulsive_rfd_n_s": propulsive_rfd,

        "relative_peak_propulsive_rfd_n_s_kg": (
            propulsive_rfd / mass
        ),

        "peak_braking_rfd_time_s": braking_rfd_time,

        "peak_propulsive_rfd_time_s": propulsive_rfd_time,

        "rfd_window_ms": window_ms
    }

    return metrics


def analyze_cmj(
    file,
    cutoff_hz=15.0,
    filter_order=4,
    weighing_start_s=0.0,
    weighing_end_s=0.75,
    onset_threshold_sd=5.0,
    onset_min_duration_ms=30.0,
    onset_backup_ms=30.0,
    contact_threshold_n=20.0,
    contact_confirmation_ms=30.0,
    rfd_window_ms=20.0,
    landing_rfd_window_ms=20.0,
    landing_analysis_window_ms=300.0,
    landing_cutoff_hz=85.0,
    g=9.81
):
    """
    Run the complete CMJ force-time analysis pipeline.
    """

    df, metadata = load_forcedecks_csv(
        file
    )

    fs = get_sampling_frequency(
        df
    )

    df = lowpass_force(
        df,
        fs=fs,
        cutoff_hz=cutoff_hz,
        filter_order=filter_order
    )

    df = lowpass_landing_force(
        df,
        fs=fs,
        cutoff_hz=landing_cutoff_hz,
        filter_order=filter_order
    )

    (
        body_weight,
        body_weight_sd,
        mass
    ) = estimate_body_weight(
        df,
        weighing_start_s=weighing_start_s,
        weighing_end_s=weighing_end_s,
        g=g
    )

    df = calculate_com_mechanics(
        df,
        body_weight=body_weight,
        mass=mass
    )

    (
        movement_onset_idx,
        movement_onset_time,
        lower_threshold,
        upper_threshold
    ) = detect_movement_onset(
        df,
        body_weight=body_weight,
        body_weight_sd=body_weight_sd,
        fs=fs,
        threshold_sd=onset_threshold_sd,
        min_duration_ms=onset_min_duration_ms,
        backup_ms=onset_backup_ms
    )

    # CMJ takeoff

    (
        takeoff_idx,
        takeoff_time,
        takeoff_threshold,
        takeoff_flight_noise_mean,
        takeoff_flight_noise_sd,
        takeoff_flight_noise_start_idx,
        takeoff_flight_noise_end_idx
    ) = detect_cmj_takeoff(
        df,
        movement_onset_idx=movement_onset_idx,
        fs=fs,
        provisional_threshold_n=contact_threshold_n,
        threshold_sd=5.0
    )

    # CMJ velocity landmarks
    #
    # These are calculated only after takeoff has
    # already been established from raw force.

    (
        min_velocity_idx,
        zero_velocity_idx
    ) = detect_cmj_phases(
        df,
        movement_onset_idx=movement_onset_idx,
        takeoff_idx=takeoff_idx
    )

    metrics = calculate_cmj_metrics(
        df,
        mass=mass,
        movement_onset_idx=movement_onset_idx,
        min_velocity_idx=min_velocity_idx,
        zero_velocity_idx=zero_velocity_idx,
        takeoff_idx=takeoff_idx,
        g=g
    )

    rfd_metrics = calculate_cmj_rfd(
        df,
        mass=mass,
        min_velocity_idx=min_velocity_idx,
        zero_velocity_idx=zero_velocity_idx,
        takeoff_idx=takeoff_idx,
        fs=fs,
        window_ms=rfd_window_ms
    )

    metrics.update(
        rfd_metrics
    )

    # Landing detection

    (
        landing_idx,
        landing_time,
        landing_threshold,
        flight_noise_mean,
        flight_noise_sd,
        flight_window_start_idx,
        flight_window_end_idx
    ) = detect_landing(
        df,
        takeoff_idx=takeoff_idx,
        fs=fs
    )

    landing_metrics = calculate_landing_metrics(
        df,
        mass=mass,
        landing_idx=landing_idx,
        fs=fs,
        landing_window_ms=landing_analysis_window_ms
    )

    landing_rfd_metrics = calculate_landing_rfd(
        df,
        mass=mass,
        landing_idx=landing_idx,
        fs=fs,
        window_ms=landing_rfd_window_ms,
        landing_window_ms=landing_analysis_window_ms
    )

    metrics.update(
        landing_metrics
    )

    metrics.update(
        landing_rfd_metrics
    )

    results = {
        "sampling_frequency_hz": fs,

        "body_weight_n": body_weight,
        "body_weight_sd_n": body_weight_sd,
        "mass_kg": mass,

        "movement_onset_idx": movement_onset_idx,
        "min_velocity_idx": min_velocity_idx,
        "zero_velocity_idx": zero_velocity_idx,
        "takeoff_idx": takeoff_idx,

        "takeoff_threshold_n": takeoff_threshold,
        "takeoff_flight_noise_mean_n": (
            takeoff_flight_noise_mean
        ),
        "takeoff_flight_noise_sd_n": (
            takeoff_flight_noise_sd
        ),
        "takeoff_flight_noise_start_idx": (
            takeoff_flight_noise_start_idx
        ),
        "takeoff_flight_noise_end_idx": (
            takeoff_flight_noise_end_idx
        ),

        "landing_idx": landing_idx,
        "landing_time_s": landing_time,

        "landing_threshold_n": landing_threshold,
        "flight_noise_mean_n": flight_noise_mean,
        "flight_noise_sd_n": flight_noise_sd,
        "flight_noise_window_start_idx": flight_window_start_idx,
        "flight_noise_window_end_idx": flight_window_end_idx,

        "onset_lower_threshold_n": lower_threshold,
        "onset_upper_threshold_n": upper_threshold,

        "filter_cutoff_hz": cutoff_hz,
        "landing_filter_cutoff_hz": landing_cutoff_hz,
        "filter_order": filter_order,

        "contact_threshold_n": contact_threshold_n,
        "contact_confirmation_ms": contact_confirmation_ms,

        "metadata": metadata
    }

    results.update(
        metrics
    )

    return results, df


# Squat jump analysis


def detect_sj_onset(
    df,
    body_weight,
    body_weight_sd,
    fs,
    threshold_sd=5.0,
    min_duration_ms=30.0
):
    """
    Detect squat jump force onset from filtered vertical force.
    """

    if "Fz_Filtered" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'Fz_Filtered'."
        )

    upper_threshold = (
        body_weight
        + threshold_sd * body_weight_sd
    )

    above_threshold = (
        df["Fz_Filtered"]
        > upper_threshold
    )

    min_samples = max(
        1,
        int(round(
            (min_duration_ms / 1000)
            * fs
        ))
    )

    sustained = (
        above_threshold
        .rolling(
            window=min_samples
        )
        .sum()
        >= min_samples
    )

    if not sustained.any():
        raise ValueError(
            "SJ force onset could not be detected."
        )

    first_sustained_idx = (
        sustained[
            sustained
        ].index[0]
    )

    movement_onset_idx = (
        first_sustained_idx
        - min_samples
        + 1
    )

    movement_onset_time = df.loc[
        movement_onset_idx,
        "Time_Rel"
    ]

    return (
        movement_onset_idx,
        movement_onset_time,
        upper_threshold
    )


def calculate_sj_metrics(
    df,
    mass,
    movement_onset_idx,
    takeoff_idx,
    g=9.81
):
    """
    Calculate core squat jump force-time metrics.
    """

    onset_time = df.loc[
        movement_onset_idx,
        "Time_Rel"
    ]

    takeoff_time = df.loc[
        takeoff_idx,
        "Time_Rel"
    ]

    propulsive_duration = (
        takeoff_time
        - onset_time
    )

    takeoff_velocity = df.loc[
        takeoff_idx,
        "Velocity"
    ]

    jump_height_m = (
        takeoff_velocity ** 2
    ) / (2 * g)

    propulsive_phase = df.loc[
        movement_onset_idx:takeoff_idx
    ]

    peak_force = (
        propulsive_phase[
            "Fz_Filtered"
        ].max()
    )

    relative_peak_force = (
        peak_force / mass
    )

    peak_propulsive_power = (
        propulsive_phase[
            "Power"
        ].max()
    )

    mean_propulsive_power = (
        propulsive_phase[
            "Power"
        ].mean()
    )

    relative_peak_power = (
        peak_propulsive_power
        / mass
    )

    relative_mean_power = (
        mean_propulsive_power
        / mass
    )

    propulsive_impulse = np.trapezoid(
        propulsive_phase[
            "Net_Force"
        ],
        propulsive_phase[
            "Time_Rel"
        ]
    )

    relative_propulsive_impulse = (
        propulsive_impulse
        / mass
    )

    metrics = {
        "movement_onset_s": onset_time,
        "takeoff_time_s": takeoff_time,

        "propulsive_duration_s": propulsive_duration,
        "contraction_time_s": propulsive_duration,

        "takeoff_velocity_m_s": takeoff_velocity,

        "jump_height_m": jump_height_m,

        "peak_force_n": peak_force,
        "relative_peak_force_n_kg": relative_peak_force,

        "peak_propulsive_power_w": peak_propulsive_power,
        "relative_peak_power_w_kg": relative_peak_power,

        "mean_propulsive_power_w": mean_propulsive_power,
        "relative_mean_power_w_kg": relative_mean_power,

        "propulsive_impulse_ns": propulsive_impulse,

        "relative_propulsive_impulse_ns_kg": (
            relative_propulsive_impulse
        ),
    }

    return metrics


def calculate_sj_rfd(
    df,
    mass,
    movement_onset_idx,
    takeoff_idx,
    fs,
    window_ms=20.0
):
    """
    Calculate peak squat jump RFD using a 20-ms moving window.
    """

    required_columns = [
        "Time_Rel",
        "Fz_Filtered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    window_samples = max(
        1,
        int(round(
            (window_ms / 1000)
            * fs
        ))
    )

    last_start_idx = (
        takeoff_idx
        - window_samples
    )

    if last_start_idx < movement_onset_idx:
        raise ValueError(
            "SJ propulsive phase is too short "
            "for the selected RFD window."
        )

    phase = df.loc[
        movement_onset_idx:
        last_start_idx
    ].copy()

    future_force = df[
        "Fz_Filtered"
    ].shift(
        -window_samples
    )

    future_time = df[
        "Time_Rel"
    ].shift(
        -window_samples
    )

    delta_force = (
        future_force.loc[
            phase.index
        ]
        - phase[
            "Fz_Filtered"
        ]
    )

    delta_time = (
        future_time.loc[
            phase.index
        ]
        - phase[
            "Time_Rel"
        ]
    )

    rfd = (
        delta_force
        / delta_time
    )

    if rfd.dropna().empty:
        raise ValueError(
            "SJ RFD could not be calculated."
        )

    peak_idx = (
        rfd.idxmax()
    )

    peak_rfd = (
        rfd.loc[
            peak_idx
        ]
    )

    peak_rfd_time = df.loc[
        peak_idx,
        "Time_Rel"
    ]

    metrics = {
        "peak_propulsive_rfd_n_s": peak_rfd,

        "relative_peak_propulsive_rfd_n_s_kg": (
            peak_rfd / mass
        ),

        "peak_propulsive_rfd_time_s": (
            peak_rfd_time
        ),

        "rfd_window_ms": window_ms
    }

    return metrics


def analyze_sj(
    file,
    cutoff_hz=15.0,
    filter_order=4,
    weighing_start_s=0.0,
    weighing_end_s=0.75,
    onset_threshold_sd=5.0,
    onset_min_duration_ms=30.0,
    contact_threshold_n=20.0,
    contact_confirmation_ms=30.0,
    rfd_window_ms=20.0,
    landing_rfd_window_ms=20.0,
    landing_analysis_window_ms=300.0,
    landing_cutoff_hz=85.0,
    g=9.81
):
    """
    Run the complete squat jump force-time analysis pipeline.
    """

    df, metadata = load_forcedecks_csv(
        file
    )

    fs = get_sampling_frequency(
        df
    )

    df = lowpass_force(
        df,
        fs=fs,
        cutoff_hz=cutoff_hz,
        filter_order=filter_order
    )

    df = lowpass_landing_force(
        df,
        fs=fs,
        cutoff_hz=landing_cutoff_hz,
        filter_order=filter_order
    )

    (
        body_weight,
        body_weight_sd,
        mass
    ) = estimate_body_weight(
        df,
        weighing_start_s=weighing_start_s,
        weighing_end_s=weighing_end_s,
        g=g
    )

    df = calculate_com_mechanics(
        df,
        body_weight=body_weight,
        mass=mass
    )

    (
        movement_onset_idx,
        movement_onset_time,
        upper_threshold
    ) = detect_sj_onset(
        df,
        body_weight=body_weight,
        body_weight_sd=body_weight_sd,
        fs=fs,
        threshold_sd=onset_threshold_sd,
        min_duration_ms=onset_min_duration_ms
    )

    (
        takeoff_idx,
        takeoff_time
    ) = detect_takeoff(
        df,
        search_start_idx=movement_onset_idx,
        fs=fs,
        contact_threshold_n=contact_threshold_n,
        confirmation_ms=contact_confirmation_ms
    )

    metrics = calculate_sj_metrics(
        df,
        mass=mass,
        movement_onset_idx=movement_onset_idx,
        takeoff_idx=takeoff_idx,
        g=g
    )

    rfd_metrics = calculate_sj_rfd(
        df,
        mass=mass,
        movement_onset_idx=movement_onset_idx,
        takeoff_idx=takeoff_idx,
        fs=fs,
        window_ms=rfd_window_ms
    )

    metrics.update(
        rfd_metrics
    )

    (
        landing_idx,
        landing_time,
        landing_threshold,
        flight_noise_mean,
        flight_noise_sd,
        flight_window_start_idx,
        flight_window_end_idx
    ) = detect_landing(
        df,
        takeoff_idx=takeoff_idx,
        fs=fs
    )

    landing_metrics = calculate_landing_metrics(
        df,
        mass=mass,
        landing_idx=landing_idx,
        fs=fs,
        landing_window_ms=landing_analysis_window_ms
    )

    landing_rfd_metrics = calculate_landing_rfd(
        df,
        mass=mass,
        landing_idx=landing_idx,
        fs=fs,
        window_ms=landing_rfd_window_ms,
        landing_window_ms=landing_analysis_window_ms
    )

    metrics.update(
        landing_metrics
    )

    metrics.update(
        landing_rfd_metrics
    )

    results = {
        "sampling_frequency_hz": fs,

        "body_weight_n": body_weight,
        "body_weight_sd_n": body_weight_sd,
        "mass_kg": mass,

        "movement_onset_idx": movement_onset_idx,
        "takeoff_idx": takeoff_idx,

        "landing_idx": landing_idx,
        "landing_time_s": landing_time,

        "landing_threshold_n": landing_threshold,
        "flight_noise_mean_n": flight_noise_mean,
        "flight_noise_sd_n": flight_noise_sd,
        "flight_noise_window_start_idx": flight_window_start_idx,
        "flight_noise_window_end_idx": flight_window_end_idx,

        "onset_upper_threshold_n": upper_threshold,

        "filter_cutoff_hz": cutoff_hz,
        "landing_filter_cutoff_hz": landing_cutoff_hz,
        "filter_order": filter_order,

        "contact_threshold_n": contact_threshold_n,
        "contact_confirmation_ms": contact_confirmation_ms,

        "metadata": metadata
    }

    results.update(
        metrics
    )

    return results, df