"""
g-Funktionen nach Carslaw & Jaeger und Eskilson.
Superposition für Sondenfelder mit Nachbarsonden.

Referenz: EWS 5.5 Manual, Anhang A, Gl. 6.8-6.18

Finite Line Source (FLS) mit Spiegelquelle für korrekte
Oberflächenrandbedingung (T_surface = const).
"""

import numpy as np
from scipy.special import exp1, erfc
from scipy.integrate import quad
from typing import List, Tuple, Optional

EULER_GAMMA = 0.5772156649015329

# Default burial depth [m] — top of borehole below ground surface.
# Typical for Swiss installations (EWS 5.5 assumes D > 0).
DEFAULT_BURIAL_DEPTH = 2.0

# Threshold factor for neighbor thermal arrival gate.
# A neighbor borehole at distance d contributes to the field g-function
# only when t > THRESHOLD_FACTOR * d^2 / a, i.e., when the thermal
# diffusion front from that neighbor has reached the project borehole.
THRESHOLD_FACTOR = 0.9


def g_carslaw_jaeger(r: float, t: float, a: float, n_terms: int = 20) -> float:
    """Dimensionslose Sprungantwort für unendliche Linienquelle (Gl. 6.11).

    g = 0.5 * E₁(r²/(4at)) where E₁ is the exponential integral.

    Parameters
    ----------
    r : Radius [m]
    t : Zeit [s]
    a : Temperaturleitfähigkeit [m²/s]
    n_terms : unused, kept for API compatibility

    Returns
    -------
    g : dimensionslose Temperatursprungantwort
    """
    if t <= 0:
        return 0.0
    x = r**2 / (4 * a * t)
    if x > 40:
        return 0.0
    return 0.5 * exp1(x)


def t_s(H: float, a: float) -> float:
    """Zeitkonstante nach Eskilson (Gl. 6.12).

    Parameters
    ----------
    H : Sondenlänge [m]
    a : Temperaturleitfähigkeit [m²/s]

    Returns
    -------
    t_s : Zeitkonstante [s]
    """
    return H**2 / (9 * a)


def g_eskilson_single(r1: float, H: float, t: float, a: float) -> float:
    """g-Funktion Einzelsonde nach Eskilson (Gl. 6.14/6.15).

    Gültig für 5*r1²/a < t < t_s (Gl. 6.14).
    Für t > t_s konvergiert gegen Gl. 6.15.

    Parameters
    ----------
    r1 : Bohrlochradius [m]
    H  : Sondenlänge [m]
    t  : Zeit [s]
    a  : Temperaturleitfähigkeit [m²/s]

    Returns
    -------
    g : dimensionslose Temperatursprungantwort
    """
    if t <= 0:
        return 0.0
    ts = t_s(H, a)
    Es = t / ts
    g_eq = np.log(H / (2 * r1))  # Gl. 6.15 equilibrium
    if Es >= 1.0:
        return g_eq
    g = np.log(H / (2 * r1)) + 0.5 * np.log(Es)  # Gl. 6.14
    return g


def g_fls_image(r: float, H: float, t: float, a: float,
                D: float = DEFAULT_BURIAL_DEPTH) -> float:
    """Finite Line Source g-function with image source.

    Computes the g-function for a single borehole of length H starting
    at depth D, using the method of images to enforce constant surface
    temperature (T_surface = const boundary condition).

    The real source extends from depth D to D+H. The image source is
    its mirror about the surface (from -D-H to -D). The g-function is
    the average temperature response along the borehole length.

    Parameters
    ----------
    r : radial distance [m] (r1 for self-response, d_ij for pair response)
    H : Sondenlänge [m]
    t : Zeit [s]
    a : Temperaturleitfähigkeit [m²/s]
    D : burial depth [m] (top of borehole below surface)

    Returns
    -------
    g : dimensionslose g-Funktion
    """
    if t <= 0:
        return 0.0

    sqrt_at = np.sqrt(a * t)

    def integrand_real(s):
        """Real source contribution (borehole-to-borehole)."""
        R = np.sqrt(r**2 + s**2)
        return (H - abs(s)) * erfc(R / (2 * sqrt_at)) / R

    def integrand_image(s):
        """Image source contribution (mirror about surface)."""
        R = np.sqrt(r**2 + s**2)
        s_min = 2 * D
        s_max = 2 * D + 2 * H
        if s < s_min or s > s_max:
            return 0.0
        mid = s_min + H
        w = (s - s_min) if s <= mid else (s_max - s)
        return w * erfc(R / (2 * sqrt_at)) / R

    I_real, _ = quad(integrand_real, -H, H,
                     limit=200, epsabs=1e-10, epsrel=1e-10)

    s_min_img = 2 * D
    s_max_img = 2 * D + 2 * H
    I_image, _ = quad(integrand_image, s_min_img, s_max_img,
                      limit=200, epsabs=1e-10, epsrel=1e-10)

    return (I_real - I_image) / (2 * H)


def g_field(positions: List[Tuple[float, float]],
            r1: float, H: float, t: float, a: float,
            neighbor_positions: Optional[List[Tuple[float, float]]] = None,
            D: float = DEFAULT_BURIAL_DEPTH,
            threshold_factor: float = THRESHOLD_FACTOR) -> float:
    """g-Funktion Sondenfeld via FLS-Superposition (Gl. 6.17/6.18).

    Berechnet die mittlere g-Funktion über alle Projektsonden unter
    Berücksichtigung der gegenseitigen thermischen Beeinflussung
    und optionaler Nachbarsonden.

    Uses the Finite Line Source with image source for accurate
    computation across all time scales. Neighbor contributions are
    threshold-gated: a borehole at distance d only contributes when
    the thermal diffusion front has arrived (t > threshold_factor * d^2/a).

    Parameters
    ----------
    positions : Liste von (x, y) Koordinaten der Projektsonden [m]
    r1 : Bohrlochradius [m]
    H  : Sondenlänge [m]
    t  : Zeit [s]
    a  : Temperaturleitfähigkeit [m²/s]
    neighbor_positions : Liste von (x, y) Koordinaten der Nachbarsonden [m]
    D  : burial depth [m] (top of borehole below surface)
    threshold_factor : thermal arrival threshold (default 0.9)

    Returns
    -------
    g : mittlere g-Funktion des Feldes
    """
    n_project = len(positions)
    if n_project == 0:
        return 0.0

    # Self-response at borehole wall (same for all probes)
    g_self = g_fls_image(r1, H, t, a, D=D)

    # All borehole positions (project + neighbor)
    all_positions = list(positions)
    if neighbor_positions:
        all_positions.extend(neighbor_positions)
    n_total = len(all_positions)

    # Superposition: for each project borehole, sum contributions
    # from all other boreholes. Average over project boreholes only
    # (Gl. 6.18: divide by n-m = n_project).
    sum_corrections = 0.0
    for ix in range(n_project):
        pos_x = positions[ix]
        for iy in range(n_total):
            if iy == ix:
                continue
            pos_y = all_positions[iy]
            d = np.sqrt((pos_x[0] - pos_y[0])**2 +
                        (pos_x[1] - pos_y[1])**2)
            if d < 1e-6:
                continue

            # Threshold gate: only include this neighbor's contribution
            # when the thermal front has arrived
            t_arrival = threshold_factor * d**2 / a
            if t > t_arrival:
                sum_corrections += g_fls_image(d, H, t, a, D=D)

    g = g_self + sum_corrections / n_project
    return g


def compute_g_values(positions: List[Tuple[float, float]],
                     r1: float, H: float, a: float,
                     neighbor_positions: Optional[List[Tuple[float, float]]] = None,
                     ln_ts_values: Optional[List[float]] = None,
                     D: float = DEFAULT_BURIAL_DEPTH) -> dict:
    """Berechnet g-Werte für Standard ln(t/ts) Stützstellen.

    Parameters
    ----------
    positions : Projektsonden-Koordinaten [m]
    r1 : Bohrlochradius [m]
    H  : Sondenlänge [m]
    a  : Temperaturleitfähigkeit [m²/s]
    neighbor_positions : Nachbarsonden-Koordinaten [m]
    ln_ts_values : ln(t/ts) Stützstellen (default: [-4, -2, 0, 2, 3])
    D  : burial depth [m] (top of borehole below surface)

    Returns
    -------
    dict : {ln(t/ts): g-Wert}
    """
    if ln_ts_values is None:
        ln_ts_values = [-4, -2, 0, 2, 3]

    ts = t_s(H, a)
    result = {}
    for ln_val in ln_ts_values:
        t = ts * np.exp(ln_val)
        g = g_field(positions, r1, H, t, a, neighbor_positions, D=D)
        result[ln_val] = g
    return result


def interpolate_g(t: float, H: float, a: float, g_table: dict,
                  r1: float = None) -> float:
    """Interpoliert g-Wert aus Tabelle für gegebene Zeit t.

    Für Zeiten unterhalb der Tabelle wird die Einzelsonden-Formel
    nach Eskilson (Gl. 6.14) verwendet. Für sehr kurze Zeiten
    die Carslaw & Jaeger Formel (Gl. 6.11).

    Parameters
    ----------
    t : Zeit [s]
    H : Sondenlänge [m]
    a : Temperaturleitfähigkeit [m²/s]
    g_table : {ln(t/ts): g-Wert} Tabelle (Feld-g-Funktion)
    r1 : Bohrlochradius [m] (optional, für kurze Zeiten)

    Returns
    -------
    g : interpolierter g-Wert
    """
    if t <= 0:
        return 0.0
    ts = t_s(H, a)
    ln_val = np.log(t / ts)

    keys = sorted(g_table.keys())
    values = [g_table[k] for k in keys]

    if ln_val >= keys[-1]:
        return values[-1]

    if ln_val >= keys[0]:
        # Linear interpolation within table
        for i in range(len(keys) - 1):
            if keys[i] <= ln_val <= keys[i + 1]:
                frac = (ln_val - keys[i]) / (keys[i + 1] - keys[i])
                return values[i] + frac * (values[i + 1] - values[i])
        return values[-1]

    # Below table range: use single-borehole formula
    # For short times, neighboring boreholes haven't contributed yet
    if r1 is None:
        r1 = H / (2 * np.exp(values[0] - 0.5 * keys[0]))  # Estimate from table

    # Check if C&J is more appropriate (very short times)
    t_min_eskilson = 5 * r1**2 / a
    if t < t_min_eskilson:
        # Carslaw & Jaeger (Gl. 6.11) at r = r1
        return g_carslaw_jaeger(r1, t, a)
    else:
        # Eskilson single borehole (Gl. 6.14)
        return g_eskilson_single(r1, H, t, a)
