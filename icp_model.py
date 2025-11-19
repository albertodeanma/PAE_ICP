
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, List, Dict, Any


# Importa el módulo Algorithms 
try:
    import Algorithms
    vf = getattr(Algorithms, 'vf', None)
except Exception:
    vf = None
    
#----------------1) Extracción de señal ----------------
def get_icp_signal(vf_obj=None, track_name: str = 'Intellivue/ICP') -> Tuple[np.ndarray, np.ndarray]:
#Extrae la señal ICP desde el objeto vf (VitalFile).
#Devuelve (timestamps, values) como numpy arrays.
#Lanza ValueError si no encuentra la señal.
    vf_local = vf_obj or vf
    if vf_local is None:
        raise ValueError("Objeto VitalFile 'vf' no disponible.")

    try:
        vals = vf_local.to_numpy(track_names='ICP', interval=1)
    except Exception as e:
        raise ValueError(f"No se encontró la señal {'ICP'}: {e}")

    timestamps = None
    try:
        timestamps = vf_local.to_numpy(track_names='timestamp', interval=1, return_timestamp=True)
    except Exception:
        # Si no existe, generamos un vector de índices (1,2,...)
        timestamps = np.arange(len(vals))


    return np.asarray(timestamps), np.asarray(vals)


# ---------------------- 2) Discretización ----------------------
#Discretiza la serie de ICP en estados 0..(n-1) según thresholds ordenados.
#Ejemplo: thresholds=[15,20] -> states: 0 (<15), 1 (15-19.999), 2 (>=20)
def discretize_icp(values: np.ndarray, thresholds: List[float] = [15.0, 20.0]) -> np.ndarray:

    bins = [-np.inf] + list(thresholds) + [np.inf]
    states = np.digitize(values, bins) - 1
    return states

#----------------3) Estimación parametros ---------------- 

 #estimar Pij

def estimate_transition_matrix(states: np.ndarray, n_states: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cuenta transiciones entre estados y calcula la matriz de probabilidad de transición P.
    Args:
        states: 1D array de enteros (estados etiquetados, p.ej. salida de discretize_icp).
        n_states: número de estados. Si None se infiere como max(states)+1.
    Returns:
        counts: matriz (n_states x n_states) con conteos de transiciones i->j.
        P: matriz (n_states x n_states) con probabilidades de transición (filas suman 1, salvo filas sin datos).
    """
    if states is None or len(states) < 2:
        return np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=float)

    # filtrar NaN o valores inválidos
    valid_mask = ~np.isnan(states)
    states = np.asarray(states)[valid_mask]

    if len(states) < 2:
        return np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=float)

    if n_states is None:
        n_states = int(np.nanmax(states)) + 1

    counts = np.zeros((n_states, n_states), dtype=int)

    # contar transiciones consecutivas
    from_states = states[:-1].astype(int)
    to_states = states[1:].astype(int)
    for a, b in zip(from_states, to_states):
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a, b] += 1

    # normalizar por filas para obtener probabilidades
    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(counts, row_sums, where=(row_sums != 0))
    P[np.isnan(P)] = 0.0

    return counts, P


def estimate_transition_probabilities_from_values(values: np.ndarray,
                                                  thresholds: List[float] = [15.0, 20.0],
                                                  n_states: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conveniencia: discretiza una señal de ICP y estima counts y P.
    Args:
        values: señal cruda (puede contener NaNs).
        thresholds: umbrales para discretización.
        n_states: opcional, fuerza número de estados.
    Returns:
        (counts, P)
    """
    # manejar NaNs en values: discretize_icp funciona con numpy array pero NaNs producen estado = len(bins)-1
    vals = np.asarray(values, dtype=float)
    # treat values outside [0,50] as missing (do not consider them)
    out_of_range_mask = (vals < 0) | (vals > 50)
    nan_mask = np.isnan(vals) | out_of_range_mask
    if nan_mask.any():
        # temporarily fill with very negative to fall into state 0, then mask later
        tmp = vals.copy()
        tmp[nan_mask] = -np.inf
    else:
        tmp = vals

    states = discretize_icp(tmp, thresholds=thresholds).astype(float)

    # mark out-of-range / NaN positions as NaN so they are ignored in transition counting
    states[nan_mask] = np.nan

    return estimate_transition_matrix(states, n_states=n_states)


if __name__ == "__main__":
    import vitaldb
    import os

    print("\n=== Testing ICP Model with Real Data ===")
    
    # Get the latest vital file
    vital_path = 'D:/UPC/CUATRI/PAE/data/kuigebjtu_250530_110926.vital'
    print(f"\nUsing vital file: {vital_path}")
    
    try:
        # Load the vital file
        vf_obj = vitaldb.VitalFile(vital_path)
        
        # Get ICP signal
        timestamps, icp_values = get_icp_signal(vf_obj)

        # Normalize arrays and trim to common length if needed
        icp_arr = np.asarray(icp_values, dtype=float).ravel()
        ts_arr = np.asarray(timestamps, dtype=float).ravel() if timestamps is not None else None
        if ts_arr is not None:
            min_len = min(len(icp_arr), len(ts_arr))
            icp_arr = icp_arr[:min_len]
            ts_arr = ts_arr[:min_len]

        # Filter to valid ICP range [0,50]
        valid_mask = (~np.isnan(icp_arr)) & (icp_arr >= 0.0) & (icp_arr <= 50.0)
        if not np.any(valid_mask):
            print("\nNo valid ICP samples in range [0,50]. Skipping analysis.")
            raise RuntimeError("No valid ICP samples in range [0,50]")

        icp_valid = icp_arr[valid_mask]
        ts_valid = ts_arr[valid_mask] if ts_arr is not None else None

        print("\n1. ICP Signal Statistics (only values in [0,50]):")
        print(f"Number of samples: {len(icp_valid)}")
        print(f"Mean ICP: {np.mean(icp_valid):.2f}")
        print(f"Min ICP: {np.min(icp_valid):.2f}")
        print(f"Max ICP: {np.max(icp_valid):.2f}")
        
        # Test discretization
        print("\n2. Testing discretization:")
        states = discretize_icp(icp_valid)
        unique_states, state_counts = np.unique(states[~np.isnan(states)], return_counts=True)
        print("State distribution:")
        for state, count in zip(unique_states, state_counts):
            print(f"State {state}: {count} samples")

        # Test transition matrix
        print("\n3. Testing transition matrix estimation:")
        counts, P = estimate_transition_probabilities_from_values(icp_valid)
        print("\nTransition counts matrix:")
        print(counts)
        print("\nTransition probability matrix:")
        print(P)

        # Verify the results
        print("\n4. Verification:")
        if len(states) == len(icp_valid):
            print("OK: Discretization produced correct number of states")
        if P.shape == (3, 3):  # We expect 3 states with default thresholds
            print("OK: Transition matrix has correct shape (3x3)")
        if np.allclose(np.sum(P, axis=1)[np.sum(counts, axis=1) > 0], 1.0):
            print("OK: Probability matrix rows sum to 1 where there are transitions")
    except Exception as e:
        print(f"Error during testing: {str(e)}")