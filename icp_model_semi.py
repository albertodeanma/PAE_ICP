import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, List, Dict, Any
np.seterr(all='ignore')  # Suppress numpy warnings

try:
    import Algorithms
    vf = getattr(Algorithms, 'vf', None)
except Exception:
    vf = None

# add scipy import for fitting
try:
    from scipy import stats
except Exception:
    stats = None

#----------------1) Extracción de señal ----------------
def get_icp_signal(vf_obj=None, track_name: str = 'Intellivue/ICP') -> Tuple[np.ndarray, np.ndarray]:
    #Extrae la señal ICP desde el objeto vf (VitalFile).
    #Devuelve (timestamps, values) como numpy arrays.
    #Lanza ValueError si no encuentra la señal.
    vf_local = vf_obj or vf
    if vf_local is None:
        raise ValueError("Objeto VitalFile 'vf' no disponible.")

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


    # Return 1-D arrays
    return np.asarray(timestamps).ravel(), np.asarray(vals).ravel()


# ---------------------- 2) Discretización ----------------------
#Discretiza la serie de ICP en estados 0..(n-1) según thresholds ordenados.
#Ejemplo: thresholds=[15,20] -> states: 0 (<15), 1 (15-19.999), 2 (>=20)
def discretize_icp(values: np.ndarray, thresholds: List[float] = [15.0, 20.0]) -> np.ndarray:
    # Ensure values is a numpy array
    values = np.asarray(values, dtype=float)
    bins = [-np.inf] + list(thresholds) + [np.inf]
    states = np.digitize(values, bins) - 1
    return states.astype(int)  # Ensure integer states

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
    # treat values outside [0,50] as missing
    out_of_range_mask = (vals < 0) | (vals > 50)
    nan_mask = np.isnan(vals) | out_of_range_mask
    if nan_mask.any():
        tmp = vals.copy()
        tmp[nan_mask] = -np.inf
    else:
        tmp = vals

    states = discretize_icp(tmp, thresholds=thresholds).astype(float)
    states[nan_mask] = np.nan

    return estimate_transition_matrix(states, n_states=n_states)

# ---------------------- 4) Semi-Markov: sojourn extraction + parametric fit ----------------------

def extract_sojourns(states: np.ndarray, timestamps: np.ndarray = None) -> Dict[int, List[float]]:
    """
    Extrae duraciones (sojourns) por estado.
    - states: array (ints or floats) where NaN means missing.
    - timestamps: optional time vector; if provided durations are in same units (end-start+dt),
      otherwise durations are integer run-lengths (# samples in state).
    Returns dict: state -> list of durations (floats).
    """    
    # Normalize inputs to 1-D arrays
    states = np.asarray(states).ravel()
    if timestamps is not None:
        timestamps = np.asarray(timestamps).ravel()

    # Synchronize lengths if needed
    if timestamps is not None and timestamps.size != states.size:
        min_len = min(states.size, timestamps.size)
        states = states[:min_len]
        timestamps = timestamps[:min_len]

    # Remove NaN state entries (and corresponding timestamps)
    valid_mask = ~np.isnan(states)
    states = states[valid_mask]
    if timestamps is not None:
        timestamps = timestamps[valid_mask]

    # Compute default dt from timestamps if available
    dt = 1.0
    if timestamps is not None and timestamps.size > 1:
        diffs = np.diff(timestamps)
        valid_diffs = diffs[~np.isnan(diffs)]
        if valid_diffs.size > 0:
            dt = float(np.median(valid_diffs))

    sojourns = defaultdict(list)
    n = len(states)
    i = 0
    
    while i < n:
        s = int(states[i])
        start_idx = i
        
        # Find end of current run
        while i + 1 < n and int(states[i + 1]) == s:
            i += 1
        end_idx = i
        
        # Calculate duration
        if timestamps is not None:
            try:
                time_duration = timestamps[end_idx] - timestamps[start_idx]
                if time_duration > 0:
                    duration = float(time_duration)
                else:
                    duration = float(end_idx - start_idx + 1) * dt
            except (IndexError, TypeError, ValueError):
                duration = float(end_idx - start_idx + 1) * dt
        else:
            duration = float(end_idx - start_idx + 1)
        
        if duration > 0:
            sojourns[s].append(duration)
        
        i = end_idx + 1
    
    return dict(sojourns)


def fit_parametric_sojourns(sojourns: Dict[int, List[float]],
                            distributions: List[str] = ['weibull_min', 'gamma', 'lognorm'],
                            fix_loc_zero: bool = True) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Ajusta distribuciones paramétricas por MLE para las duraciones de cada estado.
    Devuelve: { state: { dist_name: {params, loglik, aic, bic, n} } }
    Requiere scipy.stats.
    """
    if stats is None:
        raise RuntimeError("scipy is required for distribution fitting. Install with: pip install scipy")

    results: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for state, samples in sojourns.items():
        # Convert input to 1D numpy array and remove NaNs
        arr = np.asarray(samples, dtype=np.float64).ravel()
        arr = arr[~np.isnan(arr)]
        res_state: Dict[str, Dict[str, Any]] = {}
        n = int(max(1, arr.size))
        if arr.size < 2:
            # not enough data to fit reliably
            res_state['_note'] = {'n': n, 'error': 'too few samples'}
            results[int(state)] = res_state
            continue

        for dist_name in distributions:
            dist = getattr(stats, dist_name, None)
            if dist is None:
                res_state[dist_name] = {'error': f'distribution {dist_name} not found in scipy.stats'}
                continue
            try:
                if fix_loc_zero:
                    params = dist.fit(arr, floc=0)
                else:
                    params = dist.fit(arr)
                
                # Convert params to native Python floats safely
                params_list = []
                for p in params:
                    if isinstance(p, np.ndarray):
                        if p.size == 1:
                            params_list.append(float(p.item()))
                        else:
                            params_list.append(float(p[0]))  # Take first element if array
                    else:
                        params_list.append(float(p))
                
                # Calculate log-likelihood using the converted parameters
                logpdf = dist.logpdf(arr, *params_list)
                loglik = float(np.sum(logpdf).item())  # Ensure scalar
                
                # Calculate information criteria using scalar operations
                k = len(params_list) - (1 if fix_loc_zero else 0)
                n_scalar = float(n)
                aic = float(2.0 * k - 2.0 * loglik)
                bic = float(k * np.log(n_scalar) - 2.0 * loglik)
                
                res_state[dist_name] = {
                    'params': params_list,
                    'loglik': loglik,
                    'aic': aic,
                    'bic': bic,
                    'n': int(n)
                }
            except Exception as e:
                res_state[dist_name] = {'error': str(e)}
        results[int(state)] = res_state
    return results


def select_best_distributions(fits: Dict[int, Dict[str, Dict[str, Any]]], criterion: str = 'aic') -> Dict[int, Dict[str, Any]]:
    """
    Selecciona la mejor distribución por estado usando 'aic' or 'bic' (lower is better).
    Devuelve { state: best_result_dict }.
    """
    chosen: Dict[int, Dict[str, Any]] = {}
    for state, dist_results in fits.items():
        best_name = None
        best_score = float('inf')
        best_entry = {}
        
        for name, entry in dist_results.items():
            if not isinstance(entry, dict) or 'error' in entry:
                continue
                
            try:
                score = entry.get(criterion)
                if score is None:
                    continue
                    
                # Convert score to float if it's a numpy type
                if hasattr(score, 'item'):
                    score = float(score.item())
                else:
                    score = float(score)
                    
                if score < best_score:
                    best_score = score
                    best_name = name
                    # Create a new dict with converted values
                    best_entry = {}
                    for k, v in entry.items():
                        if hasattr(v, 'tolist'):
                            best_entry[k] = v.tolist()
                        elif hasattr(v, 'item'):
                            best_entry[k] = v.item()
                        else:
                            best_entry[k] = v
                            
            except Exception as e:
                continue
                
        if best_name is not None:
            chosen[int(state)] = {'dist': best_name, **best_entry}
        else:
            chosen[int(state)] = {'error': 'no valid fit'}
            
    return chosen


def estimate_semi_markov_from_values(values: np.ndarray,
                                     thresholds: List[float] = [15.0, 20.0],
                                     timestamps: np.ndarray = None,
                                     dist_candidates: List[str] = None,
                                     n_states: int = None,
                                     fix_loc_zero: bool = True) -> Dict[str, Any]:
    """
    Pipeline convenience:
      - discretize values (keeps NaN positions)
      - estimate counts and P (Markov)
      - extract sojourns
      - fit parametric sojourn models by MLE
    Returns dict with keys: counts, P, sojourns, fits, best, states
    """
    if dist_candidates is None:
        dist_candidates = ['weibull_min', 'gamma', 'lognorm']

    # Convert inputs to 1-D numpy arrays
    vals = np.asarray(values, dtype=float).ravel()
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float).ravel()
        # Synchronize lengths first (use sizes to be robust against shapes)
        min_len = min(vals.size, timestamps.size)
        vals = vals[:min_len]
        timestamps = timestamps[:min_len]

    # Treat values outside [0,50] as missing and handle NaNs
    out_of_range_mask = (vals < 0) | (vals > 50)
    invalid_mask = np.isnan(vals) | out_of_range_mask
    if timestamps is not None:
        invalid_mask = invalid_mask | np.isnan(timestamps)
    valid_mask = ~invalid_mask
    vals = vals[valid_mask]
    if timestamps is not None:
        timestamps = timestamps[valid_mask]

    # Discretize values (NaNs already removed)
    states = discretize_icp(vals)
    states = np.asarray(states, dtype=float)

    # Compute transitions and sojourns
    counts, P = estimate_transition_matrix(states, n_states=n_states)
    sojourns = extract_sojourns(states, timestamps=timestamps)
    fits = fit_parametric_sojourns(sojourns, distributions=dist_candidates, fix_loc_zero=fix_loc_zero)
    best = select_best_distributions(fits, criterion='aic')
    # Ensure all numpy arrays are converted to Python native types for serialization
    try:
        result = {
            'counts': counts.tolist() if hasattr(counts, 'tolist') else counts,
            'P': P.tolist() if hasattr(P, 'tolist') else P,
            'sojourns': {k: [float(d) for d in v] for k, v in sojourns.items()},
            'fits': fits,
            'best': best,
            'states': states.tolist() if hasattr(states, 'tolist') else list(states)
        }
    except Exception as e:
        result = {
            'error': 'Failed to convert results to native Python types',
            'message': str(e)
        }
    return result

    # ...existing code...
# ---------------------- 5) Semi-Markov: Simulation ----------------------

def sample_next_state(current_state: int, P: np.ndarray) -> int:
    """
    Sample next state using transition probability matrix P.
    
    Args:
        current_state: Current state index
        P: Transition probability matrix (n_states x n_states)
    
    Returns:
        Sampled next state index
    """
    if not (0 <= current_state < P.shape[0]):
        raise ValueError(f"Invalid current_state {current_state}")
    
    # Get transition probabilities for current state
    probs = P[current_state]
    
    # Handle absorbing states or invalid probabilities
    if np.sum(probs) == 0:
        return current_state
    
    # Normalize probabilities (in case they don't sum to 1)
    probs = probs / np.sum(probs)
    
    # Sample next state
    return np.random.choice(len(probs), p=probs)

def sample_sojourn_duration(state: int, fits: Dict[int, Dict[str, Dict[str, Any]]]) -> float:
    """
    Sample sojourn duration for given state using fitted distribution.
    
    Args:
        state: State index
        fits: Dictionary of fitted distributions per state
    
    Returns:
        Sampled duration (float)
    """
    if state not in fits:
        raise ValueError(f"No fitted distribution for state {state}")
        
    state_fits = fits[state]
    
    # Get best fit distribution
    best_dist = None
    best_params = None
    best_aic = float('inf')
    
    for dist_name, fit_info in state_fits.items():
        if isinstance(fit_info, dict) and 'aic' in fit_info:
            if fit_info['aic'] < best_aic:
                best_dist = dist_name
                best_params = fit_info['params']
                best_aic = fit_info['aic']
    
    if best_dist is None:
        raise ValueError(f"No valid distribution fit for state {state}")
    
    # Get distribution from scipy.stats
    dist = getattr(stats, best_dist)
    
    # Sample duration (ensure positive)
    duration = 0
    while duration <= 0:
        duration = float(dist.rvs(*best_params))
    
    return duration

def simulate_trajectory(initial_state: int,
                      duration: float,
                      P: np.ndarray,
                      fits: Dict[int, Dict[str, Dict[str, Any]]],
                      dt: float = 1.0,
                      seed: int = None) -> Tuple[List[int], List[float]]:
    """
    Simulate a semi-Markov trajectory.
    
    Args:
        initial_state: Starting state index
        duration: Total duration to simulate
        P: Transition probability matrix
        fits: Fitted sojourn distributions per state
        dt: Time step for discretization (default=1.0)
        seed: Random seed (optional)
    
    Returns:
        Tuple of (states, timestamps) where states is list of visited states
        and timestamps is list of transition times
    """
    if seed is not None:
        np.random.seed(seed)
    
    states = [initial_state]
    timestamps = [0.0]
    current_time = 0.0
    
    while current_time < duration:
        current_state = states[-1]
        
        # Sample sojourn duration
        sojourn = sample_sojourn_duration(current_state, fits)
        
        # Update time
        current_time += sojourn
        if current_time <= duration:  # Only add if within simulation window
            # Sample next state
            next_state = sample_next_state(current_state, P)
            states.append(next_state)
            timestamps.append(current_time)
    
    return states, timestamps

if __name__ == "__main__":
    import vitaldb
    import os
    import importlib

    print("\n=== Testing ICP Model with Real Data ===")
    
    # Get the latest vital file
    vital_path = 'D:/UPC/CUATRI/PAE/data/kuigebjtu_250530_110926.vital'
    print(f"\nUsing vital file: {vital_path}")
    
    try:
        # Load the vital file
        vf_obj = vitaldb.VitalFile(vital_path)
        
        # Get ICP signal
        timestamps, icp_values = get_icp_signal(vf_obj)

        # Normalize and trim to common length
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
        states = discretize_icp(icp_values)
        unique_states, state_counts = np.unique(states[~np.isnan(states)], return_counts=True)
        print("State distribution:")
        for state, count in zip(unique_states, state_counts):
            print(f"State {state}: {count} samples")
        
        # Test transition matrix
        print("\n3. Testing transition matrix estimation:")
        counts, P = estimate_transition_probabilities_from_values(icp_values)
        print("\nTransition counts matrix:")
        print(counts)
        print("\nTransition probability matrix:")
        print(P)
        
        # Semi‑Markov: sojourn extraction & parametric fits (if icp_model_semi available)
        try:
            icp_semi = importlib.import_module('icp_model_semi')
            print("\n4. Extracting sojourns and fitting parametric sojourn models:")
            
            # Ensure timestamps and values have the same length (and already filtered)
            min_len = min(len(ts_valid), len(icp_valid)) if ts_valid is not None else len(icp_valid)
            timestamps_trim = ts_valid[:min_len] if ts_valid is not None else None
            icp_values_trim = icp_valid[:min_len]

            print(f"Using {min_len} samples for semi-Markov analysis")
            semi_res = icp_semi.estimate_semi_markov_from_values(
                icp_values_trim,
                thresholds=[15.0, 20.0],
                timestamps=timestamps_trim,
                dist_candidates=['weibull_min', 'gamma', 'lognorm'],
                n_states=None,
                fix_loc_zero=True
            )
            sojourns = semi_res.get('sojourns', {})
            fits = semi_res.get('fits', {})
            best = semi_res.get('best', {})

            print("\nSojourn counts per state:")
            for s, lst in sorted(sojourns.items()):
                arr = np.asarray(lst, dtype=float)
                print(f" State {s}: {len(arr)} sojourns, durations (min/med/max): "
                      f"{np.nanmin(arr):.3g}/{np.nanmedian(arr):.3g}/{np.nanmax(arr):.3g}")

            print("\nBest-fit distribution per state (AIC):")
            for s, info in sorted(best.items()):
                if 'error' in info:
                    print(f" State {s}: fit error: {info['error']}")
                else:
                    dist = info.get('dist', '<unknown>')
                    params = info.get('params', None)
                    print(f" State {s}: {dist}, params={params}")
        except ModuleNotFoundError:
            print("\nicp_model_semi not found -> skipping sojourn extraction / parametric fits.")
        except RuntimeError as e:
            # e.g., scipy missing
            print(f"\nSemi-Markov fitting skipped due to runtime error: {e}")
        except Exception as e:
            print(f"\nError during semi-Markov estimation: {e}")
        
        # Verify the results
        verification_results = {
            'discretization': len(states) == len(icp_values),
            'matrix_shape': P.shape == (3, 3),  # We expect 3 states with default thresholds
            'probability_sums': np.allclose(np.sum(P, axis=1)[np.sum(counts, axis=1) > 0], 1.0)
        }
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
# ...existing code...