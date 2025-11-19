import numpy as np
from scipy.stats import exponweib
import vitaldb
import os
import json
import argparse

# === CONFIGURACIÓN ===
VITAL_PATH = "D:/UPC/CUATRI/PAE/data/kuigebjtu_250530_160048.vital"

# === FUNCIONES AUXILIARES ===

def discretize_icp(icp_values, thresholds=None):
    """Discretiza valores ICP en 3 estados usando dos thresholds.
    thresholds: iterable with two numeric thresholds [t1, t2].
    icp_values: 1-D numpy array."""
    if thresholds is None:
        t1, t2 = 15.0, 20.0
    else:
        t1, t2 = float(thresholds[0]), float(thresholds[1])
    icp = np.asarray(icp_values, dtype=float).ravel()
    states = np.zeros_like(icp, dtype=int)
    states[icp < t1] = 0
    states[(icp >= t1) & (icp < t2)] = 1
    states[icp >= t2] = 2
    return states


def estimate_markov_matrix(states, n_states=3):
    """Calcula la matriz de transición de Markov."""
    states = np.asarray(states).ravel().astype(int)
    P = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        a = states[i]; b = states[i + 1]
        if 0 <= a < n_states and 0 <= b < n_states:
            P[a, b] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    # evitar división por cero: si fila cero la dejamos a 0
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(P, np.where(row_sums == 0, 1.0, row_sums))
    return P


def estimate_state_durations(states, timestamps, n_states=3):
    """
    Obtiene las duraciones (en segundos) de cada estado.
    states: 1-D int array; timestamps: 1-D float array (mismo length) o None.
    """
    states = np.asarray(states).ravel().astype(int)
    if timestamps is None:
        # devolver conteos de muestras como duraciones
        durations = [[] for _ in range(n_states)]
        start = 0
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                dur = float(i - start)
                durations[states[i - 1]].append(dur)
                start = i
        # último run
        if len(states) > 0:
            durations[states[-1]].append(float(len(states) - start))
        return durations

    ts = np.asarray(timestamps).ravel().astype(float)
    # asegurar misma longitud
    min_len = min(len(states), len(ts))
    states = states[:min_len]
    ts = ts[:min_len]

    durations = [[] for _ in range(n_states)]
    start = 0
    for i in range(1, min_len):
        if states[i] != states[i - 1]:
            dur = float(ts[i] - ts[start])
            durations[states[i - 1]].append(dur)
            start = i
    # último run
    if min_len > 0:
        durations[states[min_len - 1]].append(float(ts[min_len - 1] - ts[start]))
    return durations


def fit_duration_distributions(durations):
    """Ajusta distribuciones Weibull (ExponWeib con a=1) a cada estado.
       Devuelve lista de (shape, scale) por estado."""
    best_fits = []
    for durs in durations:
        if len(durs) < 2:
            best_fits.append((1.0, 1.0))
            continue
        durs = np.asarray(durs, dtype=float)
        # ajustamos exponweib con fa=1 => equivalente a Weibull (exponweib with fa=1)
        try:
            a, loc, scale = exponweib.fit(durs, floc=0, fa=1)  # returns (a, c, loc, scale)? but with fa=1 returns (a, loc, scale)
            # exponweib.fit may return 3 or 4 params depending; normalize:
            # ensure a and scale extracted:
            params = exponweib.fit(durs, floc=0, fa=1)
            # params could be length 3 or 4; we try to extract a and scale robustly
            if len(params) == 3:
                shape = float(params[0])
                scale = float(params[2])
            else:
                shape = float(params[0])
                scale = float(params[-1])
            # defend against non-positive
            if scale <= 0 or shape <= 0:
                raise ValueError("Invalid fit params")
            best_fits.append((shape, scale))
        except Exception:
            # fallback a valores por defecto razonables
            mean_dur = float(np.nanmean(durs)) if durs.size > 0 else 1.0
            best_fits.append((1.5, max(1.0, mean_dur)))
    return best_fits


def semi_markov_next_probabilities(current_state, duration, P, best_fits):
    """
    Ajusta las probabilidades Markov con una función semi-Markov
    basada en "survival" escalar calculado desde la distribución Weibull.
    """
    n_states = P.shape[0]
    markov_probs = np.asarray(P[current_state], dtype=float).ravel()
    # defensas
    if current_state < 0 or current_state >= n_states:
        # fallback: devolver distribución uniforme
        out = np.ones(n_states) / float(n_states)
        return out

    # obtener fit
    try:
        shape, scale = best_fits[current_state]
        shape = float(shape); scale = float(scale)
        if scale <= 0 or shape <= 0:
            raise ValueError
        survival = float(np.exp(-((duration / scale) ** shape)))
    except Exception:
        # fallback a supervivencia basada en exponencial simple con media=scale_est
        survival = 0.5

    survival = np.clip(survival, 0.0, 1.0)

    adjusted = markov_probs * (1.0 - survival)
    # reasignar la probabilidad de permanecer en el mismo estado
    adjusted[current_state] += survival
    # normalizar
    s = np.sum(adjusted)
    if s <= 0:
        # fallback a markov_probs normalizadas
        row = markov_probs
        rsum = np.sum(row)
        if rsum <= 0:
            return np.ones(n_states) / float(n_states)
        return row / rsum
    return adjusted / s


def save_transition_model(P: np.ndarray, thresholds: list, path: str):
    """Save transition matrix P and thresholds to a JSON file."""
    obj = {
        'P': np.asarray(P).tolist(),
        'thresholds': thresholds
    }
    with open(path, 'w') as f:
        json.dump(obj, f)
    print(f"Saved transition model to {path}")


def load_transition_model(path: str):
    """Load transition matrix and thresholds from JSON file. Returns (P, thresholds) or (None, None) on error."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
        P = np.asarray(obj.get('P', []), dtype=float)
        thresholds = obj.get('thresholds', [15.0, 20.0])
        print(f"Loaded transition model from {path}")
        return P, thresholds
    except Exception as e:
        print(f"Failed to load transition model from {path}: {e}")
        return None, None


def evaluate_model(states, timestamps, P, best_fits):
    """
    Evalúa el modelo semi-Markov: calcula la precisión de predicción
    del siguiente estado. Utiliza step-by-step predictions (para cada sample).
    """
    states = np.asarray(states).ravel().astype(int)
    if timestamps is not None:
        timestamps = np.asarray(timestamps).ravel().astype(float)
    n = len(states)
    if n < 2:
        return 0.0

    correct = 0
    total = 0
    # calculamos durations como tiempo pasado desde el inicio del run hasta el índice i
    run_start = 0
    for i in range(n - 1):
        if i > 0 and states[i] != states[i - 1]:
            run_start = i
        # sojourn time up to index i: either number of samples or time difference
        if timestamps is None:
            duration = float(i - run_start + 1)
        else:
            duration = float(timestamps[i] - timestamps[run_start] + 1e-9)
        s = states[i]
        next_state = states[i + 1]
        probs = semi_markov_next_probabilities(s, duration, P, best_fits)
        predicted = int(np.argmax(probs))
        if predicted == int(next_state):
            correct += 1
        total += 1
    return (correct / total * 100.0) if total > 0 else 0.0


# === MAIN ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Markov ICP prediction (single-file).")
    parser.add_argument('--vital', type=str, default=VITAL_PATH, help='Path to .vital file')
    parser.add_argument('--model-file', type=str, default='transition_model.json', help='Path to saved transition matrix JSON')
    parser.add_argument('--save-model', action='store_true', help='Save computed transition matrix to --model-file')
    args = parser.parse_args()

    print("=== Testing Semi-Markov ICP model ===")
    print(f"File: {args.vital}")

# Try to load a previously saved transition model (P and thresholds). If found, we'll reuse it.
    P_loaded, thresholds_loaded = load_transition_model(args.model_file)

    # Leer archivo vital
    rec = vitaldb.VitalFile(args.vital if args.vital is not None else VITAL_PATH)

    icp = None
    timestamps = None

    # 1) Intentar to_numpy('ICP', 0, return_timestamp=True)
    try:
        # Algunos implementations devuelven un ndarray 2-col [ts, val] si se llama así
        res = rec.to_numpy('ICP', 0, return_timestamp=True)
        # puede devolver tuple (timestamps, values) o 2-col array; cubrimos ambos
        if isinstance(res, tuple) and len(res) >= 2:
            ts_raw, vals_raw = res[0], res[1]
            timestamps = np.asarray(ts_raw).ravel().astype(float)
            icp = np.asarray(vals_raw).ravel().astype(float)
            print("Using ICP via rec.to_numpy(track, interval, return_timestamp=True) -> tuple")
        else:
            arr = np.asarray(res)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                timestamps = arr[:, 0].astype(float)
                icp = arr[:, 1].astype(float)
                print("Using ICP via rec.to_numpy(...) -> 2-col ndarray")
            elif arr.ndim == 1:
                icp = arr.astype(float)
                print("Using ICP via rec.to_numpy(...) -> 1-col ndarray")
    except TypeError:
        # algunos bindings no soportan return_timestamp kw arg; intentar sin named arg
        try:
            res = rec.to_numpy('ICP', 0)
            arr = np.asarray(res)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                timestamps = arr[:, 0].astype(float)
                icp = arr[:, 1].astype(float)
                print("Using ICP via rec.to_numpy(track, interval) -> 2-col ndarray")
            else:
                icp = arr.ravel().astype(float)
                print("Using ICP via rec.to_numpy(track, interval) -> 1-col ndarray")
        except Exception:
            pass
    except Exception:
        pass

    # 2) Si no hemos conseguido, intentar get_samples o introspección (diagnóstico anterior sugiere get_samples devolviendo tuple)
    if (icp is None or icp.size == 0) and hasattr(rec, 'get_samples'):
        try:
            res = rec.get_samples('ICP', 0)
            # según diagnóstico puede devolver tuple ([array(vals)], ['ICP'])
            if isinstance(res, tuple) and len(res) >= 1:
                # buscar el primer array dentro del tuple
                possible = [r for r in res if isinstance(r, (list, tuple, np.ndarray))]
                if possible:
                    vals = possible[0]
                    # si es lista con un array dentro
                    if isinstance(vals, list) and len(vals) > 0:
                        vals = vals[0]
                    icp = np.asarray(vals).ravel().astype(float)
                    print("Using ICP via rec.get_samples('ICP', 0) (parsed tuple/list)")
            else:
                arr = np.asarray(res)
                if arr.ndim >= 1:
                    icp = arr.ravel().astype(float)
                    print("Using ICP via rec.get_samples('ICP', 0) -> ndarray")
        except Exception:
            pass

    # 3) Si aún no, intentar llamar rec.to_numpy sin args (diagnóstico mostró que funciona)
    if (icp is None or icp.size == 0):
        try:
            arr = rec.to_numpy()  # suele devolver ndarray (maybe 2-cols)
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                # intentar detectar cual columna es timestamp y cual valor (heurística)
                col0 = arr[:, 0]
                col1 = arr[:, 1]
                # si col0 valores grandes (ts ~ 1e9) considerarlo timestamp
                if np.nanmin(col0) > 1e6 and np.nanmax(col0) > 1e6:
                    timestamps = col0.astype(float)
                    icp = col1.astype(float)
                    print("Using rec.to_numpy() -> interpreted col0 as timestamps, col1 as values")
                else:
                    # si sólo hay una columna de valores
                    icp = col0.astype(float).ravel()
                    print("Using rec.to_numpy() -> single-column values")
            elif arr.ndim == 1:
                icp = arr.ravel().astype(float)
                print("Using rec.to_numpy() -> 1-D array (values)")
        except Exception:
            pass

    # 4) Si timestamps no están disponibles pero icp sí, intentar rec.timestamps o construir índices
    if icp is not None and (timestamps is None or len(timestamps) != len(icp)):
        # intentar rec.timestamps si existe y tiene longitud apropiada
        if hasattr(rec, 'timestamps'):
            try:
                ts_try = np.asarray(rec.timestamps).ravel().astype(float)
                if ts_try.size >= icp.size:
                    timestamps = ts_try[:icp.size]
                    print("Using rec.timestamps trimmed to icp length")
            except Exception:
                timestamps = None
        # si aún no, construir vector de índices (segundos relativos)
        if timestamps is None:
            timestamps = np.arange(len(icp), dtype=float)
            print("No timestamps found: using index-based timestamps (0,1,2,...)")

    # validar lectura
    if icp is None or icp.size == 0:
        # Repetimos diagnóstico (útil para depuración)
        print("\n--- Diagnostic dump: VitalFile inspection ---")
        try:
            attrs = dir(rec)
            print("Attributes available on VitalFile (first 200 chars):", ", ".join(attrs)[:200])
        except Exception as e:
            print("Error listing attributes:", repr(e))

        # Algunas pruebas comunes (seguras)
        def safe_call(name, fn):
            try:
                res = fn()
                print(f" {name}: OK, type={type(res)}, repr={repr(res)[:200]}")
            except Exception as e:
                print(f" {name}: ERROR: {repr(e)}")

        safe_call('to_numpy_noargs', lambda: rec.to_numpy())
        safe_call('to_numpy_icp_return_ts', lambda: rec.to_numpy('ICP', 0, return_timestamp=True))
        safe_call('to_numpy_icp', lambda: rec.to_numpy('ICP', 0))
        safe_call('get_samples_icp', lambda: rec.get_samples('ICP', 0) if hasattr(rec, 'get_samples') else None)
        safe_call('tracks', lambda: list(rec.tracks) if hasattr(rec, 'tracks') else None)
        safe_call('signals', lambda: list(rec.signals) if hasattr(rec, 'signals') else None)
        safe_call('timestamps', lambda: rec.timestamps if hasattr(rec, 'timestamps') else None)

        print("\n--- End diagnostic dump ---\n")
        raise ValueError("No se pudo leer la señal ICP del archivo tras varios intentos.")

    # A partir de aquí icp y timestamps deben ser 1-D numpy arrays de la misma longitud
    icp = np.asarray(icp).ravel().astype(float)
    timestamps = np.asarray(timestamps).ravel().astype(float)
    min_len = min(len(icp), len(timestamps))
    if len(icp) != len(timestamps):
        icp = icp[:min_len]
        timestamps = timestamps[:min_len]
        print(f"Trimmed icp and timestamps to common length = {min_len}")

    # Remove out-of-range ICP values: keep only values within [0,50]
    valid_mask = (~np.isnan(icp)) & (icp >= 0.0) & (icp <= 50.0)
    removed = np.count_nonzero(~valid_mask)
    if removed > 0:
        print(f"Removed {removed} ICP samples outside [0,50]")
    icp = icp[valid_mask]
    timestamps = timestamps[valid_mask]
    if icp.size == 0:
        raise ValueError("No valid ICP samples within range [0,50] after filtering.")

    print(f"Samples: {len(icp)}, mean ICP = {np.nanmean(icp):.2f}")

    # Procesamiento: attempt to reuse loaded model if available
    if P_loaded is not None:
        thresholds = thresholds_loaded if thresholds_loaded is not None else [15.0, 20.0]
        print(f"Using loaded transition model and thresholds={thresholds}")
        states = discretize_icp(icp, thresholds=thresholds)
        P = np.asarray(P_loaded, dtype=float)
    else:
        thresholds = [15.0, 20.0]
        states = discretize_icp(icp, thresholds=thresholds)
        P = estimate_markov_matrix(states, n_states=3)
        # optionally save computed model
        if args.save_model:
            try:
                save_transition_model(P, thresholds, args.model_file)
            except Exception as e:
                print(f"Failed to save transition model: {e}")
    print("\nTransition matrix (Markov, 3x3):")
    print(P)

    durations = estimate_state_durations(states, timestamps, n_states=3)
    best_fits = fit_duration_distributions(durations)

    acc = evaluate_model(states, timestamps, P, best_fits)
    print(f"\n=== Semi-Markov Prediction Accuracy: {acc:.2f}% ===")

    # Ejemplo: probabilidades para el último estado
    last_state = int(states[-1])
    # computar duración del último run
    # buscar inicio del run
    run_start = len(states) - 1
    while run_start > 0 and states[run_start - 1] == states[-1]:
        run_start -= 1
    last_duration = float(timestamps[-1] - timestamps[run_start]) if len(timestamps) > 1 else 1.0
    probs = semi_markov_next_probabilities(last_state, last_duration, P, best_fits)
    print(f"\nNext state probabilities from state {last_state} (sojourn={last_duration:.1f}): {probs}")
