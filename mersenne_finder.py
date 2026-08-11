#!/usr/bin/env python3
"""
Continuous Mersenne prime finder.

A Mersenne number has the form  M = 2^n - 1.  Such a number can be
prime only when n itself is prime, so the search iterates over prime
exponents n.

This script fetches its list of prime exponents from a remote
primes.txt (by default the one produced by the companion 6k+/-1
prime finder).  For each prime exponent n it:

  1. Computes  M = 2^n - 1.
  2. Runs a **trial-division** pre-filter (divisors must be of the
     form 2kn+1, which is the necessary form for factors of a
     Mersenne number).
  3. Runs the **Lucas-Lehmer test** — the deterministic primality
     test for Mersenne numbers — to confirm or refute primality.

It is designed for short, resumable sessions: it loads its position
from state.json, works until a wall-clock budget expires, then
flushes state and results so the next session continues exactly
where this one stopped.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

# --- configuration -----------------------------------------------------------
DEFAULT_BUDGET_SECONDS = int(os.environ.get("MERSENNE_BUDGET_SECONDS", "270"))

# Source of prime exponents n.  This is the primes.txt produced by
# the 6k+/-1 prime finder running in the companion repo.
PRIMES_URL = os.environ.get(
    "PRIMES_URL",
    "https://raw.githubusercontent.com/dipali1249/Prime/refs/heads/main/primes.txt",
)

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "state.json"
MERSENNE_FILE = BASE_DIR / "mersenne_primes.txt"
LOG_FILE = BASE_DIR / "session.log"
EXPONENT_CACHE = BASE_DIR / "exponents.cache"


# ---------------------------------------------------------------------------#
#  Number-theory primitives
# ---------------------------------------------------------------------------#
def lucas_lehmer_test(p: int) -> bool:
    """Deterministic Lucas-Lehmer primality test for M_p = 2^p - 1.

    Returns True iff M_p is prime.  The Lucas-Lehmer sequence is
    defined for odd primes p >= 3; p == 2 (M=3, prime) is handled as
    a special case.
    """
    if p == 2:
        return True          # M_2 = 3 is prime
    if p < 3:
        return False
    M = (1 << p) - 1          # 2^p - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % M
    return s == 0


def trial_division_mersenne(p: int, limit: int = 200_000) -> bool:
    """Quick trial-division screen for M_p = 2^p - 1.

    Any factor q of M_p must satisfy  q = 2kp + 1  and  q ≡ ±1 (mod 8).
    We test such candidates up to ``limit``.  If a factor is found the
    number is definitely composite; if none is found we still cannot
    conclude primality — that is the Lucas-Lehmer test's job.

    Returns True if NO small factor was found (candidate still alive),
    False if a factor was found (definitely composite).
    """
    if p < 3:
        return p == 2  # M_2 = 3 is prime
    M = (1 << p) - 1
    # Any prime divisor q of M_p has q ≡ 1 or 7 (mod 8) and q ≡ 1 (mod 2p).
    # So we step by 2p and test q = 2kp+1 for k = 1, 2, ...
    # Additionally only k where 2k+1 ≡ ±1 (mod 4) survive the mod-8 test,
    # but the simple loop below is clear and correct; the mod-8 check
    # just skips some k values for speed.
    k = 1
    two_p = 2 * p
    while True:
        q = two_p * k + 1
        if q * q > M or q > limit:
            return True  # no small factor found
        if q % 8 in (1, 7) and M % q == 0:
            return False  # factor found → composite
        k += 1


def is_probable_prime_small(n: int) -> bool:
    """Plain primality test for small n (used to validate exponents)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ---------------------------------------------------------------------------#
#  Exponent fetching
# ---------------------------------------------------------------------------#
def fetch_exponents(url: str) -> list[int]:
    """Download the remote primes.txt and return the list of prime n values.

    The file is one prime per line.  We read it, parse to ints, and
    cache a local copy so subsequent sessions can fall back to it if
    the network is unavailable.
    """
    try:
        text = urllib.request.urlopen(url, timeout=60).read().decode("ascii")
        nums = [int(line.strip()) for line in text.splitlines() if line.strip()]
        # Cache locally for resilience
        EXPONENT_CACHE.write_text("\n".join(str(n) for n in nums), encoding="ascii")
        return nums
    except Exception as exc:
        log(f"WARNING: could not fetch {url} ({exc}); using cache")
        if EXPONENT_CACHE.exists():
            text = EXPONENT_CACHE.read_text(encoding="ascii")
            return [int(line.strip()) for line in text.splitlines() if line.strip()]
        raise RuntimeError(
            "No exponent source available (fetch failed and no cache)."
        )


# ---------------------------------------------------------------------------#
#  State management
# ---------------------------------------------------------------------------#
def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "exponent_index": 0,       # position in the exponents list
        "exponents_tested": 0,
        "mersenne_primes_found": 0,
        "largest_mersenne_prime": "0",
        "largest_exponent": 0,
        "sessions_run": 0,
    }


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(STATE_FILE)


def append_mersenne_prime(n: int, M: int) -> None:
    with MERSENNE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"n={n}\tM={M}\n")


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------#
#  Main session
# ---------------------------------------------------------------------------#
def run_session(budget_seconds: int) -> None:
    state = load_state()
    idx = state["exponent_index"]
    exponents_tested = state["exponents_tested"]
    mersenne_found = state["mersenne_primes_found"]
    largest_M = int(state["largest_mersenne_prime"])
    largest_n = state["largest_exponent"]
    sessions_run = state.get("sessions_run", 0)

    log(f"Session start: index={idx}, exponents_tested={exponents_tested}, "
        f"mersenne_primes_found={mersenne_found}, budget={budget_seconds}s")

    exponents = fetch_exponents(PRIMES_URL)
    log(f"Fetched {len(exponents)} prime exponents from {PRIMES_URL}")

    start = time.monotonic()

    while idx < len(exponents) and time.monotonic() - start < budget_seconds:
        n = exponents[idx]

        # Skip non-prime exponents defensively (M can only be prime if n is prime)
        if not is_probable_prime_small(n):
            idx += 1
            continue

        # Handle the trivial case n = 2  →  M = 3
        if n == 2:
            M = 3
            # M=3 is prime; record if not already present
            if mersenne_found == 0:
                append_mersenne_prime(n, M)
                mersenne_found += 1
                largest_M = M
                largest_n = n
                log(f"  Mersenne prime: n={n}  M={M}")
            idx += 1
            exponents_tested += 1
            continue

        M = (1 << n) - 1

        # --- Step 1: trial division pre-filter ---------------------------
        td_passed = trial_division_mersenne(n)

        # --- Step 2: Lucas-Lehmer definitive test -------------------------
        ll_passed = lucas_lehmer_test(n)

        exponents_tested += 1

        if ll_passed:
            append_mersenne_prime(n, M)
            mersenne_found += 1
            largest_M = M
            largest_n = n
            log(f"  ★ Mersenne PRIME: n={n}  M has {len(str(M))} digits  "
                f"(trial_division={'passed' if td_passed else 'N/A'}, LL=confirmed)")
        else:
            # Log only periodically to avoid flooding
            if exponents_tested % 100 == 0:
                log(f"  n={n}  M composite  (trial_division={'factor_found' if not td_passed else 'no_small_factor'}, LL=rejected)")

        idx += 1

    elapsed = time.monotonic() - start
    state = {
        "exponent_index": idx,
        "exponents_tested": exponents_tested,
        "mersenne_primes_found": mersenne_found,
        "largest_mersenne_prime": str(largest_M),
        "largest_exponent": largest_n,
        "sessions_run": sessions_run + 1,
    }
    save_state(state)

    log(f"Session end: index={idx}/{len(exponents)}, exponents_tested={exponents_tested}, "
        f"mersenne_primes_found={mersenne_found}, elapsed={elapsed:.1f}s")


def main() -> None:
    run_session(DEFAULT_BUDGET_SECONDS)


if __name__ == "__main__":
    main()
