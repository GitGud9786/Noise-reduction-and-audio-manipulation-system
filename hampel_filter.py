import numpy as np

def robust_mad(x, eps=1e-12):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med, max(1.4826 * mad, eps)

def hampel_detect_mask(x, window_size=17, n_sigma=6.0):
    """
    Returns a boolean mask of detected impulsive outliers.
    """
    # Ensure array is 1-D numeric and window_size is an integer
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.ravel()
    try:
        window_size = int(window_size)
    except Exception:
        raise TypeError("window_size must be convertible to int")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    n = len(x)
    half = window_size // 2
    mask = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = int(max(0, i - half))
        hi = int(min(n, i + half + 1))
        med, scale = robust_mad(x[lo:hi])
        z = abs(x[i] - med) / scale
        if z > n_sigma:
            mask[i] = True
    return mask

def merge_segments(mask):
    """Yield (start, end) for consecutive True runs in mask; end is exclusive."""
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        yield i, j
        i = j

def crossfade_edges(y, seg_start, seg_end, fade=3):
    """
    Apply small linear crossfade at boundaries to avoid micro steps.
    seg_end is exclusive.
    """
    n = len(y)
    # Left boundary fade
    if seg_start > 0:
        Ls = max(seg_start - fade, 0)
        Le = seg_start
        if Le - Ls > 1:
            t = np.linspace(0, 1, Le - Ls, endpoint=False)
            y[Ls:Le] = (1 - t) * y[Ls:Le] + t * y[Ls:Le]  # noop placeholder to keep structure

    # Right boundary fade
    if seg_end < n:
        Rs = seg_end
        Re = min(seg_end + fade, n)
        if Re - Rs > 1:
            t = np.linspace(0, 1, Re - Rs, endpoint=False)
            y[Rs:Re] = (1 - t) * y[Rs:Re] + t * y[Rs:Re]  # noop; boundary continuity handled by interpolation

def repair_impulses(
    x,
    window_size=17,
    n_sigma=6.0,
    max_interp_len=16,
    prefer_interpolation=True,
    edge_fade=3
):
    """
    Segment-aware impulsive repair:
    - Detect with Hampel (MAD).
    - For short segments (<= max_interp_len), interpolate between clean neighbors.
    - For single-sample outliers, interpolate (or median if interpolation not possible).
    - For longer segments, fall back to median-in-window fill as last resort.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = x.copy()

    # 1) Detect
    mask = hampel_detect_mask(x, window_size=window_size, n_sigma=n_sigma)

    # 2) Process segments
    half = window_size // 2
    for s, e in merge_segments(mask):
        seg_len = e - s

        # Find clean neighbors
        left = s - 1
        right = e
        # Move left to last clean sample
        while left >= 0 and mask[left]:
            left -= 1
        # Move right to first clean sample
        while right < n and mask[right]:
            right += 1

        can_interp = left >= 0 and right < n

        if prefer_interpolation and can_interp and seg_len <= max_interp_len:
            # Linear interpolation is robust and click-safe
            y[s:e] = np.interp(np.arange(s, e), [left, right], [y[left], y[right]])
            # Small crossfade to avoid boundary ticks (mostly a guard)
            crossfade_edges(y, s, e, fade=edge_fade)
        else:
            # Fallback: local robust median fill per sample in segment
            for i in range(s, e):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                y[i] = np.median(y[lo:hi])

            # Optional: small smoothing just within the segment
            if seg_len >= 3:
                # 3-sample moving median/mean as a gentle smoother inside the segment
                k = 3
                for i in range(s, e):
                    lo = max(s, i - k//2)
                    hi = min(e, i + k//2 + 1)
                    y[i] = np.median(y[lo:hi])

            crossfade_edges(y, s, e, fade=edge_fade)

    return y, mask
