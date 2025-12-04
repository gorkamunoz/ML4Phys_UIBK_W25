


def is_valid_srv(srv, d=None):
    """
    Check necessary (and for n=3, essentially sufficient) conditions for a Schmidt Rank Vector (SRV)
    of an n-qudit pure state.

    Args:
        srv (Iterable[int]): local ranks (r1, r2, ..., rn), each >= 1
        d (int | Iterable[int] | None): local dimensions. If int, all subsystems have dim d.
                                        If iterable, must have same length as srv. If None, skip dim checks.

    Returns:
        (ok: bool, reason: str)
    """
    from math import prod

    

    # Basic form checks
    try:
        r = [int(x) for x in srv]
    except Exception:
        return False, "SRV must be an iterable of integers."
    if any(x < 1 for x in r):
        return False, "All ranks must be positive integers."

    n = len(r)
    if n < 2:
        return False, "SRV must have length >= 2."

    # Extra: not considering (1,3,3) or (3,3,1) (too long sequences)
    if srv == (1,3,3) or srv == (3,3,1) or srv == (3,1,3):
        return False, "SRV is either (1,3,3) or (3,3,1)"
    

    # Local dimension checks (if provided)
    if d is not None:
        if isinstance(d, int):
            dims = [d] * n
        else:
            dims = list(d)
            if len(dims) != n:
                return False, "Local-dimension list must match SRV length."
        if any(di < 1 for di in dims):
            return False, "All local dimensions must be >= 1."
        for i, (ri, di) in enumerate(zip(r, dims)):
            if ri > di:
                return False, f"Rank r[{i}]={ri} exceeds local dimension d[{i}]={di}."

    # Submultiplicativity constraints: r_i <= Π_{j≠i} r_j
    for i in range(n):
        rhs = prod(r[j] for j in range(n) if j != i)
        if r[i] > rhs:
            return False, f"Violates submultiplicativity at index {i}: r[{i}]={r[i]} > product of others {rhs}."

    # Extra necessary constraint for exactly three parties:
    # if one party is pure (rank 1), the other two form a bipartite pure state ⇒ their single-party ranks are equal.
    if n == 3:
        for i in range(3):
            if r[i] == 1:
                j, k = (i + 1) % 3, (i + 2) % 3
                if r[j] != r[k]:
                    return False, (
                        f"For 3 parties with r[{i}]=1, the other two ranks must be equal "
                        f"(found r[{j}]={r[j]} and r[{k}]={r[k]})."
                    )

    

    # Note: For n > 3 these are necessary but not sufficient conditions.
    return True, "SRV passes all general necessary checks."