"""Whole-month blocked validation across a complete completed history (stdlib)."""
from datetime import date
from math import ceil


def month_index(value):
    return value.year * 12 + value.month - 1


def month_date(index):
    from calendar import monthrange
    year, month = divmod(index, 12)
    return date(year, month + 1, monthrange(year, month + 1)[1])


def capped_calendar_blocks(dates, *, minimum_blocks=5, maximum_months=96):
    """Return (date blocks, boundaries), including calendar gaps and empty blocks.

    Equal calendar widths use integer endpoints, not the number of observed rows.
    Every eligible month belongs to exactly one held-out block. With a history
    shorter than five months, empty blocks remain explicit rather than changing
    the declared minimum. Callers omit folds with no usable training/validation.
    """
    dates = sorted(set(dates))
    if minimum_blocks < 2 or maximum_months < 1:
        raise ValueError('At least two blocks and a positive calendar cap required')
    if any(d != month_date(month_index(d)) for d in dates):
        raise ValueError('Completed month-end dates required')
    if not dates:
        return [[] for _ in range(minimum_blocks)], []
    start, end = month_index(dates[0]), month_index(dates[-1]) + 1
    span = end - start
    count = max(minimum_blocks, ceil(span / maximum_months))
    edges = [start + (i * span) // count for i in range(count + 1)]
    blocks, boundaries = [], []
    for i, (left, right) in enumerate(zip(edges, edges[1:])):
        if right - left > maximum_months:
            raise ValueError('Calendar validation block exceeds declared cap')
        blocks.append([d for d in dates if left <= month_index(d) < right])
        boundaries.append(dict(fold=i + 1, first_month=str(month_date(left)),
            last_month=str(month_date(right - 1)) if right > left else None,
            calendar_months=right - left))
    return blocks, boundaries
