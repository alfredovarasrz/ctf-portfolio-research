"""Versioned logical capitalization digest, independent of Arrow buffers."""
from datetime import date
from hashlib import sha256
from math import isnan, isinf
import struct

FORMAT = 'logical-cap-v1'
HEADER = b'PS3 logical-cap-v1\0id:int64,eom:date-ordinal32,market_equity:float64\0'


def logical_cap_digest(rows, count):
    """Hash sorted unique (id, month, cap) VALUES, never allocation padding.

    Null and NaN remain distinct. All NaN payloads share one logical token;
    signed zero shares the positive-zero representation. Finite values retain
    every binary64 bit. Infinity signs, dates, IDs, ordering and count are bound.
    """
    if type(count) is not int or count < 0:
        raise ValueError('Nonnegative capitalization row count required')
    result = sha256(HEADER + struct.pack('>Q', count))
    previous = None
    seen = 0
    for security, month, capital in rows:
        if type(security) is not int or not -(1 << 63) <= security < (1 << 63) or type(month) is not date:
            raise ValueError('Exact nonnull int64/date capitalization keys required')
        key = (month, security)
        if previous is not None and key <= previous:
            raise ValueError('Sorted unique capitalization keys required')
        previous = key
        result.update(struct.pack('>qi', security, month.toordinal()))
        if capital is None:
            result.update(b'N')
        else:
            if type(capital) is not float:
                raise ValueError('Float64 or null capitalization value required')
            if isnan(capital):
                result.update(b'A')
            elif isinf(capital):
                result.update(b'P' if capital > 0 else b'M')
            else:
                result.update(b'V' + struct.pack('>d', 0. if capital == 0 else capital))
        seen += 1
    if seen != count:
        raise ValueError('Capitalization row count differs from logical stream')
    return result.hexdigest()
