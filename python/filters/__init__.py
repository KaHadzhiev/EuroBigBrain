"""EuroBigBrain signal/bar filters.

Exports session gates, news blackouts, and other per-bar veto logic that
must run BEFORE a strategy's entry conditions are evaluated. Filters
return a boolean (``True`` = block this bar) so callers can compose them
with simple ``any(...)`` / ``all(...)`` semantics.
"""
