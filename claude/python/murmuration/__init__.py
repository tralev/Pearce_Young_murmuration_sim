"""Murmuration — Python bindings over the murmur_core Rust simulation engine.

Re-exports the compiled `_core` extension (design/03_observables_bindings.md §2.1).
"""

from ._core import Simulation, Snapshot

__all__ = ["Simulation", "Snapshot"]
