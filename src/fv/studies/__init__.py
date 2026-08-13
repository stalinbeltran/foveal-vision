"""I — the OAT study (schedule): a plan of ordered axes over H with B fixed.

A study automates the human who swept one axis at a time (barrido-por-ejes.md
§0): it describes the ORDER of axes and their dependencies and GUIDES the user
step by step — it does NOT execute (D-H1). Per step it derives the base from the
problem (fv.models.derive), fixes the carried winners, and generates a sweep H
(base inline) with the existing machinery (contract ⑫). The length is DYNAMIC:
a confirmed winner can unlock sub-axes (n_layers=3 -> channels[0..2]).
"""

from fv.studies.driver import (StudyError, advance, confirm, create_study,
                               status)
from fv.studies.store import StudyStore

__all__ = ["StudyStore", "StudyError", "create_study", "status", "advance",
           "confirm"]
