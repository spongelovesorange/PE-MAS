"""Checked-in flyback PLECS assets used by the Studio workflow."""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
FLYBACK_PLECS_MODEL = MODEL_DIR / "Flyback_effi.plecs"
MOSFET_THERMAL_XML = MODEL_DIR / "C2M0080120D.xml"
BODY_DIODE_THERMAL_XML = MODEL_DIR / "C2M0080120D_bodydiode.xml"

__all__ = [
    "MODEL_DIR",
    "FLYBACK_PLECS_MODEL",
    "MOSFET_THERMAL_XML",
    "BODY_DIODE_THERMAL_XML",
]
