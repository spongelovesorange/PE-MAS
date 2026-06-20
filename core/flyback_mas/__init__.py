try:
    from .graph import app
    from .graph import app_headless
except Exception:
    app = None
    app_headless = None

from .state import PowerSupplyState
