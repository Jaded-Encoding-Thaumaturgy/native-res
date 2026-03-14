import importlib
from logging import getLogger

from vsview.api import run_in_background

logger = getLogger(__name__)

QCHART_IMPORTED = False


@run_in_background(name="WarmupImportPlots")
def warmup_plots() -> None:
    """Warmup import of QChart to avoid long startup times."""
    global QCHART_IMPORTED

    if not QCHART_IMPORTED:
        QCHART_IMPORTED = True
        logger.debug("Importing Plots")
        importlib.import_module("nativeres.plotting")
        logger.debug("Importing Plots done")
