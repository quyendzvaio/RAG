import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Cau hinh root logger mot lan; uvicorn chi cau hinh logger rieng cua no."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
