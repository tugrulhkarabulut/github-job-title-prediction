import logging
from pathlib import Path

fmt = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
logging.basicConfig(level=logging.DEBUG, format=fmt)
fmt = logging.Formatter(fmt=fmt)

logger = logging.getLogger(Path(__file__).stem)
handler = logging.FileHandler('logger.log')
handler.setFormatter(fmt)
logger.setLevel(logging.INFO)
logger.addHandler(handler)