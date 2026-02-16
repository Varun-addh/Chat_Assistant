import logging
import sys


def configure_logging() -> None:
	root_logger = logging.getLogger()
	if root_logger.handlers:
		return

	handler = logging.StreamHandler(sys.stdout)
	formatter = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	handler.setFormatter(formatter)
	root_logger.addHandler(handler)
	root_logger.setLevel(logging.getLevelName("INFO"))
