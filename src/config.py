"""Load analysis configuration (data paths supplied by the user, not shipped)."""
import logging
import yaml

logger = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg
