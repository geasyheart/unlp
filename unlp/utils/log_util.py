# -*- coding: utf8 -*-

#
import logging
import sys

logger = logging.getLogger("unlp")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
