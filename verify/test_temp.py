
import json
import os
import re
import librosa
import sys
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
import numpy as np

# 添加父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
from youdub.step042_tts_xtts import tts as xtts_tts

print(inspect.signature(xtts_tts))