import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)
from metrics.registry import DETECTOR

from .xception_detector import XceptionDetector
from .efficientnetb4_detector import EfficientDetector
from .ucf_detector import UCFDetector
from .spsl_detector import SpslDetector
from .f3net_detector import F3netDetector
from .srm_detector import SRMDetector
from .ffd_detector import FFDDetector
from .core_detector import CoreDetector
from .recce_detector import RecceDetector
from .capsule_net_detector import CapsuleNetDetector
from .meso4Inception_detector import Meso4InceptionDetector
