from .converse_attack import ConverseAttack
from .max_attack import MaxAttack
from .guassian_attack import GuassianAttack
from .hidden_attack import HiddenAttack
from .litter_attack import LitterAttack
from .empire_attack import EmpireAttack


attacks = {
    "none": None,
    "converse": ConverseAttack,
    "max": MaxAttack,
    "gaussian": GuassianAttack,
    "hidden": HiddenAttack,
    "litter": LitterAttack,
    "empire": EmpireAttack,
}
