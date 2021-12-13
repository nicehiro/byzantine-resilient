from .converse_attack import ConverseAttack
from .max_attack import MaxAttack
from .guassian_attack import GaussianAttack
from .hidden_attack import HiddenAttack
from .litter_attack import LitterAttack
from .empire_attack import EmpireAttack


attacks = {
    "converse": ConverseAttack,
    "max": MaxAttack,
    "gaussian": GaussianAttack,
    "hidden": HiddenAttack,
    "litter": LitterAttack,
    "empire": EmpireAttack,
}
