from .average import Average
from .bridge import BRIDGE
from .d_bulyan import DBulyan
from .d_krum import DKrum
from .d_median import DMedian
from .mozi import MOZI
from .opdpg import OPDPG
from .qc import QC
from .zeno import Zeno

pars = {
    "average": Average,
    "bridge": BRIDGE,
    "bulyan": DBulyan,
    "krum": DKrum,
    "median": DMedian,
    "mozi": MOZI,
    "opdpg": OPDPG,
    "qc": QC,
    "zeno": Zeno,
}
