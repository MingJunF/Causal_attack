from .WALL_q_learner import WALLQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .Continous_learner import CausalQLearner
# Try to import ODE learner
try:
    from .ode_q_learner import ODEQLearner
    ODE_AVAILABLE = True
    print("ODE learner successfully imported")
except (ImportError, OSError, AttributeError) as e:
    print(f"Warning: ODE learner disabled due to dependency issue: {e}")
    ODE_AVAILABLE = False

REGISTRY = {}
REGISTRY['Causal_q_learner'] = CausalQLearner
REGISTRY["WALL_q_learner"] = WALLQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner

if ODE_AVAILABLE:
    REGISTRY["ode_q_learner"] = ODEQLearner