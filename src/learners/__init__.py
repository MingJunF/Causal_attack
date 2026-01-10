from .q_learner import QLearner
from .WALL_q_learner import WALLQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .Continous_learner import CausalQLearner

REGISTRY = {}
REGISTRY['Causal_q_learner'] = CausalQLearner
REGISTRY["WALL_q_learner"] = WALLQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner"] = QLearner
