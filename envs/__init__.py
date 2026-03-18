try:
    from envs.boosterT12v2.BoosterT12v2Env import BoosterT12v2Env
    from envs.components.LowLevelController import LowLevelController
    from envs.components.MultiAgentLowLevelController import MultiAgentLowLevelController
    from envs.chaseBall.ChaseBallEnv import ChaseBallEnv
    from envs.passBall.PassBallEnv import PassBallEnv
    from envs.trapBall.TrapBallEnv import TrapBallEnv
except ModuleNotFoundError:
    # Allow lightweight modules like hyperGym to be imported without Isaac Gym.
    BoosterT12v2Env = None
    LowLevelController = None
    MultiAgentLowLevelController = None
    ChaseBallEnv = None
    PassBallEnv = None
    TrapBallEnv = None
