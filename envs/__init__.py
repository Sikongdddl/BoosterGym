try:
    from envs.components.LowLevelController import LowLevelController
    from envs.chaseBall.ChaseBallEnv import ChaseBallEnv
    from envs.passBall.PassBallEnv import PassBallEnv
    from envs.trapBall.TrapBallEnv import TrapBallEnv
except ModuleNotFoundError:
    # Allow lightweight modules like hyperGym to be imported without Isaac Gym.
    LowLevelController = None
    ChaseBallEnv = None
    PassBallEnv = None
    TrapBallEnv = None
