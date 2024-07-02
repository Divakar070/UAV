This is the AI base class. This module contains:

- `State` class: It is a data structure uniquely describes the state of
  a system.
- `Action` class: In our implementation, we use an integer to describe 
  an action [0,1,...,COUNT-1], where COUNT is the total number of actions.
- `Pos` class: It manages the coordination of the map.
- `RL` class: It is a base class to be inherited by other actual 
  algorithm. It initializes several visible properties and methods.
- `DecayingFloat` class: It provides a convenient means to support
  a float number that decays over the time.
'''

## define position class to manage coordination
class Pos:
    CELL_SIZE = None # to be provided
    def __init__(self, col, row):
        self.col = col
        self.row = row
    def __eq__(self, other):
        return self.col==other.col and self.row==other.row
    def copy(self):
        return Pos(self.col,self.row)
    def xy(self):
        '''Return the center (x,y) coordinate of the cell.'''
        x = self.col*Pos.CELL_SIZE + Pos.CELL_SIZE//2
        y = self.row*Pos.CELL_SIZE + Pos.CELL_SIZE//2
        return (x,y)

## action class
class Action:
    '''
    Define Action in this class. It mainly defines constants for actions and 
    a constant named `COUNT` carrying the number of actions.
    '''
    COUNT = 0 # number of total actions, to be provided

## state class
class State:
    '''
    Define State in this class. It must provide `valid_action_matrix` array
    and `__str__()`.
    '''

    valid_action_matrix = None # to be provided

    def __init__(self, col, row, step=0):
        ## state = (x,y,t)
        self.col = col   # x
        self.row = row   # y
        self.step = step # t

    def valid_actions(self):
        '''It returns a list of valid actions.'''
        return State.valid_action_matrix[self.col][self.row]
    
    def __str__(self):
        '''It returns a str representation of the state. It can be useful
        for indexing a state and showing meaningful states for debugging.'''
        return f"({self.col},{self.row},{self.step})"

# Extend the RL base class to include placeholders for DQN-specific methods
class RL:
    def __init__(self, name="Base-Class"):
        self.name = name
        self.is_exploration = True
        self.epsilon = 0.0

    def q(self, state):
        return [0.0]*Action.COUNT
    
    def load_data(self) -> int:
        return -1 # failed or NA

    def save_data(self, round) -> bool:
        return False # failed or NA

    def store_transition(self, state, action, reward, next_state, done):
        pass # Placeholder for DQN experience replay

    def learn(self):
        pass # Placeholder for DQN learning step


## decaying float number for epsilon
class DecayingFloat:
    '''
    This class provides a delaying float number. It is disguised as a 
    `float` but provides methods to trigger a decay.

    The constructor takes the following inputs parameters.

    Parameters:
    value : float
        The initial value of the decaying float number. We assume it
        is a positive value.
    factor : float, optional, default=None
        The decaying factor. If None is specified, the float number will
        not decay.
    minval : float, optional, default=None
        The minimum value of the float. If None is specified, the float
        number can reach zero which is the lowest.
    mode : str
        It can be either "exp" for exponential decaying or "linear" for
        linear decaying. An unrecognized string will cause the value 
        not to decay.
    '''
    def __init__(self, value:float, factor:float=None, minval:float=None,
                 mode:str="exp"):
        self.init = value
        self.value = value
        self.factor = factor
        self.minval = minval
        self.mode = mode

    def __float__(self) -> float:
        '''
        This method performs the type casting operation to return a float.
        '''
        return float(self.value)

    def reset(self):
        '''
        To start over the decaying function from the beginning.
        '''
        self.value = self.init

    def decay(self):
        '''
        To perform a step of decay. The decaying depends on the `factor`
        and the `mode`. If `factor` is not given or `mode` string is
        unrecognized, the method simply does nothing.
        '''
        if self.factor==None: return

        if self.mode=="exp":      self.value *= self.factor
        elif self.mode=="linear": self.value -= self.factor
        
        if self.minval==None: 
            return
        elif self.value<self.minval:
            self.value = self.minval

-------------------------------------------------------------------------------

This is the AI base class. This module contains:

- `State` class: It is a data structure uniquely describes the state of
  a system.
- `Action` class: In our implementation, we use an integer to describe 
  an action [0,1,...,COUNT-1], where COUNT is the total number of actions.
- `Pos` class: It manages the coordination of the map.
- `RL` class: It is a base class to be inherited by other actual 
  algorithm. It initializes several visible properties and methods.
- `DecayingFloat` class: It provides a convenient means to support
  a float number that decays over the time.
'''

## define position class to manage coordination
class Pos:
    CELL_SIZE = None # to be provided
    def __init__(self, col, row):
        self.col = col
        self.row = row
    def __eq__(self, other):
        return self.col==other.col and self.row==other.row
    def copy(self):
        return Pos(self.col,self.row)
    def xy(self):
        '''Return the center (x,y) coordinate of the cell.'''
        x = self.col*Pos.CELL_SIZE + Pos.CELL_SIZE//2
        y = self.row*Pos.CELL_SIZE + Pos.CELL_SIZE//2
        return (x,y)

## action class
class Action:
    '''
    Define Action in this class. It mainly defines constants for actions and 
    a constant named `COUNT` carrying the number of actions.
    '''
    COUNT = 0 # number of total actions, to be provided

## state class
class State:
    '''
    Define State in this class. It must provide `valid_action_matrix` array
    and `__str__()`.
    '''

    valid_action_matrix = None # to be provided

    def __init__(self, col, row, step=0):
        ## state = (x,y,t)
        self.col = col   # x
        self.row = row   # y
        self.step = step # t

    def valid_actions(self):
        '''It returns a list of valid actions.'''
        return State.valid_action_matrix[self.col][self.row]
    
    def __str__(self):
        '''It returns a str representation of the state. It can be useful
        for indexing a state and showing meaningful states for debugging.'''
        return f"({self.col},{self.row},{self.step})"

# Extend the RL base class to include placeholders for DQN-specific methods
class RL:
    def __init__(self, name="Base-Class"):
        self.name = name
        self.is_exploration = True
        self.epsilon = 0.0

    def q(self, state):
        return [0.0]*Action.COUNT
    
    def load_data(self) -> int:
        return -1 # failed or NA

    def save_data(self, round) -> bool:
        return False # failed or NA

    def store_transition(self, state, action, reward, next_state, done):
        pass # Placeholder for DQN experience replay

    def learn(self):
        pass # Placeholder for DQN learning step


## decaying float number for epsilon
class DecayingFloat:
    '''
    This class provides a delaying float number. It is disguised as a 
    `float` but provides methods to trigger a decay.

    The constructor takes the following inputs parameters.

    Parameters:
    value : float
        The initial value of the decaying float number. We assume it
        is a positive value.
    factor : float, optional, default=None
        The decaying factor. If None is specified, the float number will
        not decay.
    minval : float, optional, default=None
        The minimum value of the float. If None is specified, the float
        number can reach zero which is the lowest.
    mode : str
        It can be either "exp" for exponential decaying or "linear" for
        linear decaying. An unrecognized string will cause the value 
        not to decay.
    '''
    def __init__(self, value:float, factor:float=None, minval:float=None,
                 mode:str="exp"):
        self.init = value
        self.value = value
        self.factor = factor
        self.minval = minval
        self.mode = mode

    def __float__(self) -> float:
        '''
        This method performs the type casting operation to return a float.
        '''
        return float(self.value)

    def reset(self):
        '''
        To start over the decaying function from the beginning.
        '''
        self.value = self.init

    def decay(self):
        '''
        To perform a step of decay. The decaying depends on the `factor`
        and the `mode`. If `factor` is not given or `mode` string is
        unrecognized, the method simply does nothing.
        '''
        if self.factor==None: return

        if self.mode=="exp":      self.value *= self.factor
        elif self.mode=="linear": self.value -= self.factor
        
        if self.minval==None: 
            return
        elif self.value<self.minval:
            self.value = self.minval



--------------------------------------------------------------------------------------

'''
This program reproduces the simulation done in the following paper:
- H. Bayerlein, P. De Kerret and D. Gesbert, "Trajectory Optimization for 
  Autonomous Flying Base Station via Reinforcement Learning," IEEE 19th 
  International Workshop on Signal Processing Advances in Wireless 
  Communications (SPAWC), 2018, pp. 1-5.

The scenario has a map of 15-by-15 cells. The UAV starts from the left
bottom and moves up/down/left/right to the next cell to maximize the
sum rate. Each cell is indexed by (col,row) as shown below:
```
            0   1   2   3   ... COLS-1
          +---+---+---+---+ ... +---+
        0 |   |   |   |   |     |   | 
          +---+---@---+---+ ... +---+   <-- @:UE1
        1 |   |   |   |   |     |   | 
          +---+---+---+---+ ... @---+   <-- @:UE2
        2 |   |   | O |   |     |   |   <-- O:UAV
          +---+---+---+---+ ... +---+
          .                         .
          .                         .
          +---+---+---+---+ ... +---+
   ROWS-1 |   |   |   |   |     |   |
          +---+---+---+---+ ... +---+
```
'''

import math
import pygame
from   shapely.geometry import LineString, Polygon

#from ai_base import State, Action, Pos
#from ai_random import RandomMove
#from ai_qlearning import Q_Learning
#from ai_sarsa import SARSA
#from ai_DQN import DeepQLearning

## switches
SHOW_ANIMATION = 0   # 1:Yes, 0:No
LOAD_DATA      = 1   # 1:Yes, 0:No, to load the file: f"{ai.name}-load.json"
SAVE_DATA      = 1   # 1:Yes, 0:No
EXPLORATION    = 1   # 1:Yes, 0:No more exploration, do testing
SHOW_RATE      = 0   # 1:Yes, 0:No, showing rate on the cells

SAMPLE_MODE     = 1      # 1:Yes, 0:No, to print rewards of sampled rounds only
SAMPLE_INTERVAL = 500  # number of interval for each sampling, in this mode, 
                         # during the sampling, the sim performs full exploitation

## stateless mode
# SHOW_ANIMATION = 1
# LOAD_DATA      = 0 # don't load 
# SAVE_DATA      = 0 # don't save
# EXPLORATION    = 1
# SAMPLE_MODE    = 0 # don't sample

## check result mode
# SHOW_ANIMATION = 1
# LOAD_DATA      = 1 
# SAVE_DATA      = 0 # don't save
# EXPLORATION    = 0 # don't explore
# SAMPLE_MODE    = 0 # don't sample

## continue progress mode
SHOW_ANIMATION = 0 # no animation please
LOAD_DATA      = 1 # load progress
SAVE_DATA      = 1 # also save progress
EXPLORATION    = 1
SAMPLE_MODE    = 1 # sample outcome

#############################################
## setup the simulation world
#############################################

## define world layout
FPS = 60                   # frame per second
WIDTH, HEIGHT = 800, 880   # screen dimensions
ROWS, COLS = 15, 15        # grid dimensions
ALTITUDE = 20              # altitude of the UAV (in pixels)
CELL_SIZE = WIDTH // COLS  # cell size (in pixels)
METER_PER_PIXEL = 2        # use it to convert to actual distance

Pos.CELL_SIZE = CELL_SIZE  # initialize Pos class

if True: # show layout info?
    print(f"Simulation info:")
    print(f"- The area is {CELL_SIZE*COLS*METER_PER_PIXEL}-by-{CELL_SIZE*ROWS*METER_PER_PIXEL} in meters")
    print(f"- UAV is flying at {ALTITUDE*METER_PER_PIXEL} meters above ground")
    print(f"- The flying altitude is about the height of a {ALTITUDE*METER_PER_PIXEL*0.3:.0f}-story building")

## define world objects
class CELL:
    SIZE = CELL_SIZE              # size of each grid cell
    BACKGROUND = (255, 255, 255)  # white color
    START_COLOR = (0, 100, 0)     # dark green
    END_LABEL = "X"               # label to mark in the final cell
    GRID_COLOR = (0, 0, 0)        # black color for grid lines

class OBSTACLE:
    POS_LIST = []                 # list of positions of the cells
    COLOR = (70, 70, 70)          # dark grey
    for x in range(9,11):
        for y in range(8,12):
            POS_LIST.append(Pos(x,y))

class UAV:
    COLOR = (255, 0, 0)           # red color for the UAV
    SIZE = CELL.SIZE // 3         # size of the UAV to draw
    START_POS = Pos(0, ROWS-1)    # left-bottom
    END_POS = Pos(0, ROWS-1)      # left-bottom
    FLIGHT_TIME = 50              # flight time of the UAV including returning

class UE:
    COLOR = (0, 0, 255)           # blue color for UEs
    SIZE = CELL.SIZE // 3         # size of the UE to draw
    POS = {0:Pos(4.5, 2.5),       # list of users, including their location
           1:Pos(11.5, 6.5)}      #   which is on the grid
    RATE = {}                     # to be filled with the rate per UE per cell

class SHADOW:
    COLOR1 = (200, 200, 200)      # very light grey
    COLOR2 = (128, 128, 128)      # light grey for more than 1 blockage
    NLOS = [] # 3D array [ue_id][col][row] -> 0:NoBlockage, 1:Blocked
    BLOCKAGE = [[0 for _ in range(ROWS)] for _ in range(COLS)] # blockage count
    ## detect blockage, i.e. non-line-of-sight
    for id,pos in UE.POS.items():
        NLOS.append([[0 for _ in range(ROWS)] for _ in range(COLS)])
        p1 = pos.xy()
        for col in range(COLS):
            for row in range(ROWS):
                NLOS[id][col][row] = 0   # no blockage, line-of-sight (LOS)
                p2 = Pos(col,row).xy()   # center position of cell (col,row)
                line1 = LineString([p1, p2])
                for obstacle in OBSTACLE.POS_LIST:
                    x,y = obstacle.col*CELL.SIZE, obstacle.row*CELL.SIZE
                    rect1 = Polygon([(x, y), (x+CELL.SIZE, y), 
                                     (x+CELL.SIZE, y+CELL.SIZE), (x, y+CELL.SIZE)])
                    if line1.intersects(rect1):
                        NLOS[id][col][row] = 1 # blocked, set 1 to Non line-of-sight
                        break
    ## consolidate blockage from all UEs into `BLOCKAGE[col][row]`
    for id in UE.POS:
        for col in range(COLS):
            for row in range(ROWS):
                BLOCKAGE[col][row] += NLOS[id][col][row]

#############################################
## configure State & Action classes
#############################################

## initialize action class by providing valid actions
Action.UP    = 0   # define all actions here starting from 0
Action.DOWN  = 1
Action.LEFT  = 2
Action.RIGHT = 3
Action.COUNT = 4   # COUNT must be the total number of actions

## initialize state class by providing valid moves for each cell in a matrix
def get_state_valid_move_matrix():
    valid_actions = [[[Action.UP,Action.DOWN,Action.LEFT,Action.RIGHT] 
                                    for _ in range(ROWS)] for _ in range(COLS)]
    for col in range(COLS):
        valid_actions[col][0].remove(Action.UP) # can't move up at top row
        valid_actions[col][ROWS-1].remove(Action.DOWN) # can't move down at bottom row
    for row in range(ROWS):
        valid_actions[0][row].remove(Action.LEFT) # can't move left at leftmost col
        valid_actions[COLS-1][row].remove(Action.RIGHT) # can't move right at rightmost col
    for pos in OBSTACLE.POS_LIST:
        valid_actions[pos.col-1][pos.row].remove(Action.RIGHT) # can't enter the obstacle
        valid_actions[pos.col+1][pos.row].remove(Action.LEFT)  # around it from any 
        valid_actions[pos.col][pos.row-1].remove(Action.DOWN)  # directions
        valid_actions[pos.col][pos.row+1].remove(Action.UP)
    return valid_actions

State.valid_action_matrix = get_state_valid_move_matrix() # install into State class

#############################################
## statistic collector
#############################################

class STAT:
    round = 1  # record which round the sim is currently running
    step  = 0  # record the current flight time in this round
    reward = 0 # total reward so far in this round
    def reset():
        STAT.step = 0
        STAT.reward = 0

#############################################
## simulation world class
#############################################

class SimWorld:
    
    def __init__(self, display=True):

        self.reset()
        pygame.init()
        self.clock = pygame.time.Clock()

        self.pause = False
        self.display = display
        if not self.display: return

        self.font = pygame.font.SysFont(None, 55)   # label font
        self.font2 = pygame.font.SysFont(None, 36)  # message font
        self.font3 = pygame.font.SysFont(None, 24)  # tiny font
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("UAV Trajectory Optimization")

    def reset(self):
        self.uav_pos = UAV.START_POS.copy()

    def toggle_pause(self):
        self.pause = not self.pause

    def _label_cell(self, text, row, col):
        '''Function to label a specific cell with text.'''
        img = self.font.render(text, True, CELL.GRID_COLOR)
        x = (CELL.SIZE - img.get_rect().width) / 2
        y = (CELL.SIZE - img.get_rect().height) / 2
        self.screen.blit(img, (col*CELL.SIZE+x, row*CELL.SIZE+y))

    def put_message(self, text):
        '''Function to write a message.'''
        if not self.display: return
        img = self.font2.render(text, True, CELL.GRID_COLOR)
        self.screen.blit(img, (int(CELL.SIZE/10), ROWS*CELL.SIZE+5))

    def show_three_floats(self, col, row, values):
        '''Show up to three floats on this cell, for debugging purpose.'''
        if not self.display: return
        margin = 3
        if values[0] is not None:
            img = self.font3.render(f"{values[0]:.1f}", True, CELL.GRID_COLOR)
            x = (CELL.SIZE - img.get_rect().width) / 2
            y = margin
            self.screen.blit(img, (col*CELL.SIZE+x, row*CELL.SIZE+y))
        if values[1] is not None:
            img = self.font3.render(f"{values[1]:.1f}", True, CELL.GRID_COLOR)
            x = (CELL.SIZE - img.get_rect().width) / 2
            y = (CELL.SIZE - img.get_rect().height) / 2
            self.screen.blit(img, (col*CELL.SIZE+x, row*CELL.SIZE+y))
        if values[2] is not None:
            img = self.font3.render(f"{values[2]:.1f}", True, CELL.GRID_COLOR)
            x = (CELL.SIZE - img.get_rect().width) / 2
            y = CELL.SIZE - img.get_rect().height - margin
            self.screen.blit(img, (col*CELL.SIZE+x, row*CELL.SIZE+y))

    def render_world(self):
        '''Function to render the world.'''
        if self.display:
            ## background
            self.screen.fill(CELL.BACKGROUND)

            ## set grid properties for drawing
            grid = []
            ## - background
            for row in range(ROWS):
                for col in range(COLS):
                    grid.append((row,col,CELL.BACKGROUND))
            ## - shadow
            for row in range(ROWS): 
                for col in range(COLS):
                    if SHADOW.BLOCKAGE[col][row]==1:
                        grid.append((row,col,SHADOW.COLOR1))
                    elif SHADOW.BLOCKAGE[col][row]>1:
                        grid.append((row,col,SHADOW.COLOR2))
            ## - obstacle cells
            for pos in OBSTACLE.POS_LIST:
                grid.append((pos.row,pos.col,OBSTACLE.COLOR))
            ## - start cell
            grid.append((UAV.END_POS.row,UAV.END_POS.col,CELL.START_COLOR))

            ## draw grid
            for (row,col,color) in grid:
                rect = pygame.Rect(col*CELL.SIZE, row*CELL.SIZE, CELL.SIZE, CELL.SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, CELL.GRID_COLOR, rect, 1)
            
            ## label the final cell
            self._label_cell(CELL.END_LABEL, UAV.END_POS.row, UAV.END_POS.col)

            ## draw users
            for pos in UE.POS.values():
                (x,y) = Pos(pos.col,pos.row).xy()
                pygame.draw.circle(self.screen, UE.COLOR, (x,y), UE.SIZE)

            ## draw UAV
            (x,y) = Pos(self.uav_pos.col,self.uav_pos.row).xy()
            pygame.draw.circle(self.screen, UAV.COLOR, (x,y), UAV.SIZE)

            ## show status
            status = f"FPS = {FPS}  |  Step = {STAT.step}"
            img = self.font2.render(status, True, CELL.GRID_COLOR)
            self.screen.blit(img, (10,HEIGHT-img.get_rect().height))

            ## show rates? 
            if SHOW_RATE:
                for col in range(COLS):
                    for row in range(ROWS):
                        v1 = UE.RATE[0][col][row]
                        v2 = UE.RATE[1][col][row]
                        # v3 = v1+v2 # sum rate
                        v3 = min(v1,v2) # the paper seems using this instead of sum rate
                        self.show_three_floats(col,row,[v1,v3,v2])

    def move_uav(self, direction):
        '''Function to move the UAV given a direction.'''
        if direction==Action.UP and self.uav_pos.row>0:
            self.uav_pos.row -= 1
        elif direction==Action.DOWN and self.uav_pos.row<ROWS-1:
            self.uav_pos.row += 1
        elif direction==Action.LEFT and self.uav_pos.col>0:
            self.uav_pos.col -= 1
        elif direction==Action.RIGHT and self.uav_pos.col<COLS-1:
            self.uav_pos.col += 1

    def show(self):
        '''Show the world on the screen.'''
        if self.display:
            pygame.display.flip()
            self.clock.tick(FPS if FPS<120 else 0)

#############################################
## communication model
#############################################

class COMM:

    FREQ  = 2.4e9  # operating frequency in Hz (2.4GHz)
    ALPHA = 2      # pathloss exponent
    SIGMA = 1      # Rayleigh fading scaling factor
    BETA_LOS  = 1     # shadowing attenuation for LOS
    BETA_NLOS = 0.01  # shadowing attenuation for NLOS

    NOISE_dBm = -174  # dBm, this is the thermal noise per Hz
    NOISE = (10**(NOISE_dBm/10)) / 1000       # in linear scale

    TX_POWER_dBm = 15     # dBm, transmit power
    TX_POWER = (10**(TX_POWER_dBm/10)) / 1000 # in linear scale

    def distance(p1,p2):
        x1,y1 = p1
        x2,y2 = p2
        return math.sqrt(ALTITUDE**2 + (x1-x2)**2 + (y1-y2)**2)

    def get_rate(d, NLOS):
        beta = COMM.BETA_NLOS if NLOS else COMM.BETA_LOS
        pl = math.pow(d,-COMM.ALPHA) * beta       # pathloss
        sn_ratio = COMM.TX_POWER*pl / COMM.NOISE  # signal-to-noise ratio, linear scale
        return math.log2(1+sn_ratio)              # Shannon's rate bps per Hz

    def get_ue_rate_matrix():
        ue_rate = {}
        for ue_id,ue_pos in UE.POS.items():
            ue_rate[ue_id] = [[0 for _ in range(ROWS)] for _ in range(COLS)] # create a 2D array with zeroes
            for col in range(COLS):
                for row in range(ROWS):
                    d = COMM.distance(ue_pos.xy(),Pos(col,row).xy()) * METER_PER_PIXEL
                    ue_rate[ue_id][col][row] = COMM.get_rate(d, SHADOW.NLOS[ue_id][col][row])
        return ue_rate

## fill UE.RATE matrix, to avoid repeating the same calculation over and over again
UE.RATE = COMM.get_ue_rate_matrix()

#############################################
## reward function
## - input: state
## - output: reward
#############################################

def get_reward(state):
    all_rates = []
    for ue_id in UE.RATE:
        all_rates.append(UE.RATE[ue_id][state.col][state.row])
    return min(all_rates) # the paper seems using this

#############################################
## simulation entry point
#############################################

def main_loop(ai):
    global SHOW_RATE, FPS

    ## choose whether to run with animation
    sim = SimWorld(display=SHOW_ANIMATION)

    ## main game loop
    repeat_round = {"flag": False, "exploration": False}
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    if event.pos[1] < CELL.SIZE * ROWS:
                        SHOW_RATE = 1 if SHOW_RATE == 0 else 0
                    else:
                        sim.toggle_pause()
                elif event.button == 4:  # scroll up
                    FPS += 1
                elif event.button == 5:  # scroll down
                    FPS -= 1
                    if FPS < 1:
                        FPS = 1

        ## render the world
        sim.render_world()
        if sim.pause:
            sim.put_message("Pause...")
            sim.show()
            continue

        ## call AI agent
        state = State(sim.uav_pos.col, sim.uav_pos.row, STAT.step)
        reward = get_reward(state)
        action = ai.execute(state, reward)

        if isinstance(ai, DeepQLearning):
            ## DQN specific logic
            next_state = State(sim.uav_pos.col, sim.uav_pos.row, STAT.step + 1)
            sim.move_uav(action)
            STAT.step += 1
            STAT.reward += reward

            ## store transition in replay buffer
            ai.store_transition(state, action, reward, next_state, STAT.step == UAV.FLIGHT_TIME or sim.uav_pos == UAV.END_POS)

            ## end of round? either flight time is over or UAV has returned
            if STAT.step == UAV.FLIGHT_TIME or sim.uav_pos == UAV.END_POS:

                ## collect the final reward
                state = State(sim.uav_pos.col, sim.uav_pos.row, STAT.step)
                reward = get_reward(state)

                ## return too early? then apply penalty
                if sim.uav_pos == UAV.END_POS:
                    reward -= reward * (UAV.FLIGHT_TIME - STAT.step) # penalty

                ## not reaching the end position at the end of the flight time?
                if STAT.step == UAV.FLIGHT_TIME and not sim.uav_pos == UAV.END_POS:
                    reward -= reward * 10 # penalty

                ## store final transition in replay buffer
                next_state = State(sim.uav_pos.col, sim.uav_pos.row, STAT.step + 1)
                ai.store_transition(state, action, reward, next_state, True)

                ## sample a batch and train the network
                if len(ai.replay_buffer) > ai.batch_size:
                    print("Training")
                    ai.train()

                ## print reward outcome
                STAT.reward += reward
                is_print_reward = True
                if repeat_round["flag"]:
                    repeat_round["flag"] = False # done, turn it off
                    ai.is_exploration = repeat_round["exploration"] # restore the flag
                elif SAMPLE_MODE:
                    if STAT.round != 1: is_print_reward = False
                    if STAT.round % SAMPLE_INTERVAL == 0:
                        repeat_round["flag"] = True # turn on repeat round
                        repeat_round["exploration"] = ai.is_exploration # keep the flag
                        STAT.round -= 1           # step one round back to repeat same round
                        ai.is_exploration = False # repeat round with full exploitation
                if is_print_reward:
                    print(f"Round {STAT.round}: reward = {STAT.reward:.2f}, "
                          f"epsilon = {float(ai.epsilon):.4f}, "
                          f"flight time = {STAT.step}, "
                          f"returned = {'Yes' if sim.uav_pos == UAV.END_POS else 'No'}")

                ## start a new episode or round
                print(f"Round {STAT.round}: reward = {STAT.reward:.2f}, "
                    f"epsilon = {float(ai.epsilon):.4f}, "
                    f"flight time = {STAT.step}, "
                    f"returned = {'Yes' if sim.uav_pos == UAV.END_POS else 'No'}")
                STAT.round += 1
                STAT.reset()
                sim.reset()
            ## show simulation world
            sim.show()

        else:
            ## Q-learning, SARSA, and other similar algorithms logic
            sim.move_uav(action)
            STAT.step += 1
            STAT.reward += reward

            ## end of round? either flight time is over or UAV has returned
            if STAT.step == UAV.FLIGHT_TIME or sim.uav_pos == UAV.END_POS:

                ## collect the final reward
                state = State(sim.uav_pos.col, sim.uav_pos.row, STAT.step)
                reward = get_reward(state)

                ## return too early? then apply penalty
                if sim.uav_pos == UAV.END_POS:
                    reward -= reward * (UAV.FLIGHT_TIME - STAT.step) # penalty

                ## not reaching the end position at the end of the flight time?
                if STAT.step == UAV.FLIGHT_TIME and not sim.uav_pos == UAV.END_POS:
                    reward -= reward * 10 # penalty

                ## update the AI with the reward
                action = ai.execute(state, reward)

                ## print reward outcome
                STAT.reward += reward
                is_print_reward = True
                if repeat_round["flag"]:
                    repeat_round["flag"] = False # done, turn it off
                    ai.is_exploration = repeat_round["exploration"] # restore the flag
                elif SAMPLE_MODE:
                    if STAT.round != 1: is_print_reward = False
                    if STAT.round % SAMPLE_INTERVAL == 0:
                        repeat_round["flag"] = True # turn on repeat round
                        repeat_round["exploration"] = ai.is_exploration # keep the flag
                        STAT.round -= 1           # step one round back to repeat same round
                        ai.is_exploration = False # repeat round with full exploitation
                if is_print_reward:
                    print(f"Round {STAT.round}: reward = {STAT.reward:.2f}, "
                          f"epsilon = {float(ai.epsilon):.4f}, "
                          f"flight time = {STAT.step}, "
                          f"returned = {'Yes' if sim.uav_pos == UAV.END_POS else 'No'}")

                ## start a new episode or round
                STAT.round += 1
                STAT.reset()
                sim.reset()
            ## show simulation world
            sim.show()


#############################################
## main launcher
#############################################

## choose a strategy below
#ai = RandomMove()
#ai = Q_Learning(exploration=EXPLORATION)
#ai = SARSA(exploration=EXPLORATION)
ai = DeepQLearning(exploration=EXPLORATION)
print(f"Runnning {ai.name} algorithm...")
if not EXPLORATION: print("- NO exploration mode is in place")

## load existing data to resume progress?
if LOAD_DATA:
    print("- Load data requested")
    progress = ai.load_data()
    if progress!=-1:
        STAT.round = progress+1
        print(f"Restarting from round = {STAT.round}")
    else:
        print("- Failed to load data")

## run the simulation
try:
    main_loop(ai)
except KeyboardInterrupt:
    pass

## end of simulation
pygame.quit()

## save existing data for later use?
print(f"Stopping {ai.name} algorithm...")
if SAVE_DATA: 
    print("- Save data requested")
    if ai.save_data(STAT.round-1):
        print(f"- Progress saved, number of rounds = {STAT.round-1}")
    else:
        print("- Failed to save data")
