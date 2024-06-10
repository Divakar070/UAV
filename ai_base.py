'''
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

## define a base reinforcement learning class
## must implement `execute(state,reward)` method
class RL:
    def __init__(self, name="Base-Class"):
        ## declare all visible properties
        self.name = name
        self.is_exploration = True
        self.epsilon = 0.0

    def q(self, state):
        return [0.0]*Action.COUNT
    
    def load_data(self) -> int:
        return -1 # failed or NA

    def save_data(self, round) -> bool:
        return False # failed or NA

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
