# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    start = problem.getStartState()
    frontier = Stack()
    frontier.push((start, []))          # (state, path)
    visited = set()

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if state in visited:
            continue

        if problem.isGoalState(state):
            return path

        visited.add(state)

        for succ, action, cost in problem.getSuccessors(state):
            if succ not in visited:
                frontier.push((succ, path + [action]))

    return []    # no solution found


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue

    start = problem.getStartState()
    frontier = Queue()
    frontier.push((start, []))          # (state, path)
    visited = set([start])

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if problem.isGoalState(state):
            return path

        for succ, action, cost in problem.getSuccessors(state):
            if succ not in visited:
                visited.add(succ)
                frontier.push((succ, path + [action]))

    return []    # no solution found


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    #    PriorityQueue : ((state, actions), priority)
    frontier = util.PriorityQueue()
    expanded = set()
    
    # 起始節點放入 frontier
    startState = problem.getStartState()
    frontier.push((startState, []), 0 ) # ((狀態, 動作列表), 成本 )

    while not frontier.isEmpty():
        
        currentState, actions = frontier.pop()# 成本最低的
        
        if problem.isGoalState(currentState):
            return actions
            
        if currentState not in expanded:
            
            expanded.add(currentState)
            successors = problem.getSuccessors(currentState)
            
            # 後繼節點加入 frontier
            for nextState, action, stepCost in successors:
                newActions = actions + [action]

                newCost = problem.getCostOfActions(newActions)
                
                # 將 new.state加入優先佇列
                frontier.push( (nextState, newActions), newCost )

    # 若frontier 為空，代表找不到路徑
    return []


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    frontier = util.PriorityQueue()
    expanded = set()
    
    startState = problem.getStartState()
    startCost = 0
    
    startHeuristic = heuristic(startState, problem)
    
    # 優先級 = cost + h
    startPriority = startCost + startHeuristic
    
    frontier.push( (startState, [], startCost), startPriority ) # ( , 動作列表, )

    while not frontier.isEmpty():
        
        currentState, actions, currentCost_g = frontier.pop() # 最低節點
        
        if problem.isGoalState(currentState):
            return actions
            
        if currentState not in expanded:
            
            expanded.add(currentState)
            
            successors = problem.getSuccessors(currentState)
            
            for nextState, action, stepCost in successors:
                
                newActions = actions + [action]
                newCost_g = currentCost_g + stepCost
                
                newHeuristic_h = heuristic(nextState, problem)
                
                newPriority_f = newCost_g + newHeuristic_h
                
                frontier.push( (nextState, newActions, newCost_g), newPriority_f )

    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

'''

python pacman.py -l bigSearch -p ClosestDotSearchAgent

'''