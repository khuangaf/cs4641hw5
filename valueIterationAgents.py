# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
from copy import deepcopy
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    
    "*** YOUR CODE HERE ***"
    for i in range(iterations):
      previousValues = deepcopy(self.values)
      for currentState in mdp.getStates():
        # if mdp.getPossibleActions(currentState) == None:
        #   continue
        if mdp.isTerminal(currentState):
          continue
        maxValue = -1
        for action in mdp.getPossibleActions(currentState):
          nextValue = 0.0
          for nextState, prob in mdp.getTransitionStatesAndProbs(currentState, action):
            reward = mdp.getReward(currentState, action, nextState)

            nextValue += prob*(reward + discount * previousValues[nextState])
          if nextValue > maxValue:
            # print nextValue
            maxValue = nextValue
        self.values[currentState] = maxValue

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    qvalue = 0.0
    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
      reward = self.mdp.getReward(state, action, nextState)
      
      qvalue += prob * (reward + self.discount * self.getValue(nextState))
    return qvalue

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    # mdp = self.mdp
    maxValue = -9999999
    bestAction = None
    if self.mdp.isTerminal(state):
      return bestAction
    # for action in mdp.getPossibleActions(state):
    #   print action
    # print "---------------------"      
    for action in self.mdp.getPossibleActions(state):
      # print action, self.getQValue(state, action)
      # print "------"
      if self.getQValue(state, action) > maxValue:
        bestAction = action
        maxValue = self.getQValue(state, action)
    
    return bestAction

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
