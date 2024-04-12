# qlearningAgents.py
# ------------------
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

#############
# Citations :
#https://www.youtube.com/watch?v=9g32v7bK3Co&ab_channel=StanfordOnline
#https://www.youtube.com/watch?v=HpaHTfY52RQ&ab_channel=StanfordOnline
#https://gibberblot.github.io/rl-notes/single-agent/value-iteration.html
#https://medium.com/@ngao7/markov-decision-process-value-iteration-2d161d50a6ff
##############

from game import *
from learningAgents import ReinforcementAgent


import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Creating a dictionary of state-action values 
        self.q_values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalaction = self.getLegalActions(state)
        # If no actions possible it means its a terminal state
        if not legalaction:
          return 0.0
        # If actions possible
        else: 
          maxV = -1000000
          for action in legalaction:
            if maxV < self.getQValue(state, action):
              maxV = self.getQValue(state, action)
          return maxV


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalaction = self.getLegalActions(state)
        # If no actions possible it means its a terminal state
        if not legalaction:
            return None
        else:
          maxQ = -1000000
          bestAction = None
          for action in legalaction:
            qValue = self.getQValue(state, action)
            if maxQ < qValue:
               maxQ = qValue
               bestAction = action
          return bestAction
      
            

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob = 0.1) to get a True value prob percentage of the times.
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        # If no actions possible it means its a terminal state
        if not legalActions:
            return None
        else:
          if not util.flipCoin(self.epsilon):
            return self.computeActionFromQValues(state) 
          else:
            return random.choice(legalActions)



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        """
          QLearning update algorithm:
          Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample
          ***sample = R(s,a,s') + gamma*max(Q(s',a'))***
        """

        currentQValue = self.getQValue(state, action)
        nextQValue = self.computeValueFromQValues(nextState)

        # Update Q-value
        self.q_values[(state, action)] = (1-self.alpha)*currentQValue + self.alpha*(reward + self.discount*nextQValue)


    def getPolicy(self, state):
        "Returns the policy at the state."
        "*** YOUR CODE HERE ***"
        return self.computeActionFromQValues(state)


    def getValue(self, state):
        return self.computeValueFromQValues(state)

