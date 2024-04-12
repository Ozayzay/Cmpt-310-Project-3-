# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

#################
#test

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Get all states from the MDP 
        states = self.mdp.getStates()
        # initialize values to zero once 
        self.values = util.Counter()

        # Loop for the number of self.iterations
        for iteration in range(self.iterations):
            # creating a new dictionary of tempvalues with same keys as self.values
            tempvalues = self.values.copy()
            for state in states:
                if not self.mdp.isTerminal(state):
                    maxActionV = float("-inf")
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        actionvalue = self.computeQValueFromValues(state, action)
                        if maxActionV <= actionvalue:
                            maxActionV = actionvalue 
                    tempvalues[state] = maxActionV # Assigning the value for a given state as max value
            self.values = tempvalues # store the updated values 
    


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Get a list of next state and probability pairs i.e the states the agent could transition to
        # and their probabilities of transitioning 
        transitionstates = self.mdp.getTransitionStatesAndProbs(state, action)
        qvalue= 0

        for nextstate, prob in transitionstates:
            # self.mdp.getrewards returns the immediate reward for transition from current 
            # state to the next by taking action 
            # Self.values[next state] is the value of state we are transitioning into from
            # the previous iteration of the algorithm
            qvalue += prob * (self.mdp.getReward(state, action, nextstate) + self.discount * self.values[nextstate])

        return qvalue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        if self.mdp.isTerminal(state):
            return None
    
        maxQValue = float("-inf")
        bestAction = None

        for action in self.mdp.getPossibleActions(state):
            # Computer q-value for the current state and action 
            qValue = self.computeQValueFromValues(state, action)
            # Check if this q-value is greater than the current
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action
        return bestAction




    def getPolicy(self, state):
        "Returns the policy at the state."
        "*** YOUR CODE HERE ***"
        return self.computeActionFromValues


    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        priorityQueue = util.PriorityQueue()
        predecessors = {}

        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextstate in predecessors:
                            predecessors[nextstate].add(state)
                        else:
                            predecessors[nextstate] = {state}

        for state in states:
            if not self.mdp.isTerminal(state):
                maxAValue = -111111111
                for action in self.mdp.getPossibleActions(state):
                    actionvalue = self.computeQValueFromValues(state, action)
                    if maxAValue <= actionvalue:
                        maxAValue = actionvalue
                    diff = abs(self.values[state] - maxAValue)
                    priorityQueue.update(state, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                return 
            highestPriorityState = priorityQueue.pop()
            if not self.mdp.isTerminal(highestPriorityState):
                maxActionV = -1111111111
                actions = self.mdp.getPossibleActions(highestPriorityState)
                for action in actions:
                    actionvalue = self.getQValue(highestPriorityState, action)
                    if maxActionV <= actionvalue:
                        maxActionV = actionvalue 
                self.values[highestPriorityState] = maxActionV

            # This block of code should be inside the iterations loop
            for nextState in predecessors[highestPriorityState]:
                if not self.mdp.isTerminal(nextState):
                    maxAValue = -1111111111
                    for pAction in self.mdp.getPossibleActions(nextState):
                        pActionValue = self.getQValue(nextState, pAction)
                        if maxAValue <= pActionValue:
                            maxAValue = pActionValue
                    difference = abs( self.values[nextState] - maxAValue)
                    if difference > self.theta:
                        priorityQueue.update(nextState, -difference)