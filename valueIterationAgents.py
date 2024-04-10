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
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*****Your code here ****"
        
        states = self.mdp.getStates()

        # Initialize the priority queue
        priorityQueue = util.PriorityQueue()

       # intialize a dictionary to store predecessors for every state
        predecessors = {}
        for state in states:
            predecessors[state] = set()

        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for nextstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    # For each next state add the current state as a predecessor 
                    predecessors[nextstate].add(state)

        # For each state calculate the maximum Q value difference between each state's current value 
        # and the maximum Q value of all its actions to populate priority queue

        for state in states:
            if not self.mdp.isTerminal(state):
                maxAValue = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    actionvalue = self.computeQValueFromValues(state, action)
                    if maxAValue <= actionvalue:
                        maxAValue = actionvalue
                    diff = abs(self.values[state] - maxAValue)
                    priorityQueue.push(state, -diff)

        
        # Perform the required number of loops

        for iteration in range(self.iterations):
            # return if priority queue is empty
            if priorityQueue.isEmpty():
                return 
            # Pop the highest priority state off the queue
            highestPriorityState = priorityQueue.pop()
            if not self.mdp.isTerminal(highestPriorityState):
                maxActionV = float("-inf")
                actions = self.mdp.getPossibleActions(highestPriorityState)
                for action in actions:
                    actionvalue = self.computeQValueFromValues(highestPriorityState, action)
                    if maxActionV <= actionvalue:
                        maxActionV = actionvalue 
                self.values[highestPriorityState] = maxActionV
        
        # Update the priorities of predecessors by reevaluation Max Q value
        # and add them to the priority queue again

        for nextState in predecessors[highestPriorityState]:
            maxAValue = float("-inf")
            for pAction in self.mdp.getPossibleActions(nextState):
                pActionValue = self.computeQValueFromValues(nextState, pAction)
                if maxAValue <= pActionValue:
                    maxAValue = pActionValue
            # Re calculate the difference 
            difference = abs( self.getValue(nextstate) - maxAValue)
            if difference > self.theta:
                priorityQueue.update(nextState, -difference)
        
