# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0000001
    return answerDiscount, answerNoise

def question3a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    answerDiscount = .4
    answerNoise = 0
    # cannot do 0 because then it has no sense of urgency and stay put
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = -0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    answerDiscount = .9
    answerNoise = 0.3
    answerLivingReward = 0.02
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = 6
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = 0
    answerLearningRate = 0
    return 'Not Possible'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
