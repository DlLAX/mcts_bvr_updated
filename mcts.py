import numpy
import copy
import jax.numpy as jnp
import time
from collections import deque

class Node():
    def __init__(self, state, step, parent=None, parentAction=None, team = 0):
        self._state = state
        self._parent = parent
        self._children = []
        self._parentAction = parentAction
        self._nextStep = step

        self._team = team
        self._reward = 0
        self._visits = 0
        self._unexploredMoves = [0,1,2,3,4,5]

    def getReward(self):
        return self._reward

    def setState(self, state):
        self._state = state

    def setTeam(self, team):
        self._team = team

    def getVisits(self):
        return self._visits

    def isTerminalNode(self):
        return self._state.all_done == 1

    def getParentAction(self):
        return self._parentAction

    def getValidMoves(self):
        return [0,1,2,3,4,5] ##

    def fullyExpanded(self):
        return len(self._unexploredMoves) == 0

    def searchUCT(self):
        c = 1
        choices = []
        for child in self._children:
            r = child.getReward()
            v = child.getVisits()
            if v == 0:
                choices.append(float("inf"))
            else:
                choices.append((r / v) + c*(numpy.sqrt(numpy.log(self._visits + 1) / v)))
        return self._children[numpy.argmax(choices)]

    def expand(self):
        if not self.fullyExpanded():
            action = self._unexploredMoves.pop()
            childStep = self._nextStep
            nxtPlayer = self.getNextPlayer(self._state)
            if nxtPlayer == self._state.BLUE_plane.team:
                childState = childStep(self._state, jnp.array([action]), jnp.array([-1]))[0] ##
            else:
                childState = childStep(self._state, jnp.array([-1]), jnp.array([action]))[0]
            child = Node(childState, childStep, self, action, nxtPlayer)
            self._children.append(child)
            return child
        else:
            return None

    def isLeafNode(self):
        return len(self._children) == 0

    def backpropagate(self, reward, player):
        self._visits += 1
        if player == self._team:
            self._reward += reward
        else:
            self._reward -= reward
        if self._parent is not None:
            self._parent.backpropagate(reward, player) ##

    def simulate(self, player):
        node = self
        while not node.isTerminalNode(): ##
            if not node.isLeafNode():
                node = node.searchUCT()
            else:
                break
        reward = node.startRollout(player)
        node.backpropagate(reward, player)


    def startRollout(self, player):
        rolloutState = self._state ##
        rolloutStep = self._nextStep
        terminal = rolloutState.all_done == 1 ##
       # depth = 0
       # t0 = time.time()
        while not terminal:
            move = self.getValidMoves()[numpy.random.randint(len(self.getValidMoves()))] #Rollout policy ##
            nxtPlayer = self.getNextPlayer(rolloutState)
            if nxtPlayer == rolloutState.BLUE_plane.team: ## om blå
                rolloutState = rolloutStep(rolloutState, jnp.array([move]), jnp.array([-1]))[0] ##
            else: ## om röd
                rolloutState = rolloutStep(rolloutState, jnp.array([-1]), jnp.array([move]))[0] ##
            terminal = rolloutState.all_done == 1
            #depth += 1
        #print("Steps:", depth, "time:", time.time() - t0)
        if player == rolloutState.BLUE_plane.team:
            return rolloutState.result[0]
        else:
            return -rolloutState.result[0]

    def bestChildVisits(self):
        if len(self._children) == 0:
            return None
        return max(self._children, key=lambda x: x.getVisits())

    def bestChildReward(self):
        if len(self._children) == 0:
            return None
        return max(self._children, key=lambda x: x.getReward())

    def bestChildAction(self):
        if len(self._children) == 0:
            return None
        return self.bestChildVisits().getParentAction()

    def runSearch(self, iterations, player):
        #self.expandDepth(self, int(numpy.log(iterations)))
        t0 = time.time()
        self.expandIter(iterations)
        #print("expand time:", time.time() - t0, "for", iterations, "iterations")
        for iteration in range(iterations):
            self.simulate(player)
        #print("tree size:", self.treeSize())
        return self.bestChildAction()

    def promoteToRoot(self, action):
        for child in self._children:
            if child.getParentAction() == action:
                child._parent = None
                return child

    def getNextPlayer(self, state):
        next_time_planes = numpy.concatenate((state.time_to_next_action_blue_plane, state.time_to_next_action_red_plane), axis=0)
        if next_time_planes[0] <= next_time_planes[1]:
            return state.BLUE_plane.team
        else:
            return state.RED_plane.team

    def expandDepth(self, node, depth):
        if depth == 0 or node.isTerminalNode():
            return
        while not node.fullyExpanded():
            node.expand()
        for child in node._children:
            child.expandDepth(node, depth - 1)

    def treeSize(self):
        size = 0
        for child in self._children:
            size += child.treeSize()
        size += 1
        return size

    def expandIter(self, iterations):
        queue = deque([self])
        expansions = 0
        while queue and expansions < iterations:
            node = queue.popleft()
            if node.isTerminalNode():
                continue
            while not node.fullyExpanded() and expansions < iterations:
                node.expand()
                expansions += 1
            for child in node._children:
                queue.append(child)