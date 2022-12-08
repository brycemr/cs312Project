#!/usr/bin/python3
import random

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import copy
import random
import itertools




class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None

	def setupWithScenario(self, scenario):
		self._scenario = scenario

	# This runs in O(n) time and space complexity as it randomly generates a number permutation
	# of size n to determine the route.
	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(ncities)
			route = []
			# Now build the route using the random permutation
			for i in range(ncities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	# This algorithm runs in O(n^3) time as it finds the minimum edge
	# in an nxn matrix for n edges. It runs in O(n^2) space complexity
	# as it creates a nxn matrix.


	def greedy(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = True
		visitedCities = []
		count = 1
		start_time = time.time()
		route = []
		numCities = len(cities)
		edges = self.calcAllEdges(cities)

		costMatrix = np.empty((ncities, ncities))

		# Initialize Cost Matrix
		for i in range(ncities):
			for j in range(ncities):
				costMatrix[i, j] = cities[i].costTo(cities[j])
		bssf = None
		count = 0
		startIndex = 0
		startCity = cities[startIndex]

		# Find minimal edge length in cost matrix and add these cities to route
		start_cities = np.where(costMatrix == np.amin(costMatrix))

		# If there is one minimal edge the value returned is formatted differently
		if start_cities[0].size > 1:
			index = random.randint(0, len(start_cities) - 1)
			start_cities = start_cities[index]
		else:
			start_cities = [start_cities[0][0], start_cities[1][0]]

		visitedCities.append(start_cities[0])
		visitedCities.append(start_cities[1])
		route.append(cities[start_cities[0]])
		route.append(cities[start_cities[1]])
		costMatrix[start_cities[0], :] = np.inf
		costMatrix[:, start_cities[0]] = np.inf
		costMatrix[:, start_cities[1]] = np.inf
		costMatrix[start_cities[1], start_cities[0]] = np.inf
		current_city = start_cities[1]

		# Iterate through the graph, following the minimal edge that does not
		# go to a city that has already been visited
		for i in range(ncities - 2):
			# Find minimal edge from current city
			min = np.inf
			next_city = -1
			for j in range(len(costMatrix[current_city])):
				if costMatrix[current_city, j] < min and j not in visitedCities:
					next_city = j
					min = costMatrix[current_city, j]

			# If the required num edges have not been found and another cannot be added
			if next_city == -1:
				foundTour = False
				break
			elif costMatrix[current_city, next_city] == np.inf:
				# No available route out of this city
				foundTour = False
				break

			route.append(cities[next_city])
			costMatrix[:, next_city] = np.inf
			costMatrix[current_city, :] = np.inf
			costMatrix[next_city, current_city] = np.inf
			current_city = next_city

		bssf = TSPSolution(route)
		start_time = time.time()
		while startIndex < len(cities) - 1 and time.time() - start_time < time_allowance:
			route = []
			route.append(startCity)
			routeCost = 0

			startPoint = startCity
			currCity = None
			# keep going until we make a full cycle back to the starting city
			while (currCity != startCity):
				currCity = startPoint
				currEdges = edges[currCity]

				# find the smallest edge out of the current city
				minEdge = self.getMinEdge(currEdges, route, numCities)
				if minEdge == None:
					break

				# add the new city to the route and update vals
				route.append(minEdge[0])
				routeCost += minEdge[1]
				startPoint = minEdge[0]
				currCity = startPoint

			# if we've made it back to the starting city, check that we've made a full cycle and everything connects
			newBSSF = TSPSolution(route)
			if newBSSF.cost != np.inf and len(route) == numCities:
				count += 1
				if bssf is None:
					bssf = newBSSF
				elif newBSSF.cost < bssf.cost:
					bssf = newBSSF

			# try starting with a new start city and repeat
			startIndex += 1
			startCity = cities[startIndex]

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	def getMinEdge(self, edges, route: list, numCities):

		minEdge = (None, np.inf)

		# find the min edge to an unvisited city
		for currEdge in edges:
			if currEdge[1] < minEdge[1] and currEdge[0] not in route:
				minEdge = currEdge
		if minEdge[1] == np.inf:
			return None

		return minEdge

	def calcAllEdges(self, cities):
		# calculate edges between all cities
		edges = {}
		for startCity in cities:
			for endCity in cities:
				if startCity == endCity:
					continue
				cost = startCity.costTo(endCity)
				if cost != np.inf:
					if edges.get(startCity) is None:
						edges.update({startCity: []})
					edges[startCity].append((endCity, cost))
		return edges
	# This algorithm runs in O(n^2 b^n) on average, and has the same space complexity
	# b represents the average number of states put on to the queue with each expansion
	# it will expand n times, and for each state it must reduce the cost matrix in O(n^2) time.
	# and uses O(n^2) space.
	def branchAndBound(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		total_states = 1
		pruned_states = 0
		max_q_size = 0

		curr_time = 0

		costMatrix = np.empty((ncities, ncities))

		# Initialize cost matrix
		for i in range(ncities):
			for j in range(ncities):
				costMatrix[i, j] = cities[i].costTo(cities[j])

		initial_results = self.greedy(time_allowance)
		bssf = initial_results['soln']
		bestCost = initial_results['cost']

		for i in range(10*ncities):
			solution = self.greedy(time_allowance)
			if bestCost > solution['cost']:
				bssf = solution['soln']
				bestCost = solution['cost']

		if bestCost == np.inf:
			solution = self.defaultRandomTour(time_allowance)
			if bestCost > solution['cost']:
				bssf = solution['soln']
				bestCost = solution['cost']

		if bssf.cost < np.inf:
			foundTour = True

		# Reduce the initial cost matrix to create initial state
		costMatrix, initial_cost = self.reduceCostMatrix(costMatrix, -1, -1)

		# Create priority queue
		queue = []

		# Add Initial Cost Matrix and Bound to priority Queue
		start_time = time.time()
		startCity = 0
		depth = 1

		heapq.heappush(queue, (initial_cost/depth, initial_cost, -total_states, (costMatrix, startCity, []), depth))

		# Expand states that are part of priority queue until queue is empty or time passes allowed time
		while queue and curr_time < time_allowance:
			max_q_size = max(len(queue), max_q_size)

			currState = heapq.heappop(queue)

			currMatrix = currState[3][0]
			depth = currState[4] + 1
			curr_city = currState[3][1]
			curr_cost = currState[1]
			curr_time = time.time() - start_time
			# If the curr_cost < bestCost, this state is pruned from the priority queue
			if curr_cost < bestCost:
				# Create a child state that gets pushed onto priority queue if cost < bssf cost
				for i in range(len(currMatrix[curr_city, :])):


					if currMatrix[curr_city, i] != np.inf:
						# Create a new state and add it to the queue
						total_states += 1
						boundCost = curr_cost + currMatrix[curr_city, i]
						if boundCost < bestCost:
							newMatrix, additionalCost = self.reduceCostMatrix(currMatrix, curr_city, i)
							boundCost += additionalCost
							route = copy.deepcopy(currState[3][2])
							route.append(i)

						if len(route) < ncities:
							if boundCost < bestCost:
								heapq.heappush(queue, (boundCost/depth, boundCost, -total_states, (newMatrix, i, route), depth))
							else:
								pruned_states += 1
						else:
							# Check if bssf should be updated
							if boundCost < bestCost:
								cityRoute = []
								for city in route:
									cityRoute.append(cities[city])
								bssf = TSPSolution(cityRoute)
								bestCost = bssf.cost
								count += 1
			else:
				pruned_states += 1

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_q_size
		results['total'] = total_states
		results['pruned'] = pruned_states

		return results

	# This method has a total runtime of O(n^2) as it moves through each
	# cost in the nxn cost matrix to verify that there is a 0 in each row
	# and column, and reduce the matrix if not. It has a space complexity of
	# O(n^2) as it creates a copy of the nxn cost matrix to modify and return
	# as the reducedCostMatrix.
	def reduceCostMatrix(self, costMatrix, row, column):
		reducedCostMatrix = np.copy(costMatrix)

		# Update cost matrix with np.inf along column and row of current city
		if row != -1 and column != -1:
			reducedCostMatrix[row, :] = np.inf
			reducedCostMatrix[:, column] = np.inf
			reducedCostMatrix[column, row] = np.inf
		additionalCost = 0

		# Reduce the matrix so each row has a 0 in it
		for i in range(len(reducedCostMatrix)):
			if 0 in reducedCostMatrix[i, :]:
				pass
			else:
				min = np.amin(reducedCostMatrix[i, :])
				if min < np.inf:
					additionalCost += min
					for j in range(len(reducedCostMatrix[i, :])):
						reducedCostMatrix[i, j] -= min

		# Reduce the matrix so each column has a 0 in it
		for i in range(len(reducedCostMatrix[:, 0])):
			if 0 in reducedCostMatrix[:, i]:
				pass
			else:
				min = np.amin(reducedCostMatrix[:, i])
				if min < np.inf:
					additionalCost += min
					for j in range(len(reducedCostMatrix[:, i])):
						reducedCostMatrix[j, i] -= min

		return reducedCostMatrix, additionalCost

	def fancy(self, time_allowance=60.0):
		print("\n\n BEGINNING FANCY")
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		total_states = 1
		pruned_states = 0
		max_q_size = 0
		MAX_ITERATIONS_WITHOUT_SWAP = ncities

		curr_time = 0

		costMatrix = np.empty((ncities, ncities))

		# Initialize cost matrix
		for i in range(ncities):
			for j in range(ncities):
				costMatrix[i, j] = cities[i].costTo(cities[j])

		initial_results = self.greedy(time_allowance)
		bssf = initial_results['soln']
		bestCost = initial_results['cost']


		if bssf.cost < np.inf:
			foundTour = True

		start_time = time.time()

		route = bssf.route

		currBest = copy.deepcopy(bssf)

		numUnchangedSolutions = 0
		innerSwaps = 0

		# I added this outer while loop so that we can continue to loop within the time constraint and keep track of the current bssf
		# we will break out if we have had 10 solutions that don't help or we run out of time. We can change this number from 10
		while numUnchangedSolutions < 10 and time.time() - start_time < time_allowance:
			# After the first solution we are going to swap some random paths to mix it up. This should swap half of them
			numSwaps = 0

			if numUnchangedSolutions > 0:
				for i in range(math.ceil(len(route) / 3)): #MAYBE CHANGE TO WHILE LOOP? GO UNTIL WE HIT # SWAPS WE WANT
					randA = random.randrange(0, len(route))
					randB = random.randrange(0, len(route))
					# if swapping edges u->v and a->b put route index of node v and route index of node a into twoOptSwap
					if randA == 0:
						u = route[-1]._index
					else:
						u = route[randA-1]._index
					v = route[randA]._index
					a = route[randB]._index
					if randB == len(route)-1:
						b = route[0]._index
					else:
						b = route[randB+1]._index
					# Verify that an edge exists from u -> a, and from v->b
					if cities[u].costTo(cities[a]) != np.inf and cities[v].costTo(cities[b]) != np.inf and randA != randB:
						temp_route = self.twoOptSwap(route, randA, randB)
						#print("Swapped v:{} and a:{}".format(randA, randB))
						new_solution = TSPSolution(temp_route)
						if new_solution.cost != np.inf:
							route = temp_route
						numSwaps += 1

						if new_solution.cost < currBest.cost:
							print("Updating CurrBest above")
							currBest = new_solution
			print("Performed " + str(numSwaps) + " initial swaps")
			#self.printRoute(route)
			solution = TSPSolution(route)
			bssf = solution
			improvementFound = foundTour
			print("Initial Route: " + str(solution.cost))
			while improvementFound:
				improvementFound = False
				#self.printRoute(route)
				for i in range(len(cities)):
					# Iterate through all the edges
					u = route[i]._index
					#print("\nu: " + str(u), end="  ")
					if i >= len(route)-1:
						v = route[0]._index
						v_route_index = i
					else:
						v = route[i+1]._index
						v_route_index = i
					#print("v: " + str(v), end="  ")
					# I changed this from ncities to length of route
					for j in range(len(route)):
						# Iterate through all the edges
						a_route_index = j
						a = route[j]._index
						if a != u and a != v:
							#print("a: " + str(a), end="  ")
							if j >= len(route) - 1:
								b = route[0]._index
							else:
								b = route[j + 1]._index
							#print("b: " + str(b), end="  ")
							if b != u and b != v:

								# Verify that an edge exists from u -> a, backward from a -> v, and from v->b
								if cities[u].costTo(cities[a]) == np.inf or cities[v].costTo(cities[b]) == np.inf:
									break

								temp_route = self.twoOptSwap(route, v_route_index, a_route_index)
								new_solution = TSPSolution(temp_route)
								print("Swap Attempted, new cost:{} bssf cost:{}".format(new_solution.cost, bssf.cost))
								if new_solution.cost < bssf.cost:
									innerSwaps += 1
									#print("BSSF UPDATED, old BSSF COST: " + str(bssf.cost))
									#print("Old Route: ", end="")
									#self.printRoute(route)
									total_states += 1
									bssf = TSPSolution(temp_route)
									route = temp_route
									improvementFound = True
									if bssf.cost < currBest.cost:
										currBest = bssf
										print("Updating currBest")
									numIterationsWithoutSwap = 0
									#print("Swap Completed, new BSSF COST: " + str(bssf.cost))
									#print("New Route: ", end="")
									#self.printRoute(route)
									break
								elif new_solution.cost == bssf.cost:
									print("Placeholder")
					if improvementFound:
						break
			# if this solution is better let's use it
			bssf = TSPSolution(route)
			if bssf.cost < currBest.cost:
				print("Updating CurrBest below")
				currBest = bssf
				numUnchangedSolutions = 0
			else:
				# if it's not lets increment the amount of solutions we have found that aren't helping
				print("Performed " + str(innerSwaps) + " inner swaps")
				print("best route found: " + str(bssf.cost) + "\n")
				innerSwaps = 0
				numUnchangedSolutions += 1

		end_time = time.time()
		results['cost'] = currBest.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = currBest
		results['max'] = None
		results['total'] = total_states
		results['pruned'] = None

		return results

	def twoOptSwap(self, route, v, wo):
		#print("Swapping from " + str(v) + " to " + str(wo))
		newRoute = []
		# Add cities in route up to edges to be switched
		if v < wo:
			#print("Straight swap")
			for i in range(v):
				newRoute.append(route[i])

			# Add cities between edges to be swapped in reverse order, thus swapping edges
			for i in range(wo, v-1, -1):
				newRoute.append(route[i])

			# Add remaining cities in route
			for i in range(wo+1, len(route)):
				newRoute.append(route[i])
		else:
			#print("Wrap-around")
			#if wo < v then it wraps around the current route, start by appending from wo + 1 to v-1, then append wo to 0, then end to v
			for i in range(wo + 1, v):
				newRoute.append(route[i])

			for i in range(wo, -1, -1):
				newRoute.append(route[i])

			for i in range(len(route)-1, v-1, -1):
				newRoute.append(route[i])

		return newRoute

	def printRoute(self, route):
		for city in route:
			print(city._index, end=" ")
		print()

