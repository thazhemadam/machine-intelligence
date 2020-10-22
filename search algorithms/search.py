import queue as Q

def DFS_Traversal(cost, start_point, goals):
	visited = set()
	stack = [(start_point, [start_point])]
	while stack:
		(node, path) = stack.pop()
		if node not in visited:
			if node in goals:
				return path
			visited.add(node)
			for i in range(len(cost[node])-1, 0, -1):
				if(cost[node][i] > 0 and i not in visited):
					stack.append((i, path + [i]))
	return []


def UCS_Traversal(cost, start_point, goals):
	frontier = Q.PriorityQueue()
	initial_cost = 0
	frontier.put((initial_cost, start_point, [start_point]))
	path = []
	visited = set()
	while not frontier.empty():
		initial_cost, node, path = frontier.get()
		if node in goals:
			return path
		visited.add(node)
		for i in range(1, len(cost[node])):
			if cost[node][i] > 0 and i not in visited:
				cost_1 = initial_cost + cost[node][i]
				new_path = path.copy()
				new_path.append(i)
				frontier.put((cost_1, i, new_path))
	return []

def A_star_Traversal(cost, heuristic, start_point, goals):
	frontier = Q.PriorityQueue()
	initial_cost = heuristic[start_point]
	frontier.put((initial_cost, start_point, [start_point]))
	path = []
	visited = set()
	while not frontier.empty():
		initial_cost, node, path = frontier.get()
		if node in goals:
			return path
		visited.add(node)
		for i in range(1, len(cost[node])):
			if cost[node][i] > 0 and i not in visited:
				cost_1 = initial_cost + cost[node][i] + \
					heuristic[i] - heuristic[node]
				new_path = path.copy()
				new_path.append(i)
				frontier.put((cost_1, i, new_path))
	return []



def tri_traversal(cost, heuristic, start_point, goals):
	l = []

	t1 = DFS_Traversal(cost, start_point, goals)
	t2 = UCS_Traversal(cost, start_point, goals)
	t3 = A_star_Traversal(cost, heuristic, start_point, goals)

	l.append(t1)
	l.append(t2)
	l.append(t3)
	return l
