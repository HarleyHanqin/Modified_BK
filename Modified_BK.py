import numpy as np
import sys
import random
import math
import copy
import time

C1, C2, C3, C4 = 2, 1, 1, 2

# Initialise the Node
class Node:
    def __init__(self, data, value):
        self.item = set()
        if data != None:
            self.item.add(data)
        self.next = None
        self.prev = None
        self.value = value

    def add(self, data):
        self.item.add(data)

    def remove(self, data):
        self.item.remove(data)

    def pop(self):
        return self.item.pop()

    def empty(self):
        return self.item == set()

# Class for doubly Linked List
class doublyLinkedList:
    def __init__(self):
        self.start_node = None
        self.last_node = None


    def InsertToStart(self, data, value):
        if self.last_node is None:
            new_node = Node(data, value)
            self.start_node = new_node
            self.last_node = new_node
            return
        n = self.start_node
        new_node = Node(data, value)
        n.prev = new_node
        new_node.next = n
        self.start_node = new_node

    # Insert element at the end
    def InsertToEnd(self, data, value):
        # Check if the list is empty
        if self.start_node is None:
            new_node = Node(data, value)
            self.start_node = new_node
            self.last_node = new_node
            return
        n = self.last_node

        new_node = Node(data, value)
        n.next = new_node
        new_node.prev = n
        self.last_node = new_node

    # Insert a new node right after the given node, return the new node
    def InsertInMiddle(self, node, data, value):
        new_node = Node(data, value)
        next_node = node.next
        node.next = new_node
        new_node.prev = node
        new_node.next = next_node
        next_node.prev = new_node
        return new_node

    # Delete the elements from the start
    def DeleteAtStart(self):
        if self.start_node is None:
            print("The Linked list is empty, no element to delete")
            return
        if self.start_node.next is None:
            self.start_node = None
            self.last_node = None

            return
        n = self.start_node.next
        n.prev = None
        self.start_node = n


    # Delete the elements from the end
    def DeleteAtEnd(self):
        # Check if the List is empty
        if self.start_node is None:
            print("The Linked list is empty, no element to delete")
            return
        if self.start_node.next is None:
            self.start_node = None
            self.last_node = None
            return
        n = self.last_node.prev
        n.next = None
        self.last_node = n

    def Delete(self, node):
        if node.next == None:
            self.DeleteAtEnd()
            return
        if node.prev == None:
            self.DeleteAtStart()
            return
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    # Traversing and Displaying each element of the list
    def Display(self):
        if self.start_node is None:
            print("The list is empty")
            return
        else:
            n = self.start_node
            while n is not None:
                print("Element is: ", n.item, "Value: ", n.value)
                n = n.next
        print("\n")



class Graph:
    def __init__(self, vertices, edges):
        self.V = vertices
        self.E = edges

    def copy(self):
        return Graph(copy.deepcopy(self.V), copy.deepcopy(self.E))

#1
V = {i + 1 for i in range(13)}
E = dict()
E[1] = {(1,2):1, (1,3):1, (1,5):1, (1,7):1, (1,8):1, (1,9):1, (1,10):1, (1,11):1}
E[2] = {(2,1):1, (2,3):1, (2,4):1, (2,9):1, (2,12):1}
E[3] = {(3,1):1, (3,2):1, (3,4):1, (3,5):1}
E[4] = {(4,2):1, (4,3):1, (4,5):1, (4,6):1}
E[5] = {(5,1):1, (5,3):1, (5,4):1, (5,6):1, (5,7):1}
E[6] = {(6,4):1, (6,5):1, (6,7):1, (6,8):1}
E[7] = {(7,1):1, (7,5):1, (7,6):1, (7,8):1}
E[8] = {(8,1):1, (8,6):1, (8,7):1, (8,9):1, (8,10):1, (8,11):1, (8,13):1}
E[9] = {(9,1):1, (9,2):1, (9,8):1, (9,10):1, (9,11):1, (9,12):1}
E[10] = {(10,1):1, (10,8):1, (10,9):1, (10,11):1, (10,12):1, (10,13):1}
E[11] = {(11,1):1, (11,8):1, (11,9):1, (11,10):1, (11,12):1, (11,13):1}
E[12] = {(12,2):1, (12,9):1, (12,10):1, (12,11):1, (12,13):1}
E[13] = {(13,8):1, (13,10):1, (13,11):1, (13,12):1}


G1 = Graph(V, E)


V = {i for i in range(2)}
E = dict()
E[0] = {(0,1):2}
E[1] = {(1,0):2}
G2 = Graph(V, E)


V = {i + 1 for i in range(8)}
E = dict()
E[1] = {(1,2):2, (1,3):1, (1,4):1, (1,7):1, (1,8):1}
E[2] = {(2,1):2, (2,3):3, (2,8):1}
E[3] = {(3,1):1, (3,2):3, (3,4):3}
E[4] = {(4,1):1, (4,3):3, (4,5):2, (4,7):1}
E[5] = {(5,4):2, (5,6):4, (5,7):2}
E[6] = {(6,5):4, (6,8):2}
E[7] = {(7,1):1, (7,4):1, (7,5):2, (7,8):4}
E[8] = {(8,1):1, (8,2):1, (8,6):2, (8,7):4}

G3 = Graph(V, E)

n = 100
V = {i for i in range(n)}
E = dict()
for i in range(n):
    E[i] = dict()
    for j in range(n):
        if i == j:
            continue
        E[i][(i, j)] = 1
complete_graph = Graph(V, E)


# input:
# output:
def Forest(G):
    new_graph = G.copy()
    vtxs = new_graph.V
    edges = new_graph.E
    E = dict()

    #scanned_edges = set()

    buckets = doublyLinkedList()
    r_bucket_map = dict()
    vertex_r_map = dict()

    # Insert the element to empty list
    buckets.InsertToEnd(None, 0)
    for i in vtxs:
        buckets.start_node.add(i)
        vertex_r_map[i] = 0
        r_bucket_map[0] = buckets.start_node


    while buckets.last_node:
        #print("r(v): ", buckets.last_node.value)
        x_key = buckets.last_node.pop()
        buckets.last_node.add(x_key)
        while edges[x_key]:
            (x, y) = next(iter(edges[x_key].keys()))
            if vertex_r_map[y] + 1 not in E:
                E[vertex_r_map[y] + 1] = set()
            if x < y:
                E[vertex_r_map[y] + 1].add((x, y))
            else:
                E[vertex_r_map[y] + 1].add((y, x))
            if vertex_r_map[x] == vertex_r_map[y]:
                vertex_r_map[x] += 1
                # update x in r_bucket_map
                x_old_bucket = r_bucket_map[vertex_r_map[x] - 1]

                if x_old_bucket.next == None:
                    buckets.InsertToEnd(x, vertex_r_map[x])
                elif x_old_bucket.next.value > vertex_r_map[x]:
                    buckets.InsertInMiddle(x_old_bucket, x, vertex_r_map[x])
                else:
                    x_old_bucket.next.add(x)
                r_bucket_map[vertex_r_map[x]] = x_old_bucket.next
                x_old_bucket.remove(x)
                if x_old_bucket.empty():
                      buckets.Delete(x_old_bucket)


            vertex_r_map[y] += 1
            edges[x][(x, y)] -= 1
            edges[y][(y, x)] -= 1
            if edges[x][(x, y)] == 0:
                edges[x].pop((x, y))
            if edges[y][(y, x)] == 0:
                edges[y].pop((y, x))
            # update y in r_bucket_map
            y_old_bucket = r_bucket_map[vertex_r_map[y] - 1]


            if y_old_bucket.next == None:
                buckets.InsertToEnd(y, vertex_r_map[y])
            elif y_old_bucket.next.value > vertex_r_map[y]:
                buckets.InsertInMiddle(y_old_bucket, y, vertex_r_map[y])
            else:

                y_old_bucket.next.add(y)
            r_bucket_map[vertex_r_map[y]] = y_old_bucket.next

            y_old_bucket.remove(y)
            if y_old_bucket.empty():
                buckets.Delete(y_old_bucket)



        # update x in r_bucket_map and vertex_r_map
        x_bucket = r_bucket_map[vertex_r_map[x_key]]

        x_bucket.remove(x_key)



        if x_bucket.empty():
            buckets.Delete(x_bucket)
            r_bucket_map.pop(vertex_r_map[x_key])

        vertex_r_map.pop(x_key)
        #print(vertex_r_map)

        #print(x_bucket.value)
        #buckets.Display()


        #print("\n\n")
    return E


G1_test = G1.copy()
output = Forest(G1_test)
assert(output[1] == {(2, 4), (1, 2), (1, 5), (1, 11), (1, 8), (4, 6), (8, 13), (1, 7), (2, 12), (1, 10), (1, 3), (1, 9)})
assert(output[2] == {(9, 10), (3, 4), (9, 12), (5, 7), (10, 13), (2, 3), (2, 9), (8, 9), (5, 6), (9, 11), (3, 5)})
assert(output[3] == {(11, 13), (6, 8), (8, 10), (6, 7), (4, 5), (10, 12), (8, 11)})
assert(output[4] == {(7, 8), (10, 11), (12, 13), (11, 12)})



def Certificate(G, k):
    #print(G.E)
    forest_output = Forest(G)
    edges = dict()
    i = 1
    while i <= k:
        if i in forest_output:
            for (x, y) in forest_output[i]:
                if (x, y) in edges:
                    edges[(x, y)] += 1
                else:
                    edges[(x, y)] = 1
        i += 1
    return edges

G1_test = G1.copy()

assert(Certificate(G1_test, 4) == {(2, 4):1, (1, 2):1, (1, 5):1, (1, 11):1, (1, 8):1, (4, 6):1, (8, 13):1, (1, 7):1, (2, 12):1, (1, 10):1, (1, 3):1, (1, 9):1, (9, 10):1, (3, 4):1, (9, 12):1, (5, 7):1, (10, 13):1, (2, 3):1, (2, 9):1, (8, 9):1, (5, 6):1, (9, 11):1, (3, 5):1, (11, 13):1, (6, 8):1, (8, 10):1, (6, 7):1, (4, 5):1, (10, 12):1, (8, 11):1, (7, 8):1, (10, 11):1, (12, 13):1, (11, 12):1})


# input: An n-vertex m-edge graph G
# output: a set containing all the k-weak edges of G
def WeakEdgesCertificate(graph, k):
    G = graph.copy()
    n = len(G.V)
    iter = int(math.log2(n))
    edge_dict = dict()
    for i in range(iter):
        new_dict = Certificate(G, C2 * k)
        for (x, y) in new_dict:
            if (x, y) in edge_dict:
                edge_dict[(x, y)] += new_dict[(x, y)]
            else:
                edge_dict[(x, y)] = new_dict[(x, y)]
            if G.E[x][(x, y)] > new_dict[(x, y)]:
                G.E[x][(x, y)] -= new_dict[(x, y)]
                G.E[y][(y, x)] -= new_dict[(x, y)]
            else:
                G.E[x].pop((x, y))
                G.E[y].pop((y, x))
        #print(edge_dict)
    return edge_dict


G1_test = G1.copy()

def contract(G, edge):
    (x, y) = edge
    G.E[x].pop((x, y))
    G.E[y].pop((y, x))
    while G.E[y] != dict():
        ((y, node), weight) = G.E[y].popitem()
        if (x, node) in G.E[x]:
            G.E[x][(x, node)] += weight
            G.E[node][(node, x)] += weight
        else:
            G.E[x][(x, node)] = weight
            G.E[node][(node, x)] = weight
        G.E[node].pop((node, y))


# input: An n-vertex m-edge graph G
# output: low connectivity edges
def Partition(graph, k):
    #print("Partition_K:", k)
    G = graph.copy()
    contracted_map = {i : [i] for i in G.V}
    contract_to_map = {i : i for i in G.V}
    allEdges = dict()
    #count = 0
    total_time = 0

    while True:
        n = len(G.V)
        m = 0
        allEdges = dict()
        for vertex in G.V:
            for (x, y) in G.E[vertex].keys():
              if x < y:
                allEdges[(x, y)] = G.E[x][(x, y)]
                m += G.E[x][(x, y)]
        #print(m, k, n)
        if m <= C1 * k * (n - 1) or n <= 2:
            #print("????")
            break
        certEdges = Certificate(G, k)
        #print(certEdges)
        for (x, y) in certEdges.keys():
            if allEdges[(x, y)] == certEdges[(x, y)]:
                allEdges.pop((x, y))
            else:
                allEdges[(x, y)] -= certEdges[(x, y)]

        while len(allEdges.keys()) > 0:
            (x, y) = allEdges.popitem()[0]
            if x not in G.V or y not in G.V:
                continue
            contract(G, (x, y))
            #cur_time = time.time()

            G.E.pop(y)
            G.V.remove(y)
            if G.E[x] == dict():
                G.E.pop(x)
                G.V.remove(x)
            contracted_map[x] += contracted_map[y]
            for vertex in contracted_map[y]:
                contract_to_map[vertex] = x
            contracted_map[y] = list()

            # t = time.time() - cur_time
            # total_time += t
            # print(total_time)

    original_edges = dict()

    for x in graph.E:
        for (x, y) in graph.E[x]:
            a = contract_to_map[x]
            b = contract_to_map[y]
            if a >= b:
                continue
            if (a, b) in G.E[a].keys():
                if x < y:
                    original_edges[(x, y)] = graph.E[x][(x, y)]
                else:
                    original_edges[(y, x)] = graph.E[x][(x, y)]

    return original_edges


# input: An n-vertex m-edge graph G
# output: a set containing all the k-weak edges of G
def WeakEdgesPartition(graph, k):
    G = graph.copy()
    n = len(G.V)
    iter = int(math.log2(n))
    edge_dict = dict()
    for i in range(iter):
        new_dict = Partition(G, C2 * k)
        for (x, y) in new_dict:
            if (x, y) in edge_dict:
                edge_dict[(x, y)] += new_dict[(x, y)]
            else:
                edge_dict[(x, y)] = new_dict[(x, y)]
            if G.E[x][(x, y)] > new_dict[(x, y)]:
                G.E[x][(x, y)] -= new_dict[(x, y)]
                G.E[y][(y, x)] -= new_dict[(x, y)]
            else:
                G.E[x].pop((x, y))
                G.E[y].pop((y, x))
        #print(edge_dict)
    return edge_dict


G1_test = G1.copy()
assert(WeakEdgesPartition(G1_test, 1) == {(1, 2): 1, (1, 3): 1, (1, 5): 1, (1, 7): 1, (1, 8): 1, (1, 9): 1, (1, 10): 1, (1, 11): 1, (2, 3): 1, (2, 4): 1, (2, 9): 1, (2, 12): 1, (3, 4): 1, (3, 5): 1, (4, 5): 1, (4, 6): 1, (5, 6): 1, (5, 7): 1, (6, 7): 1, (6, 8): 1, (7, 8): 1, (8, 9): 1, (8, 10): 1, (8, 11): 1, (8, 13): 1, (9, 10): 1, (9, 11): 1, (9, 12): 1, (10, 11): 1, (10, 12): 1, (10, 13): 1, (11, 12): 1, (11, 13): 1, (12, 13): 1})


def Estimation(subG, k, edge_c_map):
    print(k)
    H = subG.copy()
    weak_edges = WeakEdgesPartition(H, C3 * k)
    #print(weak_edges)
    #print(weak_edges)
    for (x, y) in weak_edges.keys():
        edge_c_map[(x, y)] = k
        H.E[x][(x, y)] -= weak_edges[(x, y)]
        H.E[y][(y, x)] -= weak_edges[(x, y)]
        if H.E[x][(x, y)] <= 0:
            H.E[x].pop((x, y))
            H.E[y].pop((y, x))
            if H.E[x] == dict():
                H.E.pop(x)
                H.V.remove(x)
            if H.E[y] == dict():
                H.E.pop(y)
                H.V.remove(y)
    #print(H.E)
    isChosen = set()
    for vertex in H.V:
        if vertex in isChosen:
            continue
        connected_comp = set()
        connected_comp.add(vertex)
        queue = set()
        queue.add(vertex)
        isChosen.add(vertex)
        while queue != set():
            x = queue.pop()
            for (x, y) in H.E[x].keys():
                if y not in connected_comp:
                    queue.add(y)
                    connected_comp.add(y)
                    isChosen.add(y)
        #print(connected_comp)
        V = set()
        E = dict()
        for v in connected_comp:
            V.add(v)
            E[v] = H.E[v]
        Estimation(Graph(V, E), 2 * k, edge_c_map)

def load_data(path="facebook_combined.txt"):
    V = set()
    E = dict()
    with open(path) as file:
        for line in file:
            l = line.rstrip().split(" ")
            v1, v2 = int(l[0]), int(l[1])
            if v1 not in V:
                V.add(v1)
            if v2 not in V:
                V.add(v2)
            if v1 not in E:
                E[v1] = dict()
            if v2 not in E:
                E[v2] = dict()
            if (v1, v2) not in E[v1]:
                E[v1][(v1, v2)] = 0
            E[v1][(v1, v2)] += 1
            if (v2, v1) not in E[v2]:
                E[v2][(v2, v1)] = 0
            E[v2][(v2, v1)] += 1
            #print(v1, v2)
    #print(E)
    return Graph(V, E)


def test_estimation():
    edge_c_map = dict()
    G = load_data()
    Estimation(G, 1, edge_c_map)
    print(edge_c_map)

edge_c_map = dict()
def test_complete_graph(G):
    edge_c_map = dict()
    Estimation(G, 1, edge_c_map)
    print(edge_c_map)


cnt = 0
def random_sample(G, edge_c_map):
    count = 0
    n = len(G.V)
    rho = math.log(n) / C4
    print('rho:', rho)
    V = set()
    E = dict()
    for x in G.V:
        for (x, y) in G.E[x]:
            if x > y:
                continue
            prob = min(1, rho / edge_c_map[(x, y)])
            #print(rho, edge_c_map[(x, y)], p)
            if random_edge(prob):
                if x not in E:
                    E[x] = dict()
                if y not in E:
                    E[y] = dict()
                E[x][(x, y)] = 1 / prob
                E[y][(y, x)] = 1 / prob
                count += 1
            if x not in V:
                V.add(x)
            if y not in V:
                V.add(y)
    print("Edge count:", count)
    return (Graph(V, E), count)

def random_edge(prob):
    num = random.random()
    return num < prob
  
original_time = []
compressed_time = []

def find_cut(G, lower, upper):
    cut_size = 0
    cur = time.time()
    for x in range(lower, upper):
        if x not in G.E or G.E[x] == dict():
            continue
        for (x, y) in G.E[x]:
            if y < lower or y >= upper:
                cut_size += G.E[x][(x, y)]
    return (cut_size, time.time() - cur)

def graph_number(height, tick_label):
    import matplotlib.pyplot as plt

    plt.clf()

    # x-axis values
    x = [i + 1 for i in range(len(height))]

    # plotting a bar chart
    plt.bar(x, height, tick_label = tick_label,
         width = 0.8)

    # x-axis label
    plt.xlabel('x - axis, value of C1, C2, C3, C4')
    # frequency label
    plt.ylabel('y - axis, number of sampled edges')
    # plot title
    plt.title('Average number of sampled edges on some graph')
    plt.xticks(fontsize=8)
    # showing legend
    plt.legend()

    # function to show the plot
    plt.show()
    #path = 'Figures\FigureNumber.png'
    #plt.savefig(path)



def graph_cuts(epsilons):

    import matplotlib.pyplot as plt

    plt.clf()
    
    # frequencies
    #epsilons = [0.2,0.5,0.70,0.40,0.30,0.45,0.50,0.45,0.43,0.40,0.44,0.60,0.7,0.13,0.57,0.18,0.90,0.77,0.32,0.21,0.20,0.40,-0.57,-0.18,-0.90,-0.77,-0.32,-0.21,-0.20,-0.40]

    # setting the ranges and no. of intervals
    range = (-0.21, 0.21)
    bins = 21

    color = ''
    if C4 == 1:
        color = 'green'
    if C4 == 2:
        color = 'yellow'
    if C4 == 4:
        color = 'blue'
    if C4 == 8:
        color = 'purple'
    if C4 == 16:
        color = 'red'

    # plotting a histogram
    plt.hist(epsilons, bins, range, color = color, histtype = 'bar', rwidth = 0.8)
    plt.ylim([0, 5000])
    # x-axis label
    plt.xlabel('epsilon')
    # frequency label
    plt.ylabel('No. of epsilons lie in this range')
    # plot title
    plt.title('epsilon distribution when C1, C2, C3, C4 = ' + str(C1) + ',' + str(C2) + ',' + str(C3) + ',' + str(C4))

    # function to show the plot
    #plt.show()
    path = 'Figures\Figure' + str(C1) + str(C2) + str(C3) + str(C4) + '.png'
    print(path)
    plt.savefig(path)


#height = [39602.0, 42470.85, 32887.950000000004, 49184.899999999994, 32896.65, 49176.2, 77524.25, 57419.375, 36865.5, 21812.875, 11560.125]
#tick_label = ['C1=1', 'C1=2', 'C2=1', 'C2=2', 'C3=1', 'C3=2', 'C4=2', 'C4=4', 'C4=8', 'C4=16', 'C4=32']
#graph_number(height, tick_label)

def graph_epsilons(x, y):
    import matplotlib.pyplot as plt

    plt.clf()

    # plotting a bar chart
    plt.scatter(x, y)

    # x-axis label
    plt.xlabel('x - axis, random sampling probability p')
    # frequency label
    plt.ylabel('y - axis, epsilon ^ 2 * m')
    # plot title
    plt.title('Correlation between number of edges and strong connectivity')
    # showing legend
    # function to show the plot
    #plt.show()
    path = 'Figures\FigureEpsilons.png'
    plt.savefig(path)

    
def random_graph(n, p):
    V = {i for i in range(n)}
    E = dict()
    E[0] = dict()
    for i in range(n - 1):
        E[i][(i, i + 1)] = 1
        E[i + 1] = dict()
        E[i + 1][(i + 1, i)] = 1
    m = n - 1
    for i in range(n):
        for j in range(i + 2, n):
            if random.random() < p:
                E[i][(i, j)] = 1
                E[j][(j, i)] = 1
                m += 1
        
    G = Graph(V, E)
    return (G, m)


def graph_sides(eps):
    import matplotlib.pyplot as plt
    plt.clf()
    bins = np.linspace(-0.21, 0.21, 21)

    plt.hist([eps[1], eps[3]], bins, label=['C4 = 2', 'C4 = 8'])
    plt.legend(loc='upper right')
    #plt.show()
    path = 'Figures\FigureSides' + str(C1) + str(C2) + str(C3) + '.png'
    print(path)
    plt.savefig(path)

def graph_time(x, y):
    import matplotlib.pyplot as plt
    plt.clf()

    bins = np.arange(len(x))
    # plotting a bar chart
    plt.bar(bins - 0.2, x, tick_label = ['C1=1', 'C1=2', 'C2=1', 'C2=2', 'C3=1', 'C3=2', 'C4=1', 'C4=2', 'C4=4', 'C4=8'], label = 'Original', width = 0.4)
    plt.bar(bins + 0.2, y, tick_label = ['C1=1', 'C1=2', 'C2=1', 'C2=2', 'C3=1', 'C3=2', 'C4=1', 'C4=2', 'C4=4', 'C4=8'], label = 'Compressed', width = 0.4)

    # x-axis label
    plt.xlabel('x - axis, value of C1, C2, C3, C4')
    # frequency label
    plt.ylabel('y - axis, running time for finding 5000 cuts (sec)')
    # plot title
    plt.title('Running time for finding cuts')
    plt.xticks(fontsize=8)
    # showing legend
    plt.legend()

    # function to show the plot
    plt.show()
    #path = 'Figures\FigureNumber.png'
    #plt.savefig(path)


edges = [5884.9375, 11640.0, 5912.8125, 11612.125, 5883.75, 11641.1875, 18535.75, 9360.625, 4741.375, 2412.125]
tick_label = ['C1=1', 'C1=2', 'C2=1', 'C2=2', 'C3=1', 'C3=2', 'C4=1', 'C4=2', 'C4=4', 'C4=8']
graph_number(edges, tick_label)
original_time = [40.68678426742554, 47.349268436431885, 46.02222800254822, 46.84225535392761, 49.04663109779358, 46.3486807346344, 50.29137897491455, 46.3716995716095, 45.49734830856323, 46.73969912528992, 45.329935789108276, 46.20896124839783, 47.57650017738342, 47.28434634208679, 45.51829934120178, 47.608367919921875, 41.60974049568176, 47.23435139656067, 66.01438784599304, 46.396673917770386, 45.09207725524902, 45.450703620910645, 45.63970875740051, 42.2633376121521, 46.01473259925842, 24.76508402824402, 25.51754856109619, 24.99923038482666, 33.16427993774414, 26.36267852783203, 25.7923481464386, 26.55401349067688]
compressed_time = [7.11810564994812, 5.211449384689331, 3.417501449584961, 2.2415361404418945, 15.560708284378052, 8.35545563697815, 5.41646933555603, 3.304965019226074, 14.352323532104492, 8.435422420501709, 5.088474750518799, 3.2014808654785156, 27.65720772743225, 14.509180545806885, 8.399126529693604, 5.350762367248535, 13.677829027175903, 9.02605414390564, 7.5663533210754395, 3.350040912628174, 26.833964824676514, 14.040723085403442, 8.422135591506958, 4.7569191455841064, 27.435237646102905, 9.451233386993408, 5.833110570907593, 3.3516180515289307, 39.925588607788086, 18.265429973602295, 10.371063709259033, 5.946408033370972]
x = [[], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]
y = [[], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]
i = 0
for T1 in range(1, 3):
    for T2 in range(1, 3):
        for T3 in range(1, 3):
            for T4 in range(4):
                count = original_time[i]
                x[1][T1] += count / 20
                x[2][T2] += count / 20
                x[3][T3] += count / 20
                x[4][T4] += count / 8
                count = compressed_time[i]
                y[1][T1] += count / 20
                y[2][T2] += count / 20
                y[3][T3] += count / 20
                y[4][T4] += count / 8
                i += 1

a = x[1][1:] + x[2][1:] + x[3][1:] + x[4][:4]
b = y[1][1:] + y[2][1:] + y[3][1:] + y[4][:4]

#a = [80.39922394752502, 98.43159370422363, 89.07053347826005, 89.76028417348863, 89.50010575056076, 89.33071190118791, 87.58295068144798, 74.34048491716385, 97.7900339961052, 92.6037386059761, 94.75983592867851]
#b = [34.9451346039772, 42.10978373289108, 31.68225481510162, 45.37266352176666, 31.095222020149226, 45.959696316719054, 73.74349236488342, 44.7913621366024, 36.82239109277725, 23.356272488832474, 13.923777759075165]
print(a, b)
graph_time(a, b)
#[3.753683090209961, 4.601454257965088, 5.951353311538696, 5.748924016952515, 6.482445478439331, 6.886459112167358, 5.554856777191162, 6.6592912673950195, 7.033122777938843, 7.5204126834869385, 5.237765789031982, 5.991740942001343, 6.802107810974121, 14.909975051879883, 11.793938398361206, 12.807167053222656, 11.975993156433105, 12.430103302001953, 11.272271871566772, 11.811980724334717, 12.59028697013855, 13.3734712600708, 10.940871477127075, 12.849722146987915, 14.057748079299927, 14.277053594589233, 13.149798154830933, 14.146259069442749, 14.010929584503174, 13.987962484359741, 11.796565532684326, 14.60891604423523]
#[3.036228656768799, 2.2116825580596924, 1.773503303527832, 0.9656541347503662, 6.872568845748901, 4.3612799644470215, 2.854464292526245, 1.7676424980163574, 7.522841930389404, 4.569177627563477, 2.3376331329345703, 1.4399023056030273, 8.51669692993164, 13.990816354751587, 8.690956354141235, 6.158030033111572, 11.546245336532593, 7.716180801391602, 4.313502311706543, 2.7588109970092773, 13.599654912948608, 11.729640483856201, 6.708925724029541, 4.310466527938843, 15.481214046478271, 12.782101392745972, 8.01132845878601, 5.359836578369141, 15.198915958404541, 14.668111562728882, 13.394903659820557, 11.39792275428772]

#numbers = [32843.65, 35623.2, 26703.35, 41763.50000000001, 26744.449999999997, 41722.40000000001, 73413.875, 49356.875, 30909.375, 17487.0]
#tick_label = ['C1=1', 'C1=2', 'C2=1', 'C2=2', 'C3=1', 'C3=2', 'C4=1', 'C4=2', 'C4=4', 'C4=8']
#graph_number(numbers, tick_label)
'''
(G, cnt) = random_graph(1000, 0.1)
print(cnt)
edge_c_map = dict()
edge_ratios = []
edge_count = [[], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]
for T1 in range(1, 3):
    for T2 in range(1, 3):
        for T3 in range(1, 3):
            eps = []
            for T4 in range(4):
                C1, C2, C3, C4 = T1, T2, T3, 2 ** T4
                edge_c_map = dict()
                Estimation(G, 1, edge_c_map)
                #print(edge_c_map)
                epsilons = []
                (new_graph, count) = random_sample(G, edge_c_map)
                time1, time2 = 0, 0
                for i in range(5000):
                    x = random.randint(0, 1000)
                    y = random.randint(0, 1000)
                    if x == y:
                        epsilons.append(0)
                        continue
                    if x > y:
                        temp = x
                        x = y
                        y = temp
                    (a, t) = find_cut(G, x, y)
                    time1 += t
                    (b, t) = find_cut(new_graph, x, y)
                    time2 += t
                    if a != b:
                        if i % 1000 == 999:
                            print(x, y)
                            print(a, b, b / a)
                    epsilons.append(b / a - 1)
                edge_count[1][T1] += count / 16
                edge_count[2][T2] += count / 16
                edge_count[3][T3] += count / 16
                edge_count[4][T4] += count / 8
                #print(epsilons)
                graph_cuts(epsilons)
                n = len(G.V)
                print(edge_count)
                original_time.append(time1)
                compressed_time.append(time2)
                print(original_time, compressed_time)
                eps.append(epsilons)
            graph_sides(eps)

height = edge_count[1][1:] + edge_count[2][1:] + edge_count[3][1:] + edge_count[4][:5]
print(height)
tick_label = []
for T1 in range(1, 3):
    tick_label.append('C1=' + str(T1))
for T2 in range(1, 3):
    tick_label.append('C2=' + str(T2))
for T3 in range(1, 3):
    tick_label.append('C3=' + str(T3))
for T4 in range(5):
    tick_label.append('C4=' + str(2 ** T4))
print(tick_label)
graph_number(height, tick_label)
#original_time = [55.55534815788269, 73.69494485855103, 91.14576005935669, 92.87783598899841, 104.00985860824585, 104.98981165885925, 83.62545585632324, 87.08545541763306, 88.99979281425476, 81.39956212043762, 71.32221913337708, 59.460729360580444, 74.86364603042603, 78.03516483306885, 81.39922261238098, 77.3391010761261, 64.21184849739075, 80.85734272003174, 75.58721208572388, 81.52416706085205, 86.02365374565125, 72.61619567871094, 113.05175447463989, 85.0840892791748, 90.66033387184143, 88.18178296089172, 62.597129821777344, 94.07373857498169, 112.65279173851013, 113.085373878479, 117.03972029685974, 97.16381859779358, 128.21994376182556, 107.66763973236084, 110.11023592948914, 100.21196842193604, 81.35375666618347, 113.0226309299469, 99.92538237571716, 95.88993334770203]
#compressed_time = [42.1554856300354, 32.667773485183716, 20.202919483184814, 12.79626750946045, 8.249198913574219, 109.66784000396729, 55.06597828865051, 32.63332414627075, 19.209187984466553, 9.832087755203247, 74.54994535446167, 37.636906147003174, 27.303617238998413, 15.73625922203064, 9.436418771743774, 71.5849380493164, 42.302964210510254, 36.44201302528381, 25.922200202941895, 15.50736665725708, 44.378540992736816, 28.923712015151978, 24.006507873535156, 10.822461128234863, 7.176854610443115, 63.44485807418823, 39.96890187263489, 32.309781312942505, 24.786173343658447, 15.347241878509521, 85.2751841545105, 57.62649369239807, 43.751691579818726, 23.92238426208496, 15.285818338394165, 98.89114665985107, 64.13816738128662, 77.92927408218384, 53.65524625778198, 30.555235147476196]

# generate a random graph with each edge included with probability p
coeA, coeB = 0, 0



C4List = []
kList = []
nList = []
productList = []
averageList = []
C1, C2, C3 = 2, 1, 1
for runCount in range(100):
    n = random.randint(40, 4000)
    p = random.uniform(min(0.6, 100000 / n / n) / 10, min(0.6, 100000 / n / n))
    (G, m) = random_graph(n, p)
    print(n, m)
    edge_c_map = dict()
    Estimation(G, 1, edge_c_map)
    value = 0
    print(edge_c_map)
    aveStrCon = 0
    for edge in edge_c_map.keys():
        aveStrCon += edge_c_map[edge]
    aveStrCon /= len(edge_c_map.keys())
    
    
    C4 = random.uniform(1, max(aveStrCon / 4, 3))
    (new_graph, new_m) = random_sample(G, edge_c_map)
    for i in range(100):
        x = random.randint(0, n)
        y = random.randint(0, n)
        if x == y:
            continue
        if x > y:
            temp = x
            x = y
            y = temp
        (a, t1) = find_cut(G, x, y)
        (b, t2) = find_cut(new_graph, x, y)
        if a == 0:
            print(a, x, y, b)
        value += abs(b / a - 1)
    epsilon = value / 100
    k = epsilon * epsilon * m + 2
    product = k * C4 / aveStrCon
    if product <= 1 or product >= 8:
        continue
    C4List.append(C4 / aveStrCon)
    kList.append(k)
    productList.append(k * C4 / aveStrCon)
    print(C4List)
    print(kList)
    print(productList)
    graph_epsilons(C4List, kList)

'''




# G = random_graph(10, 0.1)
# print(G.V)
# print(G.E)

