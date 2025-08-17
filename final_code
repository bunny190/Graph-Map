# Google Maps 
# Core Concepts Used: Graphs, Dijkstra's Algorithm, Priority Queue, HashMaps, Adjacency List
# Extension: Real GPS Coordinates + Haversine Distance + Visualization using networkx and matplotlib

import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import math

class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)  # Adjacency list for graph
        self.coordinates = {}  # Store GPS coordinates {node: (lat, lon)}

    def add_node(self, node, coord):
        self.coordinates[node] = coord

    def haversine_distance(self, coord1, coord2):
        # Calculate distance between two (lat, lon) pairs using Haversine formula
        R = 6371  # Radius of the Earth in km
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c  # in kilometers

    def add_edge(self, u, v, use_haversine=False):
        if use_haversine:
            w = self.haversine_distance(self.coordinates[u], self.coordinates[v])
        else:
            w = 1  # default dummy weight
        self.adj_list[u].append((v, w))
        self.adj_list[v].append((u, w))

    def dijkstra(self, source):
        min_heap = [(0, source)]
        distances = {node: float('inf') for node in self.adj_list}
        distances[source] = 0
        parent = {source: None}

        while min_heap:
            current_distance, current_node = heapq.heappop(min_heap)

            for neighbor, weight in self.adj_list[current_node]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parent[neighbor] = current_node
                    heapq.heappush(min_heap, (distance, neighbor))

        return distances, parent

    def shortest_path(self, source, target):
        distances, parent = self.dijkstra(source)
        if distances[target] == float('inf'):
            return None, float('inf')

        path = []
        current = target
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()

        return path, distances[target]

    def visualize(self, path=None):
        G = nx.Graph()
        for node, coord in self.coordinates.items():
            G.add_node(node, pos=coord)
        for u in self.adj_list:
            for v, w in self.adj_list[u]:
                if G.has_edge(u, v):
                    continue
                G.add_edge(u, v, weight=round(w, 2))

        pos = nx.get_node_attributes(G, 'pos')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
        plt.title("Map Graph with GPS Coordinates")
        plt.show()

# Sample usage
if __name__ == "__main__":
    g = Graph()
    g.add_node("A", (12.9716, 77.5946))  # Bangalore
    g.add_node("B", (13.0827, 80.2707))  # Chennai
    g.add_node("C", (17.3850, 78.4867))  # Hyderabad
    g.add_node("D", (19.0760, 72.8777))  # Mumbai
    g.add_node("E", (28.6139, 77.2090))  # Delhi
    g.add_node("F", (22.5726, 88.3639))  # Kolkata

    g.add_edge("A", "B", use_haversine=True)
    g.add_edge("A", "C", use_haversine=True)
    g.add_edge("B", "C", use_haversine=True)
    g.add_edge("B", "D", use_haversine=True)
    g.add_edge("C", "D", use_haversine=True)
    g.add_edge("D", "E", use_haversine=True)
    g.add_edge("E", "F", use_haversine=True)

    source = "A"
    destination = "F"
    path, distance = g.shortest_path(source, destination)

    print(f"\nShortest path from {source} to {destination}: {path}")
    print(f"Total Distance: {round(distance, 2)} km")

    g.visualize(path)
