import pandas as pd
import networkx as nx
import math
import os
import matplotlib.pyplot as plt
import heapq

# ==========================================
# 1. LOAD THE CSV FILES
# ==========================================

roads = pd.read_csv("data/map_road_ways.csv")
nodes = pd.read_csv("data/tezpur_map_nodes.csv", dtype={"node_id": str})

# ==========================================
# 2. CREATE A NODE COORDINATE DICTIONARY
# ==========================================

coordinates = {}

for _, row in nodes.iterrows():
    node_id = str(row["node_id"])

    coordinates[node_id] = (
        float(row["latitude"]),
        float(row["longitude"])
    )

# ==========================================
# 3. HAVERSINE DISTANCE FUNCTION
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371000  # metres

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


# ==========================================
# 4. ASSUMED SPEEDS BY ROAD TYPE
# ==========================================

# These are project assumptions, not measured
# speeds or live traffic information.

road_speeds = {
    "motorway": 50,
    "trunk": 40,
    "primary": 35,
    "secondary": 30,
    "tertiary": 25,
    "residential": 20,
    "unclassified": 20,
    "service": 10,
    "living_street": 10
}

# ==========================================
# 5. CREATE A DIRECTED WEIGHTED GRAPH
# ==========================================

G = nx.MultiDiGraph()

skipped_segments = 0
created_segments = 0

for _, road in roads.iterrows():

    way_id = str(road["osm_way_id"])
    road_name = str(road["road_name"])
    road_type = str(road["road_type"])

    # Read ordered OSM node IDs
    raw_node_ids = road["osm_node_ids_in_order"]

    if pd.isna(raw_node_ids):
        continue

    node_ids = [
        node.strip()
        for node in str(raw_node_ids).split(";")
        if node.strip()
    ]

    if len(node_ids) < 2:
        continue

    # Check one-way information
    oneway = str(road["oneway"]).strip().lower()

    # Reverse node order for one-way roads
    if oneway == "-1":
        node_ids.reverse()

    # Choose an assumed average speed
    speed = road_speeds.get(road_type, 20)

    # Connect consecutive nodes
    for i in range(len(node_ids) - 1):

        start = node_ids[i]
        end = node_ids[i + 1]

        # Skip segments with missing coordinates
        if start not in coordinates or end not in coordinates:
            skipped_segments += 1
            continue

        lat1, lon1 = coordinates[start]
        lat2, lon2 = coordinates[end]

        # Calculate actual geometric segment distance
        distance = calculate_distance(
            lat1, lon1, lat2, lon2
        )

        # Estimate travel time in minutes
        travel_time = (distance / 1000) / speed * 60

        edge_data = {
            "distance_m": distance,
            "speed_kmh": speed,
            "travel_time_min": travel_time,
            "osm_way_id": way_id,
            "road_name": road_name,
            "road_type": road_type
        }

        # Add forward direction
        G.add_edge(start, end, **edge_data)
        created_segments += 1

        # Add reverse direction unless this is one-way
        if oneway not in ["yes", "1", "true", "-1"]:
            G.add_edge(end, start, **edge_data)
            created_segments += 1


# ==========================================
# 6. DISPLAY GRAPH INFORMATION
# ==========================================

print("\n========== ROAD NETWORK GRAPH ==========")

print("Total graph nodes:", G.number_of_nodes())
print("Total directed edges:", G.number_of_edges())
print("Segments skipped:", skipped_segments)
print("Edges added during construction:", created_segments)

print("\n========== SAMPLE GRAPH EDGES ==========")

for u, v, data in list(G.edges(data=True))[:10]:

    print(
        f"{u} -> {v} | "
        f"Distance: {data['distance_m']:.2f} m | "
        f"Estimated time: {data['travel_time_min']:.3f} min | "
        f"Road: {data['road_type']}"
    )


# ==========================================
# 7. SAVE THE GRAPH
# ==========================================

os.makedirs("output", exist_ok=True)

nx.write_graphml(G, "output/road_network.graphml")

print("\nGraph saved successfully!")
print("Location: output/road_network.graphml")

# ==========================================
# 8. VISUALIZE THE ACTUAL ROAD NETWORK
# ==========================================

import matplotlib.pyplot as plt

# Use longitude as X and latitude as Y
# so the graph follows real geographic positions.

positions = {}

for node in G.nodes():

    if node in coordinates:

        latitude, longitude = coordinates[node]

        positions[node] = (longitude, latitude)

# Keep only nodes with valid positions
visual_nodes = list(positions.keys())

# Create figure
plt.figure(figsize=(12, 10))

# Draw road connections
nx.draw_networkx_edges(
    G,
    positions,
    edgelist=[
        (u, v) for u, v in G.edges()
        if u in positions and v in positions
    ],
    edge_color="gray",
    width=0.7,
    alpha=0.6,
    arrows=False
)

# Draw road nodes
nx.draw_networkx_nodes(
    G,
    positions,
    nodelist=visual_nodes,
    node_size=8,
    node_color="blue",
    alpha=0.8
)

plt.title(
    "OSM-Based Road Network of Tezpur, Assam",
    fontsize=15
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.axis("equal")
plt.grid(True, alpha=0.2)

# Save image
plt.savefig(
    "output/road_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nRoad network visualization saved!")
print("Location: output/road_network.png")

# ==========================================
# 9. FIND THE HOSPITAL AND NEARBY ROAD NODES
# ==========================================

hospitals = pd.read_csv("data/tezpur_map_hospitals.csv")

print("\n========== HOSPITAL INFORMATION ==========")

print(
    hospitals[
        ["name", "latitude", "longitude"]
    ].to_string(index=False)
)

# Find Kanaklata Civil Hospital
hospital_matches = hospitals[
    hospitals["name"].str.strip().str.lower()
    == "kanaklata civil hospital"
]

if hospital_matches.empty:

    print("\nHospital not found by that exact name.")
    print("Check the hospital names printed above.")

else:

    hospital = hospital_matches.iloc[0]

    hospital_lat = float(hospital["latitude"])
    hospital_lon = float(hospital["longitude"])

    print("\nSelected hospital:", hospital["name"])

    # Calculate distance from hospital to every graph node
    nearby_nodes = []

    for node in G.nodes():

        if node not in coordinates:
            continue

        lat, lon = coordinates[node]

        distance = calculate_distance(
            hospital_lat,
            hospital_lon,
            lat,
            lon
        )

        nearby_nodes.append((node, distance))

    nearby_nodes.sort(key=lambda item: item[1])

    print("\n========== FIVE NEAREST ROAD NODES ==========")

    for node, distance in nearby_nodes[:5]:

        print(
            f"Node ID: {node} | "
            f"Distance from hospital: {distance:.2f} metres"
        )


# ==========================================
# 10. CHECK ROAD NETWORK CONNECTIVITY
# ==========================================

components = list(nx.weakly_connected_components(G))

components.sort(key=len, reverse=True)

print("\n========== NETWORK CONNECTIVITY ==========")

print("Number of weakly connected components:", len(components))

print("\nFive largest components:")

for i, component in enumerate(components[:5], start=1):

    print(
        f"Component {i}: {len(component)} nodes"
    )
strong_components = list(
    nx.strongly_connected_components(G)
)

strong_components.sort(key=len, reverse=True)

print("\n========== DIRECTED CONNECTIVITY ==========")

print(
    "Number of strongly connected components:",
    len(strong_components)
)

print(
    "Largest strongly connected component:",
    len(strong_components[0])
)

# ==========================================
# 12. DIJKSTRA SHORTEST PATH ALGORITHM
# ==========================================

def dijkstra(graph, start, destination):

    # Store minimum known travel time
    distances = {
        node: float("inf")
        for node in graph.nodes()
    }

    # Store previous node for route reconstruction
    previous = {}

    distances[start] = 0

    # Priority queue: (travel_time, node)
    priority_queue = [(0, start)]

    while priority_queue:

        current_distance, current_node = heapq.heappop(
            priority_queue
        )

        # Ignore outdated queue entries
        if current_distance > distances[current_node]:
            continue

        # Destination reached
        if current_node == destination:
            break

        # Examine outgoing road connections
        for _, neighbour, key, data in graph.out_edges(
            current_node,
            keys=True,
            data=True
        ):

            edge_time = data["travel_time_min"]

            new_distance = current_distance + edge_time

            if new_distance < distances[neighbour]:

                distances[neighbour] = new_distance

                previous[neighbour] = current_node

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbour)
                )

    # Check whether destination is reachable
    if distances[destination] == float("inf"):
        return None, float("inf")

    # Reconstruct shortest path
    path = []
    current = destination

    while current != start:

        path.append(current)
        current = previous[current]

    path.append(start)
    path.reverse()

    return path, distances[destination]


# ==========================================
# 13. SELECT HOSPITAL ACCESS NODE
# ==========================================

hospital_access_node = "5622971534"

print("\nHospital access node:", hospital_access_node)


# ==========================================
# 14. SELECT AMBULANCE STARTING NODE
# ==========================================

print("\nAvailable example road nodes:")

for node in list(G.nodes())[:10]:
    print(node)

start_node = input(
    "\nEnter ambulance starting node ID: "
).strip()

if start_node not in G:

    print("Error: Starting node does not exist.")

else:

    # Run Dijkstra
    shortest_path, total_time = dijkstra(
        G,
        start_node,
        hospital_access_node
    )

    if shortest_path is None:

        print("No route found to the hospital.")

    else:

        print("\n========== SHORTEST AMBULANCE ROUTE ==========")

        print("Starting node:", start_node)
        print("Hospital access node:", hospital_access_node)

        print("\nRoute:")

        print(" -> ".join(shortest_path))

        print(
            "\nTotal estimated travel time:",
            round(total_time, 3),
            "minutes"
        )

        print(
            "Total road segments:",
            len(shortest_path) - 1
        )

        print(
            "\nNote: Hospital access distance of approximately "
            "43.66 metres is not included."
        )
# STAGE 7: VISUALIZE THE SHORTEST AMBULANCE ROUTE

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if shortest_path is not None:

    # Create a figure
    plt.figure(figsize=(12, 10))

    # Draw all road network edges in light gray
    nx.draw_networkx_edges(
        G,
        positions,
        edge_color="lightgray",
        width=0.8,
        arrows=False
    )

    # Draw all road network nodes in light blue
    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=5,
        node_color="lightblue"
    )

    # Create a list of consecutive route edges
    route_edges = list(zip(shortest_path, shortest_path[1:]))

    # Highlight the shortest route in red
    nx.draw_networkx_edges(
        G,
        positions,
        edgelist=route_edges,
        edge_color="red",
        width=3,
        arrows=False
    )

    # Mark the starting point in green
    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=[start_node],
        node_color="green",
        node_size=100,
        label="Starting Point"
    )

    # Mark the hospital access node in blue
    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=[hospital_access_node],
        node_color="blue",
        node_size=120,
        label="Hospital Access Point"
    )

    # Add title and legend
    plt.title("Shortest Ambulance Route to Kanaklata Civil Hospital")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    legend_items = [
        Line2D([0], [0], color="lightgray", lw=2, label="Road Network"),
        Line2D([0], [0], color="red", lw=3, label="Shortest Route"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="green", markersize=9,
               label="Starting Point"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="blue", markersize=9,
               label="Hospital Access Point")
    ]

    plt.legend(handles=legend_items)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    # Save the visualization
    plt.savefig(
        "output/shortest_ambulance_route.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Shortest route visualization saved successfully!")

else:
    print("No route is available to visualize.")
    # STAGE 8: ROAD CONGESTION SIMULATION

print("\n========== ROAD CONGESTION SIMULATION ==========")

# Store the original route
original_route = shortest_path.copy()
original_time = total_time

# Select a road segment from the original route
congested_start = original_route[1]
congested_end = original_route[2]

print("Road affected by congestion:")
print(congested_start, "->", congested_end)

# Save the original edge travel time
edge_data = G.get_edge_data(congested_start, congested_end)

if edge_data:

    # Find the fastest parallel edge, if multiple edges exist
    selected_key = min(
        edge_data,
        key=lambda key: edge_data[key]["travel_time_min"]
    )

    original_edge_time = edge_data[selected_key]["travel_time_min"]

    # Simulate heavy congestion: increase travel time 10 times
    G[congested_start][congested_end][selected_key]["travel_time_min"] = (
        original_edge_time * 10
    )

    print("\nOriginal edge time:", round(original_edge_time, 3), "minutes")
    print("Congested edge time:",
          round(original_edge_time * 10, 3), "minutes")

    # Calculate a new shortest route
    new_route, new_time = dijkstra(
        G,
        start_node,
        hospital_access_node
    )

    print("\n========== RESULT AFTER CONGESTION ==========")

    if new_route is not None:
        print("New route:")
        print(" -> ".join(new_route))

        print("New estimated travel time:",
              round(new_time, 3), "minutes")

        if new_route != original_route:
            print("\nAlternative route selected!")
        else:
            print("\nThe same route is still optimal.")

        print("Original estimated time:",
              round(original_time, 3), "minutes")

        print("New estimated time:",
              round(new_time, 3), "minutes")

    else:
        print("No route available after congestion.")

    # Restore the original edge weight
    G[congested_start][congested_end][selected_key]["travel_time_min"] = (
        original_edge_time
    )

    print("\nOriginal road weight restored.")

else:
    print("Selected road segment was not found.")
# STAGE 9: ROAD CLOSURE SIMULATION

# ==========================================
# STAGE 9: ROAD CLOSURE SIMULATION
# ==========================================

print("\n========== ROAD CLOSURE SIMULATION ==========")

if shortest_path is None or len(shortest_path) < 2:
    print("Cannot simulate road closure: no valid original route.")
else:
    closure_start = shortest_path[0]
    closure_end = shortest_path[1]

    edge_data = G.get_edge_data(closure_start, closure_end)

    if not edge_data:
        print("Selected road segment does not exist.")
    else:
        # Select the edge with the lowest travel time
        selected_key = min(
            edge_data,
            key=lambda key: edge_data[key]["travel_time_min"]
        )

        # SAVE the edge attributes BEFORE removing the edge
        saved_attributes = dict(edge_data[selected_key])

        print(f"Closing road: {closure_start} -> {closure_end}")

        # Remove the selected directed edge
        G.remove_edge(
            closure_start,
            closure_end,
            key=selected_key
        )

        # Calculate a new route
        new_path, new_time = dijkstra(
            G,
            start_node,
            hospital_access_node
        )

        print("\n========== RESULT AFTER ROAD CLOSURE ==========")

        if new_path is None:
            print("No route is available after road closure.")
        else:
            print("New route:")
            print(" -> ".join(new_path))
            print(f"New estimated time: {new_time:.3f} minutes")

        # RESTORE the removed edge using the saved attributes
        G.add_edge(
            closure_start,
            closure_end,
            key=selected_key,
            **saved_attributes
        )

        print("\nOriginal road network restored.")
# Add this near the END of main.py, after the shortest route has been calculated
# and after any congestion/closure simulation has restored the graph.


# STAGE 10: EXPORT GRAPH DATA FOR WEBSITE

import json

website_data = {
    "nodes": [
        {
            "id": str(node_id),
            "lat": float(coordinates[node_id][0]),
            "lon": float(coordinates[node_id][1])
        }
        for node_id in G.nodes
        if node_id in coordinates
    ],

    "edges": [
        {
            "source": str(u),
            "target": str(v),
            "travel_time_min": float(edge_data["travel_time_min"]),
            "road_type": str(edge_data.get("road_type", "unknown"))
        }
        for u, v, key, edge_data in G.edges(keys=True, data=True)
        if u in coordinates and v in coordinates
    ],

    "hospital": {
        "name": "Kanaklata Civil Hospital",
        "lat": hospital_lat,
        "lon": hospital_lon
    },

    "hospital_access_node": str(hospital_access_node),
    "default_start_node": str(start_node),

    "suggested_start_nodes": [
        str(node) for node in list(G.nodes())[:10]
    ],

    "default_route": [
        str(node) for node in shortest_path
    ] if shortest_path else [],

    "default_route_time_min": (
        float(total_time) if shortest_path else None
    )
}

with open("output/road_network.json", "w", encoding="utf-8") as file:
    json.dump(website_data, file, indent=2)

print("\nWebsite graph data exported successfully!")
