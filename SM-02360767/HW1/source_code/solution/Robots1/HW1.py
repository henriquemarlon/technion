import argparse
import os
from typing import List, Tuple

from Plotter import Plotter
from shapely.geometry.polygon import Polygon, LineString

import math
import heapq



# TODO
def get_minkowsky_sum(original_shape: Polygon, r: float) -> Polygon:
    
    vertices_P = list(original_shape.exterior.coords)
    vertices_Q = [(0, -r), (r, 0), (0, r), (-r, 0)]
    
    vertices_P += vertices_P[:2]
    vertices_Q += vertices_Q[:2]
    
    i, j = 0, 0
    minkowski_vertices = []
    max_i, max_j = len(vertices_P) - 2, len(vertices_Q) - 2
    
    while not (i == max_i and j == max_j):
        v_i = vertices_P[i]
        w_j = vertices_Q[j]
        minkowski_vertices.append((v_i[0] + w_j[0], v_i[1] + w_j[1]))

        if(i == max_i):
            j += 1
            continue
        elif(j == max_j):
            i += 1
            continue
        
        angle_P = math.degrees(math.atan2(vertices_P[i + 1][1] - v_i[1], vertices_P[i + 1][0] - v_i[0]))
        angle_Q = math.degrees(math.atan2(vertices_Q[j + 1][1] - w_j[1], vertices_Q[j + 1][0] - w_j[0]))
        if(angle_P < 0):
            angle_P += 360
        if(angle_Q < 0):
            angle_Q += 360
        
        if angle_P < angle_Q:
            i += 1
        elif angle_P > angle_Q:
            j += 1
        else:  # Equal angles
            i += 1
            j += 1

    return Polygon(minkowski_vertices)


# TODO
def get_visibility_graph(obstacles: List[Polygon], source=None, dest=None) -> List[LineString]:

    vertices = []
    for obstacle in obstacles:
        vertices.extend(obstacle.exterior.coords) 
    
    if source:
        vertices.append((source[0], source[1]))
    if dest:
        vertices.append((dest[0], dest[1]))
    
    lines = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            line = LineString([vertices[i], vertices[j]])
            
            is_valid = True
            for obstacle in obstacles:
                if line.intersects(obstacle) and not line.touches(obstacle):
                    is_valid = False
                    break
            
            if is_valid:
                lines.append(line)
    
    return lines

def get_shortest_path_and_cost(lines: List[LineString], source: Tuple[float, float], dest: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
    graph = {}
    for line in lines:
        start = tuple(line.coords[0])
        end = tuple(line.coords[1])
        weight = line.length

        if start not in graph:
            graph[start] = []
        if end not in graph:
            graph[end] = []
        
        graph[start].append((end, weight))
        graph[end].append((start, weight))

    priority_queue = [(0, source, [])]
    visited = set()

    while priority_queue:
        current_cost, current_vertex, path = heapq.heappop(priority_queue)

        if current_vertex in visited:
            continue
        visited.add(current_vertex)

        path = path + [current_vertex]

        if current_vertex == dest:
            return path, current_cost

        for neighbor, weight in graph.get(current_vertex, []):
            if neighbor not in visited:
                heapq.heappush(priority_queue, (current_cost + weight, neighbor, path))

    return [], float('inf')
    


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)


def get_points_and_dist(line):
    source, dist = line.split(' ')
    dist = float(dist)
    source = tuple(map(float, source.split(',')))
    return source, dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Robot", help="A file that holds the starting position of the robot, and the distance from the center of the robot to any of its vertices")
    parser.add_argument("Obstacles", help="A file that contains the obstacles in the map")
    parser.add_argument("Query", help="A file that contains the ending position for the robot.")
    args = parser.parse_args()
    obstacles = args.Obstacles
    robot = args.Robot
    query = args.Query
    is_valid_file(parser, obstacles)
    is_valid_file(parser, robot)
    is_valid_file(parser, query)
    workspace_obstacles = []
    with open(obstacles, 'r') as f:
        for line in f.readlines():
            ob_vertices = line.split(' ')
            if ',' not in ob_vertices:
                ob_vertices = ob_vertices[:-1]
            points = [tuple(map(float, t.split(','))) for t in ob_vertices]
            workspace_obstacles.append(Polygon(points))
    with open(robot, 'r') as f:
        source, dist = get_points_and_dist(f.readline())

    # step 1:
    c_space_obstacles = [get_minkowsky_sum(p, dist) for p in workspace_obstacles]
    plotter1 = Plotter()

    plotter1.add_obstacles(workspace_obstacles)
    plotter1.add_c_space_obstacles(c_space_obstacles)
    plotter1.add_robot(source, dist)

    plotter1.show_graph()

    # step 2:

    lines = get_visibility_graph(c_space_obstacles)
    plotter2 = Plotter()

    plotter2.add_obstacles(workspace_obstacles)
    plotter2.add_c_space_obstacles(c_space_obstacles)
    plotter2.add_visibility_graph(lines)
    plotter2.add_robot(source, dist)

    plotter2.show_graph()

    # step 3:
    with open(query, 'r') as f:
        dest = tuple(map(float, f.readline().split(',')))

    lines = get_visibility_graph(c_space_obstacles, source, dest)
    #TODO: fill in the next line
    shortest_path, cost = get_shortest_path_and_cost(lines, source, dest)

    plotter3 = Plotter()
    plotter3.add_robot(source, dist)
    plotter3.add_obstacles(workspace_obstacles)
    plotter3.add_robot(dest, dist)
    plotter3.add_visibility_graph(lines)
    plotter3.add_shorterst_path(list(shortest_path))


    plotter3.show_graph()