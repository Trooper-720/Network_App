import tkinter as tk
from ttkbootstrap import Style
from queue import PriorityQueue

#Graph Visualization libraries
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#class for information about the graph's nodes and edges
class Graph:
    def __init__(self, num_of_vertices):
        self.num_of_vertices = num_of_vertices
        self.adjacency_matrix = [[float('inf')] * num_of_vertices for _ in range(num_of_vertices)]
        for i in range(num_of_vertices):
            self.adjacency_matrix[i][i] = 0
        self.source_nodes = {}

    def add_edge(self, source, destination, weight):
        self.adjacency_matrix[source][destination] = weight
        self.adjacency_matrix[destination][source] = weight


def bellman_ford_algorithm(source, graph):
    #print("Source Node:", source)
    temp_list_for_dvrVisualization = []
    distance = [float('inf')] * graph.num_of_vertices
    distance[source] = 0
    for i in range(graph.num_of_vertices - 1):
        for u in range(graph.num_of_vertices):
            for v in range(graph.num_of_vertices):
                if graph.adjacency_matrix[u][v] != float('inf'):
                    #print("\t\tCurrent distance:", distance)
                    #print("\t\tReplace", distance[v], "with:", distance[u] + graph.adjacency_matrix[u][v])
                    old_distance = distance[v]
                    if distance[u] + graph.adjacency_matrix[u][v] < distance[v]:
                        distance[v] = distance[u] + graph.adjacency_matrix[u][v]
                        temp_list_for_dvrVisualization.append((source, i, u, v, old_distance, distance[v]))
                        #print("\t\t\t\tShorter path found! New distance:", distance[v])
                    else:
                        temp_list_for_dvrVisualization.append((source, i, u, v, old_distance, "nochange"))
    dvr_appendToVisualizationList(temp_list_for_dvrVisualization)
    return distance

#returns a dictionary with all the vertices mapped to the lowest cost from source
def dijkstra_algorithm(graph, source):
    #Initializes the dictionary with infinity as the minimum cost for each node
    costs={}
    for node in range(graph.num_of_vertices):
        costs[node] = float('inf')
    costs[source] = 0
    
    queue = PriorityQueue()
    queue.put((0, source))

    visited_nodes = []

    while not queue.empty():
        (cost, current_node) = queue.get()
        visited_nodes.append(current_node)

        for adjacent_vertex in range(graph.num_of_vertices):
            if graph.adjacency_matrix[current_node][adjacent_vertex] != float('inf') and adjacent_vertex not in visited_nodes:
                edge_cost = graph.adjacency_matrix[current_node][adjacent_vertex]
                if (costs[current_node] + edge_cost) < costs[adjacent_vertex]:
                    queue.put((costs[current_node] + edge_cost, adjacent_vertex))
                    costs[adjacent_vertex] = costs[current_node] + edge_cost
                    graph.source_nodes[adjacent_vertex] = current_node             
    return costs

def distance_vector_algorithm(graph):
    distance = []
    for i in range(graph.num_of_vertices):
        distance.append(i)
    for source in range(graph.num_of_vertices):
        distance[source] = bellman_ford_algorithm(source, graph)
    return distance

def dvr_appendToVisualizationList(list):
    dvr_list_for_visualization.append(list)

def visualizing_dijkstra(nodes, edges, route):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        fig, ax = plt.subplots(num="Visualizing Dijkstra")
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='#ADD8E6', ax=ax)
        nx.draw_networkx_edges(G, pos,edge_color='gray', ax=ax)
        weights = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=11)
        nx.draw_networkx_labels(G, pos)
        

        for node in route:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='r', ax=ax)
            nx.draw_networkx_edges(G, pos,  edge_color='gray',ax=ax)

            plt.pause(1)

            nx.draw_networkx_nodes(G, pos, node_color='#ADD8E6', ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray',ax=ax)

        color_map = []
        color_map_edge = []
        for n in G.nodes:
            if n in route:
                color_map.append('red')
            else:
                color_map.append('gray')

        for e in G.edges:
            if e[1] in route and e[0] in route:
                    if (route.index(e[1]) - route.index(e[0]) == 1)  or (route.index(e[1]) - route.index(e[0]) == -1):
                        color_map_edge.append('red')
                    else:
                        color_map_edge.append('gray')
            else:
                color_map_edge.append('gray')    

        nx.draw_networkx_nodes(G, pos, node_color=color_map)
        nx.draw_networkx_edges(G, pos, edge_color = color_map_edge)
        plt.show()
        
def focus_next_widget(event):
        event.widget.tk_focusNext().focus()
        return("break")        

class App:
    def __init__(self,master):
        self.master = master
        self.master.geometry("1000x700")
        self.mainFrame = tk.Frame(self.master)
        self.mainFrame.pack(side="top", expand=True, fill="both")
        self.master.winfo_toplevel().title("Interactive Tool For Routing Algorithms")
          
        #Enter the main menu
        self.mainMenu()
        
        #GLOBALS
        self.nodes_global = []
        
    #main menu
    def mainMenu(self):
          # Title
        self.title = tk.Label(self.mainFrame, text="Interactive Tool For Routing Algorithms", font=('Arial',24))
        self.title.pack(pady=130)
        # Centeralized Algorithm Button
        self.button = tk.Button(self.mainFrame, text="Centeralized Algorithm", command=self.on_click_cbutton, font=('Arial',18), width=20, highlightcolor='#FF0000')
        self.button.pack(padx=10,pady=20) 
        self.button.bind("<Return>", lambda command: self.on_click_cbutton())
        # Decentralized Algorithm Button
        self.button2 = tk.Button(self.mainFrame, text="Decentralized Algorithm", command=self.on_click_dbutton,  font=('Arial',18), width=20)
        self.button2.pack(padx=10) 
        self.button2.bind("<Return>", lambda command: self.on_click_dbutton())
        
        # Exit Button
        self.exit_B = tk.Button(self.mainFrame, text="Exit", command=self.exit_App,  font=('Arial', 14), width=10)
        self.exit_B.pack(padx=10,pady=20) 
        self.exit_B.bind("<Return>", lambda command: self.master.destroy())
        
    def exit_App(self):
        plt.close('all')
        self.master.destroy()
        exit(0)
        
    # On click event (Centeralized Algorithm Button)
    def on_click_cbutton(self):
        self.title.destroy()
        self.button.destroy()
        self.button2.destroy()
        self.exit_B.destroy()
        
        #title algorithm
        self.new_title = tk.Label(self.mainFrame, text="Dijkstra Algorithm", font=('Arial', 24))
        self.new_title.pack(padx=10, pady=50)
        
        # Text box for vertices
        self.label_vertices = tk.Label(self.mainFrame, text="Enter number of routers:", font=('Arial', 14))
        self.label_vertices.pack(padx=10)
        self.entry_vertices = tk.Text(self.mainFrame, width=5, height=1, font=('Arial', 16))
        self.entry_vertices.insert(tk.END,"9")
        self.entry_vertices.pack(padx=10, pady=10)
        self.entry_vertices.bind("<Tab>", focus_next_widget)
        self.entry_vertices.bind("<Return>", focus_next_widget)
        
        # Text box for edges
        self.label_edges = tk.Label(self.mainFrame, text="Enter paths and their costs (V1, V2, costV1-V2):" , font=('Arial',14))
        self.label_edges.pack(padx=10, pady=20)
        self.entry_edges = tk.Text(self.mainFrame, width=60, height=4, font=('Arial',16))
        self.entry_edges.insert(tk.END,"[(0, 1, 4),(0, 6, 7),(1, 6, 11),(1, 7, 20),(1, 2, 9),(2, 3, 6),(2, 4, 2),(3, 4, 10),(3, 5, 5),(4, 5, 15), (4, 7, 1), (4, 8, 5), (5, 8, 12), (6, 7, 1), (7, 8, 3)]")
        self.entry_edges.pack(padx=10)
        self.entry_edges.bind("<Tab>", focus_next_widget)
        self.entry_edges.bind("<Return>", focus_next_widget)
        
        #===============================================================
        self.frame2 = tk.Frame(self.mainFrame, width = 100, height= 100)
        self.frame2.pack(padx=10, pady=20)
        
        # Text box for sourcce vertex
        self.label_source = tk.Label(self.frame2, text="Enter source router:", font=('Arial', 14))
        self.entry_source = tk.Text(self.frame2, width=5, height=1 , font=('Arial', 16))
        self.entry_source.insert(tk.END,"0")
        self.entry_source.bind("<Tab>", focus_next_widget)
        self.entry_source.bind("<Return>", focus_next_widget)
        
        # Text box for destination vertex
        self.label_target = tk.Label(self.frame2, text="Enter destination router:" , font=('Arial', 14))
        self.entry_target = tk.Text(self.frame2, width=5, height=1, font=('Arial', 16))
        self.entry_target.insert(tk.END,"7")
        self.entry_target.bind("<Tab>", focus_next_widget)
        self.entry_target.bind("<Return>", focus_next_widget)
        
        #grid layout for above 4 widgets
        self.label_source.grid(row=0, column=0, padx=10, pady=10)
        self.entry_source.grid(row=1, column=0, padx=10, pady=10)
        self.label_target.grid(row=0, column=1, padx=10, pady=10)
        self.entry_target.grid(row=1, column=1, padx=10, pady=10)
        
        #=============================================================
        
        self.button3 = tk.Button(self.mainFrame, text="Generate", command=self.on_click_cgenerate, font=('Arial', 18), width=15)
        self.button3.pack(padx=10)
        self.button3.bind("<Return>", lambda command: self.on_click_cgenerate())
        
        self.back_B = tk.Button(self.mainFrame, text="Back", command=self.on_click_back_from_centralized , font=('Arial', 14), width=10)
        self.back_B.pack(padx=10,pady=20)
        self.back_B.bind("<Return>", lambda command: self.on_click_back_from_centralized())
    
    def on_click_back_from_centralized(self):
        self.clear_frame()
        self.mainMenu()
    
    # On-click event for generate
    def on_click_cgenerate(self):
        # Get number of vertices from entry box
        self.input_vertices = int(self.entry_vertices.get("1.0", "end"))
        # Get edges from entry box
        self.input_edges = list(eval(self.entry_edges.get("1.0", "end")))
        # Get source from entry box
        self.input_source = int(eval(self.entry_source.get("1.0", "end")))
        # Get destination from entry box
        self.input_destination = int(eval(self.entry_target.get("1.0", "end")))
        self.button3.destroy()
        self.label_source.destroy()
        self.label_target.destroy()
        self.entry_source.destroy()
        self.entry_target.destroy()
        self.new_title.destroy()
        self.label_vertices.destroy()
        self.entry_vertices.destroy()
        self.label_edges.destroy()
        self.entry_edges.destroy()
        self.back_B.destroy()
        graph = Graph(self.input_vertices) # Initialize graph with the vertices.
        nodes = [i for i in range(self.input_vertices)]
        edges = list(self.input_edges)
        for e in edges:                 
            graph.add_edge(e[0], e[1], e[2])  
        #Perform Dijkstra's algorithm to find the minimum cost/path from source to target 
        D = dijkstra_algorithm(graph, self.input_source)

        flag=0
        route = []
        route.append(self.input_destination)
        #finding the route from source to destination that results in dikstra's minimum cost
        if self.input_source!=self.input_destination:
            current_node=graph.source_nodes[self.input_destination]
            while flag == 0:
                if current_node == self.input_source:
                    route.append(current_node)
                    flag=1
                else:
                    route.append(current_node)
                    current_node=graph.source_nodes[current_node]
            route.reverse()
        
        self.result_label = tk.Label(self.mainFrame, text="Cost from router [" + str(self.input_source) + "] to router [" + str(self.input_destination) + "] is " + str(D[self.input_destination]) + "." , font=('Arial', 24))
        self.result_label.pack(padx=10, pady=120)

        # Button to visualize the algorithm.
        self.visualize_button = tk.Button(self.mainFrame, text="Visualize Path Taken",command=lambda: visualizing_dijkstra(nodes, edges, route), font=('Arial', 18), width=20)
        self.visualize_button.pack(padx=10,pady=0)
        self.visualize_button.bind("<Return>", lambda command: visualizing_dijkstra(nodes, edges, route))
    
        # Button to go back.
        self.back_button = tk.Button(self.mainFrame, text="Back",command=self.on_click_back_from_generate_centralized, font=('Arial', 14), width=10)
        self.back_button.pack(padx=10,pady=20)
        self.back_button.bind("<Return>", lambda command: self.on_click_back_from_generate_centralized())

    def on_click_back_from_generate_centralized(self):
        plt.close('all')
        self.clear_frame()
        self.on_click_cbutton()
    
    # Create default value for vertices (delete on user input)
    def default_vertices(self,event):
        self.entry_vertices.delete(0,'end')

    # Create default value for edges (delete on user input)
    def default_edges(self,event):
        self.entry_edges.delete(0,'end')

    # On-click event for decentralized algorithm
    def on_click_dbutton(self):
        # Destroy current text and widgets
        self.title.destroy()
        self.button.destroy()
        self.button2.destroy()
        self.exit_B.destroy()
        # Begin new window format
        self.new_title = tk.Label(self.mainFrame, text="Bellman Ford Algorithm", font=('Arial', 24))
        self.new_title.pack(padx=10, pady=50)
        # Text box for vertices
        self.label_vertices = tk.Label(self.mainFrame, text="Enter number of vertices:" , font=('Arial', 14))
        self.label_vertices.pack(padx=10)
        self.entry_vertices = tk.Text(self.mainFrame, width=5, height=1, font=('Arial', 16))
        self.entry_vertices.insert(tk.END,"3")
        self.entry_vertices.pack(padx=10,pady=20)
        self.entry_vertices.bind("<Tab>", focus_next_widget)
        self.entry_vertices.bind("<Return>", focus_next_widget)
       
        # Text box for edges
        self.label_edges = tk.Label(self.mainFrame, text="Enter number of edges (V1, V2, cost V1-V2):", font=('Arial', 14))
        self.label_edges.pack(padx=10)
        self.entry_edges = tk.Text(self.mainFrame, width=60, height=4, font=('Arial', 16))
        self.entry_edges.insert(tk.END,"[(0, 1, 1), (1, 2, 2), (0, 2, 5)]")
        self.entry_edges.pack(padx=10,pady=20)
        self.entry_edges.bind("<Tab>", focus_next_widget)
        self.entry_edges.bind("<Return>", focus_next_widget)
    
    
        self.button3 = tk.Button(self.mainFrame, text="Generate", command=self.on_click_generate, font=('Arial', 18), width=15)
        self.button3.pack(padx=10)
        self.button3.bind("<Return>", lambda command: self.on_click_generate())
        
        self.back_B = tk.Button(self.mainFrame, text="Back", command=self.on_click_back_from_decentralized, font=('Arial', 14), width=10)
        self.back_B.pack(padx=10, pady=20)
        self.back_B.bind("<Return>", lambda command: self.on_click_back_from_decentralized())

    def on_click_back_from_decentralized(self):
        self.clear_frame()
        self.mainMenu()
    

    # On-click event for generate
    def on_click_generate(self):
        # Get number of vertices from entry box
        self.input_vertices = int(self.entry_vertices.get("1.0", "end"))
        # Get edges from entry box
        self.input_edges = list(eval(self.entry_edges.get("1.0", "end")))
        self.button3.destroy()
        self.new_title.destroy()
        self.label_vertices.destroy()
        self.entry_vertices.destroy()
        self.label_edges.destroy()
        self.entry_edges.destroy()
        self.back_B.destroy()
        graph = Graph(self.input_vertices) # Initialize graph with 3 vertices.
        edges = list(self.input_edges)
        global dvr_list_for_visualization
        dvr_list_for_visualization = [] #The content's format would be = list[ sourceNode#List[ sourceNode#, iteration#, vertex u, vertex v, current distance, changed distance]]
        for e in edges:                 #Value of changed distance is either integer or "nochange". If it's an integer, it means the distance value was changed to the integer's value.
            graph.add_edge(e[0], e[1], e[2])    # If it's "nochange", it means the distance value wasn't changed.  
        distances = distance_vector_algorithm(graph)
       
        #add to the global nodes for visualization later
        self.nodes_global = []
        for e in distances:
            self.nodes_global.append(e)
        
        #title for nodes
        self.node_title = tk.Label(self.mainFrame, text="Distances of nodes from the source node [0]", font=('Arial', 20))
        self.node_title.pack(padx=10, pady=60)
        
        string = ""
        for node in range(len(distances)):
            string += "Node [" + str(node) + "]: " + str(distances[node][0]) + " units.\n"
            #print("Distance vector from node", node, ":", distances[node])
        
        self.result_label = tk.Text(self.mainFrame, font=('Arial', 16), width = 25, height = 13)
        self.result_label.tag_configure("tag_name", justify='center')
        self.result_label.insert(tk.END, string)
        self.result_label.tag_add("tag_name", "1.0", "end")
        self.result_label.pack(padx=10, pady=5)
        self.result_label.configure(state="disabled")
        
        #spacer
        self.space = tk.Label(self.mainFrame, text="")
        self.space.pack(padx=10, pady=15)
        
        # Button to visualize the algorithm.
        self.visualize_button = tk.Button(self.mainFrame, text="Visualize", command=self.on_click_visualize, font=('Arial', 18), width=20)
        self.visualize_button.pack(padx=10,pady=0)
        self.visualize_button.bind("<Return>", lambda command: self.on_click_visualize())
        
        # Button to go back.
        self.back_button = tk.Button(self.mainFrame, text="Back", command=self.on_click_back_from_generate_decentralized , font=('Arial', 14), width=10)
        self.back_button.pack(padx=10,pady=20)
        self.back_button.bind("<Return>", lambda command: self.on_click_back_from_generate_decentralized())
    
    
    def clear_frame(self):
        list = self.mainFrame.winfo_children()
        length = len(list)
        for widget in range(length):
            list[widget].destroy()
    
    def on_click_back_from_generate_decentralized(self):
        plt.close('all')
        self.clear_frame()
        self.on_click_dbutton()
             
        
    # On-click event for visualize
    def on_click_visualize(self): 
        #get nodes
        nodes = []
        for i in range(self.input_vertices):
            nodes.append(i)
        
        #get edges  
        edges = self.input_edges
        
        #get the route to traverse
        route = []
        
        tempRoute = []
        #print(dvr_list_for_visualization[0])
        #print(len(dvr_list_for_visualization[0]))
        for sourceNode in dvr_list_for_visualization[0]:
            if(sourceNode[5] != 'nochange'):
                tempRoute.append(sourceNode[2])
                tempRoute.append(sourceNode[3])
        
        # remove consecutive duplicates from list 
        for i in range(1, len(tempRoute)):
            if tempRoute[i] != tempRoute[i-1]:
                route.append(tempRoute[i])
                
        
        
        #show the walk of the algorithm
        visualize_bellman_ford(nodes, edges, route)
        
        #Graph FINAL after algorithms walk
        plt.clf() #clear the animation
        #==================================================
        #nodes are added in the on_click_generate
        nodes = self.nodes_global
        #====================================================
        
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for i in range(len(nodes)):
            G.add_node(i)

        # Add edges to the graph based on the array pattern
        for i in range(len(nodes)):
            for j in range(len(nodes[i])):
                #dont add the loops to self
                if(i != j):
                    G.add_edge(i, j, distance=nodes[i][j])

        # Draw the graph
        pos = nx.spring_layout(G)
        #nx.draw(G, pos, with_labels=True)
        nx.draw_networkx(G, pos, with_labels=True, arrows=True, edge_color='#000000')
        
        # Add edge labels
        labels = nx.get_edge_attributes(G, 'distance')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        
        plt.show()
    

def visualize_bellman_ford(nodes, edges, route):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    
    fig, ax = plt.subplots(num="Visualizing Bellman-Ford")
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='#ADD8E6', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax, width=2.0)
    weights = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=11)
    nx.draw_networkx_labels(G, pos)
    
    
    for node in route:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='r', ax=ax)
        nx.draw_networkx_edges(G, pos,  edge_color='gray',ax=ax)

        plt.pause(0.125)

        nx.draw_networkx_nodes(G, pos, node_color='#ADD8E6', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray',ax=ax)
    
    return None
    

style = Style(theme='darkly')
window = style.master
app = App(window)
window.mainloop()
