#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.centrality import degree_centrality, closeness_centrality


# In[6]:


path_d1 = "path to data"


# In[7]:


# We load the graph using networkx's read_edgelist attribute. We are creating the graph using nx.Graph() which gives us an undirected graph
G = nx.read_edgelist(path_d1,delimiter='\t',create_using=nx.Graph(),nodetype=int)


# ## Community Detection

# In[8]:


# We define the louvain_community_detection function which takes as a parameter the graph and applies greedy community detection algorithm.

def louvain_community_detection(graph):
    communities = list(greedy_modularity_communities(graph))
    
    # In the below code we create a dictionary and map every node to it's community.
    node_community_mapping = {node: i for i, community in enumerate(communities) for node in community}
    return node_community_mapping


# In[9]:


def landmark_selection_deg(graph, node_community_mapping, num_landmarks_per_community=3):
    # We define a dictionary for mapping highest degree nodes and landmarks.
    landmarks_degree = {}
    

    for community in set(node_community_mapping.values()):
        # We create a list community_nodes that contains the nodes of every community to be passed a parameter for subgraph.
        community_nodes = [node for node, comm in node_community_mapping.items() if comm == community]

        # We create a subgraph of the nodes of every community and apply degree based centrality measure to it.
        degree_values = degree_centrality(graph.subgraph(community_nodes))

        # We select the nodes that have the highest degree centrality in every community and add it to the landmarks_degree dictionary.
        top_degree_nodes = sorted(degree_values, key=degree_values.get, reverse=True)[:num_landmarks_per_community]
        landmarks_degree[community] = top_degree_nodes

            
    # Finally we also make one more dictionary that has the highest degree nodes as the keys and their respective communities as the values.
    landmarks_degcen = {node: i for i, community in enumerate(set(node_community_mapping.values())) for node in
                         landmarks_degree[community] }

    return (landmarks_degcen,landmarks_degree)


# In[10]:


def shortest_path_distance_matrix(graph, landmarks_combined):
    # We define two variables containing the values of the highest numbered node and no. of landmarks.
    num_nodes = max(graph.nodes) + 1
    num_landmarks = len(landmarks_combined)

    # We initialize two matrices with zeroes and with highest numbered node and no. of landmarks as shapes.

    # The distance_matrix_nodes_to_landmarks is a matrix that has the nodes as the number of rows and landmarks as the number of columns.
    # every element in this matrix is the distance or the number of steps it takes to reach a landmark from each and every node.
    distance_matrix_nodes_to_landmarks = np.zeros((num_nodes, num_landmarks))

    # The distance_matrix_landmarks_to_nodes is a matrix that has the landmarks as the number of rows and nodes as the number of columns.
    # every element in this matrix is the distance of a landmark to every node.
    distance_matrix_landmarks_to_nodes = np.zeros((num_landmarks, num_nodes))

    # In the below for loop, we iterate through every node in the graph and calculate it's path length to all the landmarks and add it in the  distance_matrix_nodes_to_landmarks matrix.    
    for node in graph.nodes:
        for j, landmark_node in enumerate(landmarks_combined):
            if nx.has_path(graph, node, landmark_node):
                distance_matrix_nodes_to_landmarks[node, j] = nx.shortest_path_length(graph, source=node, target=landmark_node)


    # In the below for loop, we iterate through every landmark from the landmarks_combined dictionary and calculate it's path length to all the nodes in the graph and add it in the distance_matrix_landmarks_to_nodes matrix
    for j, landmark_node in enumerate(landmarks_combined):
        for node in graph.nodes:
            if nx.has_path(graph, landmark_node, node):
                distance_matrix_landmarks_to_nodes[j, node] = nx.shortest_path_length(graph, source=landmark_node, target=node)
    

    return (distance_matrix_nodes_to_landmarks, distance_matrix_landmarks_to_nodes)


# In[11]:


def estimate_shortest_distance(src_nod,des_nod,comm_dict,comm_landmark_dict,landmark_comm_dict,dist_src_lnd_mat, dist_lnd_src_mat):
    # In this function we perform the task of calculating the distance of the nodes to landmarks and vice versa.

    # We first define an empty list that we will use to store the index of the landmarks that are present in the community that our source node belongs to.
    ind_of_landmrk = []

    # We obtain the community of the source node.
    comm_of_node = comm_dict[src_nod]

    # We find the landmarks of the particular community
    landmrks_of_comm = comm_landmark_dict[comm_of_node]

    # For the selected landmarks, we obtain there indexes since for both of our distance matrices, the landmarks are assigned by there index value,
    # and not by their actual node value. This is done to keep the matrix shape small.
    for l in landmrks_of_comm:
        ind_of_landmrk.append(list(landmark_comm_dict.keys()).index(l))

    # Now we find the distance of the landmark that's closest to our source node.
    dist_src_to_landmrk = min(dist_src_lnd_mat[src_nod,ind_of_landmrk])
    

    # In the below for loop we find the index of our selected landmark that's closest to the source node.
    for ind,i in enumerate(ind_of_landmrk):
        if dist_src_lnd_mat[src_nod,i] == dist_src_to_landmrk:
            #closest_landmrk_to_src_nod = landmrks_of_comm[ind]
            indx_of_closest_landmrk_to_src_nod = i
            break
            
    # We then calculate the distance of our landmark to the destination node.
    dist_landmrk_to_des = dist_lnd_src_mat[indx_of_closest_landmrk_to_src_nod,des_nod]

    # We finally return the total distance.
    return dist_src_to_landmrk + dist_landmrk_to_des


# In[12]:


# We call the function that applies the community detection algorithm on our graph.

node_community_mapping = louvain_community_detection(G)


# In[13]:


# We obtain the highest degree centrality nodes for every community in our graph.

deg_landmarks,landmarks_degree = landmark_selection_deg(graph=G,node_community_mapping=node_community_mapping)


# In[14]:


# We obtain the distance matrices.

distance_matrix_nodes_to_landmarks, distance_matrix_landmarks_to_nodes = shortest_path_distance_matrix(G, deg_landmarks)   


# In[17]:


# We randomly select nodes for testing.

source_node_list = [1,2,3,4,5,6]
dest_node_list =[7,8,9,10,11,12]


# In[18]:


# We call the estimate_shortest_distance function to obtain the distances from our source nodes to the destination nodes.

for source_node,destination_node in zip(source_node_list,dest_node_list):
    
    estimated_distance = estimate_shortest_distance( source_node, destination_node,
                                                         node_community_mapping,landmarks_degree,deg_landmarks, distance_matrix_nodes_to_landmarks,
                                                         distance_matrix_landmarks_to_nodes)
    print(f'Estimated distance between {source_node} and {destination_node} using Community based method is {estimated_distance}')


# In[ ]:





# ## Using original paper's approach

# In[19]:


def landmark_selection(graph, num_landmarks):
    # We directly find the nodes having the highest degree centrality in the graph.
    degree_centrality = nx.degree_centrality(graph)
    degree_landmarks = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:num_landmarks]
    return degree_landmarks


# In[20]:


def calculate_landmark_matrices(graph, landmarks):
    # Initialize two matrices with infinity and with highest numbered node and no. of landmarks as shapes.
    node_to_landmark_matrix = np.full((max(graph.nodes) + 1, len(landmarks)), np.inf)
    landmark_to_node_matrix = np.full((len(landmarks), max(graph.nodes)+ 1), np.inf)

    # In the below for loops, we iterate through every node in the graph and calculate it's path length to all the landmarks and also add the calculate
    # the path length landmarks to all the nodes and add it in both the matrices.
    

    for node in graph.nodes:
        for j, landmark in enumerate(landmarks):
            if nx.has_path(graph, node, landmark):
                node_to_landmark_matrix[node, j] = nx.shortest_path_length(graph, source=node, target=landmark)
                landmark_to_node_matrix[j, node] = nx.shortest_path_length(graph, source=landmark, target=node)

    return (node_to_landmark_matrix, landmark_to_node_matrix)


# In[21]:


def find_nearest_landmark_and_index_of_nearest_landmark(node, node_to_landmark_matrix):
    #  find the distance to first closest landmark from the source node.
    dist_to_nearest_landmark = min(node_to_landmark_matrix[node])  
     
    # find the index of the closest landmark
    nearest_landmark_index = np.argmin(node_to_landmark_matrix[node])
    return (dist_to_nearest_landmark,nearest_landmark_index)


# In[22]:


def estimate_shortest_path_distance(source, destination, node_to_landmark_matrix, landmark_to_node_matrix):

    # Call the function to find the nearest to landmarks from the source node.
    source_nearest_landmark_dist,indx_nearest_landmark = find_nearest_landmark_and_index_of_nearest_landmark(source, node_to_landmark_matrix)

    # Find the distance of the landmark from the source node.
    source_to_landmark_distance = node_to_landmark_matrix[source, indx_nearest_landmark]

    # Find the distance of the destination node from the landmark.
    landmark_to_destination_distance = landmark_to_node_matrix[indx_nearest_landmark, destination]

    return source_to_landmark_distance + landmark_to_destination_distance


# In[23]:


num_landmarks = len(deg_landmarks)


# In[24]:


# We obtain the highest degree centrality nodes for every community in our graph.
degree_landmarks = landmark_selection(G, num_landmarks)


# In[25]:


# We obtain the distance matrices.
node_to_landmark_matrix, landmark_to_node_matrix = calculate_landmark_matrices(G, degree_landmarks)


# In[26]:


# We call the estimate_shortest_path_distance function to obtain the distances from our source nodes to the destination nodes.
for source_node,destination_node in zip(source_node_list,dest_node_list):
    distance = estimate_shortest_path_distance(
    source_node, destination_node,
    node_to_landmark_matrix, landmark_to_node_matrix)
    print(f'Estimated distance between {source_node} and {destination_node} using original method is {distance}')


# In[ ]:





# In[27]:


for source_node,destination_node in zip(source_node_list,dest_node_list):
    print(f'The actual distance between the {source_node} and {destination_node} is {nx.shortest_path_length(G,source=source_node,target=destination_node)}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




