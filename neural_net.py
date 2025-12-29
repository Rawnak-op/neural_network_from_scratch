import numpy as np 
from pprint import pprint
#create a neural network with random weights and biases
def initialize_neural_network(num_inputs,num_hidden,num_nodes_hidden,num_nodes_outputs):
    num_previous_nodes=num_inputs
    network={}
    for layer in range(num_hidden+1):
        if layer==num_hidden:
            layer_name='output'
            num_nodes=num_nodes_outputs
        else:
            layer_name=f'hidden_{layer+1}'
            num_nodes=num_nodes_hidden[layer]

        network[layer_name]={}
        for node in range(num_nodes):
            node_name=f'node_{node+1}'
            network[layer_name][node_name]={'weights': np.around(np.random.rand(num_previous_nodes),decimals=3),
                                            'bias':np.around(np.random.rand(1),decimals=3)}
        num_previous_nodes=num_nodes 
    return network   


#compute weighted sum for a given node
def compute_weighted_sum(inputs,weights,bias):
    weighted_sum=np.sum(inputs*weights)+bias
    return weighted_sum


#node activation 
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

def forward_propagation(network,inputs):
    layer_inputs=inputs
    for layer in network.keys():#traverses each layers
        layer_outputs=[]
        for node in network[layer].keys():#traverses each nodes in the layer
            weights=network[layer][node]['weights']
            bias=network[layer][node]['bias']
            weighted_sum_node=compute_weighted_sum(layer_inputs,weights,bias)
            node_output=node_activation(weighted_sum_node)
            layer_outputs.append(np.around(node_output,decimals=3))
        layer_inputs=layer_outputs
    return layer_outputs        

#example
nn=initialize_neural_network(num_inputs=5,num_hidden=2,num_nodes_hidden=[4,5],num_nodes_outputs=2)
pprint(nn)
#output will be a dictionary representing the neural network structure with weights and biases
#generate random inputs
from random import seed
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))
predictions=forward_propagation(nn,inputs)
print(f'the prediction of the neural network is{predictions}')
    