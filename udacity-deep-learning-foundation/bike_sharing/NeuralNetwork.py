import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                                (self.hidden_nodes, self.input_nodes))
    
        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.output_nodes, self.hidden_nodes))
        
        self.learning_rate = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs)
        # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals from hidden layer
        
        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs)
        # signals into final output layer
        final_outputs = final_inputs
        # final_outputs = self.activation_function(final_inputs)
        # signals from final output layer
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        output_errors = targets_list - final_outputs
      
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors) # errors propagated to the hidden layer
        hidden_grad =  hidden_outputs * (1-hidden_outputs)   # hidden layer gradients
             
        self.weights_hidden_to_output += self.learning_rate * np.dot(output_errors, hidden_outputs.T)   # update hidden-to-output weights 
        self.weights_input_to_hidden += self.learning_rate * np.dot(hidden_errors * hidden_grad, inputs.T)   # update input-to-hidden weights 
        
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)   # signals from hidden layer

        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs   # signals from final output layer

        return final_outputs