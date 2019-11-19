"""
Edge Weighting and Tagging Perceptron
"""
import sys
import random
import numpy as np
from string import punctuation

# create the perceptron class
class Perceptron:

    def __init__(self, feature_vector_length):
        """
        Takes the self and an integer for arguments where the integer
        represents the length of the feature vectors that the perceptron
        will be receiving for inputs.
        """
        self.vec_len = feature_vector_length

        # setup the weight matrix and format the incoming sentence
        self.weights = []
        
        # create the weight array and assign random weights to each edge
        self.weights = np.array( [random.random() for i in range(self.vec_len)] )
        
        # secondary option for network weights initialization
        #self.weights = np.linspace(0, 1, self.vec_len)
            
        print("INITIAL PERCEPTRON WEIGHTS:", self.weights, '\n')
    

    def remove_punc(self, inp_str):
        return inp_str.translate( str.maketrans( dict.fromkeys(punctuation) ) )


    def gen_edge_weight(self, sentence_list, word_idx, head_idx):
        """
        Takes a list of strings and two integers for arguments where the integers
        represent the indices of the target word and head of the edge to be weighted.
        
        Returns a float value representing the weight of the target edge.
        """
        
        # generate a vector for the target edge

        #for dummy in range(self.vec_len):
            # word, head = sentence_list[word_idx], sentence_list[head_idx]
            #set features of edge vector based on words in the sent_list
            #pass
        
        # generates random vector of binary values for placeholder
        self.edge_vec = np.array([random.choice([0, 1]) for i in range(self.vec_len)])
                    
        # perceptron summation
        print("EDGE VEC from {} to {}:".format(word_idx, head_idx), self.edge_vec)
                    
        self.total = sum(self.edge_vec * self.weights)

        # consider running self.total through sigmoid function here for output

        # returns random feature vector of target edge as placeholder
        return self.total


    def gen_edge_weight_matrix(self, sentence):
        """
        Takes the self and a string for arguments where the string represents a target sentence
        to be dependency parsed.

        Returns a matrix where each row represents the index of the target word and
        the column represents the index of the target head.
        
        The values in the matrix represent the weight given to the target word-head combo.
        """
        # setup edge weight matrix
        self.edge_weights = []

        # vectorize the incoming sentence
        print("ORIGINAL SENTENCE:", sentence)
        
        self.sent_list = self.remove_punc(sentence).strip().split()
        
        print("FORMATTED SENT:", self.sent_list, '\n')
        
        # loop over each word in the sentence and assign weight
        # to each possible word-head combo
        for w_idx in range(len(self.sent_list)):
            
            self.edge_weights.append([])

            for h_idx in range(len(self.sent_list)):
                
                if w_idx == h_idx:
                    self.edge_weights[w_idx].append(0)
                else:
                    self.edge_weights[w_idx].append(self.gen_edge_weight(self.sent_list, w_idx, h_idx))
        
        return self.edge_weights
                
        
if __name__ == "__main__":
    
    # setup test sentences and perceptron
    feature_vec_len = 10
    dog_sent = "The dog ran across the park."
    sheep_sent = "Do androids dream of electric sheep?"

    my_percep = Perceptron(feature_vec_len)
    
    # testing section
    out_weights = my_percep.gen_edge_weight_matrix(dog_sent)

    print("\nFINAL EDGE WEIGHT MATRIX:\n")
    for row in out_weights:
        print(row)

    
    # Import the training and test data
    pass

    # train the weighter and tagger
    pass

    # test the accuracy of the weights and tagging ability
    pass