"""
Edge Weighting and Tagging Perceptron
"""
import os
import sys
import copy
import random
import numpy as np
from string import punctuation

# create the perceptron class
class Perceptron:

    def __init__(self):
        """
        Takes the self for an argument.
        """
        # create the weight array and other necessary variables
        self.weights = []
        self.trained = False
        self.UDP_TAGS = ["ADJ", "ADP", "PUNCT", "ADV", "AUX", "SYM", "INTJ", 
                         "CCONJ", "X", "NOUN", "DET", "PROPN", "NUM", "VERB",
                         "PART", "PRON", "SCONJ"]
    

    def remove_punc(self, inp_str):
        return inp_str.translate( str.maketrans( dict.fromkeys(punctuation) ) )


    def format_sentence(self, sentence):
        return self.remove_punc(sentence).strip().split()


    def _get_context_feats(self, child_data_idx, child_id, parent_id, train_data):
        
        # calculate the data index of the parent node based on its ID
        parent_data_idx = int(child_data_idx) + (int(parent_id) - int(child_id))
        
        
        child_pos = train_data[ child_data_idx ].strip().split()[3]
        
        if parent_id == "0":
            parent_pos = None
            parent_l_neighbor_pos = None
                
        else:
            # if the parent is a compound word, ignore it and continue
            temp = train_data[ parent_data_idx ].strip().split()
            if '-' in temp[0]:
                parent_pos = train_data[ parent_data_idx+1 ].strip().split()[3]
            else:
                parent_pos = train_data[ parent_data_idx ].strip().split()[3]

                
            # check before accessing parent's left neighbor POS
            temp_data = train_data[parent_data_idx-1].strip().split()
            if len( temp_data ) > 2 and temp_data[0] != '#':
                parent_l_neighbor_pos = temp_data[3]
            else:
                parent_l_neighbor_pos = None

            #if parent_pos not in self.UDP_TAGS:
                #print("PROBLEMATIC TEMP DATA:", train_data[parent_data_idx], "PARENT ID:", parent_id)

        # check before accessing parent's right neighbor POS
        temp_data = train_data[parent_data_idx+1].strip().split()
        if len( temp_data ) > 2 and temp_data[0] != '#':
            parent_r_neighbor_pos = temp_data[3]
        else:
            parent_r_neighbor_pos = None
        
        # check before accessing child's left neighbor POS
        temp_data = train_data[child_data_idx-1].strip().split()
        if len( temp_data ) > 2 and temp_data[0] != '#':
            child_l_neighbor_pos = temp_data[3]
        else:
            child_l_neighbor_pos = None
        
        # check before accessing child's right neighbor POS
        temp_data = train_data[child_data_idx+1].strip().split()
        if len( temp_data ) > 2:
            child_r_neighbor_pos = temp_data[3]
        else:
            child_r_neighbor_pos = None

        return [parent_l_neighbor_pos, parent_pos, parent_r_neighbor_pos,
                child_l_neighbor_pos, child_pos, child_r_neighbor_pos]


    def get_features(self, filename):
        
        # !DIGITS represents numbers
        # !X represents unknown words (not in corpus)
        self.vocab_list = {"!DIGITS": 0, "!X": 0}
        self.bigram_dict = {}
        self.context_feat_list_one = {}
        self.context_feat_list_two = {}
        self.context_feat_list_three = {}
        self.context_feat_list_four = {}
        self.trained = True
        
        # open the training file
        with open(filename) as f:
            train_data = f.readlines()

        # add unigrams, bigrams, and context features (4-grams) from corpus
        for i in range(len(train_data)):

            # UDP FORMAT:
            # [index, word, lemma, POS, xPOS, feats, head, relation, deps, misc]
            current_line = train_data[i].strip().split()

            # skip line if its a compound word, its irrelevant (#), or empty
            if not current_line:
                continue
            elif current_line[0] == "#" or '-' in current_line[0]:
                continue
            
            # pull out unigrams, bigrams, and special features
            try:
                next_line = train_data[i+1].strip().split()
            except IndexError:
                next_line = None
            

            # add unigrams
            if current_line[1].lower() not in self.vocab_list:
                self.vocab_list[ current_line[1].lower() ] = 0

            # add bigrams
            if not next_line:
                pass
            else:
                self.bigram_dict[ ( current_line[1].lower(), next_line[1].lower() ) ] = 0

            # add context feature combinations
            context_feats = self._get_context_feats(i, current_line[0], current_line[6], train_data)
            
            feat_one = tuple([context_feats[1], context_feats[2], context_feats[3], context_feats[4]])
            feat_two = tuple([context_feats[0], context_feats[1], context_feats[3], context_feats[4]])
            feat_three = tuple([context_feats[1], context_feats[2], context_feats[4], context_feats[5]])
            feat_four = tuple([context_feats[0], context_feats[1], context_feats[4], context_feats[5]])
            
            if feat_one not in self.context_feat_list_one:
                self.context_feat_list_one[ feat_one ] = 0
            if feat_two not in self.context_feat_list_two:
                self.context_feat_list_two[ feat_two ] = 0
            if feat_three not in self.context_feat_list_three:
                self.context_feat_list_three[ feat_three ] = 0
            if feat_four not in self.context_feat_list_four:
                self.context_feat_list_four[ feat_four ] = 0

        print("\nVOCAB LEN:", len(self.vocab_list))
        print("BIGRAM LEN:", len(self.bigram_dict))
        print("SPECIAL FEATURES LEN:", len(self.context_feat_list_one) + \
            len(self.context_feat_list_two) + \
            len(self.context_feat_list_three) + \
            len(self.context_feat_list_four))

        # calculate the total vector len based on extracted feature vectors 
        self.total_vec_len = len(self.vocab_list)*2 + 1 + \
            len(self.UDP_TAGS)*2 + len(self.bigram_dict) +\
            len(self.context_feat_list_one) + \
            len(self.context_feat_list_two) + \
            len(self.context_feat_list_three) + \
            len(self.context_feat_list_four)
        
        # add one more to total vec len for feat representing distance
        # between parent and child in sentence
        self.total_vec_len += 1; print("TOTAL VEC LEN:", self.total_vec_len)

        self.weights = [ 0 for i in range(self.total_vec_len) ]
    
    
    def extract_features(self, sent_list, child_index, parent_index):
        """
        Takes a list of strings and two integers for arguments where
        the list of strings is the dependencies formatted sentence 
        and the integers represent the indices of the word and head
        respectively for the edge to be weighted in the given sentence.
        
        # example:
        # final example for sparse feature representation of arc:
        # [ word1vec + direction + word2vec + word1posvec + word2posvec + \
        #   bigram_feats + context_feats + dist_between ]
        # 
        # p = parent(head), c = child(word)
        # dist_between = dist between child and parent
        #
        # the---->dog      [1, 0, 0, 0...]
        
        UDP POS TAGS:
        ADJ	ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ
        """
        # setup the feature vector and format the input sentence
        self.feat_vec = []; print("ORIGINAL FEAT VEC:", self.feat_vec) # delete later
        self.parent_info = sent_list[parent_index].split()
        self.child_info = sent_list[child_index].split()
        
        # insert feature extraction here
        print("WORD:", self.child_info[1],
              '\nHEAD:', self.parent_info[1])
        
        # establish child, parent, and direction vectors
        self.child_vec = copy.deepcopy(self.vocab_list)
        self.child_vec[ self.child_info[1].lower() ] = 1
        self.feat_vec = self.feat_vec + list(self.child_vec.values()); print("FEAT VEC W/ CHILD VEC", len(self.feat_vec)) # delete later

        self.direction = 1 if parent_index < child_index else 0
        
        self.parent_vec = copy.deepcopy(self.vocab_list)
        self.parent_vec[ self.parent_info[1].lower() ] = 1
        self.feat_vec = self.feat_vec + list(self.parent_vec.values()); print("FEAT VEC W/ PARENT VEC:", len(self.feat_vec))

        # establish POS vectors for parent and child
        self.childpos_vec = [ 0 for i in range(len(self.UDP_TAGS)) ]
        self.childpos_vec[ self.UDP_TAGS.index( self.child_info[3] ) ] = 1
        self.feat_vec = self.feat_vec + self.childpos_vec; print("FEAT VEC W/ CHILDPOS VEC", len(self.feat_vec))

        self.parentpos_vec = [ 0 for i in range(len(self.UDP_TAGS)) ]
        self.parentpos_vec[ self.UDP_TAGS.index(self.parent_info[3]) ] = 1
        self.feat_vec = self.feat_vec + self.parentpos_vec; print("FEAT VEC W/ PARENTPOS VEC", len(self.feat_vec))

        # establish bigram features
        self.bigram_feats = copy.deepcopy(self.bigram_dict)
        self.temp_bigram = (self.parent_info[1].lower(),
                            self.child_info[1].lower() )
        if self.temp_bigram in self.bigram_dict:
            self.bigram_feats[self.temp_bigram] = 1

        self.feat_vec = self.feat_vec + list(self.bigram_dict.values())

        # establish context features #
        # [parent_l_neighbor_pos, parent_pos, parent_r_neighbor_pos,
        #  child_l_neighbor_pos, child_pos, child_r_neighbor_pos]
        try:
            parent_l_neighbor_pos = sent_list[parent_index-1].split()[3]
        except IndexError:
            parent_l_neighbor_pos = None

        parent_pos = self.parent_info[3]

        try:
            parent_r_neighbor_pos = sent_list[parent_index+1].split()[3]
        except IndexError:
            parent_r_neighbor_pos = None

        try:
            child_l_neighbor_pos = sent_list[child_index-1].split()[3]
        except IndexError:
            child_l_neighbor_pos = None
        
        child_pos = self.child_info[3]

        try:
            child_r_neighbor_pos = sent_list[child_index+1].split()[3]
        except IndexError:
            child_r_neighbor_pos = None

        feat_one = (parent_pos, parent_r_neighbor_pos, child_l_neighbor_pos, child_pos)
        feat_two = (parent_l_neighbor_pos, parent_pos, child_l_neighbor_pos, child_pos)
        feat_three = (parent_pos, parent_r_neighbor_pos, child_pos, child_r_neighbor_pos)
        feat_four = (parent_l_neighbor_pos, parent_pos, child_pos, child_r_neighbor_pos)
        
        temp_one = copy.deepcopy(self.context_feat_list_one)
        temp_two = copy.deepcopy(self.context_feat_list_two)
        temp_three = copy.deepcopy(self.context_feat_list_three)
        temp_four = copy.deepcopy(self.context_feat_list_four)

        if feat_one in temp_one:
            temp_one[feat_one] = 1
        if feat_two in temp_two:
            temp_two[feat_two] = 1
        if feat_three in temp_three:
            temp_three[feat_three] = 1
        if feat_four in temp_four:
            temp_four[feat_four] = 1
        
        self.feat_vec = self.feat_vec + list(temp_one.values()) + \
            list(temp_two.values()) + \
            list(temp_three.values()) +\
            list(temp_four.values())

        print("TEST:", self.parent_info[0], self.child_info[0], "RESULT:", str( abs( int(self.parent_info[0]) - int(self.child_info[0]) ) ) )        
        self.feat_vec.append( abs( int(self.parent_info[0]) - int(self.child_info[0]) ) )

        return self.feat_vec
    
    
    def gen_edge_weight(self, sentence_list, child_idx, parent_idx):
        """
        Takes a list of strings and two integers for arguments where
        the list of strings is the tokenized sentence and the integers
        represent the indices of the word and head of the edge to be weighted
        in the given sentence.
        
        Returns a float value representing the weight of the target edge.
        """
        
        # generate a vector for the target edge
        # under construction

        # UPDATE FINAL TWO ARGUMENTS TO CORRECT POS's WHEN FIXING FUNCTION
        self.edge_vec = self.extract_features(sentence_list, child_idx, parent_idx)
                    
        # perceptron summation
        print("EDGE VEC from {} to {}:".format(child_idx, parent_idx), self.edge_vec)
                    
        self.total = sum(self.edge_vec * self.weights)

        # returns random feature vector of target edge as placeholder
        return self.total


    def gen_edge_weight_matrix(self, sentence):
        """
        Takes the self and a string for arguments where the string represents 
        a target sentence to be dependency parsed.

        Returns a matrix where each row represents the index of the target word and
        the column represents the index of the target head.
        
        The values in the matrix represent the weight given to the target word-head combo
        """
        # setup edge weight matrix
        self.edge_weights = []

        # vectorize the incoming sentence
        print("ORIGINAL SENTENCE:", sentence)
        
        self.sent_list = self.format_sentence(sentence)
        
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
    
    # change directory to current file's directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # setup test sentences and perceptron
    filepath = "UD_English-ParTUT"
    training_filename = os.path.join(filepath, "en_partut-ud-train.conllu")
    
    sheep_sent = ["1\tDo\tdo\tAUX\t_\t_\t2\taux\t_\t_",
                  "2\tandroids\tandroids\tNOUN\t_\t_\t3\tnsubj\t_\t_",
                  "3\tdream\tdream\tVERB\t_\t_\t0\troot\t_\t_",
                  "4\tof\tof\tADP\t_\t_\t3\tprep\t_\t_",
                  "5\telectric\telectric\tADJ\t_\t_\t6\tamod\t_\t_",
                  "6\tsheep\tsheep\tNOUN\t_\t_\t3\tobj\t_\t_",
                  "7\t?\t?\tPUNCT\t_\t_\t3\tpunct\t_\t_"]
    
    my_percep = Perceptron()
    my_percep.get_features(training_filename)
    
    # testing section
    test_vec = my_percep.extract_features(sheep_sent, 4, 5)
    print("TEST VEC:", len(test_vec))
    print("ONES:", test_vec.count(1))


    #out_weights = my_percep.gen_edge_weight_matrix(dog_sent)
    
    #print("\nFINAL EDGE WEIGHT MATRIX:\n")
    #for row in out_weights:
        #print(row)
    
    # train the weighter and tagger
    pass

    # test the accuracy of the weights and tagging ability
    pass