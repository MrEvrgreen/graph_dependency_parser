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
        self.weights = list()
        self.feature_library = {"!X":0} #!X represents unknown tokens
        self.START = "!START"
        self.END = "!END"
        self.UDP_TAGS = ["ADJ", "ADP", "PUNCT", "ADV", "AUX", "SYM", "INTJ",
                         "CCONJ", "X", "NOUN", "DET", "PROPN", "NUM", "VERB",
                         "PART", "PRON", "SCONJ"]


    def _normalize(self, word):
        if '-' in word and len(word) == 1:
            return "!HYPHEN"
        elif word.isdigit() and len(word) == 4:
            return "!YEAR"
        else:
            return word.lower()        


    def extract_corpus_feats(self, filename):
        
        ROOT_INFO = ['0', '!ROOT', '!ROOT', 'ROOT', '_', '_', '_', '_', '_', '_']

        def add_new_feat(name, *args):
            feat = ' '.join( tuple([name]) + tuple(args) )
            if feat not in self.feature_library:
                self.feature_library[feat] = 0

        # open the training file
        with open(filename) as f:
            train_data = f.readlines()

        # add unigrams, bigrams, and context features (4-grams) from corpus
        for child_idx in range(len(train_data)):

            child_info = train_data[child_idx].strip().split()

            # skip line if its a compound word (-), its irrelevant (#), or empty
            if not child_info:
                continue
            elif child_info[0] == "#" or '-' in child_info[0]:
                continue
            
            # find the data index of the parent's info
            parent_idx = int(child_idx) + (int(child_info[6]) - int(child_info[0]))
            
            if child_info[6] == "0":
                parent_info = ROOT_INFO
            else:
                parent_info = train_data[parent_idx].strip().split()
            
            child_next = train_data[child_idx+1].split() if train_data[child_idx+1].strip() else self.END
            parent_next = train_data[parent_idx+1].split() if train_data[parent_idx+1].strip() else self.END
            
            try:
                child_prev = train_data[child_idx-1].strip().split() if int(child_info[0]) > 1 else self.START
            except ValueError:
                temp_line = train_data[child_idx-2].strip().split()
                if not temp_line:
                    child_prev = self.START
                elif temp_line[0] == '#':
                    child_prev = self.START
                else:
                    child_prev = temp_line
                    
            try:
                parent_prev = train_data[parent_idx-1].strip().split() if int(parent_info[0]) > 1 else self.START
            except ValueError:
                temp_line = train_data[parent_idx-2].strip().split()
                if not temp_line:
                    parent_prev = self.START
                elif temp_line[0] == '#':
                    parent_prev = self.START
                else:
                    parent_prev = temp_line
            
            # check variables print statements
            #print("\nPARENT PREV:", parent_prev)
            #print("PARENT INFO:", parent_info)
            #print("PARENT NEXT:", parent_next)
            #print("CHILD PREV:", child_prev)
            #print("CHILD INFO:", child_info)
            #print("CHILD NEXT:", child_next)
            
            # store relevant features before adding to dict
            c_word = self._normalize(child_info[1])
            c_pos_prev = child_prev[3] if child_prev != self.START else self.START
            c_pos = child_info[3]
            c_pos_next = child_next[3] if child_next != self.END else self.END
            
            p_word = self._normalize(parent_info[1])
            p_pos_prev = parent_prev[3] if parent_prev != self.START else self.START
            p_pos = parent_info[3]
            p_pos_next = parent_next[3] if parent_next != self.END else self.END

            # add unigram features
            add_new_feat("p-word p-pos", p_word, p_pos)
            add_new_feat("p-word", p_word)
            add_new_feat("p-pos", p_pos)
            add_new_feat("c-word c-pos", c_word, c_pos)
            add_new_feat("c-word", c_word)
            add_new_feat("c-pos", c_pos)
            
            # add bigram features here
            add_new_feat("p-word p-pos c-word c-pos", p_word, p_pos, c_word, c_pos)
            add_new_feat("p-pos c-word c-pos", p_pos, c_word, c_pos)
            add_new_feat("p-word c-word c-pos", p_word, c_word, c_pos)
            add_new_feat("p-word p-pos c-pos", p_word, p_pos, c_pos)
            add_new_feat("p-word p-pos c-word", p_word, p_pos, c_word)
            add_new_feat("p-word c-word", p_word, c_word)
            add_new_feat("p-pos c-pos", p_pos, c_pos)
            
            # add in-between features here
            if parent_idx < child_idx:

                for i in range(parent_idx+1, child_idx):
                    b_pos = train_data[i].strip().split()[3]

                    if b_pos in self.UDP_TAGS:
                        add_new_feat("p-pos b-pos c-pos", p_pos, b_pos, c_pos)

            elif parent_idx > child_idx:

                for i in range(child_idx+1, parent_idx):
                    b_pos = train_data[i].strip().split()[3]

                    if b_pos in self.UDP_TAGS:
                        add_new_feat("p-pos b-pos c-pos", p_pos, b_pos, c_pos)
            
            # add surrounding word POS features here
            add_new_feat("p-pos p-pos+1 c-pos-1 c-pos",
                         p_pos, p_pos_next, c_pos_prev, c_pos)
            add_new_feat("p-pos-1 p-pos c-pos-1 c-pos",
                         p_pos_prev, p_pos, c_pos_prev, c_pos)
            add_new_feat("p-pos p-pos+1 c-pos c-pos+1",
                         p_pos, p_pos_next, c_pos, c_pos_next)
            add_new_feat("p-pos-1 p-pos c-pos c-pos+1",
                         p_pos_prev, p_pos, c_pos, c_pos_next)
            
        
        self.weights = np.random.rand(len(self.feature_library))
        
        #for k, v in self.feature_library.items():
            #print(k, ":", v)


    def extract_features(self, sent_list, child_index, parent_index):
        """
        Takes a list of strings in universal dependencies format and
        two integers where the two integers represent the indices (not IDs)
        of the parent and child nodes respectively.

        Assumes that dependency heads column is not filled in.

        Add 1 to get the ID of the node since this uses computer 
        standard 0-based indexing.
        """
        child_info = sent_list[child_index].strip().split()
        parent_info = sent_list[parent_index].strip().split()

        assert child_info[0] != '#', "Not a valid child for an arc. Try again."
        assert parent_info[0] != '#', "Not a valid parent for an arc. Try again."
        assert child_index != parent_index, "Nodes cannot be self-dependent."
        
        # extract relevant information from the dependency arc
        child_prev = sent_list[child_index-1].strip().split() if int(child_info[0]) > 1 else self.START
        child_next = sent_list[child_index+1].strip().split() if child_index+1 < len(sent_list) else self.END
        parent_prev = sent_list[parent_index-1].strip().split() if int(parent_info[0]) > 1 else self.START
        parent_next = sent_list[parent_index+1].strip().split() if parent_index+1 < len(sent_list) else self.END
        
        features = copy.deepcopy(self.feature_library)
        

        def add(name, *args):
            try:
                features[' '.join( tuple([name]) + tuple(args) )] += 1
            except KeyError:
                features["!X"] += 1
        
        
        # print section for inspecting variables
        #print(child_prev)
        #print(child_info)
        #print(child_next)
        #print(parent_prev)
        #print(parent_info)
        #print(parent_next)

        # p = parent, c = child (or dependent)
        c_word = self._normalize(child_info[1])
        c_pos_prev = child_prev[3] if int(child_info[0]) > 1 else self.START
        c_pos = child_info[3]
        c_pos_next = child_next[3] if child_next != self.END else self.END

        p_word = self._normalize(parent_info[1])
        p_pos_prev = parent_prev[3] if int(parent_info[0]) > 1 else self.START
        p_pos = parent_info[3]
        p_pos_next = parent_next[3] if parent_next != self.END else self.END

        # print section for inspecting variables
        #print("CHILD:", c_word)
        #print("CHILD POS:", c_pos)
        #print("CHILD PREV POS:", c_pos_prev)
        #print("CHILD NEXT POS:", c_pos_next)
        #print("PARENT WORD:", p_word)
        #print("PARENT POS:", p_pos)
        #print("PARENT PREV POS:", p_pos_prev)
        #print("PARENT NEXT POS:", p_pos_next)
        
        # count up unigram features
        add("p-word p-pos", p_word, p_pos)
        add("p-word", p_word)
        add("p-pos", p_pos)
        add("c-word c-pos", c_word, c_pos)
        add("c-word", c_word)
        add("c-pos", c_pos)

        # count up bigram features
        add("p-word p-pos c-word c-pos", p_word, p_pos, c_word, c_pos)
        add("p-pos c-word c-pos", p_pos, c_word, c_pos)
        add("p-word c-word c-pos", p_word, c_word, c_pos)
        add("p-word p-pos c-pos", p_word, p_pos, c_pos)
        add("p-word p-pos c-word", p_word, p_pos, c_word)
        add("p-word c-word", p_word, c_word)
        add("p-pos c-pos", p_pos, c_pos)

        # count up in-between POS features
        if parent_index < child_index:

            for i in range(parent_index+1, child_index):
                b_pos = sent_list[i].strip().split()[3]

                if b_pos in self.UDP_TAGS:
                    add("p-pos b-pos c-pos", p_pos, b_pos, c_pos)

        elif parent_index > child_index:

            for i in range(child_index+1, parent_index):
                b_pos = sent_list[i].strip().split()[3]

                if b_pos in self.UDP_TAGS:
                    add("p-pos b-pos c-pos", p_pos, b_pos, c_pos)

        # count up context features
        add("p-pos p-pos+1 c-pos-1 c-pos",
            p_pos, p_pos_next, c_pos_prev, c_pos)
        add("p-pos-1 p-pos c-pos-1 c-pos",
            p_pos_prev, p_pos, c_pos_prev, c_pos)
        add("p-pos p-pos+1 c-pos c-pos+1",
            p_pos, p_pos_next, c_pos, c_pos_next)
        add("p-pos-1 p-pos c-pos c-pos+1",
            p_pos_prev, p_pos, c_pos, c_pos_next)

        return np.array(list(features.values()))
    
    
    def gen_edge_weight(self, sentence_list, child_idx, parent_idx):
        """
        Takes a list of strings and two integers for arguments where
        the list of strings is the tokenized sentence and the integers
        represent the indices of the word and head of the edge to be weighted
        in the given sentence.
        
        Returns a float value representing the weight of the target edge.
        """
        
        # generate a vector for the target edge
        edge_vec = self.extract_features(sentence_list, child_idx, parent_idx)
                    
        # perceptron summation                    
        total = sum(edge_vec * self.weights)

        # returns random feature vector of target edge as placeholder
        return total


    def gen_edge_weight_list(self, sent_list):
        """
        Takes the self and a list of strings in Universal Dependencies foormat
        for arguments where the string represents a target sentence 
        to be dependency parsed.
        
        Returns a 2D matrix where each row represents the 
        index of the child and the column represents the index of the parent.
        
        The values in the matrix represent the weight given to the 
        target parent-child combo.
        """
        # setup edge weight matrix
        edge_weights = []
        
        # loop over each word in the sentence and assign weight
        # to each possible parent-child combo
        # starts at 2 to avoid # sentences in sent_list
        for c_idx in range(2, len(sent_list)):
            
            child_info = sent_list[c_idx].strip().split()

            if child_info[0] == '#':
                continue

            for p_idx in range(2, len(sent_list)):
                
                if c_idx == p_idx:
                    continue

                edge_weights.append([c_idx, p_idx, 
                                     self.gen_edge_weight(sent_list, 
                                                          c_idx, 
                                                          p_idx)])
                        
        return edge_weights
                
        
if __name__ == "__main__":
    
    # change directory to current file's directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    # setup test sentences and perceptron
    filepath = "UD_English-ParTUT"
    training_filename = os.path.join(filepath, "en_partut-ud-train.conllu")
    
    sheep_sent = ["# sent_id = en_eg-ud-1",
                  "# text = Do androids dream of electric sheep?",
                  "1\tDo\tdo\tAUX\t_\t_\t2\taux\t_\t_",
                  "2\tandroids\tandroids\tNOUN\t_\t_\t3\tnsubj\t_\t_",
                  "3\tdream\tdream\tVERB\t_\t_\t0\troot\t_\t_",
                  "4\tof\tof\tADP\t_\t_\t3\tprep\t_\t_",
                  "5\telectric\telectric\tADJ\t_\t_\t6\tamod\t_\t_",
                  "6\tsheep\tsheep\tNOUN\t_\t_\t3\tobj\t_\t_",
                  "7\t?\t?\tPUNCT\t_\t_\t3\tpunct\t_\t_"]
    
    # testing section
    my_percep = Perceptron()
    my_percep.extract_corpus_feats(training_filename)
    print("WEIGHTS LENGTH:", my_percep.weights.shape)
    
    
    test_vec = my_percep.extract_features(sheep_sent, 2, 4)
    print("TEST VEC:", len(test_vec))
    unique, counts = np.unique(test_vec, return_counts=True)
    print("TEST VECTOR VALUE COUNTS:", dict(zip(unique, counts)))
    
    edge_weights = my_percep.gen_edge_weight_list(sheep_sent)
    
    print("\n EDGES:")
    for edge in edge_weights:
        print(edge)
    
    # train the weighter and tagger
    pass

    # test the accuracy of the weights and tagging ability
    pass