import numpy as np

class term:
    def __init__(self,x,names=None):
        self.matrix = np.array(x)
        if names is not None:
            self.names = names
        else:
            pass
            # (not implemented yet.)


    ## Overloaded methods
    def __add__(self,t):
        # Check that second input is also a term.
        if type(t) is not term:
            t = term(t)

        # Grab matrices. 
        m1 = self.matrix
        m2 = t.matrix

        # Check for duplicate columns (i.e. corr(x,y)==1))
        r = m1.T @ m2 / np.linalg.norm(m1,axis=0,keepdims=1).T / np.linalg.norm(m2,axis=0,keepdims=1)
        not_duplicate = np.less(r,0.999).all(axis=0) # Not 1 to account for floating point.

        # Remove duplicates from t
        mat = [m2[:,idx] for idx in range(not_duplicate.size) if not_duplicate[idx]]
        names = [t.names[idx] for idx in range(not_duplicate.size) if not_duplicate[idx]]
   
        mat = np.array(mat).T 
        
        # Create new term
        out_matrix = np.concatenate([m1,mat],axis=1)
        out_names = self.names + names
        return term(out_matrix,out_names)

    def __sub__(self,t):
        # Check that second input is also a term.
        if type(t) is not term:
            t = term(t)

        # Grab matrices. 
        m1 = self.matrix
        m2 = t.matrix

        # Check for duplicate columns (i.e. corr(x,y)==1))
        r = m1.T @ m2 / np.linalg.norm(m1,axis=0,keepdims=1).T / np.linalg.norm(m2,axis=0,keepdims=1)
        not_duplicate = np.less(r,0.999).all(axis=1) # Not 1 to account for floating point.

        # Remove occurences of t from self
        out_matrix = [m1[:,idx] for idx in range(not_duplicate.size) if not_duplicate[idx]]
        out_matrix = np.array(out_matrix).T
        out_names = [self.names[idx] for idx in range(not_duplicate.size) if not_duplicate[idx]]
        return term(out_matrix,out_names)


        


