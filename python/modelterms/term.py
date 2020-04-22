import numpy as np

class term:
    def __init__(self,x=None,names=None):
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

    def __mul__(self,t):
        # Gotta double check this one - the matlab script seems a tad overcomplicated. 
        # Check that second input is also a term.
        if type(t) is not term:
            t = term(t)
        
        # Grab matrices. 
        m1 = self.matrix
        m2 = t.matrix

        m1r = np.expand_dims(m1,2)
        m2r = np.reshape(m2,(m2.shape[0],1,m2.shape[1]))

        # Compute multiplication
        out_matrix = np.reshape(m1r * m2r, (m1.shape[0],-1))

        # TO-DO: Add a check for colinear columns
        
        # Get new names.
        out_names = []
        for n1 in self.names:
            for n2 in t.names:
                out_names = out_names + [n1 + '*' + n2] 

        return term(out_matrix,out_names)

class random:
    # NOT FUNCTIONAL YET.
    def __init__(self,ran=None,fix=None,name_ran=None,name_fix=None,ranisvar=False):
        
        if ran is not None:
            if ranisvar:
                self.variance = term(ran,name_ran)
            else:
                variance = ran @ ran.t
                self.variance = term(np.flatten(variance),name_ran)
        else:
            self.variance = term()

        if fix is not None:
            self.mean = term(fix,name_fix)
        else:
            self.mean = term()