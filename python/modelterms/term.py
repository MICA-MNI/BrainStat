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
        m0 = self.matrix
        m1 = t.matrix

        # Check for duplicate columns (i.e. corr(x,y)==1))
        duplicate = check_colinearity(m0,m1,dim=0)

        # Remove duplicates from t
        mat = [m1[:,idx] for idx in range(duplicate.size) if not duplicate[idx]]
        names = [t.names[idx] for idx in range(duplicate.size) if not duplicate[idx]]
   
        mat = np.array(mat).T 
        
        # Create new term
        out_matrix = np.concatenate([m0,mat],axis=1)
        out_names = self.names + names
        
        return term(out_matrix,out_names)

    def __sub__(self,t):
        # Check that second input is also a term.
        if type(t) is not term:
            t = term(t)

        # Grab matrices. 
        m0 = self.matrix
        m1 = t.matrix

        # Check for duplicate columns (i.e. corr(x,y)==1))
        duplicate = check_colinearity(m0,m1,dim=1) # Not ==1 to account for floating point.

        # Remove occurences of t from self
        out_matrix = [m0[:,idx] for idx in range(duplicate.size) if not duplicate[idx]]
        out_matrix = np.array(out_matrix).T
        out_names = [self.names[idx] for idx in range(duplicate.size) if not duplicate[idx]]
        
        return term(out_matrix,out_names)

    def __mul__(self,t):
        # Gotta double check this one - the matlab script seems a tad overcomplicated. 
        # Check that second input is also a term.
        if type(t) is not term:
            t = term(t)
        
        # Grab matrices. 
        m0 = self.matrix
        m1 = t.matrix

        m0r = np.expand_dims(m0,2)
        m1r = np.reshape(m1,(m1.shape[0],1,m1.shape[1]))

        # Compute multiplication
        out_matrix = np.reshape(m0r * m1r, (m0.shape[0],-1))

         # Get new names.
        out_names = []
        for n1 in self.names:
            for n2 in t.names:
                out_names = out_names + [n1 + '*' + n2] 

        # Check for colinear columns
        duplicate = check_colinearity(out_matrix)
        out_matrix = [out_matrix[:,idx] for idx in range(duplicate.size) if not duplicate[idx]]
        out_matrix = np.array(out_matrix).T
        out_names = [out_names[idx] for idx in range(duplicate.size) if not duplicate[idx]]

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

def check_colinearity(m0,m1=None,dim=0):
    # Check for colinearity of columns of input matrices m0 and m1. If only one
    # matrix is provided, then it looks for colinearity of the columns of this
    # matrix instead.

    if m1 is None:
        r = m0.T @ m0 / np.linalg.norm(m0,axis=0,keepdims=1).T / np.linalg.norm(m0,axis=0,keepdims=1)
        r = np.tril(r,-1)
    else:
        r = m0.T @ m1 / np.linalg.norm(m0,axis=0,keepdims=1).T / np.linalg.norm(m1,axis=0,keepdims=1)

    colinear = np.greater(r,0.999).any(axis=dim) # Not 1 to account for floating point.

    return colinear