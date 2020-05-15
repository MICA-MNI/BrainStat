import sys
import numpy as np

def py_SurfStatEdge(surf):

	if 'tri' in surf:
		tri = np.sort(surf['tri'], axis=1)
		edg =  np.unique(np.concatenate((np.concatenate((tri[:,[0, 1]], \
		                 tri[:,[0, 2]])),  tri[:,[1, 2]] )) , axis=0) 
		
	elif 'lat' in surf:
		if np.ndim(surf['lat']) > 2:
			I, J, K = np.shape(surf['lat'])
			IJ = I * J
		
			a = np.arange(1, int(I)+1, dtype='int')
			b = np.arange(1, int(J)+1, dtype='int')
			
			i, j = np.meshgrid(a,b)
			i = i.T.flatten('F'); 
			j = j.T.flatten('F');
			
			n1 = (I-1) * (J-1) * 6 + (I-1) * 3 + (J-1) * 3 + 1
			n2 = (I-1) * (J-1) * 3 + (I-1) + (J-1)

			edg = np.zeros(((K-1) * n1 + n2, int(2)), dtype='int')

			for f in range(0,2):
				
				c1  = np.where((np.remainder((i+j), 2) == f) & (i < I) & (j < J))[0] 
				c2  = np.where((np.remainder((i+j), 2) == f) & (i > 1) & (j < J))[0]
				c11 = np.where((np.remainder((i+j), 2) == f) & (i == I) & (j < J))[0]
				c21 = np.where((np.remainder((i+j), 2) == f) & (i == I) & (j > 1))[0]
				c12 = np.where((np.remainder((i+j), 2) == f) & (i < I) & (j == J))[0]
				c22 = np.where((np.remainder((i+j), 2) == f) & (i > 1) & (j == J))[0]
				
				edg0 = np.block([[ c1, c1, c1, c2-1, c2-1, c2, c11, c21-I, c12, \
				                c22-1 ], [ c1+1, c1+I, c1+1+I, c2, c2-1+I, c2-1+I, \
				                c11+I, c21, c12+1, c22 ]]).T +1
				
				edg1 = np.block([[ c1, c1, c1, c11, c11, c12, c12], [c1+IJ, c1+1+IJ, \
								c1+I+IJ, c11+IJ, c11+I+IJ, c12+IJ, c12+1+IJ ]]).T +1
				
				edg2 = np.block([[c2-1, c2, c2-1+I, c21-I, c21, c22-1, c22], \
				                [c2-1+IJ, c2-1+IJ, c2-1+IJ, c21-I+IJ, c21-I+IJ, \
				                c22-1+IJ, c22-1+IJ]]).T +1
				      
			                
				if f:
					for k in range(2, K, 2):
						edg[(k-1)*n1 + np.arange(0,n1), :]  = (np.block([[edg0], \
						 [edg2], [edg1], [IJ, 2*IJ]]) + (k-1) *IJ) 
				
						print('AAAA ', k )
						print((k-1) *IJ)
				else:
					for k in range(1,2,(K-1)):
						edg[(k-1)*n1 + np.arange(0,n1), :]  = (np.block([[edg0], \
						 [edg1], [edg2], [IJ, 2*IJ]]) + (k-1) *IJ) 
										 
				if np.remainder((K+1), 2) == f:
					edg[ (k-1)*n1 + np.arange(0,n2) -1 , :]  = \
					 edg0[np.arange(0,n2),:] + (K-1) * IJ
					
			
			vid = np.array(np.multiply(np.cumsum(surf['lat'][:].T.flatten()), \
			 surf['lat'][:].T.flatten()) , dtype='int')
			vid = vid.reshape(len(vid),1)

			all_idx = np.all(np.block([[surf['lat'].T.flatten()[edg[:,0] -1]], \
			 [surf['lat'].T.flatten()[edg[:,1] -1]]]).T, axis=1)
			 
			edg = vid[edg[all_idx,:]-1].reshape(np.shape(edg[all_idx,:]-1))
			
	else:
		sys.exit('surf must have "lat" or "tri" key !!!')

	return edg

#surf['tri'] = a

#a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
#Z = np.zeros((3,4,3))
#Z[:,:,0] = a
#Z[:,:,1] = a1          
#Z[:,:,2] = a


a = np.array([[9, 10]])
Z = np.zeros((1,2,2))
Z[:,:,0] = a
Z[:,:,1] = a 


surf = {}
surf['lat'] = Z


bla = py_SurfStatEdge(surf)

print(bla)











