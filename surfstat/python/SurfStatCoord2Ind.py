import numpy as np
import sys

def py_SurfStatCoord2Ind(coord, surf):

    # if coord is a 1D array, reshape it to 2D    
    if np.ndim(coord) == 1:
        coord = coord.reshape(1, len(coord))
    
    c = np.shape(coord)[0]
    ind = np.zeros((c,1))

    if 'coord' in surf.keys():
        v = np.shape(surf['coord'])[1]
        for i in range(0, c):
            dist =  np.square(surf['coord'] - np.tile(coord[i,:], (v,1)).T)\
                    .sum(axis=0) 
            ind[i] = np.array(np.where(dist == dist.min())) 
            ind[i] = ind[i] +1   

    if 'lat' in surf.keys():
        if not 'vox' in surf:
            surf['vox'] = np.ones((1,3))
        if not 'origin' in surf.keys():
            surf['origin'] = np.zeros((1,3))
            
        i = np.around((coord[:,0] - surf['origin'][0,0]) / (surf['vox'][0,0] + \
            int(surf['vox'][0,0]==0))+1)
        j = np.around((coord[:,1] - surf['origin'][0,1]) / (surf['vox'][0,1] + \
            int(surf['vox'][0,1]==0))+1)
        k = np.around((coord[:,2] - surf['origin'][0,2]) / (surf['vox'][0,2] + \
            int(surf['vox'][0,2]==0))+1)            
        
        dim = np.shape(surf['lat'])    
        i[np.logical_or((i<1), (i>dim[0]))] = 0
        j[np.logical_or((j<1), (j>dim[1]))] = 0
        k[np.logical_or((k<1), (k>dim[2]))] = 0

        a = [m and n and l for m, n, l, in zip(i,j,k)]
        a = np.array(a, dtype=bool)
        ind = np.zeros((c,1))
        vid = np.cumsum(surf['lat'].flatten()) * surf['lat'].flatten()
        multi_ind = np.concatenate((i[a]-1, j[a]-1, k[a]-1))
        multi_ind = multi_ind.astype(int).tolist()
    
        if len(multi_ind) != 0:
            myind = np.ravel_multi_index(multi_ind, dim, order='F')           
            ind[a] = vid[myind]    

    ind = ind.flatten()
    return ind


