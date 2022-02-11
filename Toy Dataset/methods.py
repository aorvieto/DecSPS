import numpy as np
from numpy import linalg as la
import random
import math

def SGD(cost, grad, hess, K, gamma, x0, batch_size, n):   

    #batches
    batch = []
    for i in range(K): batch.append(random.sample(range(n),batch_size))   

    ## initialization
    x = [x0 for i in range(K)]
    f = np.zeros((K,))
    gammas = np.zeros((K-1,))
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        gammas[k] = gamma
        x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
        f[k+1] = cost(x[k+1],range(n))
    name = 'SGD, step='+"{:.2f}".format(gamma)
    return name, f, gammas

def SGD_decr(cost,grad,hess,nexp, K_record_times, compute_hess, gamma_init, decr, th, x0, batch_size, n):
    #number of iterations
    K = K_record_times[-1]
    
    #init
    f = np.zeros((len(K_record_times),nexp))
    gammas_rec = np.zeros((len(K_record_times),nexp))
    
    ## optimization
    for e in range(nexp):
        
        #batches        
        batch = []
        for i in range(K+1): batch.append(random.sample(range(n),batch_size))    

        #iterations
        i_record = 0
        x = [x0 for i in range(K+2)]
        gammas = np.zeros((K+2,))
        for k in range(K+1):
            # stepsize selection   
            if k<th:
                gammas[k] = gamma_init
            else:
                if decr == 'sqrt':
                    gammas[k] = gamma_init/math.sqrt(k-th+1)
                else:
                    gammas[k] = (gamma_init/(k-th+1))
            
            #record
            if k==K_record_times[i_record]:
                gammas_rec[i_record,e] = gammas[k]
                f[i_record,e] = cost(x[k],range(n))
                i_record = i_record+1
            # update
            x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
           
    ## name        
    name = r'SGD, $\gamma_k='+"{:.2f}".format(gamma_init)+'/\sqrt{k+1}$'
        
    return name, f, gammas_rec

# def SPS_max(cost,grad,hess,nexp, K_record_times, compute_hess, c, gamma_max, x0, batch_size, n):
#     #number of iterations
#     K = K_record_times[-1]
    
#     #init
#     f = np.zeros((len(K_record_times),nexp))
#     mus = np.zeros((len(K_record_times),nexp))
#     Ls = np.zeros((len(K_record_times),nexp))
#     gammas_rec = np.zeros((len(K_record_times),nexp))
    
#     ## optimization
#     for e in range(nexp):
        
#         #batches        
#         batch = []
#         for i in range(K+1): batch.append(random.sample(range(n),batch_size))    

#         #iterations
#         i_record = 0
#         x = [x0 for i in range(K+2)]
#         gammas = np.zeros((K+2,))

#         for k in range(K+1):

#             sps_grad = cost(x[k],batch[k])/la.norm(grad(x[k],batch[k]))**2
#             gammas[k] = min([sps_grad/c,gamma_max])

#             #record
#             if k==K_record_times[i_record]:
#                 gammas_rec[i_record,e] = gammas[k]
#                 f[i_record,e] = cost(x[k],range(n))
#                 if compute_hess:
#                     mus[i_record,e],Ls[i_record,e] = hess(x[k]) 
#                 i_record = i_record+1
                    
#             # update
#             x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
           
#     ## name            
#     name = r'SPS$_max$, $c='+"{:.2f}".format(c+', \gamma_{\max}='+"{:.2f}".format(gamma_max)+'$'
    
#     return name, f, gammas_rec, mus, Ls


def SPS_decr(cost,grad,hess,nexp, K_record_times, compute_hess, c_init, decr, gamma_max, x0, batch_size, n):
    #number of iterations
    K = K_record_times[-1]
    
    #init
    f = np.zeros((len(K_record_times),nexp))
    gammas_rec = np.zeros((len(K_record_times),nexp))
    
    ## optimization
    for e in range(nexp):
        
        #batches        
        batch = []
        for i in range(K+1): batch.append(random.sample(range(n),batch_size))    

        #iterations
        i_record = 0
        x = [x0 for i in range(K+2)]
        gammas = np.zeros((K+2,))
        c = np.zeros((K+2,))

        for k in range(K+1):

            sps_grad = cost(x[k],batch[k])/la.norm(grad(x[k],batch[k]))**2
            if k==0:
                c[0] = c_init
                gammas[0] = min([sps_grad,c[0]*gamma_max])/c[0]
            else:
                if decr == 'sqrt':
                    c[k] = c_init*np.sqrt(k+1)
                else:
                    c[k] = c_init*(k+1)
                
                gammas[k] = min([sps_grad, c[k-1]*gammas[k-1]])/c[k]
            
            #record
            if k==K_record_times[i_record]:
                gammas_rec[i_record,e] = gammas[k]
                f[i_record,e] = cost(x[k],range(n))
                i_record = i_record+1   
            # update
            x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
           
    ## name            
    name = r'DecSPS, $c_0='+"{:.2f}".format(c_init)+', \gamma_{b}='+"{:.0f}".format(gamma_max)+'$'
    
    return name, f, gammas_rec

def AdaNorm(cost,grad,hess,nexp, K_record_times, compute_hess, eta, b0, decr, th, x0, batch_size, n):
    #number of iterations
    K = K_record_times[-1]
    
    #init
    f = np.zeros((len(K_record_times),nexp))
    gammas_rec = np.zeros((len(K_record_times),nexp))
    
    ## optimization
    for e in range(nexp):
        
        #batches        
        batch = []
        for i in range(K+1): batch.append(random.sample(range(n),batch_size))    

        #iterations
        i_record = 0
        x = [x0 for i in range(K+2)]
        gammas = np.zeros((K+2,))
        for k in range(K+1):
            
            # stepsize selection   
            if k ==0:
                gammas[k] = 1/b0
            else:
                gammas[k] = 1/np.sqrt(1/gammas[k-1]**2 + la.norm(grad(x[k],batch[k]))**2)
                
            #record
            if k==K_record_times[i_record]:
                gammas_rec[i_record,e] = gammas[k]
                f[i_record,e] = cost(x[k],range(n))
                i_record = i_record+1
                
            # update
            x[k+1] = x[k] - eta*gammas[k]*grad(x[k],batch[k])
           
    ## name        
    name = 'AdaNorm, $b_0='+"{:.2f}".format(b0)+', \eta='+"{:.1f}".format(eta)+'$'
        
    return name, f, eta*gammas_rec





  

