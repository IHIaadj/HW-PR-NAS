from numpy.random import rand
from numpy.random import seed
from scipy.stats import kendalltau
from scipy.stats import spearmanr

ALPHA = 0.05 

'''
Compute the kendal correlation between two variables v1 & v2 
'''
def kendal_correlation(v1, v2):
    coef, p =  kendalltau(v1, v2)

    if p > ALPHA:
        print("Samples are uncorrelated (fail to reject H0)")
        return 0
    else:
        return coef 

'''
Compute the spearman correlation between two variables v1 & v2 
'''
def spearman_correlation(v1, v2):
    coef, p =  spearmanr(v1, v2)
    if p > ALPHA:
        print("Samples are uncorrelated (fail to reject H0)")
        return 0 
    else:
        return coef 

'''
Check if two variables contains ties. 
This can help us understand which one of the two rank correlation is more significant. 
Contains ties --> Spearman 
No ties --> Kendal
'''
def check_ties(v1, v2): 
    v1_set = set(v1) 
    v2_set = set(v2) 
    if len(v1_set.intersection(v2_set)) > 0: 
        return(True)  
    return(False)    