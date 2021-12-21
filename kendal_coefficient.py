import math as m
import numpy as np
np.random.seed(1)

def kendall_coeff(metric_values,test_performances):
    set_size = len(metric_values)
    coeff = 0
    count = 0
    for m1,t1 in zip(metric_values,test_performances):
        for m2,t2 in zip(metric_values,test_performances):
            if (m1,t1)!=(m2,t2):
                coeff += m.copysign(1,m1-m2)*m.copysign(1,t1-t2)
                count += 1
    return coeff/count

def granulated_kendall_coeff(metric_values,test_performances,params):
    nb_hyperparams = len(params[0].keys())
    
    granulated_coeffs = {}
    # loop over hyperparameter axes
    for hyperparam in params[0].keys():
        
        params_seen = [0 for i in range(len(params))]
        coeffs = []
        for i,(m1,t1,param1,seen1) in enumerate(zip(metric_values,test_performances,params,params_seen)):
            if seen1==0:
                # collect params with same hyperparameter values on all but current hyperparam
                metric_values_set = [m1]
                test_performances_set = [t1]
                params_seen[i]=1
                for j,(m2,t2,param2,seen2) in enumerate(zip(metric_values,test_performances,params,params_seen)):
                    if seen2==0:
                        # check hyperparameter equivalence
                        check_sum = 0
                        for key,value in param2.items():
                            if key!=hyperparam and param1[key]==value:
                                check_sum+=1
                        if check_sum==nb_hyperparams-1:
                            metric_values_set.append(m2)
                            test_performances_set.append(t2)
                            params_seen[j]=1
                
                if len(metric_values_set)>1:
                    coeffs.append(kendall_coeff(metric_values_set,test_performances_set))
        granulated_coeffs.update({hyperparam:sum(coeffs)/len(coeffs)})
    
    return granulated_coeffs
                
        