'''
Measures of intraclass clustering ability and generalization
'''
import sys
sys.path.insert(0, "../")

import warnings

import numpy as np

from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from keras import losses
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Activation
from keras.constraints import Constraint
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, Callback
from keras.engine.training_arrays import predict_loop, test_loop
from keras.preprocessing.image import ImageDataGenerator

from utils_training import history_todict, lr_schedule

def model_extract_tensors(model,input_data,tensors,batch_size=128, training_phase = 0):
    input_tensors = [model.inputs[0], # input data
                     K.learning_phase()] # train or test mode
    
    f = K.function(inputs=input_tensors, outputs=tensors)
   
    # last element of inputs is not sliced in batches thanks to keras :)
    inputs = [input_data, training_phase] 
    outputs = predict_loop(model,f, inputs, batch_size = batch_size, verbose = 0)
    return outputs

def collect_activations(model,x,batch_size = 128,training_phase = 0,preact=False):
    # collect activation layers
    relu_outputs = []
    for layer in model.layers[:-1]:
        if ('relu' in layer.name) or isinstance(layer,Activation) or isinstance(layer,ControllableReLU):
            representation = layer.input if preact else layer.output
            if len(layer.input_shape)==2:
                relu_outputs.append(representation)
            elif len(layer.input_shape)==4:
                relu_outputs.append(K.max(representation,axis = [1,2])) # global max pooling
    
    # extract relu activations
    activations = model_extract_tensors(model,x,relu_outputs,batch_size=batch_size,training_phase = training_phase)
    if not isinstance(activations,list):
        activations = [activations]
    return activations

def evaluate_in_training_mode(model,x,y,sample_weights = None, batch_size = 128, verbose = 0):
    if sample_weights == None:
        sample_weights = np.ones((x.shape[0],),np.float32)
    ins = [x, y, sample_weights, 1]
    model._make_test_function()
    f = model.test_function
    return test_loop(model, f, ins,
                     batch_size=batch_size,
                     verbose=verbose)

def blackbox_subclass(model,x,y,suby, batch_size = 128,training_phase = 0,data_subset = 1.,agg = 'max'):
    '''
    measure c_0
    '''
    if data_subset <1.: # use subset of the data to estimate metric
        subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
        x,y,suby = x[subset],y[subset],suby[subset]
        
    # class - subclass correspondence (nbclasses,nbsubclasses)
    # 1: all samples from a subclass are in a given class, 0: no samples from a subclass are in a given class
    correspondence = np.dot(y.T,suby)/suby.sum(axis = 0)
    
    metric_per_subclass = []
    for subclass_index in range(suby.shape[1]):
        class_index = np.argmax(correspondence[:,subclass_index])
        
        samples_subclass = suby[:,subclass_index].astype(bool)
        # selects samples from the class to which the subclass belongs
        samples_class = y[:,class_index]
        # remove samples from the subclass
        samples_class = (samples_class-samples_class*samples_subclass).astype(bool)
                
        x_subclass = x[samples_subclass]
        x_subclass_shuffled = x_subclass[np.random.permutation(len(x_subclass))]
        x_class = x[samples_class]
        x_class = x_class[np.random.permutation(len(x_class))[:len(x_subclass)]]
        
        scores = []
        for x1,x2 in [(x_subclass,x_subclass_shuffled),(x_subclass,x_class)]:
            interpolation_factors = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            outs = []
            for factor in interpolation_factors:
                # interpolate between samples from x1 and x2
                preds = model.predict(x1*factor + x2*(1-factor), batch_size=batch_size)
                # For each interpolation point, record prediction on the correct class
                outs.append(preds[:,class_index])
            
            if agg == 'max':
                # for each pair of samples, compute maximum deviation from perfect prediction along the interpolation points
                # average over all pairs of samples
                scores.append(np.mean(np.max(1-np.array(outs),axis = 0)))
            elif agg == 'sum':
                scores.append(np.mean(np.sum(1-np.array(outs),axis = 0)))
            else:
                raise ValueError('agg argument wrongly specified. Should be either max or sum.')
                
        # compare results for pairs of samples inside the same subclass, versus
        # pairs of samples from different subclasses (but still in the same class)
        metric_per_subclass.append(scores[1]/scores[0])
        
    return np.median(np.array(metric_per_subclass))

def neural_subclass_selectivity(model,x,y,suby, batch_size = 128,training_phase = 0, 
                                layerwise = False,subclass_agg='median', neuron_agg ='max',data_subset = 1.,preact=True):
    '''   
    measure c_1
    '''
    if data_subset <1.: # use subset of the data to estimate metric
        subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
        x,y,suby = x[subset],y[subset],suby[subset]
        
    # collect activations
    activations = collect_activations(model,x,batch_size,training_phase,preact=preact)
        
    # class - subclass correspondence (nbclasses,nbsubclasses)
    # 1: all samples from a subclass are in a given class, 0: no samples from a subclass are in a given class
    correspondence = np.dot(y.T,suby)/suby.sum(axis = 0)
    
    subclass_selectivity = []
    for subclass in range(suby.shape[1]):
        samples_subclass = suby[:,subclass].astype(bool)
        
        # selects samples from the class to which the subclass belongs
        samples_class = y[:,np.argmax(correspondence[:,subclass])]
        # remove samples from the subclass
        samples_class = (samples_class-samples_class*suby[:,subclass]).astype(bool)
        
        subclass_selectivity_neurons = []
        for layer,acts in enumerate(activations):
            mean_subclass = np.mean(acts[samples_subclass],axis = 0) 
            std_subclass = np.std(acts[samples_subclass],axis = 0)
            
            mean_class = np.mean(acts[samples_class],axis = 0) 
            std_class = np.std(acts[samples_class],axis = 0)

            selectivity = (mean_subclass-mean_class) / (std_subclass + std_class+1e-7)
            
            # ignore dead neurons
            selectivity = selectivity*(1-np.all(acts<0.,axis = 0))
    
            subclass_selectivity_neurons.append(selectivity)
        
        if not layerwise:
            # concatenate neurons of all layers
            subclass_selectivity_neurons = [np.concatenate(subclass_selectivity_neurons)]
           
        if neuron_agg == 'max':
            # max over neurons
            subclass_selectivity.append([np.max(l) for l in subclass_selectivity_neurons])
        if neuron_agg == 'topk':
            # mean of topk neurons. k is such that (nb_neurons/nb_subclasses) neurons are selected
            subclass_selectivity.append([])
            for l in subclass_selectivity_neurons:
                k = max(round(len(l)/suby.shape[1]),1) # k should be at least 1
                subclass_selectivity[-1].append( np.mean(np.partition(l,-k)[-k:]) ) # mean of top k
            
    # dimensions should be (nb_subclasses, nb_layers) if layerwise or (nb_subclasses,1) if not layerwise
    subclass_selectivity = np.array(subclass_selectivity)
    
    if subclass_agg == 'mean':
        selectivity = np.mean(subclass_selectivity,axis=0)
    elif subclass_agg == 'median':
        selectivity = np.median(subclass_selectivity,axis=0)
    elif subclass_agg == 'max':
        selectivity = np.max(subclass_selectivity,axis=0)
    
    return selectivity

def layer_subclass_clustering(model,x,y,suby, batch_size = 128,training_phase = 0, data_subset = 1.,layerwise = False,subclass_agg='median',preact=False):
    '''       
    measure c_2
    '''
    if data_subset <1.: # use subset of the data to estimate metric
        subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
        x,y,suby = x[subset],y[subset],suby[subset]
        
    # collect activations
    activations = collect_activations(model,x,batch_size,training_phase,preact=preact)
    if preact:
        for i,act in enumerate(activations):
            act = (act-np.mean(act,axis=0)) / np.std(act,axis = 0)
            
            # percentile is computed such that at least 10 neurons are activated by each sample in each layer on average
            percentile = min(round(100-100*10/act.shape[1]) , 75)
            thres = np.percentile(act,percentile,axis = 0,keepdims = True)

            activations[i] = np.maximum(act-thres,0)
        
    # class - subclass correspondence (nbclasses,nbsubclasses)
    # 1: all samples from a subclass are in a given class, 0: no samples from a subclass are in a given class
    correspondence = np.dot(y.T,suby)/suby.sum(axis = 0)
    
    subclass_clustering_per_layer = []
    for layer,acts in enumerate(activations):        
        subclass_clustering_per_layer.append([])
        for c in range(y.shape[1]):
            samples_class = y[:,c].astype(bool)
            
            # provides a silhouette score per sample
            score = silhouette_samples(acts[samples_class],
                                       np.where(suby[samples_class][:,(correspondence[c]>0.).astype(bool)])[1],
                                       metric='cosine')

            for subclass in np.where(correspondence[c]>0.)[0]:
                # compute mean silhouette score for each subclass
                subclass_clustering_per_layer[-1].append(np.mean(score[suby[samples_class][:,subclass].astype(bool)]))
                
    # dimensions should be (nb_layers, nb_subclasses)
    subclass_clustering_per_layer = np.array(subclass_clustering_per_layer)
    
    if not layerwise:
        # max over layers
        subclass_clustering = np.max(subclass_clustering_per_layer,axis = 0)
    else:
        subclass_clustering = subclass_clustering_per_layer
            
    if subclass_agg == 'mean':
        subclass_clustering = np.mean(subclass_clustering,axis = -1)
    elif subclass_agg == 'median':
        subclass_clustering = np.median(subclass_clustering,axis = -1)
    elif subclass_agg == 'max':
        subclass_clustering = np.max(subclass_clustering,axis = -1)
    
    return subclass_clustering

def neural_intraclass_selectivity(model,x,y,batch_size = 128,training_phase = 0, data_subset = 1.,layerwise = False,subclass_agg='mean',preact=True, k_neuron=None,not_all = False): 
    '''
    measure c_3
    '''
    
    if data_subset <1.: # use subset of the data to estimate metric
        subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
        x,y = x[subset],y[subset]
        
    activations = collect_activations(model,x,batch_size=batch_size,training_phase=training_phase,preact=preact)
    
    # pre-compute neuron-wise std on the data
    stds_all = []
    for layer,acts in enumerate(activations):
        stds_all.append(np.std(acts,axis = 0))
        
    # compute neural selectivity for each layer
    subclass_selectivity = []
    for c in range(y.shape[1]):
        # selects samples from the class
        samples_class = y[:,c].astype(bool)

        subclass_selectivity_layer = []
        for layer,acts in enumerate(activations):
#             mean_class = np.mean(acts[samples_class],axis = 0)
            std_class = np.std(acts[samples_class],axis = 0)
            if not_all:
                std_all = np.std(acts[(1-samples_class).astype(bool)],axis = 0)
            else:
                std_all = stds_all[layer]
            
            selectivity = std_class / (std_all+1e-7)
            
            # ignore dead neurons
            selectivity = selectivity*(1-np.all(acts<0.,axis = 0))

            subclass_selectivity_layer.append(selectivity)

        if not layerwise:
            # concatenate neurons of all layers
            subclass_selectivity_layer = [np.concatenate(subclass_selectivity_layer)]
            
        # mean of topk neurons. 
        subclass_selectivity.append([])
        for l in subclass_selectivity_layer:
            if k_neuron == None:
                # k_neuron is such that (nb_neurons/nb_classes) neurons are selected
                k_neuron = max(round(len(l)/y.shape[1]),1) # k should be at least 1
            subclass_selectivity[-1].append( np.mean(np.partition(l,-k_neuron)[-k_neuron:]) ) # mean of top k
    
    # dimensions should be (nb_subclasses, nb_layers) if layerwise or (nb_subclasses,1) if not layerwise
    subclass_selectivity = np.array(subclass_selectivity)

    if subclass_agg == 'mean':
        selectivity = np.mean(subclass_selectivity,axis=0)
    elif subclass_agg == 'median':
        selectivity = np.median(subclass_selectivity,axis=0)
    elif subclass_agg == 'max':
        selectivity = np.max(subclass_selectivity,axis=0)
    
    return selectivity

def layer_intraclass_clustering(model,x,y,batch_size = 128,training_phase = 0, data_subset = 1.,layerwise = False,subclass_agg='mean',preact=True,k_layer = 1):
    '''       
    measure c_4
    '''
#     if data_subset <1.: # use subset of the data to estimate metric
    subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
#         x,y = x[subset],y[subset]
    
    # collect activations
    activations = collect_activations(model,x,batch_size=batch_size,training_phase=training_phase,preact=preact)
    for i,act in enumerate(activations):
        # ignore dead neurons
        act = act[:,~np.all(act<0.,axis = 0)]
        
        act = (act-np.mean(act,axis=0)) / (np.std(act,axis = 0)+1e-7)
        
        if act.shape[1]!=0:
            percentile = max(min(round(100-100*10/act.shape[1]) , 75),0)
            thres = np.percentile(act,percentile,axis = 0,keepdims = True)

            activations[i] = np.maximum(act-thres,0)
        else:
            del activations[i]
            
    subclass_clustering_per_layer = []
    for layer,acts in enumerate(activations):  
        dists = cosine_distances(acts[subset])
        std_all = np.std(dists)
#         mean_all = np.mean(dists)
        
        subclass_clustering_per_layer.append([])
        for c in range(y.shape[1]):
            samples_class = y[:,c].astype(bool)
            
            dists = cosine_distances(acts[samples_class])
            std_class = np.std(dists)
#             mean_class = np.mean(dists)
            
            selectivity = std_class / (std_all+1e-7)
            subclass_clustering_per_layer[-1].append(selectivity)
                
    # dimensions should be (nb_layers, nb_classes)
    subclass_clustering_per_layer = np.array(subclass_clustering_per_layer)
    
    if not layerwise:
        # mean over topk layers
        subclass_clustering = np.mean(np.sort(subclass_clustering_per_layer,axis = 0)[-k_layer:,:],axis=0)
    else:
        subclass_clustering = subclass_clustering_per_layer
            
    if subclass_agg == 'mean':
        subclass_clustering = np.mean(subclass_clustering,axis = -1)
    elif subclass_agg == 'median':
        subclass_clustering = np.median(subclass_clustering,axis = -1)
    elif subclass_agg == 'max':
        subclass_clustering = np.max(subclass_clustering,axis = -1)
    
    return subclass_clustering

def sharpness_random(model,x,y, data_subset = .1, epsilon_weight_scale = 1e-3, nb_samplings = 10, 
                     kernel_only = False, training_phase = 0, batch_size = 300):
    '''
    training_phase=1 is useful to use batchstatistics with batchnorm. But be careful to remove dropout layers!
    
    code adapted from NeurIPS "predicting generalization in deep learning" competition starting kit
    https://competitions.codalab.org/competitions/25301 
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.),
                  metrics=['accuracy'])
    
    # collect trainable weights and their original values
    weights = model.trainable_weights
    weights_orig = [K.get_value(w) for w in weights]
    
    # m represents the bounds for the weights perturbation
    # m will be optimized such that optimizing within these bounds reaches the target deviate
    # for this optimization, h and l represent high and low tentative values of m and a bisectional method is used
    h, l = 2.0, 0.000000
    target_accuracy = 0.9
    for i in range(20): # loop to find perturbation scale 
        m = (h + l) / 2. # m fixes the bounds for the weight perturbation
        accuracy = 0.
        for k in range(nb_samplings): # loop to estimate accuracy of perturbed model given a perturbation scale
            for w,w_orig in zip(weights,weights_orig):
                if not kernel_only or len(w_orig.shape)>1.: # kernels are assumed to be the only weights with more than one dimension
                    noisy = w_orig + np.random.normal(0.,scale = m, size=list(w_orig.shape)) * (np.abs(w_orig)+epsilon_weight_scale)
                    K.set_value(w,noisy)
            
            # use subset of the data to estimate accuracy (a different subset is used for every estimation)
            subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
            if training_phase==1:
                estimate_accuracy = evaluate_in_training_mode(model,x[subset],y[subset],verbose = 0, batch_size = batch_size)[1]
            elif training_phase==0:
                estimate_accuracy = model.evaluate(x[subset],y[subset],verbose = 0, batch_size = batch_size)[1]
            accuracy += estimate_accuracy
        accuracy /=nb_samplings

        if h - l < 1e-5 or abs(accuracy - target_accuracy) < 5e-3:
            break
        if accuracy < target_accuracy:
            h = m
        else:
            l = m
    # reset original weight values
    for w,w_orig in zip(weights,weights_orig):
        K.set_value(w,w_orig)
    return m, accuracy - target_accuracy

class Clip(Constraint):
    """Element-wise clipping of weight tensors. Upper ad lower bounds are tensors of same shape as the weights"""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, w):
        return K.minimum(K.maximum(w,self.lower_bound),self.upper_bound)
    
class StoppingCriteria(Callback):
    '''
    Callback that stops training before the announced number of epochs when some criteria are met.
    '''
    def __init__(self, accuracy):
        '''
        '''
        super().__init__()
        self.acc = accuracy
        
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy')<= self.acc:
            self.model.stop_training = True
            
def sharpness_worstcase(model,x,y, data_subset = .1, epsilon_weight_scale = 1e-3, 
                        kernel_only = False, training_phase = 0,
                        batch_size = 128, epochs = 5,lr = 1.,noise = False):
    '''
    dropout layers should be removed!
    code adapted from NeurIPS "predicting generalization in deep learning" competition starting kit
    https://competitions.codalab.org/competitions/25301 
    '''
    orig_weights = model.get_weights()
    
    # collect weights and their initial values
    # prepare upper and lower bounds
    weights = model.trainable_weights
    weights_orig = [K.get_value(w) for w in weights]
    weight_upper_bounds = [K.variable(K.get_value(w)) for w in weights]
    weight_lower_bounds = [K.variable(K.get_value(w)) for w in weights]
    for w,w_upper,w_lower in zip(weights, weight_upper_bounds, weight_lower_bounds):            
        if w.constraint is not None: # keras allows only one constraint per weight
            warnings.warn("a weight constraint has been overwritten by the sharpness_worstcase() call")
        w._constraint = Clip(w_lower, w_upper)
    
    # increase the original loss
    model.compile(loss = lambda y_true,y_pred: -losses.categorical_crossentropy(y_true,y_pred), 
                  optimizer=SGD(lr),
                  metrics=['accuracy'])
    lr_sched = LearningRateScheduler(lr_schedule(lr,0.1,[3]))
    
    # m represents the bounds for the weights perturbation
    # m will be optimized such that optimizing within these bounds reaches the target deviate
    # for this optimization, h and l represent high and low tentative values of m and a bisectional method is used
    h, l = .25, 0.000000
#     h, l = .1, 0.000000
    target_accuracy = 0.9
    stop = StoppingCriteria(0.7) # training will stop if train accuracy is below 70%
    for i in range(20): # loop to find perturbation scale
        m = (h + l) / 2. # m fixes the bounds for the weight perturbation
        
        nb_samplings = 3 if noise else 1
        min_accuracy = 1.
        for k in range(nb_samplings):
            for w,w_orig,w_upper,w_lower in zip(weights, weights_orig, weight_upper_bounds, weight_lower_bounds): 
                if not kernel_only or len(w_orig.shape)>1.: # kernels are assumed to be the only weights with more than one dimension
                    if noise:
                        # add uniform noise to the kernels to accelerate training
                        noisy = w_orig+np.random.uniform(low=-m/2, high=m/2, 
                                                         size=list(w_orig.shape)) * (np.abs(w_orig)+epsilon_weight_scale)
                        K.set_value(w,noisy)
                    else:
                        K.set_value(w,w_orig)

                    # set optimization constraints
                    K.set_value(w_lower,w_orig- m*(np.abs(w_orig)+epsilon_weight_scale))
                    K.set_value(w_upper,w_orig+ m*(np.abs(w_orig)+epsilon_weight_scale))
                
            # use subset of the data to train and estimate accuracy (a different subset is used for every estimation)
            datagen = ImageDataGenerator()
            history = model.fit_generator(datagen.flow(x, y,batch_size=batch_size),
                                          steps_per_epoch=50,#int(data_subset*x_train.shape[0]/batch_size), 
                                          epochs=epochs,
                                          verbose = 0,
                                          callbacks = [lr_sched,stop])                    
            
            subset = np.random.permutation(x.shape[0])[:int(x.shape[0]*data_subset)]
            if training_phase == 1:
                # evaluation is in training mode (which is good, 'cause no need to update batchnorm running statistics)
                # but careful for dropout: should be disabled
                accuracy = history.history['accuracy'][-1] 
#                 accuracy = evaluate_in_training_mode(model,x[subset],y[subset],verbose = 0, batch_size = batch_size)[1]
            elif training_phase == 0:
                accuracy = model.evaluate(x[subset],y[subset],verbose = 0, batch_size = batch_size)[1]
            min_accuracy = min(min_accuracy,accuracy) # only useful when noise = True
            
        accuracy = min_accuracy
        
        if h - l < 1e-5 or abs(accuracy - target_accuracy) < 5e-3:
            break
        if accuracy < target_accuracy:
            h = m
        else:
            l = m
            
    model.set_weights(orig_weights)
    for w in weights:
        w._constraint = None
    return m, accuracy - target_accuracy,history_todict(history)