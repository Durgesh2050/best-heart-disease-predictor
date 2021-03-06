'''
@author Heart disease prediction
date 02.04.2022

Program to combine the model prediction
'''
import pickle
import sklearn

def combineModel(models:list,element):
    '''
    parameter
    ---------
    models: list of model prediction you want to combine
    element: scalar vector of the input feeded in the model

    return
    ------
    prediction: combine prediction
    prediction_prob: combine prediction preobability
    '''
    prediction = 0
    for model in models:
        prediction += model.predict((element)).max()
        
    prediction = prediction/len(models)
    if prediction > 0.5:
        prediction_prob = int(prediction * 100)
    else:
        prediction_prob = int(((1-prediction)*100))
    
    return prediction,prediction_prob
