from __future__ import print_function
import numpy as np
from scipy import io
from keras.models import model_from_json

#Consturct CNN model
model = model_from_json(open('srcnn_model.json').read()) 
#load weights
model.load_weights('srcnn_model_weights.h5') 
w = model.get_weights()
for i in range(0,6,2):
    w[i] = np.array(w[i])  
    w[i+1] = np.array(w[i+1])
    io.savemat('w'+str(i/2)+'.mat', {'array': w[i].transpose(2,3,1,0)})
    io.savemat('b'+str(i/2)+'.mat', {'array': w[i+1]})
