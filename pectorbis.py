#Pectorbis von Timofej D. Kazakov (Github: timdika)

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


#Kreation eines Layers:
class Layer_Dense:
    #Initalisierung des Layers
    def __init__(self, n_inputs, n_neuronen, gwicht_regularizierer_l1=0, gwicht_regularizierer_l2=0, 
    bias_regularizierer_l1=0, bias_regularizierer_l2=0):
        #Gwicht und Biases Initalisierung
        self.gwicht = 0.01 * np.random.randn(n_inputs, n_neuronen)
        self.biases = np.zeros((1, n_neuronen))

        self.gwicht_regularizierer_l1 = gwicht_regularizierer_l1
        self.gwicht_regularizierer_l2 = gwicht_regularizierer_l2
        self.bias_regularizierer_l1 = bias_regularizierer_l1
        self.bias_regularizierer_l2 = bias_regularizierer_l2
    #Forward-Pass
    def forward(self, inputs, training):
        #Berechnung des Outputs
        self.inputs = inputs
        self.output = np.dot(inputs, self.gwicht) + self.biases

    def backward(self, dvalues): 
        self.dgwicht = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.gwicht_regularizierer_l1 > 0:
            dL1 = np.ones_like(self.gwicht)
            dL1[self.gwicht < 0] = -1
            self.dgwicht += self.gwicht_regularizierer_l1 * dL1
        # L2 on weights
        if self.gwicht_regularizierer_l2 > 0:
            self.dgwicht += 2 * self.gwicht_regularizierer_l2 * \
                                self.gwicht
        # L1 on biases
        if self.bias_regularizierer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizierer_l1 * dL1
        # L2 on biases
        if self.bias_regularizierer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizierer_l2 * \
                                self.biases

        self.dinputs = np.dot(dvalues, self.gwicht.T)


#Kreation der Input-Klasse:
class Layer_Input:
    def forward(self, inputs, training): #TRAINING

        self.output = inputs

#----------------------------------------------------------

#AKTIVIERUNGSFUNKTIONEN:

#----------------------------------------------------------

class Aktivierung_ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

    def vorhersagen(self, outputs):
        return outputs


class Aktivierung_Softmax:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_werte = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        wahrscheinlichkeiten = exp_werte / np.sum(exp_werte, axis=1, keepdims=True)
        self.output = wahrscheinlichkeiten

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #Wir machen einen uninitalisierten Array

        #Enumerate outputs and gradients:
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            
            single_output = single_output.reshape(-1, 1) #Flatten out array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    def vorhersagen(self, outputs):
        return np.argmax(outputs, axis=1)


#----------------------------------------------------------

#VERLUSTFUNKTIONEN:

#----------------------------------------------------------


class Verlust:

    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0
        # L1 regularization - weights
        # calculate only when factor greater than 0
        for layer in self.trainable_layers:

            if layer.gwicht_regularizierer_l1 > 0:
                regularization_loss += layer.gwicht_regularizierer_l1 * \
                np.sum(np.abs(layer.gwicht))
            # L2 regularization - weights
            if layer.gwicht_regularizierer_l2 > 0:
                regularization_loss += layer.gwicht_regularizierer_l2 * \
                np.sum(layer.gwicht *
                layer.gwicht)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizierer_l1 > 0:
                regularization_loss += layer.bias_regularizierer_l1 * \
                np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizierer_l2 > 0:
                regularization_loss += layer.bias_regularizierer_l2 * \
                np.sum(layer.biases *
                layer.biases)
        
        return regularization_loss
    
    #Set/remember trainable layers:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def kalkulieren(self, output, y, *, include_reg=False): 
        #Calculate sample losses:
        sample_verluste = self.forward(output, y)
        #Calculate mean loss:
        data_verlust = np.mean(sample_verluste)

        self.accumulated_sum += np.sum(sample_verluste)
        self.accumulated_count += len(sample_verluste)

        if not include_reg:
            return data_verlust
        #Return the data and reg loss:
        return data_verlust, self.regularization_loss()

    def accumulated_kalkulieren(self, *, include_reg=False):

        data_verlust = self.accumulated_sum / self.accumulated_count

        if not include_reg:
            return data_verlust

        return data_verlust, self.regularization_loss()

    def neu_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Verlust_CatCrossEnt(Verlust):

    def forward(self, y_pred, y_true): 
        
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: 
            korrekte_sicherheiten = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            korrekte_sicherheiten = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log_wahrscheinlichkeiten = -np.log(korrekte_sicherheiten)
        return neg_log_wahrscheinlichkeiten

    def backward(self, dvalues, y_true):
        samples = len(dvalues) #Anzahl samples
        labels = len(dvalues[0]) #Anzahl Labels in jedem Sample

        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues 
        self.dinputs = self.dinputs / samples 


#----------------------------------------------------------

#OPTIMIZER:

#----------------------------------------------------------


class HerrAdam: #Gute Start-Lernrate = 0.001, decaying runter zu 0.00001
    #Initalisierung Optimizer. Lernrate = 1 - Basis für diesen Optimizer
    def __init__(self, lern_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.lern_rate = lern_rate
        self.momentane_lern_rate = lern_rate
        self.decay = decay
        self.iterationen = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    #Einmal aufrufen BEVOR irgendein Parameter updatet
    def pre_update_params(self):
        if self.decay:
            self.momentane_lern_rate = self.lern_rate * \
                (1. / (1. + self.decay * self.iterationen))
    #Parameter updaten
    def update_params(self, layer):
        #If we use Momentum
        if not hasattr(layer, 'gwicht_cache'):
            layer.gwicht_momenta = np.zeros_like(layer.gwicht)
            layer.gwicht_cache = np.zeros_like(layer.gwicht)
            layer.bias_momenta = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.gwicht_momenta = self.beta_1 * layer.gwicht_momenta + (1-self.beta_1) * layer.dgwicht
        layer.bias_momenta = self.beta_1 * layer.bias_momenta + (1-self.beta_1) * layer.dbiases

        gwicht_momenta_korrigiert = layer.gwicht_momenta / (1-self.beta_1 ** (self.iterationen + 1))
        bias_momenta_korrigiert = layer.bias_momenta / (1-self.beta_1 ** (self.iterationen + 1))
        
        layer.gwicht_cache = self.beta_2 * layer.gwicht_cache + (1-self.beta_2) * layer.dgwicht ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases ** 2

        gwicht_cache_korrigiert = layer.gwicht_cache / (1-self.beta_2 ** (self.iterationen + 1))
        bias_cache_korrigiert = layer.bias_cache / (1-self.beta_2 ** (self.iterationen + 1))


        layer.gwicht += -self.momentane_lern_rate * gwicht_momenta_korrigiert / (np.sqrt(gwicht_cache_korrigiert) + self.epsilon)
        layer.biases += -self.momentane_lern_rate * bias_momenta_korrigiert / (np.sqrt(bias_cache_korrigiert) + self.epsilon)
    def post_update_params(self):
        self.iterationen += 1


#----------------------------------------------------------

#GENAUIGKEIT:

#----------------------------------------------------------


class Genauigkeit:

    def kalkulieren(self, vorhersagen, y):

        vergleiche = self.vergleichen(vorhersagen, y)

        genauigkeit = np.mean(vergleiche)

        self.accumulated_sum += np.sum(vergleiche)
        self.accumulated_count += len(vergleiche)

        return genauigkeit

    def accumulated_kalkulieren(self):

        genauigkeit = self.accumulated_sum / self.accumulated_count

        return genauigkeit

    def neu_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Genauigkeit_Categorial(Genauigkeit):

    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def vergleichen(self, vorhersagen, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
            return vorhersagen == y


#----------------------------------------------------------

#M O D E L L :

#----------------------------------------------------------


class Model:

    def __init__(self):
        #Create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None
    #Add objects to the model:
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, genauigkeit): # Set loss and optimizer
        self.loss = loss
        self.optimizer = optimizer
        self.genauigkeit = genauigkeit
    
    def finalize(self):
        #Create and set the inputs layer:
        self.input_layer = Layer_Input()

        #Count all the objects:
        layer_count = len(self.layers)

        self.trainable_layers = [] #Initialize a list containing trainable layers:

        #Iterate the objects:
        for i in range(layer_count):

            #If its the first layer, the prev. layer object is the input layer:
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            #All layers except for the first and the last:
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            #The last layer - the next object is the loss:
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            #If layer contains an attribute called "gwicht", its a trainable layer - add to list
            if hasattr(self.layers[i], 'gwicht'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)


        #if isinstance(self.layers[-1], Aktivierung_Softmax) and isinstance(self.loss, Verlust_CatCrossEnt):
         #   self.softmax_classifier_output = Aktivierung_Softmax_Verlust_CatCrossEnt()

    def train(self, X, y, *, epochen=1, batch_size=None, print_every=1, validation_data=None): #Model trainieren
        
        self.genauigkeit.init(y) #Vlt von Regression auf Seite 488!!!

        train_schritte = 1

        if validation_data is not None:
            validierungs_schritte = 1

            X_val, y_val = validation_data

        if batch_size is not None:
            train_schritte = len(X) // batch_size

            if train_schritte * batch_size < len(X):
                train_schritte += 1

            if validation_data is not None:
                validierungs_schritte = len(X_val) // batch_size

                if validierungs_schritte * batch_size < len(X_val):
                    validierungs_schritte += 1

        #Main training loop:
        for epoch in range(1, epochen+1):

            print(f'epoche: {epoch}')

            self.loss.neu_pass()
            self.genauigkeit.neu_pass()

            for schritt in range(train_schritte):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[schritt*batch_size:(schritt+1)*batch_size]
                    batch_y = y[schritt*batch_size:(schritt+1)*batch_size]

                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = self.loss.kalkulieren(output, batch_y, include_reg=True)
                loss = data_loss + regularization_loss

                vorhersagen = self.output_layer_activation.vorhersagen(output)
                genauigkeit = self.genauigkeit.kalkulieren(vorhersagen, batch_y)
        
                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not schritt % print_every or schritt == train_schritte - 1:
                    print(f'schritt: {schritt},' +
                        f'genau: {genauigkeit:.3f},' +
                        f'loss: {loss:.3f}, (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f'lr: {self.optimizer.momentane_lern_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.accumulated_kalkulieren(include_reg=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_genauigkeit = self.genauigkeit.accumulated_kalkulieren()
            print(f'training, ' +
                    f'genau: {epoch_genauigkeit:.3f}, '+
                    f'loss: {epoch_loss:.3f}, (' +
                    f'data_loss: {epoch_data_loss:.3f}, ' +
                    f'reg_loss: {epoch_regularization_loss:.3f}) '+
                    f'lr: {self.optimizer.momentane_lern_rate}')
        
            if validation_data is not None:

                self.loss.neu_pass()
                self.genauigkeit.neu_pass()

                for schritt in range(validierungs_schritte):

                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val

                    else:
                        batch_X = X_val[schritt*batch_size:(schritt+1)*batch_size]
                        batch_y = y_val[schritt*batch_size:(schritt+1)*batch_size]

                
                    output = self.forward(batch_X, training=False)
                
                    self.loss.kalkulieren(output, batch_y)

                    vorhersagen = self.output_layer_activation.vorhersagen(output)

                    self.genauigkeit.kalkulieren(vorhersagen, batch_y)

                validation_loss = self.loss.accumulated_kalkulieren()
                validation_genauigkeit = self.genauigkeit.accumulated_kalkulieren()

                print(f'validation, ' + 
                    f'genau: {validation_genauigkeit:.3f}, '+
                    f'loss: {validation_loss:.3f}')

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


#----------------------------------------------------------

#DATEN:

#----------------------------------------------------------


def Datensatz_laden(datensatz, path):

    labels = os.listdir(os.path.join(path, datensatz))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, datensatz, label)):
            image = cv2.imread(os.path.join(path, datensatz, label, file), cv2.IMREAD_GRAYSCALE)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_med(path):

    X, y = Datensatz_laden('train', path)
    X_test, y_test = Datensatz_laden('test', path)

    return X, y, X_test, y_test


#----------------------------------------------------------

#KONSTRUKTION:

#----------------------------------------------------------



X, y, X_test, y_test = create_data_med('C:/Users/Timofej Kazakov/Desktop/ZweiFallDatensatz')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

#Model initalisieren:
model = Model()

#Add layers:
model.add(Layer_Dense(X.shape[1], 5))
model.add(Aktivierung_ReLU())
model.add(Layer_Dense(5, 5))
model.add(Aktivierung_ReLU())
model.add(Layer_Dense(5, 2))
model.add(Aktivierung_Softmax())


model.set(
    loss=Verlust_CatCrossEnt(), 
    optimizer=HerrAdam(decay=5e-5), #lernrate vor decay 
    genauigkeit=Genauigkeit_Categorial())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), 
            epochen=10, batch_size=5, print_every=100)