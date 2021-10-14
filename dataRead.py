import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame





filenames = ['traindata.csv', 'traindata2.csv', 'traindata3.csv', 'traindata4.csv']
with open('outfile', 'w') as outfile:
    for file in filenames:
        with open(file) as infile:
            outfile.write(infile.read())


#in the section i fouond an source for binary seperable data, i collate 4 related deta files into 
#one for easier use
aClass = '0'
bClass = '1'

f = open('outfile')
trainData = []
np.array(trainData,dtype=float)
expected_outputs = []


#-------------- read in training data and split
    #in this section the train data is read in aswell as binary identifier which will be passed to the 
    #perceptron class. From this the perceptron will hone its weights to correctly catagorise binary 
    #data 

for line in f:

    lines = line.rstrip()	
    linesplit = lines.split(',')	
    
    if linesplit[4] == aClass:
    
        trainData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
        expected_outputs.append(1)
        
    elif linesplit[4] == bClass:
    
        trainData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
        expected_outputs.append(0)

f.close()

# --------------- read in test data
    #in much similar way to previous we now read in the test data which will be used to access
    #the accuracy of a now trained perceptron 

f2 = open ('testdata.csv')

testData = []
expected_test_outputs = []

np.array(testData, dtype = float)

for line in f2:

    lines = line.rstrip()
    linesplit = lines.split(',')
    
    if linesplit[4] == aClass:
    
        testData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
        expected_test_outputs.append(linesplit[4])
        
    elif linesplit[4] == bClass:
    
        testData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
        expected_test_outputs.append(linesplit[4])
        
f2.close()

#user promp               
                
print("press 1: for perceptron")

print("press 2: for MLP")

print("press 3: for particle swarm optimization")

print("press 4: for genetic algorithm")

choice = input("Please select what you would like to run: ")

if choice == '4':

    import pygad
    import pygad.nn
    import pygad.gann
    
    #solution = variable whos fitness needs determining
    #solID = index of solution in the population 
    def fitness( sol, solID):
        #this functions determines how successfull or fit our final product will be
        #which in turn has a knock on effect on the likelihood of this population continueing
        global NeuralNet, X, Y
        #predict function comes from pygad and makes predictions
        #here it is taking last layer and data inputs as paramaters 
        predictions = pygad.nn.predict(last_layer=NeuralNet.population_networks[solID],data_inputs=X)
        correct_predictions = np.where(predictions == Y)[0].size
        solution_fitness = (correct_predictions/Y.shape[0])*100
        return solution_fitness
        

    def callback(GApopulation):
        #This function is called after each population acts as a way of checking progress between each run through
        global NeuralNet
        population_matrices = pygad.gann.population_as_matrices(population_networks=NeuralNet.population_networks,
                                    population_vectors=GApopulation.population)
        NeuralNet.update_population_trained_weights(population_trained_weights=population_matrices)
        print("Generation = {generation}".format(generation=GApopulation.generations_completed))
        print("Accuracy   = {fitness}".format(fitness=GApopulation.best_solution()[1]))
        
    
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    
   
    
    #X = np.asarray(DataFrame(trainData)).astype(float)                           
                             
    #Y = np.asarray(expected_outputs).astype(int)
    
    #neural net constructor     
    NeuralNet = pygad.gann.GANN(num_solutions = 35, num_neurons_input = 30, 
                                    num_neurons_hidden_layers=[20],
                                    num_neurons_output= 2,
                                    hidden_activations=["relu"],
                                    output_activation="softmax")

    population_vectors = pygad.gann.population_as_vectors(population_networks=NeuralNet.population_networks)

    GApopulation  = pygad.GA(num_generations=150,
                            num_parents_mating=3,
                            initial_population=population_vectors.copy(),
                            fitness_func = fitness,
                            mutation_percent_genes=5,
                            callback_generation = callback)



    GApopulation.run()

    GApopulation.plot_result()

    solution, solution_fitness, solution_idx = GApopulation.best_solution()

    print(solution)
    print(solution_fitness)
    print(solution_idx)


if choice == '1':


    class Perceptron(object):

        #percepton constructor, input number iterations and learning rate
        #to succesfully construct a number of inputs will have to be passed
        def __init__(self, inputAmount, epochs = 5, learning_rate = 0.01):
        
            self.epochs = epochs	        
            self.learning_rate = learning_rate
            self.weights = np.zeros (inputAmount + 1)
            self.weightsMem = []
            self.weightsPrev = np.zeros (inputAmount + 1)

        #using precalculated weights predict what the inputs should give
        #weights will be stored from the training phase, method will return 1 or
        #0 depending on the activation function 
        def predict(self, inputs):
            
            inputs = inputs.astype(float)	
            #activation = weights transpose inputs plus bias
            acca = np.dot(inputs, self.weights[1:]) + self.weights[0]
            
            if acca > 0:		
                binOut = 1
                
            else:			
                binOut = 0
                
            return binOut

        #training function to achieve desired weights
        def train (self, training_inputs, expected_outputs): 
            #for given iterations (ephocs) we will loop through the following,
            #this will allow us to reach an adaqute level of learning and adjustment
            for _ in range (self.epochs):	
                allcorrect = 0
                
                #we will then loop through each data object passed in 		
                for inputs, expected in zip (training_inputs, expected_outputs):
                    #must be initialised as a float to allow for further operations(transpose)
                    inputs = inputs.astype(float)	
                    #accumlator function (activation function) if the acca reaches above 0 
                    #the data object will be catagorised as 1 else a 0
                    acca = np.dot(inputs, self.weights[1:]) + self.weights[0]
                    if acca > 0:
                        binOut = 1
                    else:
                        binOut = 0				
                    #these following lines of code is arguably where the learning occurs 
                    #if expected output is equal to that of what was achieved nothing will happen 
                    #due to multiplication of 0 being negligible. If the bin out was classified
                    #wrong the weights will be adjusted accordingly (positively or negatively)
                    if ((binOut - expected) == 0):
                        allcorrect = 1
                    self.weights[1:] += self.learning_rate * (expected - binOut) * inputs
                    self.weights[0] += self.learning_rate * (expected - binOut)
                    print(self.weights)	
                if(allcorrect == 1):break

    # ------- TRAINING PORTION
        #creates perceptron and then passes data attributes asw2ell as an expected output	
    Perceptron = Perceptron(4)
    Perceptron.train(trainData,expected_outputs)

   #--------------- check accuracy of perceptron
    #now that we have a trained perceptron and operable test data we check to see the success 
    #of our binary classifier

    #Use iterator j use both a place marker and accumulator for use in operations 
    j = 0

    # We increment correctPredictions if the machine gets it right
    correctPredictions = 0

    for i in testData:

        print ("Data input: ",i)	
        print ("Expected result: ",expected_test_outputs[j])	
        if (Perceptron.predict(i) == 1):
        
            if (expected_test_outputs[j] == aClass):		
                correctPredictions += 1		
                print ("Actual result: ",aClass,"\n")
                
            else:		
                print ("Actual result: ",bClass,"\n")
            
        else:
        
            if (expected_test_outputs[j] == bClass):		
                correctPredictions += 1	
                print ("Actual result: ",bClass,"\n")
                
            else:		
                print ("Actual result: ",aClass,"\n")
        
        # Increment expected_test_outputs
        j+=1
            
    # Calculate accuracy of our binary perceptron

    accuracy = (correctPredictions / len(testData)) * 100

    print ("Classifier accuracy: ",accuracy,"%")
    
    
if choice == '2':

    #below i have hard coded in letters for my mlp to recognize
    a =[0, 0, 1, 1, 0, 0,
        0, 1, 0, 0, 1, 0,
        1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1]
    # this is b
    b =[0, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0]
    # this is c
    c =[0, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0]
    # this is
    d =[0, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0]
    

    # Creating labels of expected outputs
    y =[[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],]

    
    
    #create arrays into operational arrays 
    x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30),
	    np.array(c).reshape(1, 30), np.array(d).reshape(1, 30)]        
        
    y = np.array(y)

    #Sigmoid function for hidden layer activation
    def sig(x):
        out = (1/(1 + np.exp(-x)))
        return out

    #forward prop method
    def fPropnn(x, w1, w2):
        # hidden
        in1 = x.dot(w1)# input from layer 1
        a1 = sig(in1)# out put of layer 2 using sigmoid fucntion
        
        # Output layer
        in2 = a1.dot(w2)# input of out layer
        a2 = sig(in2)# output of out layer
        #returns output node activation
        return(a2, a1)

    
        
    # to figure our loss we will be using mean square error here
    def loss(out, Y):
        mse =(np.square(out-Y))
        return(np.sum(mse)/len(y))
        
    # Back propagation of error to then use on weights
    def back_prop(x, y, w1, w2, lr):
        
        # for rinput to hidden layer sz1 equal to pre activation a1 = post activation
        a2, a1 = fPropnn(x, w1, w2)
        
        #we now work out the erorrs within the layers, output layer first
        errorcheck =(a2-y)
        epass = np.multiply((w2.dot((errorcheck.transpose()))).transpose(),
                                    (np.multiply(a1, 1-a1)))
      
        # we now create our new weights and return
        w1 = w1-(lr*(x.transpose().dot(epass)))
        w2 = w2-(lr*(a1.transpose().dot(errorcheck)))
        
        return(w1, w2)

    #training section of my code
    def train(x, Y, w1, w2, lr = 0.01, epoch = 10):
        acc =[]
        cost =[]
        #iterate thrqough specified epochs
        for j in range(epoch):
            l =[] # tempory loss array
            #iterate through data objects
            for i in range(len(x)):
                #forward propogate data object
                output, null = fPropnn(x[i], w1, w2)
                #add our found loss for given data object and append into l
                l.append((loss(output, Y[i])))
                #we now carry out our back propogation
                #giving us our new weight ararays
                w1, w2 = back_prop(x[i], y[i], w1, w2, lr)
            #visual representation fo understanding
            print("ITERATION: ", j + 1,)
            print("ACCURACY:", (1-(sum(l)/len(x)))*100)
            #save out accuracy and loss for later mapping
            acc.append((1-(sum(l)/len(x)))*100)
            cost.append(sum(l)/len(x))
        return(acc, cost, w1, w2)

    #finally we have the prediction section to check the trained mlp
    def predict(x, w1, w2):
        #first we must forward propogate to learn output activation
        output, null = fPropnn(x, w1, w2)
        
        print(output)
        maxm = 0
        k = 0
        #runs throufh the outputs and detects what letter is being input
        for i in range(len(output[0])):
            if(maxm<output[0][i]):
                maxm = output[0][i]
                k = i
        if(k == 0):
            print("Image is letter A")
        elif(k == 1):
            print("Image is letter B")
        elif(k == 2):
            print("Image is letter C")
        else:
            print("Image is letter D")
        #plt.imshow(x.reshape(5, 6))
        #plt.show()	
        
# initializing the weights
    def generate_wt(x, y):
        l =[]
        for i in range(x * y):
            l.append(np.random.randn())
        return(np.array(l).reshape(x, y))
        
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 4)
    print(w1, "\n\n", w2)

    acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)


    plt.plot(losss)
    plt.ylabel('cost')
    plt.xlabel("Epoch")
    plt.show()
    # ploting accuraccy
    plt.plot(acc)
    plt.ylabel('Correctness')
    plt.xlabel("Epoch")
    plt.show()

    # plotting Loss
    


    # the trained weigths are
    print(w1, "\n", w2)


    #to check each letter can be correctly identified
    for i in range(len(x)): 
        predict(x[i], w1, w2)


if choice == '3':
    #more advanced data,however other parameters will have to be changed, check report
    data = load_breast_cancer()
    X = data.data
    Y = data.target

    x = np.asarray(DataFrame(trainData)).astype(float)
    y = np.asarray(expected_outputs).astype(int)
    
    
    #======================forward propogation
    #within this method we lay out a forward propogation for each of the population to carry out
    #within we specify the number of inputs an hidden layer and then outputs, no. of classification

    def forward_prop(params):
        
        n_inputs = 4
        n_hidden = 30
        n_classes = 2

        # create our weights and bias in order to use them within our forwardd propogation
        W1 = params[0:120].reshape((n_inputs,n_hidden))
        b1 = params[120:150].reshape((n_hidden,))
        W2 = params[150:210].reshape((n_hidden,n_classes))
        b2 = params[210:212].reshape((n_classes,))
        
        #print(x)

        # here we perform the forward propogation
        #z1 = preactivation of layer one which then actication at a1
        #z2 is very similar going into layer 2
        in1 = x.dot(W1) + b1  
        a1 = np.tanh(in1)     
        in2 = a1.dot(W2) + b2 
                

        # Compute for the softmax for out activation
        exp_scores = np.exp(in2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # ww compute the negative log to use loss function
        N = 157 # Number of samples found in data input
        corect_logprobs = -np.log(probs[range(N), y])
        loss = np.sum(corect_logprobs) / N

        return loss
        

    def f(x):
        
        #first we gather infotmation on the size of x
        elements = x.shape[0]
        #at this point for range of elements we complete forward propogation which we store to array j and return
        j = [forward_prop(x[i]) for i in range(elements)]
        return np.array(j)
        

    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Here we lay out the dimensions of the neyral net( 4 inputs - 30 hidden) and so on 
    dimensions = (4 * 30) + (30 * 2) + 30 + 2
    #we then use these dimensions to create an optimizer, we also then state the size of population we would like
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

    # this is where  the table is optimized and "learning" takes place, we specify iterations and our forumala f which we are pased to 
    cost, pos = optimizer.optimize(f,  iters=1000, verbose=True)
    print("p0s", pos)

    def predict(x, pos):
        n_inputs = 4
        n_hidden = 30
        n_classes = 2

        # access weights and bias 
        W1 = pos[0:120].reshape((n_inputs,n_hidden))
        b1 = pos[120:150].reshape((n_hidden,))
        W2 = pos[150:210].reshape((n_hidden,n_classes))
        b2 = pos[210:212].reshape((n_classes,))

        # Perform forward propagation
        z1 = x.dot(W1) + b1  
        a1 = np.tanh(z1)     
        z2 = a1.dot(W2) + b2          

        y_pred = np.argmax(z2, axis=1)
        return y_pred
    print()
    print("###########")
    print("Accuracy: ", ((predict(x, pos) == y).mean()*100), "%")

