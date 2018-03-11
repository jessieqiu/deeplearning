
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("exam1_train.csv", index_col=0) 
df_test = pd.read_csv("exam1_test.csv", index_col=0) 
#df.head()
#df_test.head()


# In[3]:


def one_hot_encoding(y):
    if(y.shape[1]!=1):
        print('input is wrong');
        return -1
    length = y.shape[0]
    output = np.zeros((10,length))   
    
    for i in range(length):         
        if(y[i][0]>9):
            #to prevent index out of bound
            print('label is wrong');
            return -1
        else:
            output[y[i][0]][i]=1   
    return output


# In[4]:


#split data
size = df.shape[0]
x = df.iloc[:,0:400]
X = x.values
y = df.iloc[:,-1]
y_digit = y.values.reshape(size,1)
y_one_hot = one_hot_encoding(y_digit)
print(X.shape, y_digit.shape,y_one_hot.shape)
#print('last five numbers:')
#print(y_one_hot[:, 3495:])
 


size_test = df_test.shape[0]
x_test = df_test.iloc[:,0:400]
X_test = x_test.values

y_test = df_test.iloc[:,-1].values.reshape(size_test,1)
#y_one_hot = one_hot_encoding(y_digit)
print(X_test.shape, y_test.shape)


# In[5]:


def activation_sigmoid(z): 
    #check z shape?
    a =  1.0/(1.0 + np.exp(-1*z))
    #a = np.divide(1, np.add (1 , np.exp(-z)))
    print(a.shape)
    
    return a


def activation(activation_index, z, alp=1):  #alpha is the constant for leaky relu or ELU
    
    activation_set = ["sigmoid","tanh","ReLU","Leaky ReLU","ELU"]
    
    if(activation_index >= len(activation_set)):
        print('no such function')
        return -1    
   
    #sigmoid
    if activation_index == 0:     #can change range and expand x later?
        #check z shape?
        a =  1.0/(1.0 + np.exp(-z))
        #a = np.divide(1, np.add (1 , np.exp(-z)))
    #tanh    
    elif activation_index == 1:  #can change range and expand x later?
        temp1 = np.exp(z)
        temp2 = np.exp(-z)
        a = (temp1-temp2)/(temp1+temp2)
    #ReLU
    elif activation_index == 2:
        a = np.maximum(z, np.zeros(z.shape)) #will take max(z[i],0)
    #Leaky ReLU
    elif activation_index == 3:
        #try different alp, 0.1 0.01?
        a = np.maximum(z, alp * z)
    else:
        #ELU
        a = z.copy()
        #a = np.exp(z)-1  where z<0
        #a[z < 0] filter the index for negative numbers
        #alp = 1
        a[z < 0] = alp* (np.exp(a[z < 0]) - 1)
    
    return a


# In[6]:


#test and plots for activation functions
z1 = np.arange(-5,5,0.1)
#print(z1)

a1 = activation(0,z1)
plt.plot(z1, a1)
plt.title('sigmoid')
plt.show()

a2 = activation(2,z1)
plt.plot(z1, a2)
plt.title('ReLU')
plt.show()

a3 = activation(3,z1,0.1)
#print(a3)
plt.plot(z1, a3)
plt.title('Leaky ReLU')
plt.show()

a4 = activation(1,z1)
#print(a4)
plt.plot(z1, a4)
plt.title('tanh')
plt.show()

a5= activation(4,z1)
#print(a5)
plt.plot(z1, a5)
plt.title('ELU')
#plt.plot(z1,a2,z1,a3,z1, a5)
plt.show()


# In[ ]:





# In[7]:


def forward_one_layer (weights, X, bias, activation_function,alpha): 
    # lambda ?
    z = np.dot(weights, X) + bias #bias is a one-column vector   
    a = activation(activation_function,z,alpha) 
    return z,a 

def soft_max(f):
    shift = f - np.max(f) #shift all number to the left so that the sum will not get too big or explode
    exp_f = np.exp(f) 
    return exp_f / np.sum(exp_f) 

def forward_all_layers(X_data,weights_1, bias_1, weights_2,bias_2, weights_3,bias_3,activations,alpha): # layers, what else ?  
    
    activation_set = ["sigmoid","tanh","ReLU","Leaky ReLU","ELU"] #change to 0 1 2 3 4 later,  faster?
    #a = np.array() #?
    #a.append(X_data) #?
    #a[0] = X_data.copy()
    
#     for i in layers:
#         print i
#         z[i+1], a[i+1] = forward_one_layer(weights[i],a[i],bias[i],activation_set[activations[i]])
#         print(activation_set[activations[i]])
        
    #first layer
    #print(activation_set[activations[0]])
    z1,a1 = forward_one_layer(weights_1,X_data,bias_1,activations[0],alpha[0])
    
    #second layer
    #print(activation_set[activations[1]])
    z2,a2 = forward_one_layer(weights_2,a1,bias_2,activations[1],alpha[1])
    
    #output layer
    #print(activation_set[activations[2]])
    z3,a3 = forward_one_layer(weights_3,a2,bias_3,activations[2],alpha[2])
        
        
    return z1,a1,z2,a2,z3,a3


# In[ ]:





# In[8]:


def getGradient(a1, z2, a2, previous_dldz, w, activation_function,alp=1):
    dzda = w 
    dadz = getActivationFunctionDerivative(a2,z2,activation_function,alp)
    dzdw = np.transpose(a1) #do i need transpose?
    
    #dzdw.shape (3500L, 30L)
    #dzdb = 1 
   
   
    m = 3500    # n of samples
    
    #print('1')
    #print( previous_dldz.shape,dzda.shape,dadz.shape)
    # ((10L, 3500L), (10L, 20L), (20L, 3500L))
    dldz = np.dot(previous_dldz.T,dzda).T * dadz
   # dldz = previous_dldz* np.dot(dadz,dzdw)
    #print('2')
    #print(dldz.shape,dzdw.shape)
    
    dldw = np.dot( dldz, dzdw )/m
    
    dldb = np.sum(dldz, axis = 1, keepdims = True)/(m) #  because dldb is dldz times dzdb which is 1 
    #print('3')
    #print(dldz.shape,dldw.shape,dldb.shape)
    
    return dldz, dldw, dldb 
    
def getActivationFunctionDerivative(a,z,function_index,alp=1):
    activation_set = ["sigmoid","tanh","ReLU","Leaky ReLU","ELU"]
    
    if(function_index >= len(activation_set)):
        print('no such function')
        return -1
    
    if function_index== 0:
        dadz = a * (1 - a)
    elif function_index == 1:
        dadz = 1 - a*a
    elif function_index == 2:
        dadz = np.ones(z.shape)        
        dadz[z < 0] = 0 
    elif function_index==3:
        dadz = np.ones(z.shape)
        dadz[z < 0] = alp 
    else:
        dadz = np.ones(z.shape)
        dadz[z < 0] = alp * np.exp(dadz[z < 0]) 
    return dadz
        


# In[9]:


##test getActivationFunctionDerivative

# z = np.array([[-1,1,2],[1,2,-3]])
# a = np.array([[0.1,0.2,0.3],[0.3,0.4,0.5]])
# print( z)
# print( a)
# for i in range(5):
#     print(i)
#     dadz = getActivationFunctionDerivative(a,z,i,0.01)
#     print(dadz)


# In[ ]:





# In[10]:


def lastLayerGradient(a, X, i=0, j=y_one_hot.shape[1]):  #i j can be used for mini batch, X is the input to last layer
    if(a.shape==y_one_hot[:,i:j].shape):
        dldz = a - y_one_hot[:,i:j]
        m = j-i
        dldw = np.dot(dldz,np.transpose(X))/(m) # m or -m?
        dldb = np.sum(dldz, axis = 1, keepdims = True)/(m)
        #print(dldw.shape, dldb.shape)
        #print('last layer done')
        return dldz, dldw, dldb
    else:
        print('I did something wrong, shape of a is ', a.shape, ' it should be ', y_one_hot[:,i:j].shape)
        return -1
    
def calculate_cross_entropy_loss(a, i=0, j=y_one_hot.shape[1]):
    if(a.shape==y_one_hot[:,i:j].shape):
         
        if(a.any()==0):
            a[a==0] = 0.00000001
            #print ('shift to right')
        if(a.any()==1):
            a[a==1] = 0.99999999
            #print ('shift to left')
        
            
        temp1 = y_one_hot[:,i:j] * np.log(a)
        temp2 = (1 - y_one_hot[:,i:j]) * np.log(1 - a)
        #print "why? %f" %np.sum(temp1 + temp2)
        #print 'm ', j-i
        loss = (-1 / (j-i)) * np.sum(temp1 + temp2)  #j must > i         
    else:
        print('I did something wrong again, shape of a is ', a.shape, ' it should be ', y_one_hot[:,i:j].shape)
        loss = -1
    return loss

def accuracy(prediction,label):
    if (prediction.shape!=label.shape):
        print prediction.shape
        print label.shape
        print 'wrong input size'
        return -1
    count = 0
    total = prediction.shape[0]
    #print 'size ',total
    for i in range(total):
        
        if prediction[i]!= label[i]:
            count+=1
            #print 'i', i
    #print 'error count', count
    result = 0.0
    error_rate = float(count)/total
    accuracy = 1-error_rate
    return accuracy


# In[11]:


def initialization(neurons,functions):
    
    c= [1.0,1.0,1.0]
    for i in range(len(functions)):
    
        if (functions[i]== 0):
            c[i] = math.sqrt(2)
        elif (functions[i]== 1):
            c[i] = 4 * math.sqrt(2)
        else:
            c[i] = 2    
    print 'sigma,', c
    
    
    #weights_1 = 0.001* np.random.randn(neurons[0],400)
    #weights_2 = 0.001* np.random.randn(neurons[1],neurons[0])
    #weights_3 = 0.001* np.random.randn(neurons[2],neurons[1])
    
    weights_1 = np.random.normal(0, c[0] / math.sqrt(neurons[0] + 400) ,(neurons[0],400))
    weights_2 = np.random.normal(0, c[1] / math.sqrt(neurons[1] + neurons[0]), (neurons[1],neurons[0]))
    weights_3 = np.random.normal(0, c[2] / math.sqrt(neurons[2] + neurons[1]), (neurons[2],neurons[1]))
    print(weights_1.shape, weights_2.shape,weights_3.shape)
    
#     bias_1 = 0.001*np.random.randn(neurons[0],1)
#     bias_2 = 0.001*np.random.randn(neurons[1],1)
#     bias_3 = 0.001*np.random.randn(neurons[2],1)

    bias_1 = np.random.normal(0, c[0] / math.sqrt(neurons[0]), (neurons[0],1))  #which one is correct?
    bias_2 = np.random.normal(0, c[1] / math.sqrt(neurons[1]), (neurons[1],1))
    bias_3 = np.random.normal(0, c[2] / math.sqrt(neurons[2]), (neurons[2],1))
    
#     #which one is correct?
#     bias_1 = np.random.normal(0, c[0] / math.sqrt(neurons[0] + 400) , (neurons[0],1))
#     bias_2 = np.random.normal(0, c[1] / math.sqrt(neurons[1] + neurons[0]), (neurons[1],1))
#     bias_3 = np.random.normal(0, c[2] / math.sqrt(neurons[2] + neurons[1]), (neurons[2],1))
    
    return weights_1, bias_1, weights_2,bias_2,weights_3,bias_3


# In[12]:


#initialization 
#no need to run all the time, can skip and continue training
activation_set = ["sigmoid","tanh","ReLU","Leaky ReLU","ELU"]
neurons = [500,200,10]
activations = [2,3,0]       #last layer must be 0 to be able to use cross entropy cost function
alphas = [1,1,1]      #must be corresponding to activation function
weights_1, bias_1, weights_2,bias_2,weights_3,bias_3 = initialization(neurons,activations)


# In[13]:


#main function
max_iterations = 5000
learning_rate = 0.1
previous_loss = 99999999
best_test_accuracy = 0.0
previous_accuracy = 0.0

for i in range(max_iterations):
    
    
    # X is training data
    z1,a1,z2,a2,_,a3 = forward_all_layers(np.transpose(X),weights_1, bias_1, weights_2,bias_2,weights_3,bias_3, activations,alphas)
           
    if(i%100==0):
        
        loss = calculate_cross_entropy_loss(a3)
        print 'iteration: ', i, "loss: ", loss
        if(loss - previous_loss > 10000 or abs(previous_loss - loss) < 0.00000001 or   math.isnan(loss)):
            print('loss is getting worse')
            break    
        previous_loss = loss 
    
    #back propagation
    dldz_3, dldw_3, dldb_3 = lastLayerGradient(a3,a2)

    dldz_2, dldw_2, dldb_2 = getGradient(a1, z2, a2, dldz_3, weights_3, activations[-2],alphas[-2]) #could be wrong

    dldz_1, dldw_1, dldb_1 = getGradient(X.T, z1, a1, dldz_2, weights_2, activations[-3],alphas[-3]) #could be wrong

    
    #update weights    
    weights_1 -= learning_rate*dldw_1
    weights_2 -= learning_rate*dldw_2
    weights_3 -= learning_rate*dldw_3
    
    #update bias 
    bias_1 -= learning_rate*dldb_1
    bias_2 -= learning_rate*dldb_2
    bias_3 -= learning_rate*dldb_3
    
    if(i > 0 and i % 100 == 0  ):
        _,_,_,_,_,a3_test = forward_all_layers(np.transpose(X_test),weights_1, bias_1, weights_2,bias_2,weights_3,bias_3, activations,alphas)
        predict_test = np.argmax(a3_test, axis=0)
        test_length = len(predict_test)
        test_acc = accuracy(predict_test.reshape(test_length,1),y_test)
        if (test_acc > best_test_accuracy):
            best_test_accuracy = test_acc
            print 'New best accuracy: ', test_acc, 'at i=', i, 'with neurons', neurons, 'activations,', activations, 'and alphas', alphas
            best1, best2,best3,best4,best5,best6,best7,best8= weights_1, bias_1, weights_2,bias_2,weights_3,bias_3, activations,alphas
        #early stopping
        if (test_acc < previous_accuracy - 0.01 and test_acc > 0.92  or test_acc < best_test_accuracy - 0.01):
            print('Break. Test data accuracy starts to decrease, ', test_acc, 'at iteration', i)
            break
        previous_accuracy = test_acc

print "Total number of iterations ", i+1
print 'My optimized test data accuracy', best_test_accuracy


#_,_,_,_,_,a3_test = forward_all_layers(np.transpose(X_test),weights_1, bias_1, weights_2,bias_2,weights_3,bias_3, activations,alphas)
#predict_test = np.argmax(a3_test, axis=0)
#test_length = len(predict_test)
#print('final test data accuracy', accuracy(predict_test.reshape(test_length,1),y_test))


# In[ ]:


predict = np.argmax(soft_max(a3), axis=0)
print('final train data accuracy',accuracy(predict.reshape(3500,1),y_digit))


# In[ ]:


_,_,_,_,_,a3_final = forward_all_layers(np.transpose(X_test),best1, best2,best3,best4,best5,best6,best7,best8)
predict_final = np.argmax(soft_max(a3_final), axis=0)
L = len(predict_final)
print('final test data accuracy', accuracy(predict_final.reshape(L,1),y_test))


# In[ ]:




