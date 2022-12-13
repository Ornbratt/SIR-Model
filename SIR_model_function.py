import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math

Beta = 0.8

Gamma = 0.01

Diff = 0.6


# Parameters of the simulation
n = 1000                # Number of agents 
initial_infected = 10   # Initial infected agents 
N = 100000  # Simulation time
l = 100     # Lattice size
Timesteps = 1000       # Timestep counter 
t = 0
plottingTrue = 1

def SIR_model(infectionRate,recoveryRate,diffusion,latticeSize,numberOfAgents,timeSteps,initiallyInfected,t0,PlotOnOff):
    t=t0
    
    doPlot = PlotOnOff
    Beta = infectionRate

    Gamma = recoveryRate

    Diff = diffusion

    n = numberOfAgents                # Number of agents 
    initial_infected = initiallyInfected   # Initial infected agents 
    l = latticeSize     # Lattice size
    Timesteps = timeSteps       # Timestep counter

    recoveredAmountList = []
    susceptibleAmountList = []
    infectedAmountList = []

    # Physical parameters of the system 
    x = np.floor(np.random.rand(n)*l)          # x coordinates            
    y = np.floor(np.random.rand(n)*l)          # y coordinates  
    S = np.zeros(n)                  # status array, 0: Susceptiple, 1: Infected, 2: recovered
    I = np.argsort((x-l/2)**2 + (y-l/2)**2)
    S[I[1:initial_infected]] = 1              # Infect agents that are close to center 

    nx = x                           # udpated x                  
    ny = y                           # updated y                  

    for k in range(Timesteps):
        t = t + 1
    
        B = Beta
        G = Gamma
        D = Diff
    
        steps_x_or_y = np.random.rand(n)
        steps_x = steps_x_or_y < D/2
        steps_y = (steps_x_or_y > D/2) & (steps_x_or_y < D)
        nx = (x + np.sign(np.random.randn(n)) * steps_x) % l
        ny = (y + np.sign(np.random.randn(n)) * steps_y) % l
    
        for i in np.where( (S==1) & ( np.random.rand(n) < B ))[0]:     # loop over infecting agents 
            S[(x==x[i]) & (y==y[i]) & (S==0)] = 1         # Susceptiples together with infecting agent becomes 
            #np.sum(S==2)
        S[ (S==1) & (np.random.rand(n) < G) ] = 2         # Recovery
        susceptibleAmountList.append(np.sum(S==0))
        infectedAmountList.append(np.sum(S==1))
        recoveredAmountList.append(np.sum(S==2))

        #S[ (S==1) & (np.random.rand(n) < death) ] = 3         # Death
        #if np.sum(S==1) == 0:
        #    break

        if t==500:
            if doPlot:
                it_100 = t
                x_100 = x.copy()
                y_100 = y.copy()
                z_100 = S.copy()
                colors_100 = ['mediumturquoise' if label==0 else 'coral' if label== 1 else 'palegreen' for label in z_100]
                plt.title('Agent position and status at t=' + str(t))
                plt.scatter(nx,ny,color=colors_100)
                            # 'Infected:' + str(np.sum(S==1))
                plt.show()
        x = nx                                              # Update x 
        y = ny                                              # Update y 
    #print(susceptibleAmountList[-1])
    if doPlot:
        TimestepsPlot = list(range(t))
        plt.scatter(TimestepsPlot, susceptibleAmountList, s=5, color='mediumturquoise', label='Susceptible' )
        plt.scatter(TimestepsPlot, infectedAmountList, s=5, color='coral', label='Infected' )
        plt.scatter(TimestepsPlot, recoveredAmountList, s=5, color='palegreen', label='Recovered' )
        plt.legend(['Suceptible','Infected','Recovered','Dead'])
        plt.title('Parameters used: $B$ = ' + str(B) + ', $\gamma =$' + str(G) + ', $d =$' + str(D))
        plt.ylabel('Number of agents')
        plt.xlabel('Timesteps')
        plt.show()
    global endRecoveredSum
    endRecoveredSum = recoveredAmountList[-1]+endRecoveredSum
    return endRecoveredSum

# 11.1
#SIR_model(infectionRate,recoveryRate,diffusion,latticeSize,numberOfAgents,timeSteps,initiallyInfected,t0):
endRecoveredList=[]
#SIR_model(0.7,0.01,0.6,100,1000,2000,10,0,False)
colorList=['royalblue','mediumseagreen']
labelList=['$\gamma=0.1$','$\gamma=0.2$']
meanAverageOver =10
recoveredAmountList =[]
for Ga in range(2):
    infectionRateList=[]
    Gam =(Ga+1)/100
    print(Gam)
    for Be in range(25):
        Bet=0
        Bet = (Be+1)/25
        infectionRateList.append(Bet)
        print(Bet)
        endRecoveredSum= 0
        for i in range(meanAverageOver):
            SIR_model(Bet,Gam,0.8,100,1000,3000,10,0,False)
            #endRecoveredSum = recoveredAmountList[-1]+endRecoveredSum
        endRecoveredList.append(endRecoveredSum/meanAverageOver)
        #endRecoveredSum = 0
    TimestepsPlot2 = list(range(len(endRecoveredList)))
    plt.scatter(infectionRateList, endRecoveredList, s=20,color=colorList[Ga],label=labelList[Ga])
    plt.ylabel(r'$R_\infty$')
    plt.xlabel(r'$\beta$')
    endRecoveredList=[]
plt.legend()
plt.show()
#susceptibleAmountList[-1]









