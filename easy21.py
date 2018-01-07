import random

N0 = 100
episode_over = False
Q = {}
N = {}
V = []

def initial_state():
    return (random.randint(1,10), random.randint(1,10))

def step(s, a):
    global episode_over
    dsum, psum = s
    # a = 1: stick, a = 0: hit
    if(a == 1):
        while( (dsum < 17) and (dsum > 0)):
            point = random.randint(1,10)
            color = random.randint(1,3)
            if(color == 1):
                point *= -1
            dsum += point
        episode_over = True
    else:
        point = random.randint(1,10)
        color = random.randint(1,3)
        if(color == 1):
            point *= -1
        psum += point
        if((psum > 21) or (psum < 1)):
            episode_over = True
    _s = (dsum, psum)
    r = reward(_s)
    return (_s, r)

def reward(s):
    global episode_over
    dsum, psum = s
    if(episode_over):
        if((psum > 21) or (psum <1)):
            r = -1
        elif((dsum > 21) or (dsum < 1)):
            r = 1
        else:
            if(dsum > psum):
                r = -1
            elif(dsum < psum):
                r = 1
            else:
                r = 0
    else:
        r = 0
    return r

def egreedy(s):
    s0 = (s,0)
    s1 = (s,1)
    Ns0 = N[s0] if (s0 in N) else 0
    Ns1 = N[s1] if (s1 in N) else 0
    Nsa = Ns0 + Ns1
    e = N0/(N0 + Nsa)
    if(random.random() > e):
        Q0 = Q[s0] if(s0 in Q) else 0
        Q1 = Q[s1] if(s1 in Q) else 0
        if(Q0 > Q1):
            a = 0
        else:
            a = 1
    else:
        a = random.randint(0,1)
    return a

def greedy(s):
    s0 = (s,0)
    s1 = (s,1)
    Q0 = Q[s0] if (s0 in Q) else 0
    Q1 = Q[s1] if (s1 in Q) else 0
    if(Q0 > Q1):
        a = 0
    else:
        a = 1
    return a

def update_Q(sa, G):
    N[sa] = N.get(sa, 0) + 1
    Q[sa] = Q.get(sa, 0) + (G - Q.get(sa, 0))/N[sa]

winper = []
lossper = []
drawper = []
epinum = []

def mc_train(episode = 10000):
    global episode_over
    print("Start training")
    for i in range(1, episode+1):
        #print("episode ", i)
        s = initial_state()
        steps = []
        G = 0
        episode_over = False
        while(episode_over == False):
            a = egreedy(s)
            steps.append((s,a))
            s, r = step(s,a)
            G += r
        for sa in steps:
            update_Q(sa, G)
        if( ( (i<10000) and ((i % 100)== 0)) or (i % 10000 == 0)):
            epinum.append(i)
            test()
    print("Training is over")

def test(t_episodes=100000):
    global episode_over
    win = loss = draw = 0
    for i in range(1, t_episodes+1):
        s = initial_state()
        episode_over = False
        r = 0
        while(episode_over == False):
            a = greedy(s)
            s, r = step(s,a)
        if(r == 1):
            win += 1
        elif(r == 0):
            draw += 1
        else:
            loss += 1
    total = t_episodes
    winper.append(win/total)
    drawper.append(draw/total)
    lossper.append(loss/total)
    
mc_train(3000000)

for i in range(1, 11):
    for j in range(1, 22):
        V.append( (i, j, max( Q.get(((i, j), 0), 0), Q.get(((i, j), 1), 0)) ) )


# Dump the result of training and testing
import os
import pickle
with open('mc_result/test_result', 'wb') as f1:
    pickle.dump([epinum, lossper, drawper, winper], f1)
with open('mc_result/train_result', 'wb') as f2:
    pickle.dump([Q,N,V], f2)

# Plot value function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
v = np.array(list(set(V)))
x = v[:, 0]
y = v[:, 1]
z = v[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x,y,z)
plt.show()
plt.clf()

# Plot the result of testing
plt.plot(epinum,winper, "bo-", epinum, lossper, "ro-")
plt.grid(True, linestyle = "--")
plt.show()
