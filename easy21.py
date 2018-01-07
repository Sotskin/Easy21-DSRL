import random

N0 = 100
episode_over = False
Q = {}
N = {}

def initial_state():
    return (random.randint(1,10), random.randint(1,10))

def step(s, a):
    global episode_over
    dsum, psum = s
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
    if(sa not in N):
        N[sa] = 1
    else:
        N[sa] += 1
    if(sa not in Q):
        Q[sa] = 0
    Q[sa] = Q[sa] + (G - Q[sa])/N[sa]

winper = []
lossper = []
drawper = []
epinum = []

def train(episode = 10000):
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
        for j in range(len(steps)):
            update_Q(steps[j], G)
        if( ( (i<10000) and ((i % 100)== 0)) or (i % 10000 == 0)):
            epinum.append(i)
            test()
    print("Training is over")

def test(t_episodes=10000):
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
    
train(3000000)
import pickle
with open('outfile', 'wb') as f:
    pickle.dump([epinum, lossper, drawper, winper], f)

import numpy as np
import matplotlib.pyplot as plt

plt.plot(epinum,winper, "bo-", epinum, lossper, "ro-")
plt.grid(True, linestyle = "--")
plt.show()
