import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

N0 = 100
gamma = 1 # No discounting
episode_over = False
result_testing = True
Q = {}
N = {}
V = []
E = {}

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
    Q0 = Q.get(s0, 0)
    Q1 = Q.get(s1, 0)
    if(Q0 > Q1):
        a = 0
    else:
        a = 1
    return a

winper = []
lossper = []
drawper = []
epinum = []

def mc_train(episode = 10000):
    global episode_over
    print("Start MC training")
    for i in range(1, episode+1):
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
            N[sa] = N.get(sa, 0) + 1
            Q[sa] = Q.get(sa, 0) + (G - Q.get(sa, 0))/N[sa]
        if( (result_testing == True) and (( (i<10000) and ((i % 1000)== 0)) or (i % 25000 == 0)) ):
            epinum.append(i)
            result_test()
    print("Training is over")

def result_test(t_episodes=100000):
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


mc_train(1000000)
# Calculate V = max (Q,A)
for i in range(1, 11):
    for j in range(1, 22):
        V.append( (i, j, max( Q.get(((i, j), 0), 0), Q.get(((i, j), 1), 0)) ) )

'''
with open('mc_result/mc_train_result', 'rb') as f:
    Q, N, V = pickle.load(f)
'''

# Dump the result of training and testing if needed
if( result_testing == True):
    with open('mc_result/mc_test_result', 'wb') as f1:
        pickle.dump([epinum, lossper, drawper, winper], f1)
    with open('mc_result/mc_train_result', 'wb') as f2:
        pickle.dump([Q,N,V], f2)

Q_mc = dict(Q)

sqerrors = []
lambs = []

def cal_sqr_error():
    sqerror = 0
    for i in Q_mc:
        sqerror += (Q.get(i, 0) - Q_mc[i])**2
    return sqerror/len(Q_mc)

def sarsa(l = 0, episode = 100000, plote = False):
    global episode_over, Q, N, E
    epi = []
    epierror = []
    Q = {}
    N = {}
    print("Start Sarsa trianing. Lambda = ", round(l, 1), ' ... ')
    for i in range(1, episode+1):
        E = {}
        s = initial_state()
        a = egreedy(s)
        episode_over = False
        while(episode_over == False):
            _s, r = step(s,a)
            _a = egreedy(_s)
            error = r + gamma * Q.get((_s,_a), 0) - Q.get((s,a), 0)
            E[(s,a)] = E.get((s,a), 0) + 1
            N[(s,a)] = N.get((s,a), 0) + 1
            for j in E:
                alpha = 1/N[j]
                Q[j] = Q.get(j, 0) + alpha * error * E[j]
                E[j] = E[j] * gamma * l
            s = _s
            a = _a
        if(plote == True and i % 1000 == 0):
            epi.append(i)
            epierror.append(cal_sqr_error())
    print("Training is over")
    if(plote == True):
        plt.plot(epi, epierror, "bo-")
        plt.grid(True, linestyle = '--')
        plt.show()
        plt.close()

for i in range(0, 11):
    lamb = 0.1 * i
    lambs.append(lamb)
    sarsa(l = lamb, plote = ((lamb == 0) or (lamb == 1)) )
    sqerrors.append(cal_sqr_error())

theta = np.random.rand(36, 1)
def Q_approx(sa):
    d = sa[0][0]
    p = sa[0][1]
    a = sa[1]
    phi = np.array([[1<=d<=4, 4<=d<=7, 7<=d<=10, 
        1<=p<=6, 4<=p<=9, 7<=p<=12, 10<=p<=15, 13<=p<=18, 16<=p<=21,
        a == 0, a == 1]])

def sarsa_lfa(l = 0, episode = 100000):
    global episode_over, E, N

print("lambdas: ", [round(i, 2) for i in lambs])
print("Mean square errors: ", [round(i, 4) for i in sqerrors])
# Plot value function
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
if(result_testing == True):
    plt.plot(epinum,winper, "bo-", epinum, lossper, "ro-")
    plt.grid(True, linestyle = "--")
    plt.show()
    plt.clf()

# Plot mean square errors
plt.plot(lambs, sqerrors, "bo-")
plt.grid(True, linestyle = "--")
plt.show()
plt.clf()

