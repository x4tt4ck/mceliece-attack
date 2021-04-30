#!/usr/bin/env python3

import math
import random
import numpy

def C(n, k):
    return math.factorial(n) // (math.factorial(n-k) * math.factorial(k))

def RM_height(r, m):
    height = 0
    for i in range(r + 1):
        height += C(m, i)
    return height

def is_degenerate_matrix(matr):
    matr = numpy.array(matr)
    det = round(numpy.linalg.det(matr))
    print("Det is", det)
    if det % 2 == 0:
        return True
    else:
        return False

def build_M(k):
    M = []
    for i in range(k):
        row = [random.randint(0, 1) for i in range(k)]
        M.append(row)

    while(is_degenerate_matrix(M)):
        print("Oops")
        M = []
        for i in range(k):
            row = [random.randint(0, 1) for i in range(k)]
            M.append(row)

    return M


def build_RM_basis(r, m):
    """Builds RM(r, m)"""
    result = [ [1]*pow(2, m) ]

    if r == 0:
        return result

    for i in range(m-1, -1, -1):
        result.append(([0]*pow(2, i) + [1]*pow(2, i)) * pow(2, m-1-i))

    substitution = []
    for i in range(pow(2, m)):
        binary = "{0:b}".format(i)
        binary = "0"*(r - len(binary)) + binary
        binary = [j for j, letter in enumerate(binary) if letter == '1']
        substitution.append(binary)

    substitutions = []
    for row in substitution:
        if len(row) > 1:
            substitutions += [row]

    substitutions.sort(key=len)
    #print(substitutions)

    for row in substitutions:
        if len(row) <= r:
            #print(row)
            tmp = []
            for k in range(pow(2, m)):
                mul = 1
                for elem in row:
                    mul *= result[1+elem][k]
                tmp.append(mul)
            result.append(tmp)

    return result

def build_P(m):
    shuffled = [ i for i in range(0, pow(2, m)) ]
    random.shuffle(shuffled)
    print(shuffled)

    result = []
    for i in range(pow(2, m)):
        tmp = []
        for j in range(pow(2, m)):
            if i == shuffled[j]:
                tmp.append(1)
            else:
                tmp.append(0)
        result.append(tmp)

    return result

def minor(arr,i,j):
    # ith row, jth column removed
    return arr[numpy.array(list(range(i))+list(range(i+1,arr.shape[0])))[:,numpy.newaxis],
               numpy.array(list(range(j))+list(range(j+1,arr.shape[1])))]

def invert_matrix(M):
    print(f"[*] Matr M[{len(M)} x {len(M)}]")
    result = []
    #print(M)
    for i in range(len(M)):
        row = []
        for j in range(len(M)):
            #print(f"{i},{j} tmp:")
            tmp = numpy.delete(numpy.delete(M, i, 0), j, 1)
            #print(tmp)
            row.append(round(numpy.linalg.det(tmp)) % 2)
        result.append(row)

    return result

def read_pubkey(path):
    with open(path, 'r') as f:
        Gpub = numpy.loadtxt(f, delimiter=',', dtype=int)

    return Gpub

def read_privkey(path):
    with open(path, 'r') as f:
        key = f.read()

    key = key.split('\n\n')
    Pinv = key[0].split('\n')
    Minv = key[1].split('\n')

    Ppriv = []
    Qpriv = []
    for row in Pinv:
        Ppriv.append([int(x) for x in row.split(',')])

    for row in Minv[:-1]:
        Qpriv.append([int(x) for x in row.split(',')])

    Ppriv = numpy.matrix(Ppriv)
    Qpriv = numpy.matrix(Qpriv)

    return Ppriv, Qpriv

def encrypt_message(m, p):
    return numpy.remainder(numpy.matmul(m, p), 2)

def weight(X):
    w = 0
    for i in X:
        if i == 1:
            w += 1

    return w

def add_rows(Gpub, i, j):
    for k in range(len(Gpub[0])):
        Gpub[i,k] = (Gpub[i,k] + Gpub[j,k]) % 2
    return Gpub

def Gpub_triangle(columns, Gpub):
    print('Indexos:\n', columns)
    print('Gpub:\n', Gpub)

    position = 0
    for column in columns:
        for i in range(position, len(Gpub)):
            if Gpub[i, column] == 1:
                break
        #print(i, position)
        Gpub[[position, i]] = Gpub[[i, position]]
        for i in range(position + 1, len(Gpub)):
            if Gpub[i, column] == 1:
                for k in range(len(Gpub[0])):
                    Gpub[i, k] = (Gpub[i, k] + Gpub[position, k]) % 2

        position = position + 1

    return Gpub

def add2graph(g, x, y):
    if x == y and x not in g:
        g[x] = set({})
        return g
    elif x == y:
        return g

    if x in g:
        g[x].add(y)
    else:
        g[x] = set({y})

    if y in g:
        g[y].add(x)
    else:
        g[y] = set({x})

    return g

def graphcheck(g, r, m):
    print(g)
    flag = True
    for key in g:
        if len(g[key]) > (pow(2, m-r) - 1):
            flag = False
    return flag

def graphcheckF(g, r, m):
     print(g)
     flag = len(g.keys()) == (pow(2, m) - pow(2, m-r))
     print(flag)
     for key in g:
         if len(g[key]) != (pow(2, m-r) -1):
             flag = False
     return flag

def mode_2(order, m, f):
    #order = 2
    #m = 4


    RM_basis = build_RM_basis(order, m)
    RM = numpy.matrix(RM_basis)
    '''P = build_P(m)
    k = RM_height(r, m)
    M = build_M(k)

    RMnum = numpy.matrix(RM_basis)
    Pnum = numpy.matrix(P)
    Mnum = numpy.matrix(M)

    print(RMnum)
    print(Pnum)
    print(Mnum)
    print()

    X = numpy.matmul(numpy.matmul(Mnum, RMnum), Pnum)
    X = numpy.remainder(X, 2)

    Pnum_inv = Pnum.transpose()
    print(Pnum_inv)

    Q = invert_matrix(M)

    with open('public.key', 'a') as f:
        numpy.savetxt('public.key', X, delimiter=',', fmt='%d')

    with open('private.key', 'a') as f:
        numpy.savetxt(f, P, delimiter=',', fmt='%d')
        f.write("\n")
        numpy.savetxt(f, Q, delimiter=',', fmt='%d')
    '''

    '''
    TASK 3
    print("[*] Public key (Gpub):")
    Gpub = read_pubkey('../keys/public.key')
    print(Gpub)
    #Msg = numpy.array([ random.randint(0, 1) for i in range(len(Gpub)) ])
    #Ciph = encrypt_message(Msg, Gpub)

    print("[*] Private keys (P, Q):")
    Ppriv, Qpriv = read_privkey('../keys/private.key')
    print(Ppriv)
    print(Qpriv)

    Gpub_1 = numpy.remainder(numpy.matmul(Qpriv, Gpub), 2)
    print("[*] Q * Gpub:")
    print(Gpub_1)

    Gpub_x = numpy.remainder(numpy.matmul(numpy.matmul(Qpriv, Gpub), Ppriv), 2)

    print("[*] Qpiv * Gpub * Ppriv = RM(r, m). Compare them!")
    print(Gpub_x)
    print(RM)
    #print("Message")
    #print(Msg)
    #print("Ciphertext")
    #print(Ciph)
    '''
    Gpub = read_pubkey(f)
    B = []
    for r in range(order, 1, -1):
        while True:
            rand_vec = numpy.array([random.randint(0,1) for i in range(len(Gpub))])
            X_1 = numpy.remainder(numpy.matmul(rand_vec, Gpub), 2)

            X_1_weight = weight(X_1)

            while(X_1_weight != pow(2, m-r)):
                rand_vec = numpy.array([random.randint(0,1) for i in range(len(Gpub))])
                X_1 = numpy.remainder(numpy.matmul(rand_vec, Gpub), 2)
                X_1_weight = weight(X_1)

            ones = [j for j, el in enumerate(X_1) if el == 1]

            M = 1
            #print(Gpub)
            C0 = Gpub_triangle(ones, Gpub)


            print('C0_Gpub_Triangle:\n', C0)

            #for i in range(len(ones)):

           #    C0 = numpy.delete(C0, ones[i], 1)
            C_new = []

            for i in range(len(C0)):
                flag = True
                for j in ones:
                    flag = flag and (C0[i][j] == 0)
                if flag:
                    C_new.append(C0[i])

            C0 = C_new
            C0 = numpy.delete(C0, ones, 1)
            print('C0_pre:\n', C0)


            #j = 0
            #for i in ones:
            #    C0 = numpy.delete(C0, i-j, 1)
            #    j = j + 1

            #print("C0:\n", C0)
            X_save = []
            graph = {}
            k = RM_height(r-1, m)
            while True:
                X_2 = numpy.array([random.randint(0, 1) for i in range(len(C0))])
                X_2 = numpy.remainder(numpy.matmul(X_2, C0), 2)
                X_2_weight = weight(X_2)

                epsilon = math.sqrt(1 - 1/pow(2, m - 2*r + 1))
                r_bond = pow(2, m - 2*r + 1) * (pow(2, r) - 1) * epsilon
                l_bond = pow(2, m - r)

                print('Bonds:', l_bond, r_bond)
                while(not (X_2_weight >= l_bond and X_2_weight < r_bond)):
                    X_2 = numpy.array([random.randint(0, 1) for i in range(len(C0))])
                    X_2 = numpy.remainder(numpy.matmul(X_2, C0), 2)
                    X_2_weight = weight(X_2)

                print(f'X_2 ({X_2_weight}):\n', X_2)
                X_save.append(X_2)
                HPC = numpy.zeros((pow(2, m) - pow(2, m-r), pow(2, m) - pow(2, m-r)), dtype=int)
                HPC_weight = 0

                print('HPC:\n', HPC)

                for x in X_save:
                    for i in range(len(x)):
                        for j in range(i+1, len(x)):
                            if x[i] == x[j] and x[i] == 1:
                                HPC[i,j] += 1
                                HPC_weight += 1

                l = []

                for i in range(len(HPC)):
                    for j in range(len(HPC[0])):
                        if HPC[i, j]:
                            l.append(HPC[i, j])

                c = int(numpy.sum(l) // len(l)) + 7

                print(f'HPC_{HPC_weight}:\n', HPC)


                for i in range(len(HPC)):
                    for j in range(len(HPC)):
                        if HPC[i, j] > c:
                            graph = add2graph(graph, i, j)

                graph_ok = graphcheck(graph, r, m)
                if not graph_ok:
                    M = 1
                    graph = {}
                    X_save = []
                print(graph)
                M += 1
                graph_filel = graphcheckF(graph, r, m)
                if graph_filel:
                    vs = set()
                    for key in graph:
                        vs.add(graph[key].add(key))

                    vs_new = []
                    for v in vs:
                        vs_new.append(list(v))

                    for v in vs_new:
                        for c in ones:
                            for i in len(v):
                                if v[i] >= c:
                                   v[i] +=1
                    flag1 = False
                    flag2 = False
                    for v in vs_new:
                        b = numpy.array([1 if ((i in v) or (i in ones)) else 0 for i in range(pow(2,m))])
                        B.append(b)
                        if weight(Gpub_triangle(range(pow(2,m), numpy.array(B))[-1])) == 0:
                            del B[-1]
                        if len(B) == k:
                            B_T = numpy.array(B).transpose()
                            flag1 = True

                            for s in tmp:
                                w = weight(s)
                                flag1 = flag1 and (w != 0)

                            flag2 = True
                            for row in B_T:
                                idx = (B_T == list(row)).all(axis=1).nonzero()[0]
                                flag2 = flag2 and (len(idx) == 1)

                            if flag1 and flag2:
                                break
                            del B[-1]

                    if flag1 and flag2:
                        break

        Gpub = numpy.matrix(B)
        # не до конца, просто спустился до RM(1, m)


if __name__ == "__main__":
   mode_2(2, 4, '../keys/public.key')


