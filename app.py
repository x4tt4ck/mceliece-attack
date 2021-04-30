#!/usr/bin/env python3

import sys
import numpy

from modules.utils import *

def test(P1, G, M1):
    print('[*] Mode 3 (testing)')
    print(P1)
    print(G)
    print(M1)
    print(numpy.remainder(numpy.matmul(numpy.matmul(M1, G), P1), 2))

def initialize(r, m):
    RM = numpy.matrix(build_RM_basis(r, m))
    M = numpy.matrix(build_M(RM_height(r, m)))
    P = numpy.matrix(build_P(m))

    Ppriv = P.transpose()
    Qpriv = invert_matrix(M)
    print(Qpriv)
    Gpub = numpy.remainder(numpy.matmul(numpy.matmul(M, RM), Ppriv), 2)

    print('RM')
    print(RM)

    print('P')
    print(P)

    print('M')
    print(M)

    test(Ppriv, Gpub, Qpriv)

    with open('keys/public.key', 'w') as f:
        numpy.savetxt(f, Gpub, delimiter=',', fmt='%d')

    with open('keys/private.key', 'w') as f:
        numpy.savetxt(f, Ppriv, delimiter=',', fmt='%d')
        f.write('\n')

    with open('keys/private.key', 'a') as f:
        numpy.savetxt(f, Qpriv, delimiter=',', fmt='%d')

def main(args):
    print(len(args))
    if(len(args) < 2):
        print(f"Usage: {args[0]} <MODE> <PARAMS>")
        print("  -1       key generation, params: r, m")
        print("  -2       mindler attack, params: r, m, pub file")
        print("  -3       key verification, params: pub and priv files")
        print(f"Example: {args[0]} -1 3 4")
        print(f"Example: {args[0]} -2 3 4 keys/public.key")
        print(f"Example: {args[0]} -3 keys/public.key keys/private.key")
        sys.exit(1)

    if(args[1] == '-1'):
        print("[*] You switched to mode of key generation. Please input r and m")
        r = int(args[2])
        m = int(args[3])
        initialize(r, m)
        print("[*] Done! Private and public keys can be found at keys/ dir")

    if(args[1] == '-3'):
        Gpub = read_pubkey(args[2])
        Ppriv, Qpriv = read_privkey(args[3])
        test(Ppriv, Gpub, Qpriv)

    if(args[1] == '-2'):
        r = int(args[2])
        m = int(args[3])
        f = args[4]
        mode_2(r, m, f)


if __name__ == "__main__":
    main(sys.argv)

