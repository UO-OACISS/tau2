#!/usr/bin/env python

import os
import zlib, base64, json

def querydataj(cmd='', decompress=False):
    f = os.popen("%s" % cmd, "r")
    lines=[]
    while True:
        l = f.readline()
        if not l:
            break
        lines.append(l)
    f.close()

    jtext = []
    for l in lines:
        if decompress:
            tmp=zlib.decompress(base64.b64decode(l))
            for ltmp in tmp.split():
                jtext.append(ltmp)
        else:
            jtext.append(l)

    ret = [] # return an array of dict objects
    for jt in jtext:
        try:
            j = json.loads(jt)
        except ValueError, e:
            continue
        ret.append(j)

    return ret
