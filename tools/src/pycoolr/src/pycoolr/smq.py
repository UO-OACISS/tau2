#!/usr/bin/env python

#
# simple data producer and consumer class, sharing one hash and
# limited size deque.
#

import sys, time
import socket, select
import threading

import json

from collections import deque
import random as r

class producer:
    def __init__(self, ip, port = 23458, maxdq = 1800):
        self.ip = ip
        self.port = port
        self.maxdq = maxdq

        self.dict = {}
        self.dq = deque()

        self.timeoutsec = 0.5 # timeout for select
        self.seq = 0
        self.debug = 0

    def start(self):
        self.stop_ev = threading.Event()
        self.t = threading.Thread(target = self.handler, args = [self.stop_ev])
        self.t.start()

    def stop(self):
        self.stop_ev.set()

    def active(self):
        return self.t.isAlive()

    def append(self,str):
        self.dq.append([self.seq, str])
        self.seq += 1
        while len(self.dq) >= self.maxdq:
            self.popleft()

    def dispatcher(self, c, s, stop_ev):
        while not stop_ev.is_set():
            try:
                buf = c.recv(256)
            except socket.error, e:
                if e.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    stop_ev.wait(self.timeoutsec)
                    continue
                else:
                    print e
                    break
                
            # buf should be available for read

            if len(buf) <= 0:
                # it is likely that the client closed
                # the connection, so let's quite the loop
                break
            try:
                d = json.loads(buf)
            except:
                # just ignore the request if we cannot parse it
                continue
            cmd = d['cmd']
            if len(cmd) > 0:
                if self.debug > 0:
                    print 'cmd', cmd, len(self.dq)
                if cmd == 'quit':
                    break
                if cmd == 'cfg' :
                    c.send(json.dumps(self.dict))

                if cmd == 'clear':
                    self.dq.clear()
                    ret = {}
                    ret['len'] = 0
                    c.send(json.dumps(ret))
                if cmd == 'len':
                    ret = {}
                    ret['len'] = len(self.dq)
                    c.send(json.dumps(ret))
                if cmd == 'item':
                    ret = {}
                    if len(self.dq) > 0:
                        i = self.dq.popleft()
                        ret['seq'] = i[0]
                        ret['item'] = i[1]
                    else:
                        ret['item'] = ''
                    ret['len'] = len(self.dq)
                    c.send(json.dumps(ret))

    def handler(self, stop_ev):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(0)

        s.bind((self.ip, self.port))
        s.listen(1) # accept only connection to simplify

        while not stop_ev.is_set():
            r = select.select([s], [], [], self.timeoutsec)
            if r[0]:
                c, addr = s.accept()
                self.dispatcher(c, s, stop_ev)

        s.close()
        self.stopflag = False

class consumer:
    def __init__(self, ip, port = 23458):
        self.ip = ip
        self.port = port


    def get(self,d):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.ip, self.port))
            self.s.send(json.dumps(d))
            buf = self.s.recv(1024)
            d = json.loads(buf)
        except:
            return {}
        finally:
            self.s.close()
        return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'Usage: %s p|c ip [port]'
        print ''
        print 'p is the producer mode'
        print 'c is the consumer mode'
        sys.exit(1)

    port = 23458

    ip = sys.argv[2]

    if len(sys.argv) >= 4:
        port = int(sys.argv[3])

    print 'ip:', ip
    print 'port:', port

    if sys.argv[1] == "p":
        #
        # producer mode test
        #
        dp = producer(ip, port)
        dp.start()
        dp.dict['label'] = 'test'
        cnt = 0
        while dp.active():
            str = "t=%lf cnt=%d" % (time.time(), cnt)
            print str
            dp.append(str)
            cnt += 1
            if cnt > 20:
                dp.stop()
            time.sleep(1)
        sys.exit(0)

    #
    # consumer mode test
    #

    dc = consumer(ip, port)
    d = dc.get({'cmd':'cfg'})
    if len(d) == 0:
        # assume the producer is not running or dead
        sys.exit(1)
    print 'cfg', d
    d = dc.get({'cmd':'len'})
    print 'len:', d['len']
#    d = dc.get({'cmd':'clear'})

    while True:
        d = dc.get({'cmd':'len'})
        if len(d) == 0:
            break

        while True:
            d = dc.get({'cmd':'len'})
            if len(d) == 0:
                break
            if d['len'] == 0:
                break
            d = dc.get({'cmd':'item'})
            if len(d) == 0:
                break
            print 'item:', d['item'], 'remain:', d['len']
        time.sleep(2)

    print 'done'

    sys.exit(0)
