#!/usr/bin/env python

import sys, select, termios, tty

class keypress:

    def __init__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)

    def available(self):
        if self.enabled == 0:
            return False
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def readkey(self):
        if self.available():
            return sys.stdin.read(1)

    def is_enabled(self):
        return self.enabled

    def enable(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            self.enabled = True
        except:
            print 'Failed to enable'

    def disable(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.enabled = False


if __name__ == '__main__':
    import time

    kp = keypress()

    kp.enable()
    
    i = 0
    while 1:
        print i
        i += 1
        if kp.available():
            c = kp.readkey()
            if c == 'q':
                break
        time.sleep(1)

    kp.disable()


