"""
    Copyright (c) 2011,2012,2016,2017 Merck Sharp & Dohme Corp. a subsidiary of Merck & Co., Inc., Kenilworth, NJ, USA.

    This file is part of the Deep Neural Network QSAR program.

    Deep Neural Network QSAR is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

"""
Original version by George Dahl in Dec.11, 2012.
Last modified by Yuting Xu in Jul.6, 2016.
"""

from sys import stderr

class Counter:
    def __init__(self, step=10):
        self.cur = 0
        self.step = step

    def tick(self):
        self.cur += 1
        if self.cur % self.step == 0:
            stderr.write( str(self.cur ) )
            stderr.write( "\r" )
            stderr.flush()

    def done(self):
        stderr.write( str(self.cur ) )
        stderr.write( "\n" )
        stderr.flush()

class Progress(object):
    def __init__(self, numSteps):
        self.total = numSteps
        self.cur = 0
        self.curPercent = 0
    def tick(self):
        self.cur += 1
        newPercent = (100*self.cur)/self.total
        if newPercent > self.curPercent:
            self.curPercent = newPercent
            stderr.write( str(self.curPercent)+"%" )
            stderr.write( "\r" )
            stderr.flush()
    def done(self):
        stderr.write( '100%' )
        stderr.write( "\n" )
        stderr.flush()

class DummyProgBar(object):
    def __init__(self, *args): pass
    def tick(self): pass
    def done(self): pass

def ProgressLine(line):
    stderr.write(line)
    stderr.write( "\r" )
    stderr.flush()

def main():
    from time import sleep

    print("Play with ProgressLine:")
    for i in range(100):
        s = str(2.0*i)
        ProgressLine(s)
        sleep(0.1)
    print( "\n" )

    print("Play with Counter:")
    c = Counter(5)
    for i in range(100):
        c.tick()
        sleep(.1)
    c.done()
    print( "\n" )

    print("Play with Progress:")
    p = Progress(100)
    for i in range(100):
        p.tick()
        sleep(.1)
    p.done()
    print( "\n" )

if __name__ == "__main__":
    main()

