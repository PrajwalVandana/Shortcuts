import bisect
import copy
import functools
import itertools
import math
import os
import random
import re
import turtle
from collections import UserDict, defaultdict, Counter

import inflect
from PIL import Image


class LinkedList:
    def __init__(self, val, next=None):
        """Initializes a linked list."""
        self.val = val
        self.next = next

    def __repr__(self):
        """Returns a string representation of the linked list."""
        return ' --> '.join([repr(ele) for ele in list(self)])

    def __contains__(self, val):
        """Defining how the ```in``` operator should work with a LinkedList object."""
        node = self
        if val == None:
            return True
        while node != None:
            if node.val == val:
                return True
            node = node.next
        return False

    def __iter__(self):
        """Creates an iterator from the linked list."""
        node = self
        res = []
        while node != None:
            res.append(node.val)
            node = node.next
        res.append(None)
        yield from res


class Tree:
    def __init__(self, val, left=None, right=None):
        """Initializes a tree."""
        self.val = val
        self.left = left
        self.right = right

    def __contains__(self, val):
        """Defining how the ```in``` operator should work with a Tree object."""
        return val == self.val or (type(self.left) == type(self) and val in self.left) or \
            (type(self.right) == type(self) and val in self.right)

    def __iter__(self):
        """Returns an iterator of attribute-value pairs for ```self.val```, ```self.left```,
        and ```self.right```.

        Ex: list(iter(tree)) = [(val, 1), (left, t2), (right, t3)], where
        tree, t2, t3 are Tree objects."""
        lst = list(vars(self).items())
        for i in range(len(lst)):
            if type(lst[i][1]) == type(self):
                lst[i] = (lst[i][0], dict(lst[i][1]))
        yield from lst

    def __repr__(self):
        """Returns a string representation of the linked list."""
        return str(dict(self))

    def values(self):
        """A list of values in the tree."""
        res = []
        for val in dict(self).values():
            if val != None:
                if type(val) == dict:
                    res += val.values()
                else:
                    res.append(val)
        return res


class Network:
    """A network/graph with one-way paths, where all nodes are integers."""

    def __init__(self, constructor):
        """Initializes a network/graph.

        constructor: can be a list or dictionary
            if list, then constructor should be a list of 2-tuples from one
                node to another represented as start-end pairs, e.g. (1, 2);
                nodes with no paths from them should have an empty list as a value
            if dict, then constructor should be a dictionary where each key
                contains as values a list of all nodes n for which there is a path to n
        """
        self.paths = {}
        if type(constructor) == list:
            for tup in constructor:
                self.paths[tup[0]] = self.paths.get(tup[0], []) + [tup[1]]
        elif type(constructor) == dict:
            self.paths = constructor
        else:
            raise TypeError('Constructor for a %s object cannot be type %s.' %
                            repr(type(constructor).__name__))

    def __contains__(self, val):
        """The expression ```path in network``` will evaluate to ```True``` if there is a path from the first
        element of ```path``` to the second. ```path``` should be a tuple with 2 elements.

        The expression ```int in network``` will evaluate to ```True``` if that node exists."""
        if type(val) == int:
            return val in self.paths()
        elif type(val) == tuple:
            return path[1] in self.paths.get(path[0], [])
        else:
            raise TypeError("Invalid type for 'in' with %s." %
                            repr(type(val).__name__))

    def path_list(self):
        """Returns a list of all paths as tuples in the form ```(start, stop)```."""
        res = []
        for start, stops in self.paths.items():
            res += [(start, stop) for stop in stops]
        return res

    def __iter__(self):
        """Returns an iterator of start-stop(s) pairs in ```self.paths```.

        Ex: list(iter(network)) = [(1, [2,3]), (2, [3])], where
        network is a Network object."""
        return iter(self.paths.items())

    def depict(self, r=200, node_shape=0, node_size=30, node_rot=0, node_positions=None,
               labeling=True, label_font=("Arial", 10, 'normal'), labels=None,
               label_relpos=lambda pos: pos*1.02, tr=None):
        """Draws a turtle drawing depicting the Network object.

        r: how far the nodes are drawn from the center of the screen \\
        node_shape: the number of sides of the regular polygon that represents the node; \
        ```0``` is a circle, ```1``` and ```2``` are lines of length 2*node_size \\
        node_size: circumradius of each node \\
        node_rot: rotation of the node in degrees, where 0 is the polygon laying flat on a base \\
        node_positions: dictionary mapping from nodes to locations to draw each node, as a tuple, \
        ```Point``` object, or complex number; if ```None```, nodes will be drawn \
        on the circle with radius ```r``` \\
        labeling: if ```True```, labels each node \\
        label_font: passed as the ```font``` argument to ```tr.write``` \\
        labels: a dictionary mapping from nodes to their labels \\
        label_relpos: a function for labeling that should take in a ```Point``` object and \
        return a ```Point``` object; this function will be run on each node's position to decide \
        where to put that node's label
        tr: Turtle object
        """
        if tr == None:
            tr = turtle.Turtle()

        paths = sorted(self.paths.items())
        if node_positions == None:  # adding node positions
            node_positions = {}
            for i in range(len(paths)):
                ang = math.radians(90 - (360/len(self.paths))*i)
                node_positions[paths[i][0]] = r * \
                    Point(math.cos(ang), math.sin(ang))
        else:  # converting all node positions to Point objects
            for node, pos in node_positions.items():
                typ = type(pos)
                if typ not in {Point} | point_types():
                    raise TypeError(
                        "'%s' is not a valid type for a point." % typ.__name__)
                node_positions[node] = Point(pos)

        drawn_lines = set()  # contains lines that are already drawn, to avoid drawing over
        for node, pos in node_positions.items():
            tr.pu()
            tr.goto(pos)
            tr.pd()
            tr.lt(node_rot)
            polygon(node_shape, node_size, tr)
            tr.rt(node_rot)

            if labeling:  # labeling the nodes
                if labels is None:
                    labels = {node: str(node) for node in self.paths}
                tr.pu()
                tr.goto(label_relpos(pos))
                tr.write(labels[node], font=label_font)
                tr.goto(pos)
                tr.pd()

            # drawing the paths
            for stop in self.paths[node]:
                stop_pos = node_positions[stop]
                line = Line(pos, stop_pos)

                if line not in drawn_lines:
                    drawn_lines.add(line)
                    tr.goto(stop_pos)

                # stamp the direction of the path
                tr.pu()
                tr.goto(line.midpoint())
                cur_heading = tr.heading()
                tr.setheading(line.angle() + 180*(line.start != pos))
                tr.fd(r/15)
                cur_shape = tr.shape()
                tr.shape('classic')
                tr.stamp()
                tr.shape(cur_shape)
                tr.setheading(cur_heading)
                tr.goto(pos)
                tr.pd()

    def shortest_paths(self, n1, n2):
        """Returns the shortest path(s) from n1 to n2 as a list of nodes traversed, in order."""
        if n2 in self.paths[n1]:
            return [[n1, n2]]

        res = []
        cur_len = float('inf')
        for i in self.paths[n1]:
            if n1 not in self.paths[i]:
                for path in self.shortest_paths(i, n2):
                    if len(path) == cur_len:
                        res.append([n1] + path)
                    elif len(path) < cur_len:
                        res = [[n1] + path]
                        cur_len = len(path)
            else:
                for j in self.paths[i]:
                    if n2 == j:
                        res.append([n1, i, n2])
                        cur_len = 3
                        break
                    if n1 != j:
                        for path in self.shortest_paths(j, n2):
                            if len(path) == cur_len:
                                res.append([n1, i] + path)
                            elif len(path) < cur_len:
                                res = [[n1, i] + path]
                                cur_len = len(path)

            return res


class DefaultArgDict(UserDict):
    """Similar to ```collections```'s ```defaultdict```, but the ```default_factory```
    takes the index as an argument."""

    def __init__(self, default_factory=lambda *args: None):
        self.default_factory = default_factory
        self.data = {}

    def __missing__(self, i):
        self.data[i] = self.default_factory(i)
        return self.data[i]


class Tag:
    """
    Tags a function.

    used as: ```@tag(tag1, tag2, ...)``` right before a function definition, where ```tag``` is a ```Tag``` object
    and ```tag1``` and ```tag2``` are tags.
    """

    def __init__(self):
        self.mapping = defaultdict(list)

    def __call__(self, *tags):
        def tag_decorator(func):
            for tag in tags:
                self.mapping[tag].append(func)
            return func
        return tag_decorator


class FuncWithString:
    """
    A function, but it has a custom string representation. \\
    Initialized as ```func = FuncWithString(name, parFunc)```, where parFunc is the function func will call.
    """

    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.__name__ = self.func.__name__
        self.__doc__ = self.func.__doc__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return repr(self.func)

    def attributes(self):
        """Returns the __dict__ of self.func, i.e. the function, equivalent to ```self.func.__dict__```.
        To get the attributes of the FuncWithString object, use ```self.__dict__```."""
        return self.func.__dict__


class Point:
    """Represents a point in the Cartesian plane."""

    def __init__(self, x=0, y=0):
        types = num_types()
        types.remove(complex)

        if {type(x), type(y)} <= types:
            self.point = complex(x, y)
        elif type(x) in {type(self)} | point_types():
            if type(x) == tuple:
                x = complex(*x)
            self.point = complex(x)
        else:
            raise TypeError(
                "Incorrect constructor type(s) for 'Point' object. Use a complex number, two ints, or a tuple.")

    def __mul__(self, val):
        assert type(val) in {type(self)} | num_types()
        if type(val) == type(self):
            val = complex(val)
        return self.__class__(*xy_tup(self.point*val))

    def __truediv__(self, val):
        assert type(val) in {type(self)} | num_types()
        if type(val) == type(self):
            val = complex(val)
        return self.__class__(*xy_tup(self.point/val))

    def __rmul__(self, val):
        assert type(val) in {type(self)} | num_types()
        if type(val) == type(self):
            val = complex(val)
        return self.__class__(*xy_tup(self.point*val))

    def __add__(self, val):
        return self.__class__(self.point+val)

    def __radd__(self, val):
        return self + val

    def __sub__(self, val):
        return self.__class__(self.point-val)

    def __rsub__(self, val):
        return -self + val

    def __neg__(self):
        return self.__class__(-self.point)

    def __abs__(self):
        return abs(self.point)

    def __complex__(self):
        return self.point

    def __iter__(self):
        return iter(xy_tup(self.point))

    def __lt__(self, other):
        assert type(other) in {type(self)} | point_types()
        if type(other) == tuple:
            other = complex(*other)
        return abs(self) < abs(other)

    def __le__(self, other):
        assert type(other) in {type(self)} | point_types()
        if type(other) == tuple:
            other = complex(*other)
        return abs(self) <= abs(other)

    def __gt__(self, other):
        assert type(other) in {type(self)} | point_types()
        if type(other) == tuple:
            other = complex(*other)
        return abs(self) > abs(other)

    def __ge__(self, other):
        assert type(other) in {type(self)} | point_types()
        if type(other) == tuple:
            other = complex(*other)
        return abs(self) >= abs(other)

    def __eq__(self, other):
        assert type(other) in {type(self)} | point_types()
        if type(other) == tuple:
            other = complex(*other)
        if type(other) == type(self):
            other = complex(other)
        return other == complex(self)

    def __str__(self):
        return 'Point(%f, %f)' % tuple(self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, i):
        return tuple(self)[i]

    def __setitem__(self, i, val):
        if i == 1:
            self.point += complex(0, val) - complex(0, self.point.imag)
        elif i == 0:
            self.point += val - self.point.real
        else:
            raise IndexError(
                "Index for type 'Point' must be 0 or 1, not '%s'." % str(i))

    @property
    def x(self):
        return self.point.real

    @property
    def y(self):
        return self.point.imag

    def normalized(self):
        return normalized(self)

    def quadrants(self):
        poss = {1, 2, 3, 4}

        if self.y > 0:
            poss.discard(3)
            poss.discard(4)
        elif self.y < 0:
            poss.discard(1)
            poss.discard(2)

        if self.x > 0:
            poss.discard(2)
            poss.discard(3)
        elif self.x < 0:
            poss.discard(1)
            poss.discard(4)

        return poss

    def angle(self):
        """Angle the point makes at the intersection of the line to the origin and
        the positive x-axis (going counterclockwise from the positive x-axis)."""
        quads = self.quadrants()
        norm = self.normalized()

        if len(quads) > 1:
            if quads == {1, 2}:
                return 90
            if quads == {2, 3}:
                return 180
            if quads == {3, 4}:
                return 270
            if quads == {1, 4}:
                return 0
        else:
            quad = quads.pop()
            if quad == 1:
                return math.degrees(math.asin(norm.y))
            if quad == 2:
                return math.degrees(math.acos(norm.x))
            if quad == 3:
                return 180 - math.degrees(math.asin(norm.y))
            if quad == 4:
                return 360 - math.degrees(math.acos(norm.x))

    def polar(self):
        return abs(self), self.angle()


class Line:
    """Represents a line segment in the Cartesian plane."""

    def __init__(self, p1, p2):
        """Note that ```Line(start, stop)``` == ```Line(stop, start)```; ```start``` automatically
        becomes the endpoint with lowest x-coord."""

        if type(p1) not in {Point} | point_types():
            raise TypeError("Type '%s' not a valid point." % type(p1).__name__)
        elif type(p2) not in {Point} | point_types():
            raise TypeError("Type '%s' not a valid point." % type(p2).__name__)

        p1 = Point(p1)
        p2 = Point(p2)

        lst = sorted([p1, p2], key=lambda p: (p.x, p.y))

        self.start = lst[0]
        self.stop = lst[1]

    def angle(self):
        """Angle ```Line(a, b)``` (with b.y > a.y) makes with ```Line(a, x)``` \
        such that ```Line(a, x)``` is parallel to the x-axis."""
        slope = self.slope()
        if slope != None:
            ang = math.degrees(math.atan(slope))
            if ang < 0:
                ang += 180
            return ang
        else:
            return 90

    def slope(self):
        """Returns slope, ```None``` means undefined slope."""
        if self.start.x == self.stop.x:
            return None
        return (self.stop[1]-self.start[1])/(self.stop[0]-self.start[0])

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'Line((%f, %f), (%f, %f))' % (self.start.x, self.start.y, self.stop.x, self.stop.y)

    def __getitem__(self, i):
        if i == 1:
            return self.stop
        elif i == 0:
            return self.start
        else:
            raise IndexError(
                "Index for a Line object must be 0 or 1, not '%s'." % str(i))

    def __setitem__(self, i, val):
        if i == 1:
            self.stop = val
        elif i == 0:
            self.start = val
        else:
            raise IndexError(
                "Index for a Line object must be 0 or 1, not '%s'." % str(i))

    def __iter__(self):
        return iter(self[0], self[1])

    def equation(self, x):
        if slope is None:
            return
        return slope*x+y_int

    def on(self, p):
        """Returns whether ```p``` is on the line."""
        if type(p) == complex:
            p = xy_tup(p)
        if self.slope() != None:
            return self.equation(p[0]) == p[1] and self.start[0] < p[0] < self.stop[0]
        else:
            return p[0] == self.start.x and self.start[1] < p[1] < self.stop[1]

    def y_int(self):
        if self.slope() != None:
            return self.start[1] - self.start[0]*self.slope()
        else:
            return None

    def midpoint(self):
        return (self.start + self.stop)/2

    def draw(self, tr=None):
        if tr is None:
            tr = turtle.Turtle()

        orig = tr.pos()
        tr.pu()
        tr.goto(self.start)
        tr.pd()
        tr.goto(self.stop)
        tr.pu()
        tr.goto(orig)


def convert_point_input(string, sep=' '):
    """converts separated coordinate points, e.g. (1,2) to a list of tuples

    sep: character separating each set of points"""
    lst = string.split(sep)
    for i in range(len(lst)):
        tup = tuple(lst[i].split(','))
        res = float(tup[0][1:]), float(tup[1][:-1])
        lst[i] = res
    return lst


def fibonacci_gen():
    """generator function for the Fibonacci sequence, where
    the Fibonacci sequence is ```f_0 = 0, f_1 = 1, f_2 = 1, f_3 = 2, ...```"""
    prev = 1
    total = 0
    while True:
        yield total
        total, prev = total+prev, total


def fibonacci(n):
    """returns the nth Fibonacci number,
    where the Fibonacci sequence is ```f_0 = 0, f_1 = 1, f_2 = 1, f_3 = 2, ...```
    (sum of previous two terms)"""
    assert n >= 0, "The Fibonacci numbers are not defined for negative indices."
    fib_gen = fibonacci_gen()
    for _ in range(n+1):
        x = next(fib_gen)
    return x


def digital_sum(n):
    """returns the sum of the digits of an integer"""
    assert isinstance(n, int), "Digital sum is defined for integers only."
    return sum([int(digit) for digit in str(n)])


def collapsed_digital_sum(num):
    """returns the recursive (collapsed) digital sum of a number"""
    res = num
    while res > 9:
        res = digital_sum(res)
    return res


def invert_dict(d):
    """inverts a dictionary (keys become values, values become keys)"""
    res = {}
    for key in d:
        res[d[key]] = key
    return res


def is_balanced_parentheses(string):
    """returns whether parantheses are balanced"""
    par = '()'
    for i in range(len(string)):
        if string[i] not in par:
            string = string[:i] + '%' + string[i+1:]
    s = string.replace('%', '')
    count = 0
    for i in range(len(s)):
        if s[i] == '(':
            count += 1
        else:
            count -= 1
        if count < 0:
            return False
    return not count


def average_list(lst):
    """averages a list"""
    return sum(lst)/len(lst)


def choose_from_hist(hist):
    """weighted pseudorandom choice from a dictionary that maps keys to their "weight"""
    items = []
    csum_list = []
    freq_sum = 0
    for key, val in hist.items():
        items.append(key)
        freq_sum += val
        csum_list.append(freq_sum)
    num = random.randint(1, csum_list[-1])
    index = bisect.bisect_right(csum_list, num)-1
    return items[index]


def rom_to_int(numeral):
    """converts Roman numerals to integers"""
    num_dict = {'M': 1000, 'D': 500, 'C': 100,
                'L': 50, 'X': 10, 'V': 5, 'I': 1}
    val = 0
    i = 0
    while i in range(len(numeral)):
        try:
            if num_dict[numeral[i]] < num_dict[numeral[i+1]]:
                val += num_dict[numeral[i+1]]-num_dict[numeral[i]]
                i += 1
            else:
                val += num_dict[numeral[i]]
        except IndexError:
            val += num_dict[numeral[i]]
        i += 1
    return val


def tri_area(a, b, c):
    """computes the area of a triangle given its side lengths using Heron's formula:
    https://en.wikipedia.org/wiki/Heron%27s_formula"""
    s = (a+b+c)/2
    return math.sqrt(s*(s-a)*(s-b)*(s-c))


def sum_squares(n):
    """returns the sum of the first n squares"""
    assert isinstance(n, int), 'Argument to sum_squares must be an integer.'
    return int((n*(n+1)*(2*n+1))//6)


def sum_num(start, stop, step=1):
    """Returns the sum of an arithmetic sequence, from ```start``` to ```stop```
    (inclusive), stepping by ```step```.

    Equivalent to ```sum(range(start, stop, step))``` \
    (but works better on large inputs)."""
    num_terms = (stop - start)//step + 1
    return (start+stop)*num_terms/2


def sum_cubes(n):
    """returns the sum of the first n cubes"""
    assert isinstance(n, int), '\'n\' must be an integer.'
    return int(sum_num(1, n)**2)


def median(lst, floor=True):
    """returns the median of a list

    floor: if ```True```, returns the lesser of the middle two elements
    if the list has an even number of elements, otherwise returns the mean of the middle two elements"""
    assert len(lst), 'The list cannot be empty.'
    lst.sort()
    if len(lst) % 2:
        return lst[(len(lst)-1)//2]
    else:
        sl = len(lst)//2
        return lst[sl-1] if floor else average_list(lst[sl-1:sl+1])


def atbash(msg):
    """encrypts a message using Atbash (flip the alphabet, so 'a' becomes 'z', 'b' becomes 'y', etc.)"""
    for i in range(len(msg)):
        x = ord(msg[i])
        if x in range(97, 123):
            msg = msg[:i]+chr(219-x)+msg[i+1:]
        elif x in range(65, 91):
            msg = msg[:i]+chr(155-x)+msg[i+1:]
    return msg


def caesar(msg, shift):
    """Encrypts a message with Caesar shift (shift forward by some number,
    e.g. if ```shift = 2```, 'a' becomes 'c', 'b' becomes 'd', etc.).

    msg: string \\
    shift: shift number"""
    shift = shift % 26
    for i in range(len(msg)):
        x = ord(msg[i])
        if x in range(97, 123):  # lowercase letters
            x -= 97
            num = (x+shift) % 26
            num += 97
            msg = msg[:i]+chr(num)+msg[i+1:]
        elif x in range(65, 91):  # uppercase letters
            x -= 65
            num = (x+shift) % 26
            num += 65
            msg = msg[:i]+chr(num)+msg[i+1:]
    return msg


def to_decimal(num, base, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    """Converts a number from some base to decimal.

    alphabet: 'digits' that the base system uses"""
    using = str(num)[::-1]
    res = 0
    for i in range(len(using)):
        res += alphabet.find(using[i])*base**i
    return res


def from_decimal(num, base):
    """Converts a number from decimal to a 'base-list', which can be converted into a readable format
    using ```convert_base_list```."""
    lst = []
    while num:
        lst.append(num % base)
        num = num//base
    return lst[::-1]


def convert_base_list(lst, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    """Converts a "base-list" (something like what ```from_decimal``` would output) to a number (in string form)."""
    res = ''
    for i in lst:
        res += alphabet[i]
    return res


def prod_list(lst):
    """returns the product of all numbers in a list"""
    if lst:
        res = 1
        for num in lst:
            res *= num
        return res
    else:
        raise ValueError("List cannot be empty.")


def prime_fctr(n):
    """Returns a prime factorization of a number as a dictionary in the form
    {primefactor: exponent} \\
    Ex: 60 = 2^2 3^1 5^1 => ```{2:2, 3:1, 5:1}```
    Note: ```prime_fctr(1)``` returns ```{}```

    Uses the (6k+-1) algorithm; all primes > 3 are of the form 6k+-1,
    since 2 divides (6k+0), (6k+2), (6k+4), and 3 divides (6k+3)."""
    assert isinstance(n, int), "Can only factorize integers."

    # can't prime factorize nonpositive integers
    if n <= 0:
        raise ValueError("Cannot prime factorize nonpositive integers.")

    res = defaultdict(int)

    # since 2 and 3 don't follow the 6k+-1 rule, we deal with them now
    while not n % 2:
        n //= 2
        res[2] += 1

    while not n % 3:
        n //= 3
        res[3] += 1

    i = 5
    finished = False
    while i in range(int(math.sqrt(n))+1) and not finished:
        while not n % i:
            n //= i
            res[i] += 1

        if n == 1:  # we don't need to check anymore
            finished = True

        i += (i + 3) % 6

    if n != 1:
        res[int(n)] = 1

    return dict(res)


def neat_prim_fctr(n):
    """prime factors of a number in a mathematical format"""
    return ' '.join(["%d^%d" % (key, val) for key, val in prime_fctr(n).items()])


def num_fctrs(n):
    """returns the number of factors of a number"""
    res = 1
    for exp in prime_fctr(n).values():
        res *= (exp+1)
    return res


def dist(p1, p2):
    """
    dist(p1, p2): distance between two points \\
    dist(p, line) OR dist(line, p): distance between point and line
    """
    types = set(map(type, (p1, p2)))
    if types <= point_types() | Point:
        if isinstance(p1, tuple):
            p1 = complex(*p1)
        else:
            p1 = complex(p1)
        if isinstance(p2, tuple):
            p2 = complex(*p2)
        else:
            p2 = complex(p2)
        x1, y1, x2, y2 = xy_tup(p1)+xy_tup(p2)
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    else:
        if isinstance(p1, Line):
            if isinstance(p2, tuple):
                p2 = complex(*p2)
            else:
                p2 = complex(p2)
            p = Point(*xy_tup(p2))
            line = p1
        elif isinstance(p2, Line):
            if isinstance(p1, tuple):
                p1 = complex(*p1)
            else:
                p1 = complex(p1)
            p = Point(*xy_tup(p1))
            line = p2

        m = line.slope()
        if m == None:
            p_perp = Point(line[0][0], p.y)
        elif m == 0:
            p_perp = Point(p.x, line[0][1])
        else:
            x, y = tuple(p)
            x0 = (line.y_int()-y-x/m)/(-1/m - m)
            p_perp = Point(x0, line.equation(x0))

        if line.on(p_perp):
            return dist(p, p_perp)
        return min(dist(p, line[0]), dist(p, line[1]))

    raise TypeError("Your input was in an incorrect format.")


def std_dev(lst):
    """calculates the standard deviation of a data set"""
    mean = average_list(lst)
    square_diff = []
    for num in lst:
        square_diff.append((num-mean)**2)
    return math.sqrt(average_list(square_diff))


def primorial(n):
    """calculates the product of all primes up to ```n```"""
    res = 1
    for i in range(1, n+1):
        if is_prime(i):
            res *= i
    return res


def h_mean(lst):
    """harmonic mean of a list"""
    return len(lst)/sum([1/num for num in lst])


def g_mean(lst):
    """geometric mean of a list"""
    res = 1
    for i in lst:
        res *= i
    return res**(1/len(lst))


def totient(n):
    """number of totatives (numbers less than n that are coprime to n)"""
    res = n
    for fctr in prime_fctr(n):
        res *= 1 - 1/fctr
    return int(res)


def months(numbers=True):
    """returns a list of accepted ways the months can be written
    (including the integers 1 through 12 if ```numbers``` is ```True```) as strings"""
    lst = [
        '1', 'january', 'jan', '2', 'february', 'feb', '3', 'march', 'mar', '4', 'april', 'apr',
        '5', 'may', 'may', '6', 'june', 'jun', '7', 'july', 'jul', '8', 'august', 'aug', '9',
        'september', 'sep', '10', 'october', 'oct', '11', 'november', 'nov', '12', 'december', 'dec'
    ]
    return [ele for ele in lst if not ele.isnumeric() or numbers]

# # date input
# finished = False
# while not finished:
#     date_in = input("Enter date in format mm/dd/yyyy time(24h,e.g. 12:37) timezone(e.g. +0100, -0800, etc.): ")
#
#     lst = re.split('/| |:',date_in)
#     mo,dy,yr,hr,mnt,tzh,tzm = int(lst[0]),int(lst[1]),int(lst[2]),int(lst[3]),int(lst[4]),lst[5][:3],int(lst[5][0]+'1')*int(lst[5][3:])
#     hr,mnt = divmod((hr-int(tzh))*60+mnt-int(tzm),60)
#
#     try:
#         date = datetime.datetime(month=mo,day=dy,year=yr,hour=hr,minute=mnt)
#         finished = True
#     except:
#         print("That wasn't a valid input. Try again.")


def print_attr(obj):
    """print all attributes and their values of any object"""
    for attr in vars(obj):
        print(repr(attr), repr(getattr(obj, attr)))


def sum_fctrs(n):
    """sum of factors of a number"""
    res = 1
    if n == 1:
        return 1
    for fctr, exp in prime_fctr(n).items():
        res *= (fctr**(exp+1)-1)/(fctr-1)
    return int(res)


def flatten_gen(obj):
    """flatten a nested object containing lists and/or tuples
    (generator function that yields each element one by one; see also ```flatten```)"""
    nest = copy.deepcopy(obj)

    while nest:
        ele = nest.pop(0)

        if isinstance(ele, list) or isinstance(ele, tuple):
            nest = ele + nest
        else:
            yield ele


def flatten(nest):
    """flatten a nested object containing lists and/or tuples"""
    return nest.__class__(flatten_gen(nest))


def find_def_class(obj, method):
    """finds the parent class where the method of the child object is defined"""
    for typ in type(obj).mro():
        if method in typ.__dict__:
            return typ


def lcm(lst, *args):
    """lcm of a list using ```lcm2```. Also gathers any arguments and adds them to the list."""
    if isinstance(lst, int):
        lst = [lst]
    lst += list(args)
    res = functools.reduce(lambda x, y: x*y//math.gcd(x, y), lst)
    return res


def gcd(lst, *args):
    """gcd of a list using the ```math``` module's ```gcd``` function. Also gathers any arguments and adds them to the list."""
    if isinstance(lst, int):
        lst = [lst]
    lst += list(args)
    res = functools.reduce(math.gcd, lst)
    return res


def n_mod(num, div):
    """returns nonpositive modulo residue"""
    if (res := num % div):
        return res - div
    else:
        return 0


def opt_mod(num, div):
    """returns nonnegative or negative modulo residue depending on whichever one has a lower absolute value
    (if both equal, returns nonnegative)"""
    res = num % div
    return res if res <= (div/2) else res-div


def prime_gen():
    yield 2
    yield 3
    i = 3
    while True:
        i += 2
        if is_prime(i):
            yield i


def is_prime(num):
    """Uses the (6k+-1) algorithm; all primes > 3 are of the form 6k+-1,
    since 2 divides (6k+0), (6k+2), (6k+4), and 3 divides (6k+3)."""
    if num == 2 or num == 3:
        return True

    if num % 6 not in (1, 5) or num == 1:
        return False

    i = 5
    while i in range(int(num**.5)+1):
        if not num % i:
            return False
        i += (i + 3) % 6
    return True


# # linked list input
# p = inflect.engine()
#     lst = []
#     finished = False
#     i = 1
#     while not finished:
#         num = input("Enter the "+p.ordinal(i)+" value: ")
#         if num.lower() == 'stop':
#             finished = True
#         else:
#             lst.append(i)
#             i += 1
#     list_to_linked(lst,n)


def pattern_sort(lst, pattern, key=None, reverse=False):
    """sorts lst based on pattern
    (e.g. ```pattern_sort(['a','ba','c'], [2, 0, 1], lambda x: len(x))``` would return ```['ba','a','c']```)

    lst: the list to sort \\
    pattern: the pattern to sort with
    (list of numbers, i.e. ```[2, 1, 0]``` would swap the 2th element with the 0th element) \\
    key: sorting key to sort initially before sorting using the pattern (function) \\
    reverse: whether to sort backwards during initial sort (bool)
    """
    lst.sort(key=key, reverse=reverse)
    zip_list = zip(lst, pattern)
    return [ele for ele, _ in sorted(zip_list, key=lambda x: x[1])]


def angle_to_comp(n, deg=False):
    """Returns the complex number with a magnitude of 1 that forms an angle of n with the real axis

    n: angle as float \\
    deg: bool (if ```True```, n is taken to be in degrees, if ```False```, n is taken to be in radians)"""
    if deg:
        n = math.radians(n)
    return complex(math.cos(n), math.sin(n))


def rotate_comp(z, rot, deg=False):
    """Rotates the complex number z by the angle rot

    z: complex number \\
    rot: angle as float \\
    deg: bool (if True, rot is taken to be in degrees, if False, rot is taken to be in radians)"""
    return z * angle_to_comp(rot, deg)


def string_to_linked(s, split_chars=['→', '-->']):
    """Reads a string and converts it to a linked list.

    s: string \\
    split_chars: list of characters to split at

    Returns: linked list node"""
    s += '--> None' if 'None' not in s else ''
    s = s.replace(' ', '')
    lst = re.split('|'.join(split_chars), s)
    return list_to_linked(lst)


def list_to_linked(lst):
    """Reads a list and converts it to a linked list in order.

    lst: list

    Returns: linked list node"""
    if lst[-1] != None:
        lst.append(None)

    prev = None
    for i in range(len(lst)-1, 0, -1):
        prev = LinkedList(lst[i-1], prev)
    return prev


def look_and_say(n):
    """Returns the nth term of the "look and say" sequence, which is defined as follows: beginning
    with the term 1, each subsequent term visually describes the digits appearing in the previous term.
    The first few terms are as follows:

    1
    11
    21
    1211
    111221

    Ex: the fourth term is 1211, since the third term consists of one 2 and one 1.
    """
    assert n, "There is no zeroth term."
    if n == 1:
        return 1
    else:
        s = str(look_and_say(n-1))
        res = ''
        i = 0
        while i in range(len(s)):
            count = 1
            num = s[i]
            while i in range(len(s)-1) and s[i] == s[i+1]:
                count += 1
                i += 1
            res += str(count) + num
            i += 1
        return int(res)


def alt_case(string, lower_first=True):
    """Returns the string with alternating upper and lowercase characters.

    lower_first: if True, first character is lowercase; if False, first character is uppercase"""
    string = string.lower()
    for i in range(len(string)):
        if bool(i % 2) == lower_first:
            string = string[: i] + string[i].upper() + string[i+1:]
    return string


def sort_alpha(lst):
    """Sorts a list of integers in abecedarian (alphabetical) order."""
    engine = inflect.engine()
    res = []
    for num in lst:
        word = engine.number_to_words(num)
        res.append((word, num))
    res.sort()
    return [ele for _, ele in res]


def dict_neat(d, use_repr=True, deep=True, indent=4, use_braces=False, use_commas=False, tab_braces=True, comma_braces=True, begin_indent=0):
    """Returns a human-readable string representation of a dictionary (or defaultdict).

    use_repr: if ```True```, will use ```repr(key)``` and ```repr(val)``` instead of ```str(key)``` and ```str(val)``` \\
    deep: if ```True```, will format all dictionaries that are values neatly as well \\
    indent: how many spaces to indent when formatting nested dictionaries \\
    use_braces: if ```True```, will use braces when formatting \\
    use_commas: if ```True```, will use commas when formatting \\
    tab_braces: if ```True```, items inside braces will not be on the same indentation level as the braces \\
    comma_braces: if ```True```, will use commas after closing braces \\
    begin_indent: level of indentation applied to first level; for example, if ```begin_indent=1```, then
    the entire dictionary will be indented by 1 from the left
    """
    def _dict_neat(d, use_repr=True, deep=True, indent=4, use_braces=False, use_commas=False, tab_braces=True, comma_braces=True, begin_indent=0):
        """Helper function for ```dict_neat```.
        """
        def represent(s):
            return repr(s) if use_repr else str(s)

        comma_char = ',' if use_commas and len(d.values())-1 else ''

        sp = ' '*indent*(begin_indent+(tab_braces and use_braces))

        res = ''
        if use_braces:
            res += ' '*indent*begin_indent + '{' + '\n'

        i = 1
        for key, val in d.items():
            if len(d) == i:
                comma_char = ''
            if type(val) not in dict_types() or not deep:
                res += sp + represent(key) + ': ' + \
                    represent(val) + comma_char + '\n'
            else:
                res += sp + represent(key) + ':' + '\n' + \
                    _dict_neat(val, use_repr, deep, indent,
                               use_braces, use_commas, tab_braces, comma_braces, begin_indent+1) + '\n'
                if use_braces:
                    res = res[:-1] + comma_char + res[-1]
            i += 1

        if use_braces:
            ending_brace = '}' + ','*comma_braces
            res += ' '*indent*begin_indent + ending_brace + '\n'

        return res[:-1]

    res = _dict_neat(d, use_repr, deep, indent,
                     use_braces, use_commas, tab_braces, comma_braces, begin_indent)
    if not use_braces:
        comma_braces = False
    return res[:-1] if comma_braces else res


def rand_color(start=0, stop=256**3-1):
    """Picks a random color from the inclusive range [start, stop]."""
    return convert_color(random.randint(start, stop), tuple)


def opp_color(col):
    col = convert_color(col, int)
    return convert_color(256**3-1 - col, tuple)


def turtle_gif(func, args, kwargs, fps=10, fname=None, path=None, temp_fname=None, temp_path=None, optimize=False, duration=100, tr=None):
    """Saves a gif of a turtle function.

    func: turtle function \\
    args: non-keyword arguments to function (as tuple) \\
    kwargs: keyword arguments to function (as dict) \\
    Ex: ```("'off'",7,'key=9')```\\
    fps: frames per second \\
    fname: filename for the gif \\
    path: where to save the gif \\
    temp_fname: turtle_gif saves multiple .eps images, then deletes them later;
    temp_fname can be used to make sure turtle_gif doesn't overwrite any existing files \\
    temp_path: where to save the temporary multiple images \\
    optimize, duration: passed when creating the gif, like so:
        frames[0].save(path+fname, save_all=True, append_images=frames[1:],
        optimize=optimize, duration=duration, loop=0) \\
    tr: Turtle object
    """
    if tr == None:
        tr = turtle.Turtle()

    func_name = func.__name__
    if fname == None:
        fname = func_name
    if temp_fname == None:
        temp_fname = fname

    if path == None:
        path = os.getcwd()

    if temp_path == None:
        temp_path = os.getcwd()

    path, temp_path = format_input_path(path), format_input_path(temp_path)

    running = True  # bool for whether program is running
    counter = 1  # the number for the temp file name
    frames = []  # list of images

    def save():
        nonlocal counter
        file_str = "%s%d.eps" % (temp_path + temp_fname, counter)  # file name
        tr.getcanvas().postscript(file=file_str)  # save the file
        frames.append(file_str)

        if running:
            # save the screen again after a set time
            tr.ontimer(save, 1000//fps)
        counter += 1

    save()  # start recording

    # start the program (half-second leader)
    tr.ontimer(func(*args, **kwargs))

    running = False

    for i in range(len(frames)):
        im = frames[i]
        frames[i] = Image.open(im)

        os.remove(im)

    frames[0].save(path+fname+'.gif', save_all=True,
                   append_images=frames[1:], optimize=optimize, duration=duration, loop=0)


def format_input_path(path):
    if path[-1] != '/':
        path += '/'
    if path[0] != '/':
        path = '/' + path
    return path


def sign(num):
    """Returns the sign of num as an integer;
    if ```num < 0```, returns ```-1```, ```num = 0```, returns ```0```, ```num > 0```, returns ```1```."""
    return int(num > 0) - int(num < 0)


def char_to_morse():
    """Returns a dictionary mapping from characters to their morse code representation."""
    return {
        'A': '.-',
        'B': '-...',
        'C': '-.-.',
        'D': '-..',
        'E': '.',
        'F': '..-.',
        'G': '--.',
        'H': '....',
        'I': '..',
        'J': '.---',
        'K': '-.-',
        'L': '.-..',
        'M': '--',
        'N': '-.',
        'O': '---',
        'P': '.--.',
        'Q': '--.-',
        'R': '.-.',
        'S': '...',
        'T': '-',
        'U': '..-',
        'V': '...-',
        'W': '.--',
        'X': '-..-',
        'Y': '-.--',
        'Z': '--..',
        ' ': '/',
        '0': '-----',
        '1': '.----',
        '2': '..---',
        '3': '...--',
        '4': '....-',
        '5': '.....',
        '6': '-....',
        '7': '--...',
        '8': '---..',
        '9': '----.',
        '&': '.-...',
        "'": '.----.',
        '@': '.--.-.',
        ')': '-.--.-',
        '(': '-.--.',
        ':': '---...',
        ',': '--..--',
        '=': '-...-',
        '!': '-.-.--',
        '.': '.-.-.-',
        '-': '-....-',
        '+': '.-.-.',
        '"': '.-..-.',
        '?': '..--..',
        '/': '-..-.',
    }


def morse_to_char():
    """Returns a dictionary mapping from morse code to their character representation."""
    return {
        '.-': 'A',
        '-...': 'B',
        '-.-.': 'C',
        '-..': 'D',
        '.': 'E',
        '..-.': 'F',
        '--.': 'G',
        '....': 'H',
        '..': 'I',
        '.---': 'J',
        '-.-': 'K',
        '.-..': 'L',
        '--': 'M',
        '-.': 'N',
        '---': 'O',
        '.--.': 'P',
        '--.-': 'Q',
        '.-.': 'R',
        '...': 'S',
        '-': 'T',
        '..-': 'U',
        '...-': 'V',
        '.--': 'W',
        '-..-': 'X',
        '-.--': 'Y',
        '--..': 'Z',
        '/': ' ',
        '-----': '0',
        '.----': '1',
        '..---': '2',
        '...--': '3',
        '....-': '4',
        '.....': '5',
        '-....': '6',
        '--...': '7',
        '---..': '8',
        '----.': '9',
        '.-...': '&',
        '.----.': "'",
        '.--.-.': '@',
        '-.--.-': ')',
        '-.--.': '(',
        '---...': ':',
        '--..--': ',',
        '-...-': '=',
        '-.-.--': '!',
        '.-.-.-': '.',
        '-....-': '-',
        '.-.-.': '+',
        '.-..-.': '"',
        '..--..': '?',
        '-..-.': '/',
    }


def encode_morse(msg):
    """Returns the Morse code version of a string."""
    lst = list(msg)
    encoder = char_to_morse()
    res = []
    for char in lst:
        res.append(encoder[char.upper()])
    return ' '.join(res)


def decode_morse(msg, uppercase=True):
    """Decodes Morse code and returns a string.

    uppercase: if True, returns uppercase letters; if False, lowercase
    """
    lst = msg.split()
    decoder = morse_to_char()
    res = []
    for beep in lst:
        res.append(decoder[beep])
    return ''.join(res)


def normalized(z):
    """Returns the complex number with the same argument/phase
    but with a magnitude of 1."""
    try:
        return z/abs(z)
    except ZeroDivisionError:
        raise ZeroDivisionError("Cannot normalize 0.")


def slice_length(n, s):
    """Returns the circumradius of a regular polygon with ```n``` sides of length ```s````."""
    return math.sqrt((s**2)/((1-math.cos(360*math.pi/(180*n)))*2))


def polygon(n, r, tr=None):
    """Draws a regular polygon with ```n``` sides and a circumradius of ```r```, \
    centered at the turtle's current position. ```1``` and ```2``` both draw a line with length \
    ```2r```."""
    if tr == None:
        tr = turtle.Turtle()

    if n == 0:
        tr.rt(90)
        tr.pu()
        tr.fd(r)
        tr.pd()
        tr.lt(90)
        tr.circle(r)
        tr.pu()
        tr.lt(90)
        tr.fd(r)
        tr.rt(90)
        tr.pd()
    else:
        if n in (1, 2):
            tr.pu()
            tr.bk(r)
            tr.pd()
            n = 2
            s = 2*r
            tr.fd(2*r)
            tr.pu()
            tr.bk(r)
            tr.pd()
        else:
            ang = 90/n

            tr.lt(ang)
            tr.pu()
            tr.bk(r)
            tr.pd()
            tr.rt(ang)
            s = 2*r*math.sin(math.radians(2*ang))

            for i in range(n):
                tr.fd(s)
                tr.lt(4*ang)

            tr.pu()
            tr.lt(ang)
            tr.fd(r)
            tr.rt(ang)
            tr.pd()


def exterior(n):
    """Returns the exterior angle of a regular polygon with ```n``` sides."""
    return 360/n


def interior(n):
    """Returns the interior angle of a regular polygon with ```n``` sides."""
    return 180-exterior(n)


def factors(n):
    """Returns a list of the factors of n."""
    res = set()
    res |= {1, n}
    for i in range(2, int(n**.5)+1):
        if not n % i:
            res.add(i)
            res.add(n//i)
            res |= factors(n//i)
    return set(res)


def totatives(n):
    """All numbers less than n coprime to n."""
    res = set()
    mx = totient(n)
    for i in range(n):
        if math.gcd(i, n) == 1:
            res.add(i)
        if len(res) == mx:
            return res


def order(n, div):
    """Returns the order of ```n``` mod ```div```."""
    if math.gcd(n, div) != 1:
        raise ValueError("n must be coprime to the mod.")
    i = 1
    while True:
        if n**i % div == 1:
            return i
        i += 1


def mod_powers(pwr, div):
    """Prints all possible residues when raised to ```pwr``` mod ```div```."""
    res = set()
    for i in range(div):
        res.add(pow(i, pwr, div))
    return res


def evens():
    """Generator function yielding even numbers."""
    i = 0
    while True:
        yield (i := i+2)


def odds():
    """Generator function yielding odd numbers."""
    i = -1
    while True:
        yield (i := i+2)


def sum_of_factorial(n, used=None):
    """Returns the set of distinct numbers whose factorials add up to ```n```.
    If there is no such set, returns ```None```.

    used: if not ```None```, is a set containing that cannot be in the final result"""
    assert isinstance(n, int) and n > 0, "n must be a positive integer."

    finished = False
    num, fctrl = 1, 1

    if used == None:
        used = set()

    while not finished:
        fctrl *= num
        if num not in used and fctrl == n:
            return {num}
        elif fctrl > n:
            finished = True
            num -= 1
        else:
            num += 1

    while num > 0:
        if num not in used:
            used.add(num)
            if (x := sum_of_factorial(n-math.factorial(num), used)) != None:
                return used.union(x)
            used.remove(num)
        num -= 1

    return None


def num_types():
    """Numerical types in Python."""
    return {float, int, complex}


def mat_neat(mat):
    """Neat representation of a 2D matrix."""
    res = ''
    for row in mat:
        res += ' ' + repr(row) + ',\n'
    return '[' + res[1:-2] + ']'


def powerset(s, as_set=False):
    """Returns the power set of a set (set of all subsets).

    as_set: If ```True```, returns set of frozensets, otherwise
            returns list of sets."""
    res = []
    for size in range(len(s)+1):
        for ele in map(frozenset if as_set else set, itertools.combinations(s, size)):
            res.append(ele)
    return set(res) if as_set else res


def heading(s):
    return '_'*len(s) + '\n' + s + '\n' + '\u203e'*len(s)


def num_palindromes(n, leading_zeros=False):
    """Returns the number of all n-digit palindromes.

    leading_zeros: if ```True```, 037...730 is a valid palindrome (for example)
    """
    return 10**((n-1)//2)*(9+leading_zeros)


def sum_palindromes(n, leading_zeros=False):
    """Returns the sum of all n-digit palindromes.

    leading_zeros: if ```True```, 037...730 is a valid palindrome (for example)
    """
    assert isinstance(n, int) and n > 0, 'n must be a positive integer.'

    if n == 1:
        return 45  # sum of 1 through 9
    elif n == 2:
        return 495  # 45*11
    else:
        return 45 * (10**(n-1)+1) * num_palindromes(n-2, True) + sum_palindromes(n-2, True) * 10 * (9+leading_zeros)


def lists_from_options(poss, size, repetitions=True):
    """Returns a list of all possible lists of size ```size``` that
    consist only of items from ```poss```, a set."""
    if not size:
        return [[]]

    res = []
    prev = lists_from_options(poss, size-1)

    for lst in prev:
        if repetitions:
            for ele in poss:
                res.append(lst+[ele])
        else:
            assert size <= len(poss)

            for ele in poss - set(lst):
                res.append(lst+[ele])

    return res


def xy_tup(c):
    """Returns a tuple real, imag of a complex number."""
    return c.real, c.imag


def point_types():
    """Ways a point can be represented in Python basic datatypes."""
    return {complex, tuple}


def dict_types():
    """Dictionary-like types."""
    return {dict, defaultdict, DefaultArgDict, Counter, UserDict}


def column(matrix, col):
    """Returns a column from a matrix given the (0-indexed) column number."""
    res = []
    for r in range(len(matrix)):
        res.append(matrix[r][col])
    return res


def pairings(items):
    if len(items) == 0:
        return [[]]

    res = []
    for i in range(1, len(items)):
        remaining = items[1:]
        cur = [(items[0], items[i])]
        remaining.pop(i-1)
        for pairing in pairings(remaining):
            res.append(pairing+cur)


def catalan(n):
    """Returns the ```n```-th Catalan number.
    https://en.wikipedia.org/wiki/Catalan_number"""
    return int(math.comb(2*n, n)/(n+1))


def mod_inv(n, div):
    """Returns the modular inverse of n mod div."""
    if math.gcd(n, div) != 1:
        return None
    else:
        return pow(n, sh.phi(div)-1, div)


def solve_mod_equations(*args):
    """Solves a system of modular congruences, where each argument is a tuple of the
    form ```(a, p)```. In this case, x == a (mod p).

    Returns a tuple ```(b, q)```, where x == b (mod q) is the general solution of the system."""
    def solve_mod_equation(t1, t2):
        t1, t2 = sorted((t1, t2), key=lambda x: x[1])
        a, p, b, q = t1+t2
        return (q*mod_inv(q, p)*(a-b) + b) % (q*p), q*p

    return functools.reduce(lambda res, div: solve_mod_equation(res, div), args)


def alphabet(lower=True):
    res = ''.join((chr(i) for i in range(65, 91)))
    return res.lower() if lower else res


def weighted_random_choice(choice_info):
    x = list(choice_info.items())

    num = random.random()

    csum = 0
    i = 0
    while csum < num:
        csum += x[i][1]
        i += 1

    return x[i-1][0]


def neighbors(mat, r, c, diagonals=True):
    res = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (diagonals or not(i and j)) and r+i in range(len(mat)) and c+j in range(len(mat[0])) and (i or j):
                res.append(mat[r+i][c+j])

    return res


def rotate(mat, deg):
    res = copy(mat)
    size = len(mat)
    if deg == 90:
        res = rotate(mat, 180)
        res = rotate(res, 270)
    elif deg == 180:
        res = reflect(mat, 'x')
        res = reflect(res, 'y')
    elif deg == 270:
        for r in range(size):
            for c in range(size):
                res[c][r] = mat[r][-1-c]
    elif deg == 0:
        return mat
    else:
        raise ValueError('Angle must be 0, 90, 270, or 360.')

    return res


def reflect(mat, axis):
    if axis == 'x':
        return mat[::-1]
    elif axis == 'y':
        res = []
        for row in mat:
            res.append(row[::-1])
    else:
        raise ValueError("Axis must be 'x' or 'y'.")

    return res


def convert_color(col_in, mode=tuple):
    if type(col_in) == tuple:
        col = 256**2*col_in[0] + 256*col_in[1] + col_in[2]
    elif type(col_in) == str:
        col_in = col_in.lstrip('#')
        col_in = int(col_in[:2], 16), int(col_in[2:4], 16), int(col_in[4:], 16)
        col = 256**2*int(col_in[0]) + 256*int(col_in[1]) + int(col_in[2])
    elif type(col_in) == int:
        col = col_in
    else:
        raise TypeError("Colors must be tuples, strings, or integers, not '%s'."
                        % type(col_in).__name__)

    if mode == tuple:
        return col//(256**2), col//256 % 256, col % 256
    elif mode == str:
        return '#'+convert_base_list(from_decimal(col, 256))
    elif mode == int:
        return col
    else:
        raise TypeError("Colors must be tuples, strings, or integers, not '%s'."
                        % type(col_in).__name__)


# alternative names for divisor functions
phi = totient
sigma = sum_fctrs
tau = num_fctrs
