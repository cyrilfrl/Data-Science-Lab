







import numpy as np
import pandas as pd

























"""
################################################## (include: figures\networkx 100 randomized items graph.png)
import networkx as nx
import random
import matplotlib.pyplot as plt

# Number of nodes
num_nodes = 100

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
for i in range(num_nodes):
    G.add_node(i)

# Add edges randomly
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        # Generate a random number to decide whether to add an edge
        if random.random() < 0.2:  # Adjust the probability as needed
            G.add_edge(i, j)

# Print some information about the generated graph
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Draw the graph
nx.draw(G, with_labels=False, node_size=100)
plt.show()

##################################################
history = [0] * nets
##################################################
metro_areas = [
    ('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
    ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
    ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
    ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
    ('São Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
]
def main():
    print(f'{"":15} | {"latitude":>9} | {"longitude":>9}') #//# f'{"":15}', creates 15 void spaces
    for record in metro_areas:
        match record:
            case [name, _, _, (lat, lon)] if lon <= 0:
                print(f'{name:15} | {lat:9.4f} | {lon:9.4f}')

main()
                |  latitude | longitude
Mexico City     |   19.4333 |  -99.1333
New York-Newark |   40.8086 |  -74.0204
São Paulo       |  -23.5478 |  -46.6358

################################################## MATCH CASE
def classify_number(number):
    match number:
        case 0:
            return "Zero"
        case n if n > 0:
            return "Positive"
        case _:
            return "Negative"

# Example usage
print(classify_number(5))  # Output: Positive #//# n takes the value of number
print(classify_number(-3)) # Output: Negative
print(classify_number(0))  # Output: Zero

#Positive
#Negative
#Zero
#//# more powerful than C's switch because of destructuring:


################################################## Grab excess items
a, b, *rest = range(5)
print(f"{a} | {b} | {rest}")
#0 | 1 | [2, 3, 4]
*head, b, c = range(5) #//# can appear at any position
head, b, c
print(f"{head} | {b} | {c}")
#[0, 1, 2] | 3 | 4

################################################## tuple: less memory than list, length doesn't change
################################################## TRY DIFFERENT COMBINATIONS
colors = ['black', 'white']
sizes = ['S', 'M', 'L']
tshirts = [(color, size) for color in colors for size in sizes]
print(tshirts)
#[('black', 'S'), ('black', 'M'), ('black', 'L'), ('white', 'S'), ('white', 'M'), ('white', 'L')]

################################################## LIST GENERATOR EXPRESSIONS
symbols = '$¢£¥€¤'
codes = [ord(symbol) for symbol in symbols] #//# create a list using listcomp, advised if takes less than two lines to stay readable
print(codes)
#[36, 162, 163, 165, 8364, 164] returns unicode
#ord("t") => 116



# Without walrus operator
data = [1, 2, 3, 4, 5]
result = []
for num in data:
    if num % 2 == 0:
        result.append(num * 2)

print(result)

# With walrus operator
result = []
for num in data:
    if (doubled := num * 2) % 2 == 0:
        result.append(doubled)

print(result)


# Without using the walrus operator
numbers = [1, 2, 3, 4, 5]
squared_numbers = [num ** 2 for num in numbers if num % 2 == 0]

# Using the walrus operator
squared_numbers = [num ** 2 for num in numbers if (remainder := num % 2) == 0]

################################################## EXECUTE TEXT CODE AS PYTHON
import timeit

TIMES = 10000

SETUP = """
symbols = '$¢£¥€¤'
def non_ascii(c):
    return c > 127
"""

def clock(label, cmd):
    res = timeit.repeat(cmd, setup=SETUP, number=TIMES)
    print(label, *(f'{x:.3f}' for x in res))

clock('listcomp        :', '[ord(s) for s in symbols if ord(s) > 127]')
clock('listcomp + func :', '[ord(s) for s in symbols if non_ascii(ord(s))]')
clock('filter + lambda :', 'list(filter(lambda c: c > 127, map(ord, symbols)))')
clock('filter + func   :', 'list(filter(non_ascii, map(ord, symbols)))')

################################################## SCROLL PDF WHILE READING WITH EYE TRACKING
PROJECT, make youtube video and share project on github
python program to know where i am looking at (create pygame figure and capture my face
as i am looking to a specific place on the screen and be able to predict where i am looking
with new images), explain on YouTube and share video

################################################## LATEX CODE RECOGNIZER
SCREENSHOT OR IMAGE AND RETURN LATEX CODE, unsupervised because many possibilities, create youtube video



################################################## solve_ivp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def exponential_growth(t, y):
    return 2 * y  # Exponential growth example

sol = solve_ivp(exponential_growth, [0, 5], [1], t_eval=np.linspace(0, 5, 100))

plt.plot(sol.t, sol.y[0])
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Exponential Growth')
plt.show()

################################################## interpolate.splrep()
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10)
y = np.sin(x)

tck = interpolate.splrep(x, y)
x_new = np.linspace(0, 10, 100)
y_new = interpolate.splev(x_new, tck)

plt.plot(x, y, 'ro', label='Original points')
plt.plot(x_new, y_new, label='Spline curve')
plt.legend()
plt.show()

################################################## splev
y_first_derivative = interpolate.splev(x_new, tck, der=1)
plt.plot(x_new, y_first_derivative, label='First Derivative')
plt.legend()
plt.show()

################################################## DO NOT TRY
import pyautogui
import cv2
import numpy as np

def draw_rectangle(x, y, w, h):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), 2)
    return image

def main():
    while True:
        # Check if the mouse button is clicked
        if pyautogui.mouseDown():
            # Get the current mouse position
            x, y = pyautogui.position()

            # Get the screen resolution
            screen_width, screen_height = pyautogui.size()

            # Set the width and height for the rectangle
            rect_width = 100
            rect_height = 100

            # Calculate the boundaries for the rectangle
            left = max(x - rect_width // 2, 0)
            top = max(y - rect_height // 2, 0)
            right = min(left + rect_width, screen_width)
            bottom = min(top + rect_height, screen_height)

            # Get the screenshot of the screen
            screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))

            # Convert the screenshot to OpenCV format
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Draw a rectangle around the hovered zone
            image_with_rectangle = draw_rectangle(right - left, bottom - top)

            # Overlay the rectangle on the screenshot
            overlay = cv2.addWeighted(screenshot, 1, image_with_rectangle, 0.5, 0)

            # Display the overlaid image
            cv2.imshow("Highlighted Zone", overlay)

            # Wait for the user to release the mouse button
            while pyautogui.mouseDown():
                pass

        # Check for the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
################################################## MOUSE POSITION
import pyautogui

def get_rectangle_corners():
    print("Please select the top-left corner of the rectangle.")
    top_left = pyautogui.position()
    print("Top-left corner coordinates:", top_left)
    
    print("Please select the bottom-right corner of the rectangle.")
    bottom_right = pyautogui.position()
    print("Bottom-right corner coordinates:", bottom_right)
    
    return top_left, bottom_right

# Example usage
top_left_corner, bottom_right_corner = get_rectangle_corners()
################################################## TAKE SCREENSHOT OF WHOLE WINDOW
import numpy as np
from PIL import ImageGrab
import matplotlib.pyplot as plt

def take_screenshot():
    screen = ImageGrab.grab()  # Grab the entire screen
    screenshot_array = np.array(screen)

    return screenshot_array

# Example usage:
# Capture screenshot using Pillow
screenshot_array = take_screenshot()

# Convert Pillow image array to NumPy array
if screenshot_array is not None:
    # Convert from RGB to grayscale
    screenshot_gray = np.dot(screenshot_array[..., :3], [0.2989, 0.5870, 0.1140])
    print("Screenshot shape (grayscale):", screenshot_gray.shape)
else:
    print("Screenshot capture failed.")

# Display the screenshot
plt.imshow(screenshot_gray, cmap="binary")
plt.axis("off")
plt.show()
##################################################
import math
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Vector({self.x!r}, {self.y!r})' #//# use !r to represent in original datatype

    def __abs__(self):
        return math.hypot(self.x, self.y)
    
    def __bool__(self):
        return bool(abs(self))
    
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y    
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
################################################## DATA MODEL
import collections
Card = collections.namedtuple('Card', ['rank', 'suit'])
Card # <class '__main__.Card'>

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA') # ['J', 'Q', 'K', 'A']
    suits = 'spades diamonds clubs hearts'.split()
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position): #//# used to get item with method like [x]
        return self._cards[position]

beer_card = Card('7', 'diamonds')
beer_card # Card(rank='7', suit='diamonds')

deck = FrenchDeck()
len(deck) # 52

deck[0] # Card(rank='2', suit='spades') #//# uses the __getitem__ method

from random import choice
choice(deck) #//# Card(rank='K', suit='clubs')

Card('Q', 'hearts') in deck #//# True
Card('Q', 'caca') in deck #//# False


#for card in deck:
#    print(card)
    
#Card(rank='2', suit='spades')  
#Card(rank='3', suit='spades')
#...
#Card(rank='K', suit='hearts')
#Card(rank='A', suit='hearts')

suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0) #//# {'spades': 3, 'hearts': 2, 'diamonds': 1, 'clubs': 0}

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    card.rank # J, Q, K, A, 1, 2...
    pass
    #print(card)

################################################## namedtuple factory function for creating tuple subclasses with named fields
from collections import namedtuple

# Define a namedtuple 'Person' with fields 'name', 'age', and 'gender'
Person = namedtuple('Person', ['name', 'age', 'gender'])

# Create an instance of Person
person1 = Person(name='Alice', age=30, gender='Female')

# Accessing fields
print(person1.name)  # Output: Alice
print(person1.age)   # Output: 30
print(person1.gender) # Output: Female

##################################################

import json
def load_json(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    return data

def dump_json(data, filename, indent=4):
    with open(filename, 'w+') as file:
        json.dump(data, file, indent=indent)


import time
current_timestamp = int(time.time())
print("Current timestamp:", current_timestamp)

##################################################
#                   PROJECTS                     #
##################################################


################################################## IDENTIFY PEOPLE IN PICTURE
sklearn.datasets.fetch_olivetti_faces()
https://www.kaggle.com/code/serkanpeldek/face-recognition-on-olivetti-dataset

################################################## FINANCIAL PREDICTIONS
volatility, direction, ... (build technical indicators)



"""