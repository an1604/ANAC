from parser import *
import random
import string
import pdb

a = Num(random.randint(-100, 100))
print(f'The value in a.calc() is : {a.calc()}')
b = Num(random.randint(-100, 100))
print(f'The value in b.calc() is : {b.calc()}')
c = Num(random.randint(-100, 100))
print(f'The value in c.calc() is : {c.calc()}')
d = Num(random.randint(-100, 100))
print(f'The value in d.calc() is : {d.calc()}')

if Plus(a, b).calc() != a.calc() + b.calc():
    print("problem with Plus (-10)")

if Minus(a, b).calc() != a.calc() - b.calc():
    print("problem with Minus (-10)")

if Mul(a, b).calc() != a.calc() * b.calc():
    print("problem with Mul (-10)")

x = Div(a, b).calc()
if x > a.calc() / b.calc() + 0.01 or x < a.calc() / b.calc() - 0.01:
    print("problem with Div (-10)")

if Plus(a, Mul(b, Minus(c, d))).calc() != a.calc() + b.calc() * (c.calc() - d.calc()):
    print("problem with expression (-10)")


def strf(x) -> string:
    s = str(x)
    if s.startswith('-'):
        s = "(" + s + ")"
    return s


sa = strf(a.calc())
sb = strf(b.calc())
sc = strf(c.calc())
sd = strf(d.calc())

s = sa + '+' + sb + '*(' + sc + '-' + sd + ')'
wrong = (parser(s) != eval(s))
if wrong:
    pdb.set_trace()
if parser(s) != eval(s):
    print("problem with parser (-10)")

s = sa + '*' + sa + '+' + sb + '*(' + sc + '-' + sd + '+' + sb + ')'
wrong = (parser(s) != eval(s))
if wrong:
    pdb.set_trace()
if parser(s) != eval(s):
    print("problem with parser (-20)")

s = sa + '*(' + sa + '+' + sb + '*(' + sc + '-' + sd + '+' + sb + '))'
wrong = (parser(s) != eval(s))
if wrong:
    pdb.set_trace()
if parser(s) != eval(s):
    print("problem with parser (-20)")

print("done")
