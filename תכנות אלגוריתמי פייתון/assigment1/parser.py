from abc import ABC
from numpy import double
from abc import ABC, abstractmethod


class Expression(ABC):
    @abstractmethod
    def calc(self) -> double:
        pass


# implement the classes here
class Num(Expression):
    def __init__(self, number):
        self.number = double(number)

    def calc(self) -> double:
        return self.number


class BinExp(Expression):
    def __init__(self, right: Expression, left: Expression, op: Expression):
        self.right = right
        self.left = left
        self.operator = op

    @abstractmethod
    def calc(self) -> double:
        pass


class Plus(BinExp):
    def __init__(self, right: Expression, left: Expression):
        super().__init__(right, left, self)
        self.right = right
        self.left = left

    def calc(self) -> double:
        return double(self.right.calc() + self.left.calc())


class Minus(BinExp):
    def __init__(self, right: Expression, left: Expression):
        super().__init__(right, left, self)
        self.right = right
        self.left = left

    def calc(self) -> double:
        return double(self.right.calc() - self.left.calc())


class Mul(BinExp):
    def __init__(self, right: Expression, left: Expression):
        super().__init__(right, left, self)
        self.right = right
        self.left = left

    def calc(self) -> double:
        return double(self.left.calc() * self.right.calc())


class Div(BinExp):
    def __init__(self, right: Expression, left: Expression):
        super().__init__(right, left, self)
        self.right = right
        self.left = left

    def calc(self) -> double:
        return double(self.right.calc() / self.left.calc())


# The OPERATIONS_MAP dictionary maps the operators for an easy usage
OPERATIONS_MAP = {
    '+': Plus,
    '-': Minus,
    '*': Mul,
    '/': Div
}

import re


def is_empty(op_stack):
    return len(op_stack) == 0


class expression_handler:
    def __init__(self, expression: str):
        self.expression = expression
        self.OP_PRIORITY = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '(': 0,
            ')': 0
        }

    def split_expression(self):
        tokens = re.findall(r'[-+]?\d*\.?\d+|[()+\-\*/]', self.expression)
        new_tokens = []
        for i, token in enumerate(tokens):
            if '+' in token and len(token) > 1:
                new_tokens.append('+')
                new_tokens.append(token.split('+')[-1])
            elif '*' in token and len(token) > 1:
                new_tokens.append('*')
                new_tokens.append(token.split('*')[-1])
            elif '/' in token and len(token) > 1:
                new_tokens.append('/')
                new_tokens.append(token.split('/')[-1])
            elif '-' in token and len(token) > 1:
                if not '(' in tokens[i - 1]:  # The case of not unary negative number
                    new_tokens.append('-')
                    new_tokens.append(token.split('-')[-1])
                else:  # The case of unary negative number ('(' is behind the minus sign)
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
        return new_tokens

    @staticmethod
    def is_numeric(token):
        try:
            float(token)
            return True
        except ValueError:
            return False

    def infix_to_prefix(self):
        op_stack = []
        queue = []
        tokens = self.split_expression()
        for token in tokens:
            if self.is_numeric(token):  # If the token is a digit, append him to the Q
                # Handle the negative number using unique tokenization.
                if float(token) < 0:
                    new_token = token.replace('-', 'n')
                    token = new_token
                queue.append(token)
            elif token == '(':
                op_stack.append(token)
            elif token == ')':
                while not is_empty(op_stack) and op_stack[-1] != '(':
                    queue.append(op_stack.pop())
                op_stack.pop()  # Discard the '(' after breaking the while loop.
            elif token in self.OP_PRIORITY.keys():
                while not is_empty(op_stack) and self.OP_PRIORITY[token] <= self.OP_PRIORITY[op_stack[-1]] \
                        and op_stack[-1] != '(':
                    queue.append(op_stack.pop())
                op_stack.append(token)
        # After iterating over the tokens, we're emptying the stack
        ignore_op = ['(', ')']
        expressions_queue = []
        while not is_empty(op_stack):
            e = op_stack.pop()
            if not e in ignore_op:
                expressions_queue.append(e)
                queue.append(e)
        return queue


# implement the parser function here
def parser(expression) -> double:
    expression = expression_handler(expression)
    expression_prefix_list = expression.infix_to_prefix()
    result_stack = []
    for token in expression_prefix_list:
        if 'n' in token:
            token = token.replace('n', '-')
            result_stack.append(Num(token))
        elif expression_handler.is_numeric(token):
            result_stack.append(Num(token))
        elif token in OPERATIONS_MAP.keys():
            right_num = result_stack.pop()
            left_num = result_stack.pop()
            operation_res = Num(OPERATIONS_MAP[token](left_num, right_num).calc())  # Getting the result as a Num.
            result_stack.append(operation_res)
    return result_stack.pop().calc()
