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
