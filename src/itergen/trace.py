from typing import Iterable


class TraceToken:
    """
    This class stores the information about a token in the trace. It stores the token id, the position in the generation, next and the previous token.

    Args:
    - tid: Token id
    - position: Position in the generation
    - string: String representation of the token
    - prev_token: Previous token
    - start_char: Start character location of the token in the structured generation
    - end_char: End character location of the token in the structured generation
    - prob: Probability of the token
    """
    def __init__(self, tid, position, string, prev_token, start_char, end_char, prob=None):
        self.tid = tid
        self.position = position
        self.string = string
        self.next_token: Iterable[TraceToken] = []
        self.prev_token: TraceToken = prev_token
        self.start_char: int = start_char
        self.end_char: int = end_char
        self.prob = prob


class Trace:
    """
    This class stores the whole hisory of given IterGen session. This allows IterGen to provide additional 
    functionalities where it could avoid taking the generation paths that have already been taken in the past.
    We store this history of LLM tokens as a directed graph where each node is a token and each edge is a
    transition between two tokens. Each generated token is stored is uniquely identified by the position in the generation and the token index.
    """
    def __init__(self):
        self.root_token = TraceToken(tid=-1, position=-1, string='', prev_token=None, start_char=-1, end_char=0)
        self.current_token = self.root_token
    
    def add_token(self, tid, position, string, start_char, end_char, prob=None):
        """
        Add a new token to the trace.

        Args:
        - tid: Token id
        - position: Position in the generation
        - string: String representation of the token
        - start_char: Start character location of the token in the structured generation
        - end_char: End character location of the token in the structured generation
        - prob: Probability of the token
        """
        new_token = TraceToken(tid, position, string, self.current_token, start_char, end_char, prob)
        self.current_token.next_token.append(new_token)
        self.current_token = new_token
        # print(f"Added token {tid} at position {position}")
    
    def backtrack_steps(self, k):
        """
        Backtrack k steps in the trace.
        """
        for _ in range(k):
            self.current_token = self.current_token.prev_token
    
    def backtrack_to_char_pos(self, target_char_pos, keep_trace=True):
        """
        Backtrack to a specific character position in the trace.
        """
        while self.current_token.end_char > target_char_pos and self.current_token.prev_token is not None:
            self.current_token = self.current_token.prev_token
            if not keep_trace:
                self.current_token.next_token = []
    
    def get_next_token_ids(self):
        """
        Get the next token set from the current token.
        """
        return [tok.tid for tok in self.current_token.next_token]
    
    def delete_last_token(self):
        """
        Delete the last token from the trace.
        """
        self.current_token = self.current_token.prev_token
        self.current_token.next_token = []
    
    def print_trace(self):
        current_token = self.current_token
        while current_token is not None:
            print(f"Token: {current_token.tid}, Position: {current_token.position}, String: {current_token.string}, 'Start Char': {current_token.start_char}, 'End Char': {current_token.end_char}")
            current_token = current_token.prev_token
    