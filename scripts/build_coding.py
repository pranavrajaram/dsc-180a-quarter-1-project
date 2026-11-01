"""Builds tasks_coding.json: the 5 pilot tasks plus 15 new ones.
Each new task has a reference solution and an independent brute-force check.
Expected outputs are computed by the reference, then confirmed by the brute force,
so no expected value is typed by hand. Run: python3 -m scripts.build_coding"""
import copy
import heapq
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"


# ---------- 1. expression evaluation ----------
def evaluate_expression(s):
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(int(s[i:j]))
            i = j
        else:
            tokens.append(c)
            i += 1
    pos = 0

    def div(a, b):
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q

    def expr():
        nonlocal pos
        value = term()
        while pos < len(tokens) and tokens[pos] in '+-':
            op = tokens[pos]; pos += 1
            value = value + term() if op == '+' else value - term()
        return value

    def term():
        nonlocal pos
        value = factor()
        while pos < len(tokens) and tokens[pos] in ('*', '/'):
            op = tokens[pos]; pos += 1
            value = value * factor() if op == '*' else div(value, factor())
        return value

    def factor():
        nonlocal pos
        tok = tokens[pos]
        if tok == '-':
            pos += 1
            return -factor()
        if tok == '(':
            pos += 1
            value = expr()
            pos += 1
            return value
        pos += 1
        return tok

    return expr()


def evaluate_expression_brute(s):
    # Independent: translate to Python with a truncating-division helper.
    import re
    class T:
        def __init__(self, v): self.v = v
        def __add__(self, o): return T(self.v + o.v)
        def __sub__(self, o): return T(self.v - o.v)
        def __mul__(self, o): return T(self.v * o.v)
        def __truediv__(self, o): return T(int(self.v / o.v) if abs(self.v) < 2**50 else None)
        def __neg__(self): return T(-self.v)
    code = re.sub(r'\d+', lambda m: f'T({m.group()})', s)
    return eval(code, {'T': T}).v


# ---------- 2. edit distance ----------
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_distance_brute(a, b):
    from functools import lru_cache
    @lru_cache(None)
    def go(i, j):
        if i == len(a): return len(b) - j
        if j == len(b): return len(a) - i
        return min(go(i + 1, j) + 1, go(i, j + 1) + 1, go(i + 1, j + 1) + (a[i] != b[j]))
    return go(0, 0)


# ---------- 3. decode ways ----------
def count_decodings(s):
    if not s:
        return 0
    prev, cur = 1, 1 if s[0] != '0' else 0
    for i in range(1, len(s)):
        nxt = cur if s[i] != '0' else 0
        if 10 <= int(s[i - 1:i + 1]) <= 26:
            nxt += prev
        prev, cur = cur, nxt
    return cur


def count_decodings_brute(s):
    if not s:
        return 0
    def go(rest):
        if not rest:
            return 1
        total = 0
        for k in (1, 2):
            piece = rest[:k]
            if len(piece) == k and piece[0] != '0' and 1 <= int(piece) <= 26:
                total += go(rest[k:])
        return total
    return go(s)


# ---------- 4. minimum coins ----------
def min_coins(amount, coins):
    best = [0] + [None] * amount
    for a in range(1, amount + 1):
        options = [best[a - c] for c in coins if c <= a and best[a - c] is not None]
        best[a] = min(options) + 1 if options else None
    return -1 if best[amount] is None else best[amount]


def min_coins_brute(amount, coins):
    from collections import deque
    seen, queue = {0}, deque([(0, 0)])
    while queue:
        total, n = queue.popleft()
        if total == amount:
            return n
        for c in coins:
            if total + c <= amount and total + c not in seen:
                seen.add(total + c)
                queue.append((total + c, n + 1))
    return -1


# ---------- 5. coin change combinations ----------
def count_coin_combinations(amount, coins):
    ways = [1] + [0] * amount
    for c in sorted(set(coins)):
        for a in range(c, amount + 1):
            ways[a] += ways[a - c]
    return ways[amount]


def count_coin_combinations_brute(amount, coins):
    coins = sorted(set(coins))
    def go(i, rest):
        if rest == 0: return 1
        if i == len(coins): return 0
        return sum(go(i + 1, rest - k * coins[i]) for k in range(rest // coins[i] + 1))
    return go(0, amount)


# ---------- 6. longest increasing subsequence ----------
def longest_increasing_subsequence(nums):
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        tails[i:i + 1] = [x]
    return len(tails)


def longest_increasing_subsequence_brute(nums):
    for r in range(len(nums), 0, -1):
        for idx in itertools.combinations(range(len(nums)), r):
            if all(nums[a] < nums[b] for a, b in zip(idx, idx[1:])):
                return r
    return 0


# ---------- 7. word wrap ----------
def wrap_text(text, width):
    lines, line = [], ''
    for word in text.split():
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += ' ' + word
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def wrap_text_brute(text, width):
    words, lines = text.split(), []
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and len(' '.join(words[i:j + 1])) <= width:
            j += 1
        lines.append(' '.join(words[i:j]))
        i = j
    return lines


# ---------- 8. islands ----------
def count_islands(grid):
    seen, count = set(), 0
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == '1' and (r, c) not in seen:
                count += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < len(grid) and 0 <= nx < len(grid[ny]) and grid[ny][nx] == '1' and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
    return count


def count_islands_brute(grid):
    cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == '1']
    parent = {cell: cell for cell in cells}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for (r, c) in cells:
        for other in ((r + 1, c), (r, c + 1)):
            if other in parent:
                parent[find((r, c))] = find(other)
    return len({find(cell) for cell in cells})


# ---------- 9. shortest path ----------
def shortest_path_length(n, edges, source, target):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    dist = {source: 0}
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == target:
            return d
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if d + w < dist.get(v, float('inf')):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return -1


def shortest_path_length_brute(n, edges, source, target):
    inf = float('inf')
    d = [[0 if i == j else inf for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        d[u][v] = min(d[u][v], w)
        d[v][u] = min(d[v][u], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return -1 if d[source][target] == inf else d[source][target]


# ---------- 10. roman numerals ----------
def to_roman(n):
    table = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
             (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    out = ''
    for value, sym in table:
        while n >= value:
            out += sym
            n -= value
    return out


def to_roman_brute(n):
    digits = [('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'),
              ('', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC'),
              ('', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'),
              ('', 'M', 'MM', 'MMM')]
    return ''.join(digits[i][(n // 10 ** i) % 10] for i in range(3, -1, -1))


# ---------- 11. spiral order ----------
def spiral_order(matrix):
    out = []
    m = [list(row) for row in matrix]
    while m:
        out += m.pop(0)
        m = [list(row) for row in zip(*m)][::-1]
    return out


def spiral_order_brute(matrix):
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    seen, out = set(), []
    r = c = d = 0
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for _ in range(rows * cols):
        out.append(matrix[r][c])
        seen.add((r, c))
        nr, nc = r + moves[d][0], c + moves[d][1]
        if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in seen:
            d = (d + 1) % 4
            nr, nc = r + moves[d][0], c + moves[d][1]
        r, c = nr, nc
    return out


# ---------- 12. longest palindromic substring ----------
def longest_palindrome(s):
    best = ''
    for center in range(2 * len(s) - 1):
        lo, hi = center // 2, (center + 1) // 2
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            lo -= 1
            hi += 1
        cand = s[lo + 1:hi]
        if len(cand) > len(best) or (len(cand) == len(best) and s.find(cand) < s.find(best)):
            best = cand
    return best


def longest_palindrome_brute(s):
    best = ''
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            if sub == sub[::-1] and len(sub) > len(best):
                best = sub
    return best


# ---------- 13. sliding window maximum ----------
def sliding_window_max(nums, k):
    from collections import deque
    dq, out = deque(), []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


def sliding_window_max_brute(nums, k):
    return [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]


# ---------- 14. next permutation ----------
def next_permutation(nums):
    a = list(nums)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a


def next_permutation_brute(nums):
    perms = sorted(set(itertools.permutations(nums)))
    idx = perms.index(tuple(nums))
    return list(perms[(idx + 1) % len(perms)])


# ---------- 15. range compression ----------
def compress_ranges(nums):
    values = sorted(set(nums))
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f'{values[i]}->{values[j]}')
        else:
            out.extend(str(v) for v in values[i:j + 1])
        i = j + 1
    return out


def compress_ranges_brute(nums):
    values = sorted(set(nums))
    runs = []
    for v in values:
        if runs and runs[-1][-1] == v - 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    out = []
    for run in runs:
        out += [f'{run[0]}->{run[-1]}'] if len(run) >= 3 else [str(v) for v in run]
    return out


NEW_TASKS = [
    ("evaluate_expression", "hard",
     "Implement evaluate_expression(s). s is a valid arithmetic expression containing nonnegative integer literals, the binary operators +, -, *, /, unary minus, parentheses, and spaces. Use standard precedence (* and / bind tighter than + and -; operators of equal precedence are left-associative; unary minus binds tighter than * and /). Division is integer division that truncates toward zero. No division by zero occurs. Return the integer result. Do not use eval or exec.",
     evaluate_expression, evaluate_expression_brute, False,
     [["1 + 2 * 3"], ["(1 + 2) * 3"], ["7 / 2"], ["-7 / 2"], ["10 - 4 - 3"], ["2 * (3 + 4) - -5"], ["  42  "], ["100 / (2 * -3) + 8 * (2 - 5)"]]),
    ("edit_distance", "medium",
     "Implement edit_distance(a, b). Return the minimum number of single-character insertions, deletions, or substitutions needed to turn string a into string b. Either string may be empty. Comparison is case-sensitive.",
     edit_distance, edit_distance_brute, False,
     [["", ""], ["", "abc"], ["kitten", "sitting"], ["flaw", "lawn"], ["abc", "abc"], ["Abc", "abc"], ["intention", "execution"], ["ab", "ba"]]),
    ("count_decodings", "medium",
     "Implement count_decodings(s). The letters A to Z are encoded as the numbers 1 to 26. s is a string of digits. Return the number of ways s can be split into pieces where each piece is a number from 1 to 26 written without leading zeros. Return 0 for the empty string or if no valid split exists.",
     count_decodings, count_decodings_brute, False,
     [[""], ["12"], ["226"], ["0"], ["06"], ["10"], ["2101"], ["11106"], ["27"], ["1111111111"]]),
    ("min_coins", "medium",
     "Implement min_coins(amount, coins). coins is a list of distinct positive integers and amount is a nonnegative integer. Return the minimum number of coins (each denomination may be used any number of times) that sum exactly to amount, or -1 if it is impossible. min_coins(0, coins) is 0.",
     min_coins, min_coins_brute, True,
     [[0, [1, 2]], [11, [1, 2, 5]], [3, [2]], [6, [1, 3, 4]], [7, [2, 4]], [27, [5, 9, 13]], [100, [1, 25, 50]], [12, [7, 5]]]),
    ("count_coin_combinations", "medium",
     "Implement count_coin_combinations(amount, coins). coins is a list of distinct positive integers and amount is a nonnegative integer. Return the number of different multisets of coins (each denomination usable any number of times) that sum exactly to amount. Order does not matter, so 1+2 and 2+1 count once. count_coin_combinations(0, coins) is 1.",
     count_coin_combinations, count_coin_combinations_brute, True,
     [[0, [1, 2]], [5, [1, 2, 5]], [3, [2]], [10, [10]], [12, [1, 5, 10]], [8, [3, 5, 2]], [7, []], [20, [2, 3, 7]]]),
    ("longest_increasing_subsequence", "medium",
     "Implement longest_increasing_subsequence(nums). Return the length of the longest strictly increasing subsequence of the list of integers nums (elements need not be contiguous). Return 0 for an empty list.",
     longest_increasing_subsequence, longest_increasing_subsequence_brute, True,
     [[[]], [[5]], [[10, 9, 2, 5, 3, 7, 101, 18]], [[0, 1, 0, 3, 2, 3]], [[7, 7, 7, 7]], [[-3, -1, -2, 0, -5, 4]], [[1, 2, 3, 4, 5, 6]], [[6, 5, 4, 3, 2, 1]]]),
    ("wrap_text", "medium",
     "Implement wrap_text(text, width). Split text into words on any whitespace. Greedily build lines: place as many words on a line as fit, joined by single spaces, without the line length exceeding width. A word longer than width goes on its own line unchanged. Return the list of lines; return [] if there are no words. width is a positive integer.",
     wrap_text, wrap_text_brute, False,
     [["", 10], ["   ", 5], ["the quick brown fox", 10], ["the quick brown fox", 9], ["a b c d e", 3], ["supercalifragilistic is long", 6], ["one\ttwo\nthree  four", 9], ["exact fit", 9]]),
    ("count_islands", "medium",
     "Implement count_islands(grid). grid is a list of equal-length strings containing only '0' and '1'. An island is a maximal group of '1' cells connected horizontally or vertically (not diagonally). Return the number of islands. The grid may be empty.",
     count_islands, count_islands_brute, True,
     [[[]], [["0"]], [["1"]], [["11000", "11000", "00100", "00011"]], [["101", "010", "101"]], [["111", "101", "111"]], [["1001", "1001", "1111"]], [["10101"]]]),
    ("shortest_path_length", "hard",
     "Implement shortest_path_length(n, edges, source, target). The graph is undirected with nodes 0 through n-1. Each edge is [u, v, w] with nonnegative integer weight w; there may be parallel edges and self-loops. Return the total weight of the minimum-weight path from source to target, 0 if source equals target, or -1 if target is unreachable. Do not mutate the inputs.",
     shortest_path_length, shortest_path_length_brute, True,
     [[1, [], 0, 0], [2, [], 0, 1], [3, [[0, 1, 4], [1, 2, 3], [0, 2, 10]], 0, 2], [3, [[0, 1, 5], [0, 1, 2], [1, 2, 1]], 0, 2], [4, [[0, 1, 0], [1, 2, 0], [2, 3, 7]], 3, 0], [5, [[0, 1, 1], [1, 2, 1], [3, 4, 1]], 0, 4], [4, [[0, 0, 3], [0, 1, 2], [1, 3, 9], [0, 2, 6], [2, 3, 1]], 0, 3], [6, [[0, 1, 7], [0, 2, 9], [0, 5, 14], [1, 2, 10], [1, 3, 15], [2, 3, 11], [2, 5, 2], [3, 4, 6], [4, 5, 9]], 0, 4]]),
    ("to_roman", "easy",
     "Implement to_roman(n). Given an integer n with 1 <= n <= 3999, return its standard Roman numeral using uppercase I, V, X, L, C, D, M and subtractive notation (IV, IX, XL, XC, CD, CM).",
     to_roman, to_roman_brute, False,
     [[1], [4], [9], [14], [40], [444], [1994], [3999]]),
    ("spiral_order", "medium",
     "Implement spiral_order(matrix). matrix is a list of rows, all the same length. Return all elements in clockwise spiral order starting from the top-left corner and moving right first. Return [] for an empty matrix or a matrix with empty rows. Do not mutate the input.",
     spiral_order, spiral_order_brute, True,
     [[[]], [[[]]], [[[1]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], [[[1], [2], [3]]], [[[1, 2, 3]]], [[[1, 2], [3, 4], [5, 6], [7, 8]]]]),
    ("longest_palindrome", "medium",
     "Implement longest_palindrome(s). Return the longest contiguous substring of s that reads the same forwards and backwards. If several have the maximum length, return the one that starts earliest. Return '' for an empty string. Comparison is case-sensitive.",
     longest_palindrome, longest_palindrome_brute, False,
     [[""], ["a"], ["babad"], ["cbbd"], ["abacdfgdcaba"], ["forgeeksskeegfor"], ["Aa"], ["abcd"]]),
    ("sliding_window_max", "medium",
     "Implement sliding_window_max(nums, k). Return a list containing the maximum of every contiguous window of length k in nums, in order from left to right. 1 <= k <= len(nums). Do not mutate nums.",
     sliding_window_max, sliding_window_max_brute, True,
     [[[5], 1], [[1, 3, -1, -3, 5, 3, 6, 7], 3], [[4, 2], 2], [[1, 2, 3, 4], 1], [[9, 8, 7, 6, 5], 2], [[2, 2, 2, 2], 3], [[-1, -5, -2, -8], 2], [[1, 3, 1, 2, 0, 5], 6]]),
    ("next_permutation", "medium",
     "Implement next_permutation(nums). Return a new list containing the next lexicographically greater rearrangement of the integers in nums. If nums is already the greatest rearrangement, return the smallest one (sorted ascending). nums may contain duplicates and has at least one element. Do not mutate nums.",
     next_permutation, next_permutation_brute, True,
     [[[1]], [[1, 2, 3]], [[3, 2, 1]], [[1, 1, 5]], [[1, 3, 2]], [[2, 2, 1, 1]], [[1, 5, 1]], [[0, 1, 2, 5, 3, 3, 0]]]),
    ("compress_ranges", "easy",
     "Implement compress_ranges(nums). nums is a list of integers in any order and may contain duplicates. Treat it as a set and process values in ascending order. Every maximal run of three or more consecutive integers becomes the string 'start->end'; values in runs of length one or two are listed individually as strings. Return the resulting list of strings in ascending order. Do not mutate nums.",
     compress_ranges, compress_ranges_brute, True,
     [[[]], [[5]], [[1, 2]], [[1, 2, 3]], [[3, 1, 2, 2, 7, 8, 10, 11, 12]], [[-3, -2, -1, 1]], [[0, 2, 4, 6]], [[9, 8, 7, 6, 4, 3, 1]]]),
]


def build():
    tasks = []
    for task in json.loads((ROOT / "tasks.json").read_text()):
        tasks.append({**task, "domain": "coding"})
    for task_id, difficulty, prompt, ref, brute, no_mut, arg_lists in NEW_TASKS:
        tests = []
        for args in arg_lists:
            expected = ref(*copy.deepcopy(args))
            check = brute(*copy.deepcopy(args))
            assert expected == check, (task_id, args, expected, check)
            json.dumps(expected)  # must be JSON-serializable
            tests.append({"args": args, "expected": expected})
        tasks.append({"task_id": task_id, "domain": "coding", "difficulty": difficulty, "prompt": prompt,
                      "entry_point": ref.__name__, "check_no_mutation": no_mut, "tests": tests})
    assert len(tasks) == 20 and len({t["task_id"] for t in tasks}) == 20
    (ROOT / "tasks_coding.json").write_text(json.dumps(tasks, indent=2) + "\n")
    for t in tasks:
        print(f'{t["task_id"]:32} {t["difficulty"]:7} {len(t["tests"])} tests')


if __name__ == "__main__":
    build()
