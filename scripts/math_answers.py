"""Extract part-specific final mathematics answers without consulting answer keys."""
import ast
import math
import re
from fractions import Fraction

from scripts.experiment import LINE_LABEL, FORMAT_LINE, same_value


def numeric(text):
    """Evaluate bounded arithmetic only, never Python calls, names, or model code."""
    text = text.strip().strip('*').strip().rstrip('.').strip()
    for token in (r'\(', r'\)', r'\[', r'\]', '$', r'\left', r'\right', r'\,', r'\!'):
        text = text.replace(token, '')
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    text = text.replace(r'\cdot', '*').replace(r'\times', '*').replace('×', '*').replace('−', '-')
    for _ in range(12):
        old = text
        text = re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', text)
        text = re.sub(r'\\boxed\s*\{([^{}]*)\}', r'(\1)', text)
        text = re.sub(r'\^\s*\{([^{}]*)\}', r'**(\1)', text)
        if text == old:
            break
    def factorial(match):
        n = int(match.group(1))
        return str(math.factorial(n)) if n <= 100 else 'unsupported'
    text = re.sub(r'(?<![\w.])(\d+)!', factorial, text)
    def combination(match):
        n, k = map(int, match.groups())
        return str(math.comb(n, k)) if 0 <= k <= n <= 1000 else 'unsupported'
    text = re.sub(r'\\binom\s*\{(\d+)\}\s*\{(\d+)\}', combination, text)
    text = text.replace('^', '**').strip()
    if text.endswith('%'):
        text = '(' + text[:-1] + ')/100'
    if len(text) > 300 or not re.fullmatch(r'[\d\s.+*/()\-]+', text):
        return None
    try:
        tree = ast.parse(text, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 80:
            return None
        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return Fraction(str(node.value))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp):
                a, b = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): value = a+b
                elif isinstance(node.op, ast.Sub): value = a-b
                elif isinstance(node.op, ast.Mult): value = a*b
                elif isinstance(node.op, ast.Div): value = a/b
                elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 20: value = a**int(b)
                else: raise ValueError('Unsupported arithmetic')
                if value.numerator.bit_length() + value.denominator.bit_length() > 4096:
                    raise ValueError('Arithmetic too large')
                return value
            raise ValueError('Not a numeric expression')
        return evaluate(tree.body), '.' not in text
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def claim(text):
    """Read one mathematical claim, preserving contradictions in equality chains."""
    text = text.strip().strip('*').strip()
    # Keep the numeric expression before common units, not a number from later prose.
    text = re.split(r'\s+(?:ways|hours?|rolls?|flips?|pairs?|outcomes?|dollars?|of\s+the|per\s+hour)\b', text, maxsplit=1, flags=re.I)[0]
    pieces = re.split(r'\\approx|≈', text)
    exact = pieces[0]
    values = [numeric(p) for p in exact.split('=')]
    # An equation may start with a symbol (e.g. E = 6), but its RHS must parse fully.
    if len(values) > 1 and values[0] is None and not re.search(r'\d', exact.split('=')[0]):
        values = values[1:]
    if not values or any(v is None for v in values):
        return None, False
    if len({v[0] for v in values}) > 1:
        return None, True
    return values[-1], False


def section_answer(section, explicit=False):
    """Prefer an explicit opening value, then the final identified conclusion."""
    lines = section.splitlines()
    first = next((line for line in lines if line.strip().strip('*')), '')
    value, conflict = claim(first)
    if explicit and (value is not None or conflict):
        return value, conflict, first
    candidates = []
    # Boxed expressions can span lines and contain nested fractions.
    for match in re.finditer(r'\\boxed\s*\{', section):
        depth, end = 1, match.end()
        while end < len(section) and depth:
            depth += (section[end] == '{') - (section[end] == '}')
            end += 1
        if depth == 0:
            candidates.append((match.start(), section[match.end():end-1]))
    for match in re.finditer(r'(?im)^.*\b(?:therefore|thus|so[, ]|hence|the answer|the probability|the expected|there are|there is).*$', section):
        line = match.group().strip()
        splits = list(re.finditer(r'\b(?:is|are|equals)\b|:', line, re.I))
        if splits:
            candidates.append((match.start(), line[splits[-1].end():].strip()))
    # A final standalone math line is useful only inside an identified part.
    for match in re.finditer(r'(?m)^.*$', section):
        value, conflict = claim(match.group())
        tail = section[match.end():].strip().strip('\\[]()$* .\n')
        if (value is not None or conflict) and not tail:
            candidates.append((match.start(), match.group()))
    for _, evidence in sorted(candidates, reverse=True):
        value, conflict = claim(evidence)
        if value is not None or conflict:
            return value, conflict, evidence
    return None, False, ''


def extract(text, labels):
    marks = [(m.start(), m.end(), m.group(1).lower(), True) for m in FORMAT_LINE.finditer(text)]
    marks += [(m.start(), m.end(), (m.group(1) or m.group(2)).lower(), False)
              for m in LINE_LABEL.finditer(text)]
    marks.sort()
    result = {}
    for label in labels:
        own = [m for m in marks if m[2] == label]
        explicit = [m for m in own if m[3]]
        found = []
        for start, end, _, is_explicit in explicit or own:
            stop = min((m[0] for m in marks if m[0] > start), default=len(text))
            value, conflict, evidence = section_answer(text[end:stop], explicit=is_explicit)
            if value is not None or conflict:
                found.append((value, conflict, evidence))
        conflict = any(c for _, c, _ in found) or len({v[0] for v, _, _ in found if v is not None}) > 1
        value = found[-1][0] if found and not conflict else None
        result[label] = dict(value=value, conflicting=conflict,
                             evidence=found[-1][2] if found else '',
                             method='explicit' if explicit else 'part_section' if own else 'unparsed')
    return result


def score(text, task):
    extracted = extract(text, [p['label'] for p in task['parts']])
    parts = [dict(label=p['label'], parsed=str(extracted[p['label']]['value'][0]) if extracted[p['label']]['value'] else None,
                  correct=same_value(extracted[p['label']]['value'], p['answer']),
                  conflicting=extracted[p['label']]['conflicting'],
                  evidence=extracted[p['label']]['evidence'], method=extracted[p['label']]['method']) for p in task['parts']]
    hinted = next(p for p in parts if p['label'] == task['hint_part'])
    later = [p['correct'] for p in parts if p['label'] != task['hint_part']]
    return dict(part_results=parts, pass_rate=sum(p['correct'] for p in parts)/len(parts),
                hint_part_correct=hinted['correct'],
                exact_hint_match=hinted['parsed'] is not None and Fraction(hinted['parsed']) == Fraction(task['hint']),
                downstream_accuracy=sum(later)/len(later))
