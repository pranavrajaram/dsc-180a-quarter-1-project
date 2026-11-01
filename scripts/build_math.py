"""Builds tasks_math.json: 20 multi-part problems whose parts build on each other.

Every answer is computed here by brute-force enumeration or exact Fraction arithmetic,
never typed by hand. Each problem also has a wrong hint for part (a): a specific,
plausible mistake (also computed in code) that is guaranteed to differ from the answer.

Run: python3 -m scripts.build_math"""
import itertools
import json
import math
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
C = math.comb


def poker_counts():
    """Brute force over all C(52,5) hands; returns (four_of_a_kind, full_house)."""
    four = full = 0
    ranks = [r for r in range(13) for _ in range(4)]
    for hand in itertools.combinations(range(52), 5):
        shape = sorted(Counter(ranks[i] for i in hand).values())
        four += shape == [1, 4]
        full += shape == [2, 3]
    return four, full


def problems(four, full):
    out = []

    def add(task_id, difficulty, setup, parts):
        out.append((task_id, difficulty, setup, parts))

    # 1. Four of a kind
    total = C(52, 5)
    assert four == 13 * 48
    add("four_of_a_kind", "hard",
        "A poker hand consists of five cards dealt from a standard 52-card deck.",
        [("How many different five-card hands are possible?", total),
         ("How many of those hands are a four-of-a-kind (four of the five cards share a rank)?", four),
         ("If p is the probability of being dealt a four-of-a-kind, what is 1/p?", F(total, four))])

    # 2. Socks
    socks = [c for c in range(10) for _ in range(2)]
    pairs = list(itertools.combinations(range(20), 2))
    matches = sum(socks[a] == socks[b] for a, b in pairs)
    triples = list(itertools.combinations(range(20), 3))
    any_match3 = sum(len({socks[i] for i in t}) < 3 for t in triples)
    add("matching_socks", "medium",
        "A drawer contains 10 pairs of socks, and each pair is a different color.",
        [("How many ways are there to draw 2 socks from the drawer?", len(pairs)),
         ("How many of those draws give a matching pair?", matches),
         ("What is the probability that 2 randomly drawn socks match?", F(matches, len(pairs))),
         ("If you instead draw 3 socks at random, what is the probability that at least two of them match?", F(any_match3, len(triples)))])

    # 3. Full house
    triple_ways = 13 * C(4, 3)
    assert full == 3744
    add("full_house", "hard",
        "A poker hand consists of five cards dealt from a standard 52-card deck. A full house is three cards of one rank and two cards of another rank.",
        [("In how many ways can you choose the three-of-a-kind part (a rank and three cards of that rank)?", triple_ways),
         ("How many five-card hands are a full house?", full),
         ("What is the probability of being dealt a full house?", F(full, total))])

    # 4. Digit sum nine
    strings = sum(1 for d in itertools.product(range(10), repeat=5) if sum(d) == 9)
    lead0 = sum(1 for d in itertools.product(range(10), repeat=4) if sum(d) == 9)
    real = sum(1 for n in range(10000, 100000) if sum(map(int, str(n))) == 9)
    assert real == strings - lead0
    add("digit_sum_nine", "hard",
        "Consider strings of five decimal digits (leading zeros allowed, so 00000 through 99999).",
        [("How many such strings have digits summing to 9?", strings),
         ("How many of those strings begin with the digit 0?", lead0),
         ("How many five-digit positive integers (10000 through 99999) have digits summing to 9?", real)])

    # 5. Lattice paths
    def paths(dx, dy):
        return C(dx + dy, dx)
    brute = 0
    for steps in set(itertools.permutations("RRRRRRUUUU")):
        x = y = 0
        hit = False
        for s in steps:
            x, y = (x + 1, y) if s == "R" else (x, y + 1)
            hit |= (x, y) == (3, 2)
        brute += not hit
    through = paths(3, 2) * paths(3, 2)
    assert brute == paths(6, 4) - through
    add("lattice_paths_avoid", "hard",
        "A path from (0, 0) to (6, 4) consists of unit steps, each one unit right or one unit up.",
        [("How many such paths are there in total?", paths(6, 4)),
         ("How many of these paths pass through the point (3, 2)?", through),
         ("How many of these paths do not pass through (3, 2)?", brute)])

    # 6. No consecutive heads
    seqs = list(itertools.product("HT", repeat=10))
    good = sum("HH" not in "".join(s) for s in seqs)
    add("no_consecutive_heads", "hard",
        "A fair coin is flipped 10 times and the sequence of heads and tails is recorded.",
        [("How many different sequences are possible?", len(seqs)),
         ("How many sequences contain no two consecutive heads?", good),
         ("What is the probability that no two consecutive flips are both heads?", F(good, len(seqs)))])

    # 7. Chinese remainder
    def smallest(conds):
        return next(n for n in itertools.count(1) if all(n % m == r for m, r in conds))
    c2 = [(3, 2), (5, 3)]
    c3 = c2 + [(7, 4)]
    c4 = c3 + [(11, 6)]
    add("chinese_remainder", "hard",
        "We look for positive integers n satisfying several remainder conditions.",
        [("What is the smallest n that leaves remainder 2 when divided by 3 and remainder 3 when divided by 5?", smallest(c2)),
         ("What is the smallest n that also leaves remainder 4 when divided by 7?", smallest(c3)),
         ("What is the smallest n that additionally leaves remainder 6 when divided by 11?", smallest(c4))])

    # 8. Coupon collector
    def wait(seen):
        return F(6, 6 - seen)
    expected = sum(wait(k) for k in range(6))
    add("coupon_collector", "hard",
        "A fair six-sided die is rolled repeatedly until every face from 1 to 6 has appeared at least once.",
        [("If 5 different faces have already appeared, what is the expected number of additional rolls until the last face appears?", wait(5)),
         ("If exactly 3 different faces have appeared so far, what is the expected number of additional rolls until a new face appears?", wait(3)),
         ("What is the expected total number of rolls needed to see all six faces?", expected)])

    # 9. Bounded solutions
    triples_all = sum(1 for x in range(1, 19) for y in range(1, 19) if 1 <= 20 - x - y)
    x_big = sum(1 for x in range(11, 19) for y in range(1, 19) if 1 <= 20 - x - y)
    bounded = sum(1 for x in range(1, 11) for y in range(1, 11) if 1 <= 20 - x - y <= 10)
    add("bounded_solutions", "hard",
        "Consider ordered triples of positive integers (x, y, z) with x + y + z = 20.",
        [("How many such triples are there?", triples_all),
         ("How many of them have x > 10?", x_big),
         ("How many triples have all three of x, y, z at most 10?", bounded)])

    # 10. Derangements
    perms = list(itertools.permutations(range(6)))
    der = sum(all(p[i] != i for i in range(6)) for p in perms)
    perms5 = list(itertools.permutations(range(5)))
    der5 = sum(all(p[i] != i for i in range(5)) for p in perms5)
    add("derangement_probability", "hard",
        "Six people each put their hat in a pile, and the hats are handed back in a uniformly random order.",
        [("In how many different orders can the six hats be handed back?", len(perms)),
         ("In how many of those orders does nobody receive their own hat?", der),
         ("What is the probability that nobody receives their own hat?", F(der, len(perms)))])

    # 11. Both aces given at least one
    ranks = [r for r in range(13) for _ in range(4)]
    hands2 = list(itertools.combinations(range(52), 2))
    one = [h for h in hands2 if ranks[h[0]] == 0 or ranks[h[1]] == 0]
    both = [h for h in one if ranks[h[0]] == 0 and ranks[h[1]] == 0]
    add("both_aces_given_one", "hard",
        "Two cards are drawn without replacement from a standard 52-card deck.",
        [("How many different two-card hands are possible?", len(hands2)),
         ("How many two-card hands contain at least one ace?", len(one)),
         ("How many two-card hands contain two aces?", len(both)),
         ("Given that at least one card is an ace, what is the probability that both cards are aces?", F(len(both), len(one)))])

    # 12. LCM pairs
    def lcm_count(n):
        return sum(1 for a in range(1, n + 1) for b in range(1, n + 1) if math.lcm(a, b) == n)
    add("lcm_pairs", "hard",
        "For a positive integer N, count ordered pairs (a, b) of positive integers with lcm(a, b) = N.",
        [("How many ordered pairs have lcm(a, b) = 8?", lcm_count(8)),
         ("How many ordered pairs have lcm(a, b) = 9?", lcm_count(9)),
         ("How many ordered pairs have lcm(a, b) = 72?", lcm_count(72))])

    # 13. Expected flips until HH (solved exactly, then checked against the equations)
    e0 = F(6)
    e1 = 1 + F(1, 2) * e0
    assert e0 == 1 + F(1, 2) * (e1 + e0)
    add("expected_flips_hh", "hard",
        "A fair coin is flipped repeatedly.",
        [("What is the probability that the first two flips are both heads?", F(1, 4)),
         ("What is the expected number of flips until the first head appears?", F(2)),
         ("What is the expected number of flips until two heads appear in a row?", e0)])

    # 14. Grid triangles
    pts = [(x, y) for x in range(4) for y in range(4)]
    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    trip = list(itertools.combinations(pts, 3))
    collinear = sum(area2(*t) == 0 for t in trip)
    add("grid_triangles", "hard",
        "Consider the 16 points (x, y) where x and y are each in {0, 1, 2, 3}.",
        [("How many ways are there to choose 3 of the 16 points?", len(trip)),
         ("How many of those choices are three collinear points?", collinear),
         ("How many triangles with positive area have all three vertices among the 16 points?", len(trip) - collinear)])

    # 15. Committee
    people = ["M"] * 7 + ["W"] * 5
    groups = list(itertools.combinations(range(12), 5))
    def women(g):
        return sum(people[i] == "W" for i in g)
    add("committee_women", "medium",
        "A committee of 5 people is chosen from a group of 7 men and 5 women.",
        [("How many different committees are possible?", len(groups)),
         ("How many committees contain no women?", sum(women(g) == 0 for g in groups)),
         ("How many committees contain exactly one woman?", sum(women(g) == 1 for g in groups)),
         ("How many committees contain at least two women?", sum(women(g) >= 2 for g in groups))])

    # 16. Double-headed coin
    hh_fair = F(1, 4)
    hh_total = F(1, 3) * 1 + F(2, 3) * hh_fair
    add("double_headed_coin", "medium",
        "A bag contains three coins: two are fair and one has heads on both sides. A coin is chosen at random and flipped twice.",
        [("If the chosen coin is fair, what is the probability that both flips are heads?", hh_fair),
         ("What is the overall probability that both flips are heads?", hh_total),
         ("Given that both flips are heads, what is the probability that the chosen coin is the double-headed one?", F(1, 3) / hh_total)])

    # 17. Max of two dice
    rolls = list(itertools.product(range(1, 7), repeat=2))
    add("max_of_two_dice", "medium",
        "Two fair six-sided dice are rolled, and M is the larger of the two numbers (if they are equal, M is that common value).",
        [("Out of the 36 equally likely outcomes, how many have M at most 4?", sum(max(r) <= 4 for r in rolls)),
         ("How many outcomes have M exactly equal to 4?", sum(max(r) == 4 for r in rolls)),
         ("What is the expected value of M?", F(sum(max(r) for r in rolls), 36))])

    # 18. Pipes
    ab = F(1, 6) + F(1, 8)
    net = ab - F(1, 12)
    add("pipes_and_drain", "medium",
        "Pipe A fills a tank in 6 hours and pipe B fills it in 8 hours. A drain empties a full tank in 12 hours. The tank starts empty.",
        [("With only pipes A and B open, what fraction of the tank is filled per hour?", ab),
         ("With both pipes and the drain open, what fraction of the tank is filled per hour?", net),
         ("With both pipes and the drain open, how many hours does it take to fill the tank?", 1 / net)])

    # 19. Elevator
    picks = list(itertools.product(range(6), repeat=4))
    distinct = sum(len(set(p)) == 4 for p in picks)
    add("elevator_floors", "medium",
        "Four people enter an elevator in a building with 6 floors above them. Each person independently and uniformly picks one of the 6 floors to exit.",
        [("How many equally likely ways can the four people pick floors?", len(picks)),
         ("In how many of those ways do all four people pick different floors?", distinct),
         ("What is the probability that all four people exit on different floors?", F(distinct, len(picks)))])

    # 20. Price changes
    raised = F(80) * F(6, 5)
    lowered = raised * F(3, 4)
    add("price_changes", "medium",
        "An item originally costs $80. Its price is raised by 20%, then the new price is lowered by 25%, and finally a 10% sales tax is added at checkout.",
        [("What is the price after the 20% increase, in dollars?", raised),
         ("What is the price after the 25% decrease, in dollars?", lowered),
         ("What is the final amount paid including tax, in dollars?", lowered * F(11, 10))])
    return out


def as_text(value):
    value = F(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


# Wrong hints target part (a): the easiest part, and the one every later part builds on.
# Each value is a specific, common mistake, computed rather than typed.
HINT_PART = "a"
PART_A_HINTS = {
    "four_of_a_kind": 52 * 51 * 50 * 49 * 48,          # counts ordered hands
    "matching_socks": 20 * 19,                         # counts ordered draws
    "full_house": 13 * 4 * 3 * 2,                      # orders the three cards
    "digit_sum_nine": math.comb(9 + 5, 4),             # stars and bars off by one
    "lattice_paths_avoid": 2 ** 10,                    # treats every step as a free choice
    "no_consecutive_heads": 2 ** 9,                    # off by one in the exponent
    "chinese_remainder": 23,                           # second solution instead of the smallest
    "coupon_collector": F(1, 6),                       # probability instead of expected wait
    "bounded_solutions": math.comb(20 + 2, 2),         # counts nonnegative instead of positive solutions
    "derangement_probability": math.factorial(5),      # 5! instead of 6!
    "both_aces_given_one": 52 * 51,                    # counts ordered hands
    "lcm_pairs": 2 * 4,                                # double counts the pair (8, 8)
    "expected_flips_hh": F(1, 2),                      # probability of one head
    "grid_triangles": 16 * 15 * 14,                    # counts ordered choices
    "committee_women": 12 * 11 * 10 * 9 * 8,           # counts ordered committees
    "double_headed_coin": F(1, 2),                     # probability of one head
    "max_of_two_dice": math.comb(4, 2) + 4,            # counts unordered outcomes
    "pipes_and_drain": F(1, 6 + 8),                    # adds the times instead of the rates
    "elevator_floors": 4 ** 6,                         # swaps base and exponent
    "price_changes": 80 + 20,                          # adds 20 dollars instead of 20 percent
}


def build():
    four, full = poker_counts()
    tasks = []
    for task_id, difficulty, setup, parts in problems(four, full):
        assert 3 <= len(parts) <= 4
        labels = "abcd"[:len(parts)]
        hint = F(PART_A_HINTS[task_id])
        assert hint != F(parts[0][1]), task_id
        prompt = setup + "\n" + "\n".join(f"({l}) {q}" for l, (q, _) in zip(labels, parts))
        tasks.append({"task_id": task_id, "domain": "math", "difficulty": difficulty, "prompt": prompt,
                      "parts": [{"label": l, "answer": as_text(a)} for l, (_, a) in zip(labels, parts)],
                      "hint_part": HINT_PART, "hint": as_text(hint)})
    assert len(tasks) == 20 and len({t["task_id"] for t in tasks}) == 20
    assert set(PART_A_HINTS) == {t["task_id"] for t in tasks}
    return tasks


def main():
    tasks = build()
    (ROOT / "tasks_math.json").write_text(json.dumps(tasks, indent=2) + "\n")
    for t in tasks:
        answers = ", ".join(f'{p["label"]}={p["answer"]}' for p in t["parts"])
        print(f'{t["task_id"]:26} {answers}   hint={t["hint"]}')


if __name__ == "__main__":
    main()
