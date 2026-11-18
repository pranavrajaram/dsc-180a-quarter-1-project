"""Generate 200 multipart tasks in 20 families, with exact computed answer keys."""
import itertools as it
import json
import math
from fractions import Fraction as F

from scripts.experiment import ROOT


def build():
    tasks = []

    def add(family, variant, setup, parts, wrong):
        assert F(wrong) != F(parts[0][1]), family
        labels = 'abcd'[:len(parts)]
        tasks.append(dict(task_id=f'{family}_{variant:02}', family=family, domain='math',
                          prompt=setup+'\n'+'\n'.join(f'({label}) {q}' for label,(q,_) in zip(labels,parts)),
                          parts=[dict(label=label, answer=str(F(a))) for label,(_,a) in zip(labels,parts)],
                          hint_part='a', hint=str(F(wrong))))

    C = math.comb
    for v in range(10):
        ranks = 5+v
        n = 4*ranks
        total = C(n,5)
        four = ranks*(n-4)
        full = ranks*C(4,3)*(ranks-1)*C(4,2)
        deck = f'A deck has {ranks} ranks, each with four distinct suits, for {n} distinct cards. A hand contains five cards, with order ignored.'
        add('four_of_a_kind',v,deck,[('How many five-card hands are possible?',total),
            ('How many hands contain four cards of the same rank?',four),
            ('What is the reciprocal of the probability of four of a kind?',F(total,four))], math.perm(n,5))
        add('full_house',v,deck+' A full house contains three cards of one rank and two of another.',[
            ('How many ways can you choose the three-of-a-kind part?',ranks*C(4,3)),
            ('How many full houses are possible?',full),('What is the probability of a full house?',F(full,total))],ranks*math.perm(4,3))

        pairs = 4+v
        socks = list(range(pairs))*2
        matching = sum(socks[a]==socks[b] for a,b in it.combinations(range(2*pairs),2))
        triples = sum(len({socks[i] for i in group})<3 for group in it.combinations(range(2*pairs),3))
        assert matching == pairs and triples == pairs*(2*pairs-2)
        add('matching_socks',v,f'A drawer contains {pairs} pairs of socks, each pair a different color. Individual socks are distinct.',[
            ('How many unordered draws of two socks are possible?',C(2*pairs,2)),
            ('How many give a matching pair?',matching),('What is the probability that two socks match?',F(matching,C(2*pairs,2))),
            ('If three socks are drawn, what is the probability that at least two match?',F(triples,C(2*pairs,3)))],math.perm(2*pairs,2))

        digits, target = 3+v%3, 1+v
        def digit_count(length):
            counts = [1]+[0]*target
            for _ in range(length):
                counts = [sum(counts[s-d] for d in range(10) if s>=d) for s in range(target+1)]
            return counts[target]
        all_strings, zero_first = digit_count(digits), digit_count(digits-1)
        formula = sum((-1)**j*C(digits,j)*C(target-10*j+digits-1,digits-1)
                      for j in range(min(digits,target//10)+1))
        assert all_strings == formula
        add('digit_sum',v,f'Consider strings of {digits} decimal digits, allowing leading zeros.',[
            (f'How many have digit sum {target}?',all_strings),
            ('How many of those begin with zero?',zero_first),
            (f'How many {digits}-digit positive integers have that digit sum?',all_strings-zero_first)],all_strings+C(target+digits-2,digits-2))

        x,y = 3+v%5,3+v//5
        a,b = x//2,y//2
        through = C(a+b,a)*C(x+y-a-b,x-a)
        paths = C(x+y,x)
        add('lattice_paths',v,f'A path from (0,0) to ({x},{y}) uses only unit steps right or up.',[
            ('How many paths are possible?',paths),(f'How many pass through ({a},{b})?',through),
            (f'How many avoid ({a},{b})?',paths-through)],2**(x+y))

        flips = 4+v
        good = sum('11' not in ''.join(seq) for seq in it.product('01',repeat=flips))
        f0,f1 = 1,2
        for _ in range(flips):
            f0,f1 = f1,f0+f1
        assert good == f0
        add('no_consecutive_heads',v,f'A fair coin is flipped {flips} times.',[
            ('How many ordered sequences are possible?',2**flips),
            ('How many sequences contain no consecutive heads?',good),
            ('What is the probability of no consecutive heads?',F(good,2**flips))],2**(flips-1))

        m,n = [(3,5),(3,7),(3,11),(5,7),(5,11),(7,11),(3,13),(5,13),(7,13),(11,13)][v]
        conditions = [(m,m-1),(n,n-2),(17,4),(19,6)]
        def smallest(k):
            return next(i for i in range(1,math.prod(t[0] for t in conditions[:k])+1)
                        if all(i%mod==rem for mod,rem in conditions[:k]))
        first = smallest(2)
        add('chinese_remainder',v,'Find positive integers satisfying these remainder conditions.',[
            (f'What is the smallest integer leaving remainder {m-1} modulo {m} and {n-2} modulo {n}?',first),
            ('What is the smallest integer that also leaves remainder 4 modulo 17?',smallest(3)),
            ('What is the smallest integer that additionally leaves remainder 6 modulo 19?',smallest(4))],first+m*n)

        sides = 4+v
        add('coupon_collector',v,f'A fair {sides}-sided die is rolled repeatedly until every face appears.',[
            (f'If {sides-1} different faces have appeared, what is the expected number of additional rolls until the last face appears?',sides),
            (f'If {sides//2} different faces have appeared, what is the expected number of rolls until a new face appears?',F(sides,sides-sides//2)),
            ('What is the expected total number of rolls to see all faces?',sum(F(sides,k) for k in range(1,sides+1)))],F(1,sides))

        total_sum = 8+2*v
        bound = total_sum//2
        triples = [(x,y,total_sum-x-y) for x in range(1,total_sum) for y in range(1,total_sum-x)]
        assert len(triples)==C(total_sum-1,2)
        add('bounded_solutions',v,f'Consider ordered triples of positive integers with x+y+z={total_sum}.',[
            ('How many triples are there?',len(triples)),(f'How many have x>{bound}?',sum(x>bound for x,y,z in triples)),
            (f'How many have all coordinates at most {bound}?',sum(max(t)<=bound for t in triples))],C(total_sum+2,2))

        people = 4+v
        d0,d1 = 1,0
        for k in range(2,people+1):
            d0,d1 = d1,(k-1)*(d1+d0)
        assert d1 == sum((-1)**k*math.factorial(people)//math.factorial(k) for k in range(people+1))
        add('derangements',v,f'{people} people each put one hat in a pile, and the hats are returned in uniformly random order.',[
            ('In how many orders can the hats be returned?',math.factorial(people)),
            ('In how many orders does nobody receive their own hat?',d1),
            ('What is the probability nobody receives their own hat?',F(d1,math.factorial(people)))],math.factorial(people-1))

        balls,red = 12+v,2+v%4
        draws = list(it.combinations(range(balls),2))
        one = sum(a<red or b<red for a,b in draws)
        both = sum(a<red and b<red for a,b in draws)
        assert one==C(balls,2)-C(balls-red,2) and both==C(red,2)
        add('conditional_draw',v,f'A bag holds {balls} distinct balls, of which {red} are red. Two balls are drawn without replacement.',[
            ('How many unordered two-ball draws are possible?',len(draws)),
            ('How many contain at least one red ball?',one),('How many contain two red balls?',both),
            ('Given that at least one is red, what is the probability both are red?',F(both,one))],math.perm(balls,2))

        e,f = v%5+1,v//5+1
        def lcm_count(n):
            divisors = [d for d in range(1,n+1) if n%d==0]
            return sum(math.lcm(a,b)==n for a in divisors for b in divisors)
        counts = [lcm_count(n) for n in (2**e,3**f,2**e*3**f)]
        assert counts==[2*e+1,2*f+1,(2*e+1)*(2*f+1)]
        add('lcm_pairs',v,'Count ordered pairs (a,b) of positive integers with the specified least common multiple.',[
            (f'How many pairs have lcm(a,b)={2**e}?',counts[0]),
            (f'How many pairs have lcm(a,b)={3**f}?',counts[1]),
            (f'How many pairs have lcm(a,b)={2**e*3**f}?',counts[2])],2*(e+1))

        p = F(v+1,v+3)
        hh = (1+p)/p**2
        e1 = 1+(1-p)*hh
        assert hh == 1+p*e1+(1-p)*hh
        add('expected_heads',v,f'Independent flips of a biased coin have probability {p} of heads on each flip.',[
            ('What is the probability the first two flips are both heads?',p*p),
            ('What is the expected number of flips until the first head?',1/p),
            ('What is the expected number of flips until two consecutive heads?',hh)],p)

        width,height = 3+v%5,3+v//5
        points = list(it.product(range(width),range(height)))
        triples = list(it.combinations(points,3))
        collinear = sum((b[0]-a[0])*(c[1]-a[1])==(b[1]-a[1])*(c[0]-a[0]) for a,b,c in triples)
        add('grid_triangles',v,f'Consider all {len(points)} grid points with integer coordinates 0<=x<{width} and 0<=y<{height}.',[
            ('How many unordered choices of three points are possible?',len(triples)),
            ('How many choices are collinear?',collinear),('How many form triangles of positive area?',len(triples)-collinear)],math.perm(len(points),3))

        men,women,size = 5+v,4+v%3,3+v%3
        groups = list(it.combinations(range(men+women),size))
        none = sum(all(i<men for i in g) for g in groups)
        one = sum(sum(i>=men for i in g)==1 for g in groups)
        assert none==C(men,size) and one==women*C(men,size-1)
        add('committee',v,f'A committee of {size} people is chosen from {men} men and {women} women.',[
            ('How many committees are possible?',len(groups)),('How many contain no women?',none),
            ('How many contain exactly one woman?',one),('How many contain at least two women?',len(groups)-none-one)],math.perm(men+women,size))

        fair,flips = 2+v//4,2+v%4
        p_fair = F(1,2**flips)
        p_all = F(1,fair+1)+F(fair,fair+1)*p_fair
        add('double_headed_coin',v,f'A bag contains {fair} fair coins and one double-headed coin. One coin is chosen uniformly and flipped {flips} times.',[
            ('If the coin is fair, what is the probability all flips are heads?',p_fair),
            ('What is the overall probability all flips are heads?',p_all),
            ('Given all flips were heads, what is the probability the coin is double-headed?',F(1,fair+1)/p_all)],F(1,2**(flips-1)))

        sides,k = 6+v,2+v%5
        rolls = list(it.product(range(1,sides+1),repeat=2))
        maximum = F(sum(max(r) for r in rolls),sides*sides)
        assert maximum==F((sides+1)*(4*sides-1),6*sides)
        add('maximum_dice',v,f'Two fair {sides}-sided dice numbered 1 through {sides} are rolled. M is the larger result.',[
            (f'How many of the {sides*sides} ordered outcomes have M at most {k}?',k*k),
            (f'How many have M exactly {k}?',2*k-1),('What is the expected value of M?',maximum)],C(k,2)+k)

        a,b,d = 4+v,6+v,12+v
        rate = F(1,a)+F(1,b)
        net = rate-F(1,d)
        add('pipes',v,f'Pipe A fills a tank in {a} hours, pipe B in {b} hours, and a drain empties it in {d} hours.',[
            ('With only A and B open, what fraction fills per hour?',rate),
            ('With both pipes and the drain open, what fraction fills per hour?',net),
            ('With both pipes and drain open, how many hours fill an empty tank?',1/net)],F(1,a+b))

        floors,people = 5+v,3+v%3
        add('elevator',v,f'{people} people independently choose uniformly among {floors} floors to exit an elevator.',[
            ('How many equally likely assignments of people to floors are possible?',floors**people),
            ('How many assignments give everyone different floors?',math.perm(floors,people)),
            ('What is the probability everyone exits on different floors?',F(math.perm(floors,people),floors**people))],people**floors)

        price,up,down,tax = 85+20*v,10+v,5+v,5+v%5
        raised = F(price*(100+up),100)
        lowered = raised*F(100-down,100)
        add('price',v,f'An item costs ${price}. Its price rises {up}%, then falls {down}%, then {tax}% sales tax is applied.',[
            ('What is the price after the increase, in dollars?',raised),
            ('What is the price after the decrease, in dollars?',lowered),
            ('What is the final price including tax, in dollars?',lowered*F(100+tax,100))],price+up)

    assert len(tasks)==200 and len({t['prompt'] for t in tasks})==200
    return tasks


if __name__ == '__main__':
    tasks = build()
    path = ROOT/'data/tasks_math_scale.json'
    path.write_text(json.dumps(tasks,indent=2)+'\n')
    print(f'Saved {len(tasks)} tasks in 20 families to {path}')
