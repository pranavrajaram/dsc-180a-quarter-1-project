"""Regression cases for notation, part boundaries, ambiguity, and safe arithmetic."""
import unittest
from fractions import Fraction
from scripts.math_answers import numeric, extract, score
from scripts.challenge import sign_flip_p


class AnswerTests(unittest.TestCase):
    def test_numeric_notation(self):
        cases = {r'\(13^{5}\)': 371293, r'\frac{1}{4}+\frac{1}{6}': Fraction(5,12),
                 r'\boxed{\frac{5}{12}}': Fraction(5,12), '1,234':1234,
                 '25%':Fraction(1,4), '13!':6227020800, r'\binom{48}{5}':1712304}
        for text, value in cases.items():
            with self.subTest(text=text):self.assertEqual(numeric(text)[0], value)
        for text in ['__import__("os")', '2**999999', '1/0', '1/4 or 1/2', 'x + 1', '999!']:
            self.assertIsNone(numeric(text))

    def test_prose_and_boundaries(self):
        text = '(a) Choose a rank.\nSo, there are 48 ways to choose 3 suits.\n(b) Thus, the answer is 144.'
        found = extract(text, ['a','b'])
        self.assertEqual(found['a']['value'][0],48)
        self.assertEqual(found['b']['value'][0],144)
        self.assertIsNone(extract('Some intermediate numbers: 3, 4, 48.', ['a'])['a']['value'])
        self.assertIsNone(extract('(a) Work in progress:\n2 + 3 = 5\nStill need to finish.', ['a'])['a']['value'])

    def test_section_conclusion_takes_priority_over_work(self):
        result = extract('(a) 2 + 3 = 5\nThus, the answer is 6.', ['a'])
        self.assertEqual(result['a']['value'][0], 6)

    def test_conflicts_and_final_labels(self):
        for text in ['Answer (a): 1/4\nAnswer (a): 1/2', 'Answer (a): 1/4 + 1/6 = 1/10']:
            self.assertTrue(extract(text,['a'])['a']['conflicting'])
        found=extract('(a) Thus, the answer is 12.\nAnswer (a): 10', ['a'])
        self.assertEqual(found['a']['value'][0],10)
        self.assertEqual(extract('Answer (a): 1/3 ≈ 0.333', ['a'])['a']['value'][0],Fraction(1,3))

    def test_extraction_does_not_depend_on_key_or_hint(self):
        text='(a) Thus, the answer is 48.\n(b) Answer is 2.'
        task={'parts':[{'label':'a','answer':'48'},{'label':'b','answer':'2'}], 'hint_part':'a','hint':'50'}
        first=score(text,task)
        task['parts'][0]['answer']='50';task['hint']='48'
        second=score(text,task)
        self.assertEqual(first['part_results'][0]['parsed'],second['part_results'][0]['parsed'])
        self.assertTrue(first['hint_part_correct']);self.assertTrue(second['exact_hint_match'])

    def test_sign_flip_known_results(self):
        self.assertEqual(sign_flip_p([1]*18+[0,0]), 2/2**18)
        self.assertEqual(sign_flip_p([1,-1]),1)
        self.assertEqual(sign_flip_p([0,0]),1)
