"""Experiment conducted.

This module contains the code ran for each of the experiments outlined in the paper.
"""

from typing import List


class Solution:
    def wordsInPhoneNumber(self, phone: str, words: list[str]) -> list[str]:
        # Traditional keypad mapping
        lookup = {
            "a": "2",
            "b": "2",
            "c": "2",
            "d": "3",
            "e": "3",
            "f": "3",
            "g": "4",
            "h": "4",
            "i": "4",
            "j": "5",
            "k": "5",
            "l": "5",
            "m": "6",
            "n": "6",
            "o": "6",
            "p": "7",
            "q": "7",
            "r": "7",
            "s": "7",
            "t": "8",
            "u": "8",
            "v": "8",
            "w": "9",
            "x": "9",
            "y": "9",
            "z": "9",
        }

        res = []

        for word in words:
            encoded = ""
            for ch in word:
                encoded += lookup[ch]
            if encoded in phone:
                res.append(word)

        return res
