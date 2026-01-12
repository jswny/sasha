from __future__ import annotations

from typing import Iterable, List
from pokerkit import Card, Rank, Suit


_RANK_NAMES = {
    Rank.DEUCE: "two",
    Rank.TREY: "three",
    Rank.FOUR: "four",
    Rank.FIVE: "five",
    Rank.SIX: "six",
    Rank.SEVEN: "seven",
    Rank.EIGHT: "eight",
    Rank.NINE: "nine",
    Rank.TEN: "ten",
    Rank.JACK: "jack",
    Rank.QUEEN: "queen",
    Rank.KING: "king",
    Rank.ACE: "ace",
}

_SUIT_NAMES = {
    Suit.CLUB: "clubs",
    Suit.DIAMOND: "diamonds",
    Suit.HEART: "hearts",
    Suit.SPADE: "spades",
}


def card_to_text(card: Card) -> str:
    rank = _RANK_NAMES[card.rank]
    suit = _SUIT_NAMES[card.suit]
    return f"{rank} of {suit} ({repr(card)})"


def cards_to_text(cards: Iterable[Card]) -> List[str]:
    return [card_to_text(card) for card in cards]


def cards_to_shorthand(cards: Iterable[Card]) -> List[str]:
    return [repr(card) for card in cards]
