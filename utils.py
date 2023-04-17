from datatypes import int_to_rank, int_to_suit,rank_to_int,suit_to_int
from random import shuffle

def return_deck():
    deck = []
    for suit in range(1,5):
        for rank in range(1,14):
            deck.append((rank,suit))
    shuffle(deck)
    return deck

def human_readable_cards(cards):
    readable_cards = []
    for i in range(0,len(cards),2):
        readable_cards.append([int_to_rank[cards[i]],int_to_suit[cards[i+1]]])
    return readable_cards

def readable_card_to_int(card):
    return [rank_to_int[card[0]],suit_to_int[card[1]]]