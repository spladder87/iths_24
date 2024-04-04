import random

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value} of {self.suit}"

class Deck:
    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        self.cards = [Card(suit, value) for suit in suits for value in values]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        value, aces = 0, 0
        for card in self.cards:
            if card.value in ['Jack', 'Queen', 'King']:
                value += 10
            elif card.value == 'Ace':
                aces += 1
                value += 11
            else:
                value += int(card.value)
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def __str__(self):
        return ', '.join(map(str, self.cards))

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = Hand()

    def is_hitting(self):
        return input("Do you want to hit? (y/n): ").lower() == 'y'

    def is_busted(self):
        return self.hand.get_value() > 21

class Dealer(Player):
    pass  # Inherits everything from Player

class Game:
    def __init__(self):
        self.deck, self.player, self.dealer = Deck(), Player("Player"), Dealer("Dealer")
        self.deck.shuffle()

    def _deal_initial_cards(self):
        for _ in range(2):
            self.player.hand.add_card(self.deck.deal())
            self.dealer.hand.add_card(self.deck.deal())

    def _player_turn(self):
        while self.player.is_hitting():
            print("Player hits!")
            self.player.hand.add_card(self.deck.deal())
            self._show_hands()
            if self.player.is_busted():
                print("Player busts!")
                break

    def _dealer_turn(self):
        while self.dealer.hand.get_value() < 17:
            print("Dealer hits!")
            self.dealer.hand.add_card(self.deck.deal())
            if self.dealer.is_busted():
                print("Dealer busts!")
                break

    def _show_hands(self, initial=False):
        print(f"{self.player.name} has {self.player.hand} with a value of {self.player.hand.get_value()}")
        if initial:
            print(f"{self.dealer.name} has {self.dealer.hand.cards[0]} and an unknown card")
        else:
            print(f"{self.dealer.name} has {self.dealer.hand} with a value of {self.dealer.hand.get_value()}")

    def _determine_winner(self):
        player_value, dealer_value = self.player.hand.get_value(), self.dealer.hand.get_value()
        if player_value > 21 or (dealer_value <= 21 and dealer_value > player_value):
            print("Dealer wins!")
        elif dealer_value > 21 or player_value > dealer_value:
            print("Player wins!")
        else:
            print("It's a tie!")

    def play(self):
        while True:
            self.deck, self.player.hand, self.dealer.hand = Deck(), Hand(), Hand()
            self.deck.shuffle()
            self._deal_initial_cards()
            print("\nWelcome to Blackjack!\n" + "="*20)
            self._show_hands(initial=True)

            self._player_turn()
            if not self.player.is_busted():
                self._dealer_turn()

            self._show_hands()
            self._determine_winner()

            if input("Do you want to play again? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    Game().play()

