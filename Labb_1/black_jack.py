# Skapa ett program som simulerar ett blackjack-spel mellan en spelare och en dator.

# ·         Spelet spelas med en vanlig kortlek som blandas innan varje runda.

# ·         Varje spelare får två kort i början av spelet. Datorn visar bara upp ett av sina kort.

# ·         Spelaren kan välja att ta fler kort (hit) eller stanna på sina nuvarande kort (stand).

# ·         Spelaren kan fortsätta att ta kort tills hen når 21 poäng eller över.

# ·         Om spelaren går över 21 poäng förlorar hen direkt.

# ·         När spelaren stannar, spelar datorn sin tur. Datorn måste ta kort så länge summan av korten är mindre än 17 poäng och stanna när datorns kortsumma är 17 poäng eller mer.

# ·         Om datorn går över 21 poäng vinner spelaren oavsett vilka kort spelaren har.

# ·         Om varken spelaren eller datorn går över 21 poäng så vinner den som har högst kortsumma.



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
        import random
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()


class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        value = 0
        aces = 0
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
        answear = input("Do you want to hit? (y/n): ").lower()
        if answear == 'y':
            return True
        else:
            print("Player stands!")
            return False

    def is_busted(self):
        return self.hand.get_value() > 21

class Dealer(Player):
    def is_hitting(self):
        # if the dealer has less than 17, the dealer should hit
        return True if self.hand.get_value() < 17 else False


class Game:
    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.player = Player("Player")
        self.dealer = Dealer("Dealer")

    def show_hands(self, initial=False):
        if initial:
            print(f"{self.player.name} has {self.player.hand} with a value of {self.player.hand.get_value()}")
            print(f"{self.dealer.name} has {self.dealer.hand.cards[0]} and an unknown card")
        else:
            print(f"{self.player.name} has {self.player.hand} with a value of {self.player.hand.get_value()}")
            print(f"{self.dealer.name} has {self.dealer.hand} with a value of {self.dealer.hand.get_value()}")

    def determine_winner(self):
        player_value = self.player.hand.get_value()
        dealer_value = self.dealer.hand.get_value()
        if player_value > 21:
            print("You busted!")
        elif dealer_value > 21:
            print("Dealer busted!")
        elif player_value > dealer_value:
            print("You win!")
        elif player_value < dealer_value:
            print("Dealer wins!")
        else:
            print("It's a tie!")

    def end_game(self):
        answear = input("Do you want to play again? (y/n): ").lower()
        if answear == 'y':
            return True
        else:
            return False


    def play(self):
        while True:
            self.deck = Deck()
            self.deck.shuffle()
            self.player.hand = Hand()
            self.dealer.hand = Hand()

            # Initial dealing
            self.player.hand.add_card(self.deck.deal())
            self.player.hand.add_card(self.deck.deal())
            self.dealer.hand.add_card(self.deck.deal())
            self.dealer.hand.add_card(self.deck.deal())

            print("\n" + "="*20)
            print("Welcome to Blackjack!")
            # Show initial hands, hiding dealer's first card
            self.show_hands(initial=True)
            print("="*20 + "\n")

            print("Player's turn!")
            while self.player.is_hitting():
                print("="*20)
                print("Player hits!")
                self.player.hand.add_card(self.deck.deal())
                self.show_hands()
                if self.player.is_busted():
                    print("Player busts!")
                    self.determine_winner()
                    break
            print("="*20)

            print("="*20)
            if not self.player.is_busted():
                print("Dealer's turn!")
                while self.dealer.is_hitting():
                    print("Dealer hits!")
                    self.dealer.hand.add_card(self.deck.deal())
                    self.show_hands()
                    if self.dealer.is_busted():
                        print("Dealer busts!")
                        break
                print("Dealer stands!")
                print("="*20)
                self.show_hands()
                self.determine_winner()
                print("="*20)
            print("="*20)
            if not self.end_game():
                break

if __name__ == "__main__":
    game = Game()
    game.play()
