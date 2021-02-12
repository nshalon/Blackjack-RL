import numpy as np


class Player:
    def __init__(self):
        self.sum = 0
        self.cards = []


class Game:
    def __init__(self, players, num_deck): # num_decks default = 4
        self.num_deck = num_deck
        self.deck = self.get_deck()
        self.players = players

    """
    purpose: get starting deck
    input:
    output: 
    -the full deck (with the values of the cards instead of actual cards), which
    will have more than 52 cards if num_deck > 1
    """

    def get_deck(self):
        return self.num_deck * (4 * (list(range(2, 10)) + [10, 10, 10, 10, 11]))

    """
       purpose: draw a card from the deck and update the sum for the respective agent
       input: 
       -num_draws: number of draws (2 to start, 1 everytime after that)
       -sum: the sum going into the draw
       -player: the player number (for adding to their sum)
       output:
       -cards
    """

    def draw(self, num_draws, player_ind):
        cards = []
        player = self.players[player_ind]
        for _ in range(num_draws):
            card = self.deck.pop(int(np.floor(len(self.deck) * np.random.random())))
            player.sum += self.get_sum(card, player.sum)
            cards.append(card)
        return self.check_bust(player_ind)

    """
       purpose: draw two cards from each person
       input: 
       output:
       -cards for two draws
    """

    def start_game(self):
        self.draw(1, 0)
        self.draw(2, 1)
        # for player_ind in range(len(self.players)):
        #     self.draw(2, player_ind)


    def check_bust(self, player):
        if self.players[player].sum > 21:
            return True
        else:
            return self.players[0].sum, self.players[1].sum

    """
         purpose: return the dealer policy
         input: 
         output:
         -dealer action based on sum
    """

    def dealer_policy(self):
        if self.players[0].sum >= 17:
            return "stay"
        else:
            return "hit"

    def get_sum(self, card, sum):
        if card != 11 or sum <= 10:
            return sum + card
        else:
            return sum + 1

    def dealer_iterate(self):
        dealer_action = self.dealer_policy()
        dealer_bust = False
        while not dealer_bust and dealer_action != "stay":

            self.draw(num_draws=1, player_ind=0)  # player 0 -> dealer
            dealer_bust = self.check_bust(0)
            dealer_action = self.dealer_policy()
        return dealer_bust

    def dealer_rollout(self):
        dealer_bust = self.dealer_iterate() # roll out the dealer's actions and check if they bust by the end of it
        if dealer_bust or self.players[0].sum < self.players[1].sum:
            return "win" # from perspective of the player
        elif self.players[0].sum > self.players[1].sum:
            return "loss"
        else:
            return "draw"

    def terminal_reward(self, state):
        if state == "loss":
            return -1
        else:
            return self.dealer_rollout

class Policy:
    def __init__(self, game, gamma=0.9, lr=0.05):
        self.game = game
        self.reward = {"loss" : -1,
                       "win" : 1,
                       "draw" : 0}
        self.epsilon = 0.05
        self.qval = {}
        self.actions = ["hit", "stay"]
        self.gamma = gamma
        self.lr = lr
        self.outcome = None


    def update_qval(self, state):
        if self.check_terminal(state):
            self.outcome = state
            return 0
        action = self.get_action(state)
        successor = self.get_successor_state(action, 1) # if hit then rollout (check bust) ; elif stay then dealer rollout -> successor
        reward = self.get_reward(successor)
        self.qval[(state, action)] = self.get_qval(state, action) +\
                                     self.lr * (reward + self.gamma * self.get_qval(successor, self.get_action(successor, False)) -\
                                                self.get_qval(state, action))

        return self.update_qval(successor)

    def check_terminal(self, state):
        if state == "stay" or state == "loss" or state == True or state == "win":
            return True
        elif state[1] > 21:
            return True
        else:
            return False

    def get_reward(self, state):
        return self.reward.get(state, 0)

    def get_qval(self, state, action):
        return self.qval.get((state, action), 0)

    def get_action(self, state, explore=True):
        if explore:
            if np.random.random() < self.epsilon:
                return np.random.choice(self.actions)
        values = np.array([self.get_qval(state, action) for action in self.actions])
        return self.actions[np.random.choice(np.argwhere(values == max(values)).ravel())]  # randomly break tie  breakers

    def get_successor_state(self, action, player_ind):
        if action == "hit":
            self.game.draw(1, player_ind)
            bust = self.game.check_bust(1)
            if bust is True:
                return "loss"
            else:
                return self.game.players[0].sum, self.game.players[1].sum
            # return self.game.check_bust(1)
        else:
            return self.game.dealer_rollout()

