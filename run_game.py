import argparse
from tqdm import tqdm
import rl_frame


def run_game():

    parser = argparse.ArgumentParser(description='run the RL blackjack simulation')
    parser.add_argument('iterations', metavar='<number of iterations to train on>', type=int, help='cycle_contig_table')
    parser.add_argument('alpha', metavar='<number of iterations to train on>', type=float, help='cycle_contig_table')
    parser.add_argument('gamma', metavar='<number of iterations to train on>', type=float, help='cycle_contig_table')
    parser.add_argument('num_decks', metavar='<how many decks do you want to train on>', type=int, help='cycle_contig_table', default=4)
    parser.add_argument('write_table', metavar='<write q table (T/F)>', type=bool, help='cycle_contig_table',
                        default=True)
    parser.add_argument('table_dir', metavar='<path to write q table>', type=str, help='cycle_contig_table',
                        default=True)
    parser.add_argument('train', metavar='<train? (T/F)>', type=bool, help='cycle_contig_table',
                        default=True)
    parser.add_argument('test', metavar='<test rollouts? (T/F)>', type=bool, help='cycle_contig_table',
                        default=True)

    args = parser.parse_args()

    outcomes = []
    Policy = rl_frame.Policy(game=None, lr=args.alpha, gamma=args.gamma)
    for trial in tqdm(range(args.iterations)):
        dealer = rl_frame.Player()
        player = rl_frame.Player()
        game = rl_frame.Game([dealer, player], args.num_decks)
        game.start_game()
        Policy.game = game
        Policy.update_qval((game.players[0].sum, game.players[1].sum))
        outcomes.append(Policy.outcome)
        if trial % 1000 == 0:
            percent_win = len([1 for out in outcomes[-100:] if out == "win"]) / 100
            print("Trial %s percent win in the last 100: %f" % (trial, percent_win))

    print(Policy.qval)


if __name__ == "__main__":
    run_game()
