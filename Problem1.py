import numpy as np
import scr.FigureSupport as figureLibrary
import scr.StatisticalClasses as Stat


class Game(object):
    def __init__(self, id, prob_head):
        self._id = id
        self._rnd = np.random
        #self._rnd.seed(id)
        self._probHead = prob_head  # probability of flipping a head
        self._countWins = 0  # number of wins, set to 0 to begin

    def simulate(self, n_of_flips):

        count_tails = 0  # number of consecutive tails so far, set to 0 to begin

        # flip the coin 20 times
        for i in range(n_of_flips):

            # in the case of flipping a heads
            if self._rnd.random_sample() < self._probHead:
                if count_tails >= 2:  # if the series is ..., T, T, H
                    self._countWins += 1  # increase the number of wins by 1
                count_tails = 0  # the tails counter needs to be reset to 0 because a heads was flipped

            # in the case of flipping a tails
            else:
                count_tails += 1  # increase tails count by one

    def get_reward(self):
        # calculate the reward from playing a single game
        return 100*self._countWins - 250


class SetOfGames:
    def __init__(self, prob_head, n_games):
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._lossprob = []
        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())

        self._sumStat_gameoutcome = \
        Stat.SummaryStat('Game outcomes', self._gameRewards)

    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return sum(self._gameRewards) / len(self._gameRewards)

    def get_reward_list(self):
        """ returns all the rewards from all game to later be used for creation of histogram """
        return self._gameRewards


    def get_max(self):
        """ returns maximum reward"""
        return max(self._gameRewards)

    def get_min(self):
        """ returns minimum reward"""
        return min(self._gameRewards)

    def get_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        for value in self._gameRewards:
            if value < 0:
                count_loss += 1
        self._gamelossprob = count_loss / len(self._gameRewards)
        return self._gamelossprob

    def get_CI_reward(self, alpha):
        """
        :param alpha: confidence level
        :return: t-based confidence interval
        """
        return self._sumStat_gameoutcome.get_t_CI(alpha)






class Multigame:
    """ simulates multiple cohorts with different parameters """

    def __init__(self, ids, pop_sizes, prob_head):
        """
        :param ids: a list of ids for cohorts to simulate
        :param pop_sizes: a list of population sizes of cohorts to simulate
        :param mortality_probs: a list of the mortality probabilities
        """
        self._ids = ids
        self._popSizes = pop_sizes
        self._probhead = prob_head

        self._rewards = []      # two dimensional list of patient survival time from each simulated cohort
        self._meanreward = []   # list of mean patient survival time for each simulated cohort
        self._lossprob=[]
        self._sumStat_meanreward = None

        self._sumStat_gamelossprob = None
    def simulate(self, n_time_steps):
        """ simulates all cohorts """

        for i in range(len(self._ids)):
            # create a cohort
            game = SetOfGames(self._probhead[i], self._popSizes[i])
            # simulate the cohort

            # store all patient surival times from this cohort
            self._rewards.append(game.get_reward_list())
            # store average survival time for this cohort
            self._meanreward.append(game.get_ave_reward())
            self._lossprob.append(game.get_probability_loss())

        # after simulating all cohorts
        # summary statistics of mean survival time
        self._sumStat_meanreward = Stat.SummaryStat('Mean reward', self._meanreward)

        self._sumStat_gamelossprob = Stat.SummaryStat('Mean loss prob', self._lossprob)
    def get_cohort_mean_reward(self, cohort_index):
        """ returns the mean survival time of an specified cohort
        :param cohort_index: integer over [0, 1, ...] corresponding to the 1st, 2ndm ... simulated cohort
        """
        return self._meanreward[cohort_index]

    def get_cohort_CI_mean_reward(self, cohort_index, alpha):
        """ :returns: the confidence interval of the mean survival time for a specified cohort
        :param cohort_index: integer over [0, 1, ...] corresponding to the 1st, 2ndm ... simulated cohort
        :param alpha: significance level
        """
        st = Stat.SummaryStat('', self._meanreward[cohort_index])
        return st.get_t_CI(alpha)

    def get_all_mean_reward(self):
        " :returns a list of mean survival time for all simulated cohorts"
        return self._meanreward

    def get_overall_mean_reward(self):
        """ :returns the overall mean survival time (the mean of the mean survival time of all cohorts)"""
        return self._sumStat_meanreward.get_mean()

    def get_cohort_PI_reward(self, cohort_index, alpha):
        """ :returns: the prediction interval of the survival time for a specified cohort
        :param cohort_index: integer over [0, 1, ...] corresponding to the 1st, 2ndm ... simulated cohort
        :param alpha: significance level
        """
        st = Stat.SummaryStat('', self._rewards[cohort_index])
        return st.get_PI(alpha)

    def get_PI_mean_reward(self, alpha):
        """ :returns: the prediction interval of the mean survival time"""
        return self._sumStat_meanreward.get_PI(alpha)


    def get_PI_loss_prob(self, alpha):
        """ :returns: the prediction interval of the mean survival time"""
        return self._sumStat_gamelossprob.get_PI(alpha)
    def get_loss_mean(self):
        """ :returns: the prediction interval of the mean survival time"""
        return self._sumStat_gamelossprob.get_mean()




NUM1=1000
multigame = Multigame(
    ids=range(NUM1),   # [0, 1, 2 ..., NUM_SIM_COHORTS-1]
    pop_sizes=[1000] * NUM1,  # [REAL_POP_SIZE, REAL_POP_SIZE, ..., REAL_POP_SIZE]
    prob_head=[0.5]*NUM1  # [p, p, ....]
)
# simulate all cohorts
multigame.simulate(NUM1)
print('95% CI of loss probability is',multigame.get_PI_loss_prob(0.05))
print('95% CI of expected reward is', multigame.get_PI_mean_reward(0.05))





#Problem2
print('If you repeat the simulation many times, the proportion of the CI that cover the true mean is expected to be 95%')

#Problem3
NUM=1000
multigame1 = Multigame(
    ids=range(NUM),   # [0, 1, 2 ..., NUM_SIM_COHORTS-1]
    pop_sizes=[1000] * NUM,  # [REAL_POP_SIZE, REAL_POP_SIZE, ..., REAL_POP_SIZE]
    prob_head=[0.5]*NUM  # [p, p, ....]
)
# simulate all cohorts
multigame1.simulate(NUM)
tmp = multigame1.get_PI_mean_reward(0.05)
tmp = [i * -1 for i in tmp]
tmp[0], tmp[1] = tmp[1], tmp[0]


print('The expected reward for casino owner is', -1*multigame1.get_overall_mean_reward() ,'with a projection interval of', tmp)
NUM2 = 1000
multigame2 = Multigame(
    ids=range(NUM2),   # [0, 1, 2 ..., NUM_SIM_COHORTS-1]
    pop_sizes=[10] * NUM2,  # [REAL_POP_SIZE, REAL_POP_SIZE, ..., REAL_POP_SIZE]
    prob_head=[0.5]*NUM2  # [p, p, ....]
)
# simulate all cohorts
multigame2.simulate(NUM2)


print('The expected reward of a gambler is ',multigame2.get_overall_mean_reward(),'with a projection interval of ',multigame2.get_PI_mean_reward(0.05))


print('If you repeat the simulation many times, the proportion of the CIs that cover the true average reward of the casino owner is expected to be 95%.')
print('The next realization of expected reward of the gambler will fall in the projection interval',multigame2.get_PI_mean_reward(0.05),'with probability 95%. ')
