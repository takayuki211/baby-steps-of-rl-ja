import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import gym
from fn_framework import FNAgent, Trainer, Observer


class ValueFunctionAgent(FNAgent):

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        # modelの定義
        scaler = StandardScaler() # 標準化
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1) # 価値関数 # ノード数10 * 2層
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)]) # 標準化して価値関数

        # modelのうち、scalerはexperienceに含まれる状態sで初期化する
        states = np.vstack([e.s for e in experiences])
        self.model.named_steps["scaler"].fit(states)

        # Avoid the predict before fit.
        # とりあえず1回学習させておく（エラー回避のため）
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def estimate(self, s):
        estimated = self.model.predict(s)[0]
        return estimated

    def _predict(self, states):
        # 初期化後はモデルで予測する
        if self.initialized:
            predicteds = self.model.predict(states)
        # 初回はランダム値を返す
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds


# 勉強会で説明
    # Q-learningと同様の処理
    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])
        estimateds = self._predict(states) # 現在の状態の価値を計算
        future = self._predict(n_states) # 遷移先の価値を計算

        # 現在の状態の価値を報酬+遷移先の価値で更新
        for i, e in enumerate(experiences): # １回で32データが更新される
            reward = e.r # その状態で得られるべき報酬
            if not e.d:
                reward += gamma * np.max(future[i]) # 価値が最大になるような行動を取ると仮定
            estimateds[i][e.a] = reward

        # モデルに入力できるようにデータを整形
        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)

        # モデルを更新
        self.model.named_steps["estimator"].partial_fit(states, estimateds)


class CartPoleObserver(Observer):
    # 値を整形しているだけ
    def transform(self, state):
        return np.array(state).reshape((1, -1))


class ValueFunctionTrainer(Trainer):

    def train(self, env, episode_count=220, epsilon=0.1, initial_count=-1,
              render=False):
        actions = list(range(env.action_space.n)) # アクション([0,1])
        agent = ValueFunctionAgent(epsilon, actions)

        # スーパークラスの学習ループ関数で学習
        self.train_loop(env, agent, episode_count, initial_count, render)

        return agent

    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)

    def step(self, episode, step_count, agent, experience):
        if self.training:
            # バッチサイズでサンプリング
            batch = random.sample(self.experiences, self.batch_size)
            # agentをupdate(model更新)
            agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play, episode_count):
# def main(play):

    # CartPoleの環境を構築する（env=Observer）
    env = CartPoleObserver(gym.make("CartPole-v0"))

    # Trainer生成
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_agent.pkl") # ログ出力先指定

    if play: # 実行モード
        # 学習済みのagentを読み込む
        agent = ValueFunctionAgent.load(env, path)

        # 実行
        agent.play(env)

    else: # 学習モード
        # 学習(trainedにはagentが入る)
        trained = trainer.train(env,episode_count)
        # trained = trainer.train(env)

        # ログ出力
        trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)

        # 学習済みのagentを保存する
        trained.save(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--episode", type=int, default=220,
                        help="episode number") ## 追加
    args = parser.parse_args()

    main(args.play, args.episode)
