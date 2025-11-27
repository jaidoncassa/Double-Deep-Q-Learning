import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        self.save_step_log = save_dir / "episode_metrics.csv"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        with open(self.save_step_log, "w") as f:
            f.write(
                f"{'Reward':>15}{'Loss':>15}{'Length':>15}\n"
            )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards_step_plot = save_dir / "reward_step_plot.jpg"
        self.ep_losses_step_plot = save_dir / "loss_step_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Per epsiode step metrics
        self.ep_rewards_step = []
        self.ep_losses_step = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q if q is not None else 0.0
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.ep_rewards_step.append(self.curr_ep_reward)
        self.ep_losses_step.append(ep_avg_loss)

        # Log to step file (Correctly writing raw data)
        with open(self.save_step_log, "a") as f:
            f.write(
                f"{self.curr_ep_reward:15.3f}{ep_avg_loss:15.3f}{self.curr_ep_length:15d}\n"
            )

        # # Plotting the raw episode-by-episode data (FIXED)
        # plot_metrics = {
        #     "ep_rewards_step": self.ep_rewards_step_plot,
        #     "ep_losses_step": self.ep_losses_step_plot,
        # }
        
        # for metric_name, file_path in plot_metrics.items():
        #     plt.clf()
        #     plt.plot(
        #         # Correct access: use metric_name (e.g., "ep_rewards_step")
        #         getattr(self, metric_name), 
        #         label=metric_name.replace("ep_", "raw_")
        #     )
        #     plt.legend()
        #     # Correct saving: use the pre-defined file path
        #     plt.savefig(file_path)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        # for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
        #     plt.clf()
        #     plt.plot(
        #         getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}"
        #     )
        #     plt.legend()
        #     plt.savefig(getattr(self, f"{metric}_plot"))

    def save_plots(self):
        plot_metrics = {
            "ep_rewards": self.ep_rewards_plot,
            "ep_lengths": self.ep_lengths_plot,
            "ep_avg_losses": self.ep_avg_losses_plot,
            "ep_avg_qs": self.ep_avg_qs_plot,
        }

        for metric_name, file_path in plot_metrics.items():
            plt.clf()
            plt.plot(
                # Correct access: use metric_name (e.g., "ep_rewards")
                getattr(self, f"moving_avg_{metric_name}"), 
                label=f"moving_avg_{metric_name}"
            )
            plt.legend()
            # Correct saving: use the pre-defined file path
            plt.savefig(file_path)

        plot_metrics = {
            "ep_rewards_step": self.ep_rewards_step_plot,
            "ep_losses_step": self.ep_losses_step_plot,
        }
        
        for metric_name, file_path in plot_metrics.items():
            plt.clf()
            plt.plot(
                # Correct access: use metric_name (e.g., "ep_rewards_step")
                getattr(self, metric_name), 
                label=metric_name.replace("ep_", "raw_")
            )
            plt.legend()
            # Correct saving: use the pre-defined file path
            plt.savefig(file_path)