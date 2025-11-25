import sys
import io
import re
import itertools
import time
import os
import csv
from datetime import datetime
from typing import Set
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent

try:
    from my_agent import MyNDaysNCampaignsAgent
except ImportError:
    print("!!! ERROR: Could not find 'my_agent.py'.")
    sys.exit(1)


def frange(start, stop, step):
    result = []
    current = start
    while current <= stop + 1e-9:
        result.append(round(current, 2))
        current += step
    return result


class TunableAgent(MyNDaysNCampaignsAgent):
    def __init__(self, daily_budget_rate, urgency_base, urgency_slope):
        super().__init__()
        self.name = "OptimizationBot"
        self.p_daily_budget_rate = daily_budget_rate
        self.p_urgency_base = urgency_base
        self.p_urgency_slope = urgency_slope

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        for camp in self.get_active_campaigns():
            reach = camp.reach
            done = self.get_cumulative_reach(camp)
            spent = self.get_cumulative_cost(camp)
            remaining = max(reach - done, 0)
            remaining_budget = max((camp.budget or 0) - spent, 0)

            if remaining <= 0 or remaining_budget <= 0:
                continue

            daily_budget = max(1.0, remaining_budget * self.p_daily_budget_rate)
            urgency = 1 - (remaining / reach)
            bid_per_item = self.p_urgency_base + (self.p_urgency_slope * urgency)

            bid = Bid(bidder=self, auction_item=camp.target_segment, bid_per_item=bid_per_item, bid_limit=daily_budget)
            bundle = BidBundle(campaign_id=camp.uid, limit=daily_budget, bid_entries={bid})
            bundles.add(bundle)
        return bundles


def parse_profit_from_log(output_str, agent_name):
    pattern = re.compile(rf"###\s+{re.escape(agent_name)}\s+#\s+([0-9.-]+)")
    match = pattern.search(output_str)
    if match: return float(match.group(1))
    return -99999.0


def run_optimization():

    # budget_rates = frange(0.1, 0.9, 0.2)
    # urgency_bases = frange(0.2, 1.0, 0.2)
    # urgency_slopes = frange(0.2, 2.0, 0.3)
    budget_rates = frange(0.8, 1, 0.05)
    urgency_bases = frange(0.1, 0.3, 0.05)
    urgency_slopes = frange(1.8, 2.5, 0.1)

    SIMS_PER_RUN = 50

    log_folder = "experiment_logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"Created folder: {log_folder}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = os.path.join(log_folder, f"results_{timestamp}.csv")

    print(f"Logging data to: {csv_filename}")
    print("-" * 50)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run_ID", "Budget_Rate", "Base_Bid", "Slope", "Avg_Profit", "Num_Sims"])

    combinations = list(itertools.product(budget_rates, urgency_bases, urgency_slopes))
    total_combs = len(combinations)
    best_score = -float('inf')

    start_time = time.time()

    for i, (rate, base, slope) in enumerate(combinations):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i if i > 0 else 1)
        remaining = (total_combs - i) * avg_time / 60

        print(f"[{i + 1}/{total_combs}] R={rate}, B={base}, S={slope} | Rem: {remaining:.1f}m ... ", end="", flush=True)

        my_agent = TunableAgent(rate, base, slope)
        opponents = [Tier1NDaysNCampaignsAgent(name=f"Opponent {j + 1}") for j in range(9)]
        agents = [my_agent] + opponents
        simulator = AdXGameSimulator()

        capture_string = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = capture_string

        try:
            simulator.run_simulation(agents=agents, num_simulations=SIMS_PER_RUN)
        except Exception:
            sys.stdout = original_stdout
            print("CRASH")
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + 1, rate, base, slope, "CRASH", SIMS_PER_RUN])
            continue

        sys.stdout = original_stdout
        full_log = capture_string.getvalue()
        avg_profit = parse_profit_from_log(full_log, my_agent.name)

        print(f"${avg_profit:.2f}")

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, rate, base, slope, avg_profit, SIMS_PER_RUN])

        if avg_profit > best_score:
            best_score = avg_profit

    print("\n" + "=" * 50)
    print(f"DONE. Results saved to {csv_filename}")
    print(f"Best Profit: ${best_score:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    run_optimization()