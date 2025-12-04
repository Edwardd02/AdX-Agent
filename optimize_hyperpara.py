import sys
import io
import re
import itertools
import time
import os
import csv
import math
from datetime import datetime
from typing import Set
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent

# Import your base agent so we inherit the Campaign Bidding logic
try:
    from my_agent_explained import MyNDaysNCampaignsAgent
except ImportError:
    # Fallback if file not found, usually for running in isolation
    from my_agent import MyNDaysNCampaignsAgent


def frange(start, stop, step):
    """Helper to generate float ranges."""
    result = []
    current = start
    while current <= stop + 1e-9:
        result.append(round(current, 2))
        current += step
    return result


class TunableSigmoidAgent(MyNDaysNCampaignsAgent):
    def __init__(self, peak_multiplier, steepness, base_lift):
        super().__init__()
        self.name = "SigmoidOptBot"
        # Parameters to tune:
        self.p_peak_multiplier = peak_multiplier  # e.g., 4.0
        self.p_steepness = steepness  # e.g., 10.0
        self.p_base_lift = base_lift  # e.g., 0.2

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

            # 1. Base Bid (Avg Price)
            base_bid = remaining_budget / remaining

            # 2. Progress
            progress = done / reach

            # 3. Tunable Sigmoid Logic
            #    We use self.p_steepness to stretch/shrink the curve
            x = (progress - 0.5) * self.p_steepness

            try:
                deriv = math.exp(-x) / ((1 + math.exp(-x)) ** 2)
            except OverflowError:
                deriv = 0.0

            #    We use self.p_peak_multiplier to scale the height of the bid
            #    We use self.p_base_lift as the minimum bid floor
            urgency_multiplier = self.p_base_lift + (deriv * self.p_peak_multiplier)

            bid_per_item = max(base_bid * urgency_multiplier, 0.01)

            # Keep pacing standard for this test (0.95)
            daily_budget = max(1.0, remaining_budget * 0.95)

            bid = Bid(bidder=self, auction_item=camp.target_segment, bid_per_item=bid_per_item, bid_limit=daily_budget)
            bundle = BidBundle(campaign_id=camp.uid, limit=daily_budget, bid_entries={bid})
            bundles.add(bundle)
        return bundles


def parse_profit_from_log(output_str, agent_name):
    """Extracts the final profit from the simulator's stdout logs."""
    pattern = re.compile(rf"###\s+{re.escape(agent_name)}\s+#\s+([0-9.-]+)")
    match = pattern.search(output_str)
    if match: return float(match.group(1))
    return -99999.0


def run_optimization():
    # 1. Peak Multiplier: How aggressive should we be in the middle?
    #    The derivative max is 0.25. So 4.0 means ~1.0x multiplier.
    #    We test range around 3.0 to 6.0.
    peak_mults = frange(3.0, 6.0, 1.0)

    # 2. Steepness: How "wide" is the peak window?
    #    10 is standard. Lower (5) is wider window. Higher (15) is narrower.
    steepnesses = frange(8.0, 12.0, 2.0)

    # 3. Base Lift: Minimum bid multiplier at the edges (start/end)
    #    0.1 means we bid 10% of avg value. 0.3 means 30%.
    base_lifts = frange(0.1, 0.3, 0.1)

    SIMS_PER_RUN = 15

    log_folder = "experiment_logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = os.path.join(log_folder, f"sigmoid_results_{timestamp}.csv")

    print(f"Logging data to: {csv_filename}")
    print("-" * 50)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run_ID", "Peak_Mult", "Steepness", "Base_Lift", "Avg_Profit", "Num_Sims"])

    combinations = list(itertools.product(peak_mults, steepnesses, base_lifts))
    total_combs = len(combinations)
    best_score = -float('inf')
    best_params = None

    start_time = time.time()

    for i, (pm, steep, base) in enumerate(combinations):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i if i > 0 else 1)
        remaining = (total_combs - i) * avg_time / 60

        print(f"[{i + 1}/{total_combs}] P={pm}, S={steep}, B={base} | Rem: {remaining:.1f}m ... ", end="", flush=True)

        my_agent = TunableSigmoidAgent(pm, steep, base)
        # 9 Opponents (Standard Tier 1)
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
            continue

        sys.stdout = original_stdout
        full_log = capture_string.getvalue()
        avg_profit = parse_profit_from_log(full_log, my_agent.name)

        print(f"${avg_profit:.2f}")

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, pm, steep, base, avg_profit, SIMS_PER_RUN])

        if avg_profit > best_score:
            best_score = avg_profit
            best_params = (pm, steep, base)

    print("\n" + "=" * 50)
    print(f"DONE. Results saved to {csv_filename}")
    print(f"Best Profit: ${best_score:.2f}")
    print(f"Best Params: Peak={best_params[0]}, Steep={best_params[1]}, Base={best_params[2]}")
    print("=" * 50)


if __name__ == "__main__":
    run_optimization()