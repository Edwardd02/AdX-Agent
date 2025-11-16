from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from typing import Set, Dict
from math import ceil
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent


# counts for each detailed group
ATOMIC_SEGMENT_COUNTS = {
    ("Male", "Young", "LowIncome"): 1836,
    ("Male", "Young", "HighIncome"): 517,
    ("Male", "Old", "LowIncome"): 1795,
    ("Male", "Old", "HighIncome"): 808,
    ("Female", "Young", "LowIncome"): 1980,
    ("Female", "Young", "HighIncome"): 256,
    ("Female", "Old", "LowIncome"): 2401,
    ("Female", "Old", "HighIncome"): 407,
}


def expected_daily_users(target_segment: MarketSegment) -> int:
    # sum any atomic group that fits the segment (e.g. "Male","Young")
    tset = set(target_segment)
    total = 0
    for atomic, cnt in ATOMIC_SEGMENT_COUNTS.items():
        if tset.issubset(set(atomic)):
            total += cnt
    return total


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    def __init__(self):
        super().__init__()
        self.name = "HTandRX's Agent"

    def on_new_game(self):
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        today = self.get_current_day()

        for camp in self.get_active_campaigns():
            R = camp.reach
            done = self.get_cumulative_reach(camp)
            spent = self.get_cumulative_cost(camp)

            remaining = max(R - done, 0)
            remaining_budget = max((camp.budget or 0) - spent, 0)

            # skip finished or broke campaigns
            if remaining <= 0 or remaining_budget <= 0:
                continue

            days_left = max(1, camp.end_day - today + 1)

            # spend a steady portion each day
            daily_budget = max(1.0, remaining_budget * 0.35)

            # urgency goes up when we are behind
            urgency = 1 - (remaining / R)
            bid_per_item = 0.3 + 0.7 * urgency

            # only bid on the exact segment
            bid = Bid(
                bidder=self,
                auction_item=camp.target_segment,
                bid_per_item=bid_per_item,
                bid_limit=daily_budget
            )

            bundle = BidBundle(
                campaign_id=camp.uid,
                limit=daily_budget,
                bid_entries={bid}
            )

            bundles.add(bundle)

        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        quality = max(self.get_quality_score(), 0.05)

        for camp in campaigns_for_auction:
            R = camp.reach
            duration = max(1, camp.end_day - camp.start_day + 1)

            # estimate how many users we can expect
            daily_supply = expected_daily_users(camp.target_segment)
            expected_total = daily_supply * duration

            # avoid campaigns that look too large
            if R > expected_total * 1.5:
                bids[camp] = self.clip_campaign_bid(camp, max(0.1, 0.02 * R))
                continue

            # how tight the campaign is
            difficulty = min(1.0, R / max(1.0, expected_total))

            # prefer small campaigns
            size_boost = 1.2 if R <= 300 else 1.0

            # use the budget if available
            value_anchor = camp.budget if (camp.budget and camp.budget > 0) else R

            # simple rule: bid a portion of the value,
            # increase it a bit when the campaign is tight
            raw_bid = value_anchor * (1.0 - 0.4 * (1 - difficulty)) * size_boost

            # adjust using quality so that effective bid stays reasonable
            adjusted = raw_bid / quality

            # keep it safe
            bids[camp] = self.clip_campaign_bid(camp, adjusted)

        return bids


if __name__ == "__main__":
    test_agents = [MyNDaysNCampaignsAgent()] + [
        Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    ]

    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=50)
