from typing import Set, Dict
from math import exp
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment


# A simple estimation of segment popularity (used only for rough filtering)
EST_SEG_SIZE = {
    ("Female", "Old", "LowIncome"): 2400,
    ("Male", "Young", "LowIncome"): 1800,
    ("Female", "Young", "LowIncome"): 1950,
    ("Male", "Old", "LowIncome"): 1750,
    ("Male", "Old", "HighIncome"): 800,
    ("Male", "Young", "HighIncome"): 500,
    ("Female", "Old", "HighIncome"): 400,
    ("Female", "Young", "HighIncome"): 250,
}


def estimate_size(segment: MarketSegment) -> int:
    key = tuple(sorted(segment))
    return EST_SEG_SIZE.get(key, 1500)


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "A very long Agent name"

    def on_new_game(self):
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        today = self.get_current_day()

        for camp in self.get_active_campaigns():

            reach = camp.reach # total required
            done = self.get_cumulative_reach(camp)
            spent = self.get_cumulative_cost(camp)
            remaining = max(reach - done, 0)

            days_left = camp.end_day - today + 1
            if days_left <= 0 or remaining <= 0:
                continue

            # Check if the campaign is realistic to finish
            seg_size = estimate_size(camp.target_segment)
            expected_supply = seg_size * days_left

            # If impossible to finish → skip bidding
            if expected_supply < remaining:
                continue

            remaining_budget = max(camp.budget - spent, 0)
            daily_limit = remaining_budget / days_left

            # Simple safe bid per impression
            # This helps avoid overpaying
            base_bid = remaining_budget / remaining
            bid_per_item = max(0.1, base_bid)

            # Build bids only on correct segments
            segment_bids = set()
            for seg in MarketSegment.all_segments():
                if seg.issubset(camp.target_segment):
                    segment_bids.add(
                        Bid(
                            bidder=self,
                            auction_item=seg,
                            bid_per_item=bid_per_item,
                            bid_limit=daily_limit
                        )
                    )

            bundle = BidBundle(
                campaign_id=camp.uid,
                limit=daily_limit,
                bid_entries=segment_bids
            )
            bundles.add(bundle)

        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        quality = self.get_quality_score()

        for camp in campaigns_for_auction:

            R = camp.reach
            duration = camp.end_day - camp.start_day + 1
            seg_size = estimate_size(camp.target_segment)

            # Check if the campaign is reasonable to win
            expected_supply = seg_size * duration

            # If reach is too high for that segment → avoid
            if expected_supply < R:
                bids[camp] = self.clip_campaign_bid(camp, R * 0.9)
                continue

            # Base bid = conservative fraction of reach
            base_bid = 0.3 * R

            # Longer duration → safer → small decrease
            base_bid = base_bid * (1 - (duration - 1) * 0.05)

            # Adjust with quality (small adjustment)
            base_bid *= (1 + 0.1 * (quality - 1))

            bid = self.clip_campaign_bid(camp, base_bid)
            bids[camp] = bid

        return bids



if __name__ == "__main__":
    test_agents = [MyNDaysNCampaignsAgent()] + [
        Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    ]

    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
