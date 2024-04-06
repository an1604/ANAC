class Offer:
    def __init__(self, offer, ranking_rate, lucia_rate, outcome, name):
        self.name = name
        self.offer = offer
        self.ranking_rate = ranking_rate
        self.lucia_rate = lucia_rate
        self.outcome = outcome
        self.offer_string_rep = offer[0]  # Save only the first element from the offer's tuple

    def __eq__(self, other):
        return (self.offer == other.offer and
                self.ranking_rate == other.ranking_rate and
                self.lucia_rate == other.lucia_rate and
                self.outcome == other.outcome and
                self.name == other.name and
                self.offer_string_rep == other.offer_string_rep)

    def __lt__(self, other):
        # Defining the less-than comparison based on the specified sequence
        if self.ranking_rate != other.ranking_rate:
            return self.ranking_rate < other.ranking_rate
        elif self.outcome != other.outcome:
            return self.outcome < other.outcome
        else:
            return self.lucia_rate < other.lucia_rate

    def __le__(self, other):
        # Defining the less-than-or-equal comparison based on the specified sequence
        if self.ranking_rate != other.ranking_rate:
            return self.ranking_rate <= other.ranking_rate
        elif self.outcome != other.outcome:
            return self.outcome <= other.outcome
        else:
            return self.lucia_rate <= other.lucia_rate

    def __gt__(self, other):
        # Defining the greater-than comparison based on the specified sequence
        if self.ranking_rate != other.ranking_rate:
            return self.ranking_rate > other.ranking_rate
        elif self.outcome != other.outcome:
            return self.outcome > other.outcome
        else:
            return self.lucia_rate > other.lucia_rate

    def __ge__(self, other):
        # Defining the greater-than-or-equal comparison based on the specified sequence
        if self.ranking_rate != other.ranking_rate:
            return self.ranking_rate >= other.ranking_rate
        elif self.outcome != other.outcome:
            return self.outcome >= other.outcome
        else:
            return self.lucia_rate >= other.lucia_rate

    def __hash__(self):
        # Compute hash based on specific attributes
        return hash(tuple([self.ranking_rate, self.lucia_rate, self.outcome, self.name, self.offer_string_rep]))


class Historical_Offer:
    def __init__(self, offer, current_time, relative_time, current_round, self_outcome,
                 opponent_outcome, self_lucia_rate, opponent_lucia_rate,
                 self_ranking_rate, opponent_ranking_rate):
        self.offer_string_rep = offer[0]
        self.offer = offer
        self.current_time = current_time
        self.relative_time = relative_time
        self.current_round = current_round
        self.self_outcome = self_outcome
        self.opponent_outcome = opponent_outcome
        self.self_lucia_rate = self_lucia_rate
        self.opponent_lucia_rate = opponent_lucia_rate
        self.self_ranking_rate = self_ranking_rate
        self.opponent_ranking_rate = opponent_ranking_rate


class Offer_Map:
    def __init__(self):
        self.idx_to_offer_map: dict[int, Offer] = {}
        self.offer_to_idx_map: dict[Offer, int] = {}

    def add_offer(self, offer: Offer):
        if not self.exists(offer):
            index = len(self.offer_to_idx_map)
            self.idx_to_offer_map[index] = offer
            self.offer_to_idx_map[offer] = index
            return index
        return -1

    def get_offer(self, index):
        return self.idx_to_offer_map.get(index)

    def get_index(self, offer: Offer):
        return self.offer_to_idx_map.get(offer)

    def remove_offer(self, offer: Offer):
        index = self.offer_to_idx_map.pop(offer, None)
        if index is not None:
            del self.idx_to_offer_map[index]
            return True
        return False

    def exists(self, offer: Offer) -> bool:
        return offer in self.offer_to_idx_map.keys()

    @property
    def offer_count(self):
        return len(self.offer_to_idx_map)

    def get_offer_from_tuple(self, offer: tuple) -> Offer:
        for _offer, idx in self.offer_to_idx_map.items():
            if offer == _offer.offer:
                return idx, _offer
        return None

    def sort(self, reverse):
        return sorted([offer for offer in self.offer_to_idx_map.keys()], reverse=reverse)

    def get_offers_as_list(self):
        return [offer for offer in self.offer_to_idx_map.keys()]

    def get_lowest(self):
        return self.sort(reverse=False)[0]

    def get_highest(self):
        return self.sort(reverse=True)[0]
