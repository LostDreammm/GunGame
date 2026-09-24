"""Bounded inventory planning from validated mine-closure announcements.

A closure is evidence of future scarcity, not an exact price forecast. The
collection weight is a configurable preference only; every sale still uses
observed vendor quotes. The planner never invents a future shop price.
"""
from collections import Counter
import math
import re


ORES = ('stone', 'iron', 'copper')


_ORE_WORDS = {'stone': ('石矿', '石头', 'stone'),
              'iron': ('铁矿', 'iron'), 'copper': ('铜矿', 'copper')}
_NUMBER = r'(?:[0-9]{1,2}|[一二两三四五六七八九十]{1,3})'
_CLOSED = r'(?:停工|停采|禁止采集|无法采集|关闭|封闭)'
_DAY_RANGE = re.compile(r'第\s*(' + _NUMBER + r')\s*天?\s*(?:至|到|—|–|-|~|～)\s*第?\s*('
                        + _NUMBER + r')\s*天')
_DURATION = re.compile(r'(?:需要|需|持续|为期|历时|停工|停采|关闭|封闭)\s*('
                       + _NUMBER + r')\s*天')


def _day_number(raw):
    if raw.isascii() and raw.isdigit():
        return int(raw)
    digits = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    if raw in digits:
        return digits[raw]
    if raw in ('十', '一十'):
        return 10
    return None


def _dates_in_quote(quote, published_day):
    dates, offsets = set(), set()
    # Timing must occur in the closure's own clause, so an unrelated period of
    # normal operation cannot be attached to a later closure mention.
    for clause in re.split(r'[，,。！？!？;；\n]', quote):
        if not re.search(_CLOSED, clause):
            continue
        for first, last in _DAY_RANGE.findall(clause):
            dates.add((_day_number(first), _day_number(last)))
        if re.search(r'明日|明天|次日', clause):
            offsets.add(1)
        if '后天' in clause:
            offsets.add(2)
    if offsets:
        durations = {_day_number(raw) for raw in _DURATION.findall(quote)}
        if len(durations) != 1 or None in durations or not 1 <= next(iter(durations)) <= 10:
            return set()
        duration = next(iter(durations))
        dates.update((published_day + offset, published_day + offset + duration - 1)
                     for offset in offsets)
    return dates


def grounded_closure(closure, news_history):
    """Validate proposed closed days against one exact official evidence quote.

    Supported dates are an explicit day range or tomorrow/day-after-tomorrow
    plus a stated duration, anchored to the original publication day. Unknown,
    conflicting, negated and out-of-match dates remain unverified. An LLM's date
    fields are never used to fill gaps in the announcement.
    """
    if not isinstance(closure, dict):
        return False
    ore, quote = closure.get('ore'), closure.get('evidence')
    first, last = closure.get('start_day'), closure.get('end_day')
    if (ore not in _ORE_WORDS or not isinstance(quote, str) or not quote
            or type(first) is not int or type(last) is not int or not 1 <= first <= last <= 10):
        return False
    mentioned = {kind for kind, aliases in _ORE_WORDS.items() if any(alias in quote.lower() for alias in aliases)}
    if mentioned != {ore} or not re.search(_CLOSED, quote):
        return False
    evidence_dates = set()
    for entry in news_history or ():
        if not isinstance(entry, dict):
            continue
        official, day = entry.get('officialNews'), entry.get('day')
        if not isinstance(official, str) or type(day) is not int or not 1 <= day <= 10:
            continue
        start = official.find(quote)
        while start >= 0:
            end = start + len(quote)
            # Include the original enclosing sentences: a quote cannot trim
            # away "officials deny" or a preceding negation to become evidence.
            left = max((m.end() for m in re.finditer(r'[。！？!？;；\n]', official[:start])), default=0)
            following = re.search(r'[。！？!？;；\n]', official[end:])
            right = end if re.search(r'[。！？!？;；\n]$', quote) else (
                end + following.start() if following else len(official))
            context = official[left:right]
            if (re.search(r'(?:不|没有|没|未|无需|无须|并非)[^。！？;；\n]{0,12}' + _CLOSED, context)
                    or re.search(r'否认|取消|辟谣|不实|谣言|如果|假如|可能|或许|尚未确定|至少|最多', context)):
                return False
            dates = _dates_in_quote(quote, day)
            if not dates or any(a is None or b is None or not 1 <= a <= b <= 10 for a, b in dates):
                return False
            evidence_dates.update(dates)
            start = official.find(quote, start + len(quote))
    return evidence_dates == {(first, last)}


class MarketPlanner:
    def __init__(self, config=None):
        self.config = dict((config or {}).get('market', {}))
        self.day = 0
        self.prices = {}
        self.events = {}
        self.blocked_ores = set()

    def _number(self, name, default, low, high):
        try:
            value = float(self.config.get(name, default))
        except (ValueError, TypeError, OverflowError):
            value = default
        return min(high, max(low, value)) if math.isfinite(value) else default

    @property
    def enabled(self):
        return self.config.get('enabled', True) is not False

    def observe(self, day, prices, reasoning):
        """Consume already grounded closure dates; pin the first actual quote."""
        self.day = day
        self.prices = dict(prices)
        self.events = {key: event for key, event in self.events.items()
                       if event['end_day'] >= day}
        for closure in getattr(reasoning, '_closures', ()):
            if not isinstance(closure, dict):
                continue
            ore = closure.get('ore')
            first, last = closure.get('start_day'), closure.get('end_day')
            if (ore not in ORES or type(first) is not int or type(last) is not int
                    or not 1 <= first <= last <= 10 or last < day):
                continue
            key = (ore, first, last)
            if key not in self.events:
                self.events[key] = dict(closure, observed_day=day, baseline_price=None)
            event = self.events[key]
            # Missing quotes cannot establish either a hold or a profit.
            if event['baseline_price'] is None and prices.get(ore, 0) > 0:
                event['baseline_price'] = prices[ore]
        self.blocked_ores = {e['ore'] for e in self.events.values()
                             if e['start_day'] <= day <= e['end_day']}

    def _relevant(self, ore):
        if not self.enabled:
            return []
        horizon = int(self._number('lookahead_days', 2, 0, 9))
        return [e for e in self.events.values() if e['ore'] == ore
                and self.day <= e['end_day'] and e['start_day'] <= self.day + horizon
                and e['baseline_price'] is not None]

    def take_profit(self, ore):
        """A higher *observed* quote releases this ore, including small batches."""
        return any(self.prices.get(ore, 0) > e['baseline_price'] for e in self._relevant(ore))

    def _waiting(self, ore):
        # Overlapping events must not re-lock ore already profitable against an
        # earlier recorded quote.
        return self._relevant(ore) if not self.take_profit(ore) else []

    def _hold_capacity(self, role):
        capacity = max(0, int(role.get('backPackCapability', 100)))
        fraction = self._number('max_hold_fraction', 0.5, 0, 0.9)
        return min(int(self._number('stockpile_target', 16, 0, 100)), int(capacity * fraction))

    def reserves(self, role, emergency=False, force=False):
        bag = role.get('backpack', [])
        capacity = max(0, int(role.get('backPackCapability', 100)))
        if emergency or force or len(bag) >= capacity:
            return {}
        counts, held = Counter(bag), {}
        room = self._hold_capacity(role)
        # A shared per-person cap prevents several announcements filling the bag.
        candidates = sorted((min(e['start_day'] for e in self._waiting(ore)), ore)
                            for ore in ORES if counts[ore] and self._waiting(ore))
        for _, ore in candidates:
            held[ore] = min(counts[ore], room)
            room -= held[ore]
        return {ore: count for ore, count in held.items() if count}

    def collection_weight(self, ore, role, emergency=False):
        if emergency:
            return 1.0
        counts = Counter(role.get('backpack', []))
        held = sum(counts[kind] for kind in ORES if self._waiting(kind))
        if held >= self._hold_capacity(role):
            return 1.0
        future = [e for e in self._waiting(ore) if e['start_day'] > self.day]
        if not future:
            return 1.0
        scarcity = max((e['end_day'] - e['start_day'] + 1) / (e['start_day'] - self.day)
                       for e in future)
        weight = self._number('scarcity_weight', 0.75, 0, 5)
        maximum = self._number('max_collection_weight', 3.0, 1, 10)
        return min(maximum, 1.0 + weight * scarcity)

    def has_plan(self, ore):
        return bool(self._relevant(ore))
