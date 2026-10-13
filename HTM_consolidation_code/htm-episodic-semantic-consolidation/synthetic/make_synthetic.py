"""
Generator for the episodic-to-semantic consolidation benchmark (version 2).

Each sample is one entity and one target fact, supported by 3-4 episodic sentences. Seven fact types follow
the manuscript: category generalization (food, exercise), reading preference, transport mode, temporal habit,
location preference, causal-conditional, and a no-generalization negative control. About 25% of positive samples
contain one exception episode (an item from a different category), with a reduced confidence label (0.75).

Leakage control: entities are assigned to 5 folds (entity-disjoint), and every episode is rendered with three
disjoint template sets - 'train', 'val' and 'test' - so that sentence templates never overlap between the data used
for fitting, for threshold selection and for testing.

Usage: python make_synthetic.py  -> episodic_semantic_dataset_v2.json  (deterministic, seed 2026)
"""
import json, random

SEED = 2026
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ENTITIES = ['Aarav', 'Aditi', 'Akash', 'Ananya', 'Anil', 'Arjun', 'Bhavna', 'Chetan', 'Deepa', 'Dev', 'Divya',
            'Farhan', 'Gauri', 'Harish', 'Ishaan', 'Isha', 'Jaya', 'Kabir', 'Kavya', 'Kiran', 'Lakshmi', 'Manoj',
            'Meera', 'Mohan', 'Nandini', 'Naveen', 'Neha', 'Nikhil', 'Pooja', 'Pranav', 'Priya', 'Rahul', 'Rajesh',
            'Ram', 'Ravi', 'Ritu', 'Rohan', 'Sahil', 'Sanjay', 'Sara', 'Shreya', 'Siddharth', 'Sneha', 'Sunil',
            'Swati', 'Tanvi', 'Tarun', 'Uma', 'Varun', 'Vidya', 'Vijay', 'Vikram', 'Vinita', 'Yash', 'Zoya',
            'Abhay', 'Charu', 'Geeta', 'Hemant', 'Indira', 'Jatin', 'Kunal', 'Leela', 'Madhav', 'Nisha', 'Om',
            'Parul', 'Rekha', 'Suresh', 'Tejas']
assert len(ENTITIES) == len(set(ENTITIES)) == 70

# template sets: 6 templates per type; indices 0-1 -> train, 2-3 -> val, 4-5 -> test
SPLITS = {'train': (0, 1), 'val': (2, 3), 'test': (4, 5)}

FOOD = {'sweet food': ['kheer', 'jalebi', 'gulab jamun', 'rasgulla', 'ladoo', 'halwa', 'barfi'],
        'spicy food': ['biryani', 'pani puri', 'chole bhature', 'vada pav', 'samosa', 'chilli paneer'],
        'healthy food': ['a salad', 'sprouts', 'a fruit bowl', 'oats', 'boiled eggs', 'vegetable soup'],
        'fast food': ['pizza', 'a burger', 'french fries', 'noodles', 'a hot dog', 'a sandwich']}
FOOD_WHEN = ['in the evening', 'at breakfast', 'after lunch', 'late at night']
EXER = {'cardio': ['went running', 'went cycling', 'went swimming', 'went jogging', 'did skipping'],
        'strength': ['lifted weights', 'did push-ups', 'did squats', 'did pull-ups', 'did bench presses'],
        'flexibility': ['did yoga', 'did stretching exercises', 'did pilates', 'did a mobility routine']}
EXER_WHEN = ['in the morning', 'after work', 'in the evening']
BOOKS = {'mystery': ['a detective novel', 'a crime thriller', 'a murder mystery', 'a spy story'],
         'science fiction': ['a space adventure novel', 'a story about robots', 'a time-travel novel', 'an alien invasion story'],
         'history': ['a book on ancient empires', 'a biography of a freedom fighter', 'a history of the world wars', 'a book about medieval kings'],
         'poetry': ['a collection of poems', 'a book of verses', 'a volume of sonnets', 'a poetry anthology']}
MODES = ['bus', 'train', 'metro', 'bicycle', 'auto rickshaw', 'car']
DESTS = ['office', 'college', 'market', 'gym', 'hospital', 'bank', 'temple', 'mall']
TIMES = {'early in the morning': ['5 am', '5:30 am', '6 am', '6:15 am'],
         'late at night': ['11 pm', '11:30 pm', 'midnight', '12:30 am'],
         'in the afternoon': ['2 pm', '2:30 pm', '3 pm', '4 pm']}
HABITS = [('went for a walk', 'goes for a walk'), ('studied', 'studies'), ('practised the guitar', 'practises the guitar'), ('meditated', 'meditates')]
PLACES = {'in a library': ['the city library', 'the college library', 'the public library', 'the district library'],
          'in a cafe': ['a coffee shop near the station', 'the corner cafe', 'a cafe on the main road', 'a small tea house'],
          'in a park': ['the riverside park', 'the neighbourhood park', 'the botanical garden', 'the lakeside garden'],
          'at home': ['the bedroom at home', 'the living room at home', 'the study room at home', 'the balcony at home']}
PLACE_ACTS = [('studied', 'study'), ('worked', 'work'), ('read', 'read')]
CAUSES = {('it rains', 'stays indoors'): [('it rained', 'stayed indoors'), ('it was raining heavily', 'remained inside the house'), ('there was a downpour', 'did not go out')],
          ('they have an exam the next day', 'stays up late'): [('they had an exam the next day', 'stayed up late'), ('an exam was due the following morning', 'studied past midnight'), ('they had a test the next day', 'slept very late')],
          ('they feel tired', 'drinks coffee'): [('they felt tired', 'drank coffee'), ('they were exhausted', 'had a strong cup of coffee'), ('they felt sleepy', 'made some coffee')],
          ('it is a holiday', 'visits their grandparents'): [('it was a holiday', 'visited their grandparents'), ('the office was closed for a festival', 'went to see their grandparents'), ('there was a public holiday', 'spent the day with their grandparents')]}

T = {
 'food': ['{E} ate {x} {w}.', '{W}, {E} had {x}.', '{E} enjoyed some {x} {w}.', '{E} was eating {x} {w} on {d}.',
          'On {d}, {E} ordered {x} {w}.', '{E} picked {x} {w} again on {d}.'],
 'exercise': ['{E} {x} {w}.', '{W}, {E} {x}.', 'On {d}, {E} {x} {w}.', '{E} said they {x} {w} on {d}.',
              'Yesterday {E} {x} {w}.', "{E}'s friend saw that {E} {x} {w}."],
 'reading': ['{E} read {x} on {d}.', 'On {d}, {E} was reading {x}.', '{E} borrowed {x} from a friend.',
             '{E} finished {x} last week.', '{E} bought {x} at the bookstore on {d}.', '{E} spent the afternoon reading {x}.'],
 'transport': ['{E} took the {x} to the {p} on {d}.', 'On {d}, {E} travelled to the {p} by {x}.', '{E} went to the {p} by {x}.',
               '{E} used the {x} to get to the {p} on {d}.', 'To reach the {p}, {E} chose the {x}.', '{E} rode the {x} to the {p} this {d}.'],
 'habit': ['{E} {v} at {x} on {d}.', 'On {d}, {E} {v} at {x}.', 'At {x}, {E} {v}.', '{E} {v} at around {x}.',
           'It was {x} when {E} {v}.', "{E}'s roommate noticed that {E} {v} at {x}."],
 'location': ['{E} {v} at {x} on {d}.', 'On {d}, {E} {v} at {x}.', '{E} spent the day at {x} and {v} there.',
              '{E} {v} for hours at {x}.', 'On {d} afternoon {E} went to {x} and {v}.', '{E} {v} quietly at {x} on {d}.'],
 'causal': ['When {c} on {d}, {E} {r}.', '{E} {r} because {c}.', 'Since {c}, {E} {r}.', 'On {d} {c}, so {E} {r}.',
            '{E} {r} on {d}, as {c}.', 'As {c}, {E} {r} that {d}.'],
}
FACT = {'food': '{E} eats {cat} {w}.', 'exercise': '{E} does {cat} exercise {w}.', 'reading': '{E} likes reading {cat} books.',
        'transport': '{E} usually travels by {cat}.', 'habit': '{E} {vp} {cat}.', 'location': '{E} prefers to {va} {cat}.',
        'causal': 'When {c}, {E} {r}.'}
TYPE_OF = {'food': 'category generalization', 'exercise': 'category generalization', 'reading': 'reading preference',
           'transport': 'transport mode', 'habit': 'temporal habit', 'location': 'location preference', 'causal': 'causal-conditional'}


def cap(s):
    return s[0].upper() + s[1:]


def sample_spec(rng, kind):
    """Return (category, fact-fields, function producing episode fields for an in- or out-of-category item)."""
    if kind == 'food':
        cat = rng.choice(list(FOOD)); w = rng.choice(FOOD_WHEN)
        def ep(exc): c = rng.choice([k for k in FOOD if k != cat]) if exc else cat; return dict(x=rng.choice(FOOD[c]), w=w, W=cap(w))
        return cat, dict(cat=cat, w=w), ep
    if kind == 'exercise':
        cat = rng.choice(list(EXER)); w = rng.choice(EXER_WHEN)
        def ep(exc): c = rng.choice([k for k in EXER if k != cat]) if exc else cat; return dict(x=rng.choice(EXER[c]), w=w, W=cap(w))
        return cat, dict(cat=cat, w=w), ep
    if kind == 'reading':
        cat = rng.choice(list(BOOKS))
        def ep(exc): c = rng.choice([k for k in BOOKS if k != cat]) if exc else cat; return dict(x=rng.choice(BOOKS[c]))
        return cat, dict(cat=cat), ep
    if kind == 'transport':
        cat = rng.choice(MODES)
        def ep(exc): return dict(x=rng.choice([m for m in MODES if m != cat]) if exc else cat, p=rng.choice(DESTS))
        return cat, dict(cat=cat), ep
    if kind == 'habit':
        cat = rng.choice(list(TIMES)); past, pres = rng.choice(HABITS)
        def ep(exc): c = rng.choice([k for k in TIMES if k != cat]) if exc else cat; return dict(x=rng.choice(TIMES[c]), v=past)
        return cat, dict(cat=cat, vp=pres), ep
    if kind == 'location':
        cat = rng.choice(list(PLACES)); past, base = rng.choice(PLACE_ACTS)
        def ep(exc): c = rng.choice([k for k in PLACES if k != cat]) if exc else cat; return dict(x=rng.choice(PLACES[c]), v=past)
        return cat, dict(cat=cat, va=base), ep
    if kind == 'causal':
        key = rng.choice(list(CAUSES))
        def ep(exc):
            k = rng.choice([q for q in CAUSES if q != key]) if exc else key
            c, r = rng.choice(CAUSES[k]); return dict(c=c, r=r)
        return f'{key[0]} -> {key[1]}', dict(c=key[0], r=key[1]), ep


def render(kind, fields, E, d, tidx):
    f = dict(fields, E=E, d=d)
    return T[kind][tidx].format(**f)


def main():
    rng = random.Random(SEED)
    kinds_pos = ['food', 'exercise', 'reading', 'transport', 'habit', 'location', 'causal']
    entities = ENTITIES[:]; rng.shuffle(entities)
    folds = {e: k % 5 for k, e in enumerate(entities)}
    # each entity gets 4 distinct sample types; 'negative' appears for 40 of the 70 entities' slots overall
    types7 = ['category', 'reading', 'transport', 'habit', 'location', 'causal', 'negative']
    samples, sid = [], 0
    slots = []
    for k, e in enumerate(entities):
        start = k % 7
        slots += [(e, types7[(start + j) % 7]) for j in range(4)]
    for e, t in slots:
        n_ep = rng.choice([3, 4])
        if t == 'negative':
            eps = []
            own = {k for e2, t2 in slots if e2 == e for k in (['food', 'exercise'] if t2 == 'category' else [t2])}
            free = [k for k in kinds_pos if k not in own]   # never echo one of the entity's own facts
            n_ep = min(n_ep, len(free))
            kinds = rng.sample(free, n_ep)                  # every episode from a different pattern -> no shared fact
            for kind in kinds:
                _, _, ep = sample_spec(rng, kind)
                fields, d = ep(False), rng.choice(DAYS)
                eps.append(dict(kind=kind, is_exception=False, text={s: render(kind, fields, e, d, rng.choice(ix)) for s, ix in SPLITS.items()}))
            samples.append(dict(sample_id=f's{sid:03d}', entity=e, fold=folds[e], fact_type='no_generalization (negative control)',
                                category=None, target_fact=None, confidence=0.0, has_exception=False, episodes=eps))
        else:
            kind = rng.choice(['food', 'exercise']) if t == 'category' else t
            cat, ff, ep = sample_spec(rng, kind)
            exc = rng.random() < 0.25
            exc_pos = rng.randrange(n_ep) if exc else -1
            eps = []
            for j in range(n_ep):
                fields, d = ep(j == exc_pos), rng.choice(DAYS)
                eps.append(dict(kind=kind, is_exception=(j == exc_pos), text={s: render(kind, fields, e, d, rng.choice(ix)) for s, ix in SPLITS.items()}))
            fact = FACT[kind].format(E=e, **ff)
            samples.append(dict(sample_id=f's{sid:03d}', entity=e, fold=folds[e], fact_type=TYPE_OF[kind], category=cat,
                                target_fact=fact, confidence=0.75 if exc else 1.0, has_exception=exc, episodes=eps))
        sid += 1
    out = dict(description=__doc__.strip(), seed=SEED, n_folds=5, template_splits={k: list(v) for k, v in SPLITS.items()},
               templates=T, fact_templates=FACT, samples=samples)
    json.dump(out, open('episodic_semantic_dataset_v2.json', 'w'), indent=1, ensure_ascii=False)
    from collections import Counter
    print('samples', len(samples), 'episodes', sum(len(s['episodes']) for s in samples))
    print(Counter(s['fact_type'] for s in samples))
    print('exceptions', sum(s['has_exception'] for s in samples), 'facts unique', len({s['target_fact'] for s in samples if s['target_fact']}))


if __name__ == '__main__':
    main()
