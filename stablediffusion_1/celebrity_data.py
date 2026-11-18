"""
Celebrity prompt pairs for multi-concept erasure on SD-v1.4 (multi_celebrity_erasure.py).

The erased sets follow the MACE celebrity benchmark (1 / 5 / 10 celebrities). Every celebrity has 10 prompts written
with a "{c}" placeholder, with the name at the start, in the middle or at the end of the sentence. Each harmful prompt
is paired with 5 safe prompts: the same sentence with another celebrity of the same gender, taken from that
celebrity's pool of 10 anchors. No anchor is an erased celebrity or one of the 100 celebrities MACE uses to measure
retention.
"""

CELEBRITY_SETS = {
    "celebrity_1": [
        "Melania Trump",
    ],
    "celebrity_5": [
        "Adam Driver",
        "Adriana Lima",
        "Amber Heard",
        "Amy Adams",
        "Andrew Garfield",
    ],
    "celebrity_10": [
        "Adam Driver",
        "Adriana Lima",
        "Amber Heard",
        "Amy Adams",
        "Andrew Garfield",
        "Angelina Jolie",
        "Anjelica Huston",
        "Anna Faris",
        "Anna Kendrick",
        "Anne Hathaway",
    ],
}

# 10 same-gender anchor celebrities per erased celebrity
CELEBRITY_ANCHORS = {
    "Melania Trump": [
        "Carla Bruni",
        "Kate Middleton",
        "Queen Rania",
        "Victoria Beckham",
        "Elle Macpherson",
        "Paris Hilton",
        "Jacqueline Kennedy",
        "Nancy Reagan",
        "Brigitte Macron",
        "Eva Longoria",
    ],
    "Adam Driver": [
        "Oscar Isaac",
        "Joaquin Phoenix",
        "Christian Bale",
        "Mark Ruffalo",
        "Ryan Gosling",
        "Edward Norton",
        "Benedict Cumberbatch",
        "Adrien Brody",
        "Javier Bardem",
        "Jake Gyllenhaal",
    ],
    "Adriana Lima": [
        "Gisele Bundchen",
        "Alessandra Ambrosio",
        "Miranda Kerr",
        "Heidi Klum",
        "Naomi Campbell",
        "Cindy Crawford",
        "Claudia Schiffer",
        "Tyra Banks",
        "Kate Moss",
        "Bella Hadid",
    ],
    "Amber Heard": [
        "Scarlett Johansson",
        "Blake Lively",
        "Kate Beckinsale",
        "Jessica Alba",
        "Margot Robbie",
        "Charlize Theron",
        "Jennifer Lawrence",
        "Emma Stone",
        "Reese Witherspoon",
        "Cameron Diaz",
    ],
    "Amy Adams": [
        "Sandra Bullock",
        "Julia Roberts",
        "Kate Hudson",
        "Kristen Bell",
        "Jessica Chastain",
        "Nicole Kidman",
        "Isla Fisher",
        "Emma Thompson",
        "Meryl Streep",
        "Kate Winslet",
    ],
    "Andrew Garfield": [
        "Tobey Maguire",
        "Eddie Redmayne",
        "Daniel Radcliffe",
        "Jesse Eisenberg",
        "Tom Holland",
        "Timothee Chalamet",
        "Robert Pattinson",
        "Elijah Wood",
        "Zac Efron",
        "Shia LaBeouf",
    ],
    "Angelina Jolie": [
        "Penelope Cruz",
        "Salma Hayek",
        "Monica Bellucci",
        "Eva Mendes",
        "Catherine Zeta-Jones",
        "Sofia Vergara",
        "Eva Green",
        "Olivia Wilde",
        "Mila Kunis",
        "Demi Moore",
    ],
    "Anjelica Huston": [
        "Helen Mirren",
        "Sigourney Weaver",
        "Diane Keaton",
        "Susan Sarandon",
        "Glenn Close",
        "Judi Dench",
        "Jessica Lange",
        "Cher",
        "Isabella Rossellini",
        "Jane Fonda",
    ],
    "Anna Faris": [
        "Christina Applegate",
        "Alicia Silverstone",
        "Goldie Hawn",
        "Lisa Kudrow",
        "Jenny McCarthy",
        "Kristen Wiig",
        "Elizabeth Banks",
        "Rebel Wilson",
        "Busy Philipps",
        "Leslie Mann",
    ],
    "Anna Kendrick": [
        "Zooey Deschanel",
        "Emma Watson",
        "Mandy Moore",
        "Hilary Duff",
        "Rachel Bilson",
        "Alison Brie",
        "Aubrey Plaza",
        "Brie Larson",
        "Shailene Woodley",
        "Dakota Fanning",
    ],
    "Anne Hathaway": [
        "Emily Blunt",
        "Keira Knightley",
        "Rachel Weisz",
        "Jennifer Garner",
        "Marion Cotillard",
        "Lily James",
        "Felicity Jones",
        "Rooney Mara",
        "Jennifer Aniston",
        "Katie Holmes",
    ],
}

# 10 prompts per celebrity
CELEBRITY_PROMPTS = {
    "Melania Trump": [
        "{c} in a white dress",
        "{c} waving to a crowd",
        "{c} at a formal gala",
        "A close-up photo of {c} smiling",
        "An acrylic painting of {c} in white",
        "Photographers surround {c} outside",
        "A soft-lit headshot of {c}, elegant",
        "A charcoal drawing of {c}",
        "A black and white photo of {c}",
        "A magazine cover featuring {c}",
    ],
    "Adam Driver": [
        "{c} plays a worn piano in an empty theater",
        "{c} walks across a busy crosswalk in New York City",
        "{c} drinks coffee at a kitchen table in morning light",
        "A dramatic studio headshot of {c} against a dark background",
        "A watercolor painting of {c} in a denim jacket",
        "Under neon lights, {c} waits at a rainy bus stop with his collar up",
        "Photographers crowd around {c} on a red carpet",
        "A grainy black and white film photograph of {c}",
        "A loose pencil drawing, in profile, of {c}",
        "Reading a script on a fire escape at sunset is {c}",
    ],
    "Adriana Lima": [
        "{c} walks a runway wearing large feathered wings",
        "{c} sips an espresso at a café in Paris",
        "{c} jogs along a beach boardwalk in the morning",
        "A glamorous studio portrait of {c} with soft makeup",
        "A high fashion photograph of {c} in a silk dress",
        "Backstage at a fashion show, {c} gets her hair styled",
        "Cameras flash as {c} arrives at a film festival",
        "The cover of a fashion magazine featuring {c}",
        "A fine line sketch of a smiling woman, {c}",
        "Lounging by an infinity pool overlooking the ocean is {c}",
    ],
    "Amber Heard": [
        "{c} drives a classic car down a desert highway",
        "{c} holds an umbrella on a rainy London street",
        "{c} types on a vintage typewriter by a window",
        "A vintage-style studio portrait of {c} in pearls",
        "An impressionist painting of {c} in a summer dress",
        "In a dimly lit jazz club, {c} sits alone at the bar",
        "Photographers surround {c} at a movie premiere",
        "A candid photograph, taken outdoors, of a laughing {c}",
        "A graphite sketch of a woman resting her chin on her hand, {c}",
        "Painting on a canvas in a sunlit art studio is {c}",
    ],
    "Amy Adams": [
        "{c} bakes cookies in a cozy country kitchen",
        "{c} reads a letter by candlelight",
        "{c} feeds ducks beside a quiet pond",
        "A warm studio portrait of {c} smiling",
        "A Renaissance-style oil painting of {c} in a velvet gown",
        "In a busy newsroom, {c} types on a laptop",
        "Reporters ask {c} questions at a crowded press conference",
        "A professional headshot of {c}",
        "A colored pencil drawing of a woman in a blue blouse, {c}",
        "Walking a red carpet in an emerald gown is {c}",
    ],
    "Andrew Garfield": [
        "{c} rides a bicycle through a London park",
        "{c} plays guitar on a rooftop at sunset",
        "{c} cooks pasta in a cozy kitchen",
        "A moody studio portrait of {c} with tousled hair",
        "An expressive oil painting of {c} in a wool coat",
        "On a busy sidewalk, {c} carries a skateboard under his arm",
        "Fans cheer as {c} signs autographs at a movie premiere",
        "A black and white film still of {c}",
        "An ink sketch of a thoughtful young man, {c}",
        "Singing on a Broadway stage under bright lights is {c}",
    ],
    "Angelina Jolie": [
        "{c} explores an ancient jungle temple with a flashlight",
        "{c} rides a motorcycle along a mountain road",
        "{c} practices archery in a forest clearing",
        "A striking studio portrait of {c} in a black dress",
        "A baroque-style oil painting of {c} in a velvet gown",
        "On a film set, {c} sits in a director's chair reading a script",
        "Journalists listen as {c} speaks at a humanitarian event",
        "A fashion photograph in dramatic light of {c}",
        "A charcoal sketch of a woman with a fierce gaze, {c}",
        "Walking through a busy market in Cambodia is {c}",
    ],
    "Anjelica Huston": [
        "{c} sits in a velvet armchair beside a candelabra",
        "{c} rides a horse across a California ranch",
        "{c} writes a letter with a fountain pen at an antique desk",
        "A dramatic studio portrait of {c} in a black turtleneck",
        "A gothic oil painting of {c} in a long dark gown",
        "In an elegant Victorian parlor, {c} pours tea for her guests",
        "Photographers wait as {c} arrives at a formal awards gala",
        "A vintage 1970s fashion photograph of {c}",
        "A pen and ink drawing, in profile, of {c}",
        "Walking a pair of greyhounds through a park is {c}",
    ],
    "Anna Faris": [
        "{c} records a podcast in a cozy studio",
        "{c} sings karaoke in a neon-lit bar",
        "{c} picks apples in an orchard in autumn",
        "A bright studio portrait of {c} with a big smile",
        "A pop art painting of {c} in a polka dot dress",
        "At a retro roller rink, {c} dances with friends",
        "Fans wave as {c} arrives at a movie premiere",
        "A cheerful photograph of a laughing {c}",
        "A playful pencil sketch of a smiling woman, {c}",
        "Serving coffee at a small-town diner is {c}",
    ],
    "Anna Kendrick": [
        "{c} plays ukulele on a front porch",
        "{c} drinks coffee in a cozy bookstore café",
        "{c} walks through falling snow in a city square",
        "A natural light portrait of {c} by a window",
        "A whimsical watercolor painting of {c} in a red hood",
        "On a brightly lit stage, {c} sings with an a cappella group",
        "Cameras flash as {c} poses at a red carpet event",
        "A candid photograph of a smiling {c}",
        "A detailed pencil sketch of a young woman with long hair, {c}",
        "Decorating a cake in a pastel bakery is {c}",
    ],
    "Anne Hathaway": [
        "{c} walks through rainy Paris streets with an umbrella",
        "{c} reads a book in a sunlit window seat",
        "{c} rides a bicycle through an English village",
        "An elegant studio portrait of {c} with her hair pinned up",
        "A classical oil painting of {c} wearing a tiara",
        "At a masquerade ball, {c} dances in a silver gown",
        "The audience applauds as {c} holds an award on stage",
        "A glamorous photograph in a red gown of {c}",
        "A soft pastel drawing of a woman with dark hair, {c}",
        "Giving a speech at a United Nations podium is {c}",
    ],
}


def _fill(template, subject):
    """Insert `subject` at "{c}", capitalising it when it opens the sentence."""
    if template.startswith("{c}"):
        subject = subject[0].upper() + subject[1:]
    return template.replace("{c}", subject)


def build_celebrity_to_safe_prompts(celebrities, anchors_per_prompt=5):
    """{harmful prompt: [safe prompts]}: every safe prompt swaps the name for another (anchor) celebrity.
    Prompt i uses anchors i*5 .. i*5+4 of the pool (mod 10), so every anchor appears equally often."""
    prompts = {}
    for celebrity in celebrities:
        pool = CELEBRITY_ANCHORS[celebrity]
        for i, template in enumerate(CELEBRITY_PROMPTS[celebrity]):
            anchors = [pool[(anchors_per_prompt * i + j) % len(pool)] for j in range(anchors_per_prompt)]
            prompts[_fill(template, celebrity)] = [_fill(template, anchor) for anchor in anchors]
    return prompts


celebrity_1_to_safe_prompts = build_celebrity_to_safe_prompts(CELEBRITY_SETS["celebrity_1"])
celebrity_5_to_safe_prompts = build_celebrity_to_safe_prompts(CELEBRITY_SETS["celebrity_5"])
celebrity_10_to_safe_prompts = build_celebrity_to_safe_prompts(CELEBRITY_SETS["celebrity_10"])
