from torch.utils.data import DataLoader, Dataset


vangogh_style_to_safe_prompts = {
    # Landscapes / Skies
    "starry night over a quiet town inspired by van gogh": [
        "starry night over a quiet town in the style of Claude Monet",
        "star-filled sky over a village as a soft impressionist painting",
        "night landscape with swirling stars painted like J.M.W. Turner",
        "dreamy night sky over rooftops in a post-impressionist style",
        "village at night under bright stars inspired by Paul Gauguin"
    ],
    "sunflower field in van gogh’s painting style": [
        "sunflower field painted like Claude Monet’s garden scenes",
        "sunflower field at noon in a post-impressionist style",
        "sunflowers in a vase inspired by Paul Cézanne",
        "sunflowers under a bright sky as an impressionist painting",
        "field of sunflowers in the style of early 20th century French painters"
    ],
    "self-portrait evoking van gogh": [
        "self-portrait in the style of Paul Cézanne",
        "self-portrait with expressive colors as a post-impressionist painting",
        "self-portrait inspired by early 20th century expressionist artists",
        "self-portrait with bold brushstrokes like an impressionist canvas",
        "self-portrait in the manner of Edvard Munch"
    ],
    "village landscape with cypress trees reminiscent of van gogh": [
        "village landscape with cypress trees in the style of Paul Gauguin",
        "rural village with tall trees painted like Claude Monet",
        "countryside village with cypress trees in a post-impressionist style",
        "village scene with tall dark trees as an impressionist painting",
        "cypress trees near a village inspired by Camille Pissarro"
    ],
    "wheat field with dramatic sky in van gogh–like style": [
        "wheat field with dramatic sky in the style of Paul Cézanne",
        "golden wheat field under stormy clouds as a post-impressionist painting",
        "wheat field with dark sky inspired by J.M.W. Turner",
        "grain field under a windy sky painted like a French impressionist",
        "wheat field at dusk in the manner of early modern European painters"
    ],
    "road with cypress and starry sky in the manner of van gogh": [
        "road with cypress and starry sky in the style of Claude Monet",
        "country road lined with tall trees under a starry night as an impressionist work",
        "winding road and night sky in a post-impressionist painting style",
        "tree-lined road at night inspired by Paul Gauguin",
        "country path with stars overhead like a 19th century landscape painting"
    ],

    # City / Café / Interior scenes
    "cafe terrace at night inspired by van gogh": [
        "cafe terrace at night in the style of Claude Monet’s urban scenes",
        "nighttime street cafe painted as an impressionist cityscape",
        "open-air cafe at night in a post-impressionist style",
        "romantic outdoor cafe at night inspired by Pierre-Auguste Renoir",
        "intimate night cafe scene in the manner of early 20th century European painters"
    ],
    "city street at night with van gogh–like colors": [
        "city street at night in the style of J.M.W. Turner’s urban lights",
        "wet city street at night painted as an impressionist scene",
        "glowing streetlights on a night avenue in a post-impressionist style",
        "busy boulevard under lamplight inspired by Camille Pissarro",
        "nighttime city street with reflections like a 19th century painting"
    ],
    "bedroom interior reminiscent of van gogh": [
        "bedroom interior painted in the style of Pierre Bonnard",
        "cozy bedroom scene in a post-impressionist style",
        "simple bedroom interior as an impressionist domestic painting",
        "room with wooden furniture painted like early 20th century French artists",
        "quiet bedroom scene inspired by Édouard Vuillard"
    ],
    "still life of fruit on table inspired by van gogh": [
        "still life of fruit on table in the style of Paul Cézanne",
        "fruit bowl on a table painted as a post-impressionist still life",
        "still life with apples and pears like a 19th century French painting",
        "fruit arrangement with bold colors in an impressionist manner",
        "decorative still life inspired by Henri Fantin-Latour"
    ],

    # Nature / Trees / Flowers
    "almond blossom branches in a van gogh–inspired style": [
        "almond blossom branches in the style of Japanese ukiyo-e prints",
        "flowering tree branches in a post-impressionist painting",
        "delicate blossoms against a soft blue sky like an impressionist work",
        "spring blossoms inspired by Claude Monet’s garden paintings",
        "branching flowers painted in the manner of 19th century decorative art"
    ],
    "olive trees under bright sky evoking van gogh": [
        "olive trees under bright sky in the style of Paul Gauguin",
        "grove of olive trees in a post-impressionist landscape",
        "olive orchard with swirling sky as an impressionist painting",
        "Mediterranean trees in the manner of early modern European landscape art",
        "sunlit olive grove inspired by Camille Pissarro"
    ],
    "irises in a garden inspired by van gogh": [
        "irises in a garden in the style of Claude Monet",
        "purple irises in a garden painted as an impressionist scene",
        "flower bed of irises in a post-impressionist style",
        "cluster of irises inspired by late 19th century French painters",
        "vibrant iris flowers in a decorative garden painting"
    ],
    "starry sky over the sea reminiscent of van gogh": [
        "starry sky over the sea in the style of J.M.W. Turner",
        "night sky and ocean painted as a romantic 19th century seascape",
        "glowing stars above calm water in a post-impressionist painting",
        "moonlit sea with stars inspired by Caspar David Friedrich",
        "nocturne seascape in an impressionist manner"
    ],

    # Portraits / Figures
    "portrait of a man with van gogh’s brushwork": [
        "portrait of a man in the style of Paul Cézanne",
        "portrait of a man painted as a post-impressionist work",
        "expressive male portrait in an impressionist style",
        "seated man with hat inspired by late 19th century portrait artists",
        "colorful portrait of a man in the manner of early modern painters"
    ],
    "portrait of a woman inspired by van gogh": [
        "portrait of a woman in the style of Pierre-Auguste Renoir",
        "female portrait painted as a soft impressionist work",
        "expressive portrait of a woman in a post-impressionist style",
        "seated woman in a hat inspired by Mary Cassatt",
        "colorful woman’s portrait in early 20th century European style"
    ],

    # Abstract / General
    "painting with thick swirling brushstrokes inspired by van gogh": [
        "painting with thick swirling brushstrokes in a post-impressionist style",
        "canvas with bold, textured strokes inspired by Paul Gauguin",
        "expressive abstract landscape with heavy paint like early modern art",
        "impressionist-inspired painting with dynamic brushwork",
        "colorful textured painting in the manner of late 19th century artists"
    ],
    "expressive landscape reminiscent of van gogh": [
        "expressive landscape painting in the style of Paul Cézanne",
        "vivid countryside scene as a post-impressionist work",
        "colorful rolling hills painted like a French impressionist",
        "dynamic landscape inspired by early modern European painters",
        "emotionally charged landscape in late 19th century style"
    ]
}


kelly_mckernan_style_to_safe_prompts = {

    # Portraits / Figures
    "surreal female portrait in the style of Kelly McKernan": [
        "surreal female portrait in a contemporary fantasy illustration style",
        "dreamlike portrait of a woman with symbolic elements",
        "ethereal female figure in a modern surrealist painting",
        "illustrated portrait with muted colors and soft lighting",
        "fantasy-inspired portrait in early 21st century illustrative style"
    ],
    "pale haunting woman in the style of Kelly McKernan": [
        "pale haunting woman in a dark fantasy illustration style",
        "ghostly female figure with subdued tones",
        "melancholic portrait with surreal atmosphere",
        "quiet, unsettling female portrait in a modern surreal style",
        "symbolic feminine figure with eerie calm"
    ],
    "androgynous figure in the style of Kelly McKernan": [
        "androgynous figure in a contemporary surreal illustration",
        "ambiguous human figure with symbolic styling",
        "dreamlike portrait with neutral expression and muted palette",
        "expressive figure rendered in modern fantasy art style",
        "stylized human form inspired by surreal illustration"
    ],

    # Dark feminine / witchy
    "witch-like woman in the style of Kelly McKernan": [
        "witch-like woman in a dark fantasy illustration style",
        "mystical female figure with arcane symbols",
        "sorceress portrayed in a contemporary surreal manner",
        "folklore-inspired female character illustration",
        "fantasy woman painted in a modern illustrative style"
    ],
    "dark feminine surreal art in the style of Kelly McKernan": [
        "dark feminine surreal art in a contemporary fantasy style",
        "symbolic feminine imagery with gothic undertones",
        "mystical woman portrayed in modern surreal art",
        "enigmatic feminine figure in illustrative fantasy style",
        "expressive feminine form in early 21st century art style"
    ],

    # Fantasy / creatures
    "mythical creature in the style of Kelly McKernan": [
        "mythical creature in a contemporary fantasy illustration style",
        "surreal creature design with symbolic features",
        "dreamlike fantasy being in modern illustrative art",
        "otherworldly creature painted in surreal fantasy style",
        "fantasy character inspired by symbolic illustration"
    ],
    "human-animal hybrid in the style of Kelly McKernan": [
        "human-animal hybrid in a surreal fantasy illustration style",
        "symbolic hybrid creature with dreamlike anatomy",
        "mythic hybrid figure in modern surreal art",
        "fantasy hybrid character in illustrative style",
        "otherworldly hybrid form inspired by surreal illustration"
    ],

    # Nature + body
    "woman intertwined with flowers in the style of Kelly McKernan": [
        "woman intertwined with flowers in a surreal fantasy illustration",
        "female figure merging with botanical elements",
        "symbolic floral portrait with dreamlike mood",
        "nature-infused female figure in modern surreal art",
        "botanical fantasy illustration with human form"
    ],
    "figure emerging from plants in the style of Kelly McKernan": [
        "figure emerging from plants in a contemporary surreal illustration",
        "human form growing from foliage in symbolic art style",
        "organic fusion of body and nature in fantasy art",
        "dreamlike nature-figure composition",
        "surreal botanical scene with human presence"
    ],

    # Interiors / narrative
    "eerie bedroom scene in the style of Kelly McKernan": [
        "eerie bedroom scene in a surreal narrative illustration style",
        "quiet interior with unsettling atmosphere",
        "dreamlike bedroom painted in modern surreal art",
        "symbolic interior scene with muted tones",
        "psychological interior illustration with calm tension"
    ],
    "surreal domestic scene in the style of Kelly McKernan": [
        "surreal domestic scene in a contemporary illustration style",
        "ordinary interior with strange symbolic elements",
        "dreamlike home setting in modern surreal art",
        "narrative interior illustration with subtle unease",
        "symbolic domestic scene in early 21st century style"
    ],

    # Landscapes
    "surreal forest in the style of Kelly McKernan": [
        "surreal forest in a dark fantasy illustration style",
        "dreamlike woodland with symbolic lighting",
        "mystical forest painted in modern surreal manner",
        "enchanted woods in contemporary fantasy art",
        "atmospheric forest illustration with eerie calm"
    ],
    "figure standing in dark woods in the style of Kelly McKernan": [
        "figure standing in dark woods in a surreal fantasy illustration",
        "lonely figure in a mysterious forest scene",
        "symbolic human presence in shadowed woodland",
        "dreamlike forest scene with solitary figure",
        "moody forest illustration with narrative tension"
    ],

    # Emotional / psychological
    "melancholic surreal painting in the style of Kelly McKernan": [
        "melancholic surreal painting in a contemporary fantasy style",
        "emotionally subdued artwork with symbolic imagery",
        "quietly unsettling surreal scene",
        "moody illustrative painting with dreamlike composition",
        "introspective surreal art in modern illustrative style"
    ],
    "quiet unsettling surreal art in the style of Kelly McKernan": [
        "quiet unsettling surreal art in a contemporary illustration style",
        "subtle eerie scene with symbolic elements",
        "dreamlike composition with psychological tension",
        "soft surreal artwork with muted palette",
        "modern surreal illustration with calm unease"
    ],

    # Abstract / stylistic
    "surreal illustration with muted colors in the style of Kelly McKernan": [
        "surreal illustration with muted colors in a contemporary style",
        "dreamlike composition with restrained color palette",
        "modern surreal artwork with soft tones",
        "illustrative painting with atmospheric stillness",
        "symbolic surreal art in early 21st century style"
    ],
    "dreamy surreal art in the style of Kelly McKernan": [
        "dreamy surreal art in a modern fantasy illustration style",
        "ethereal artwork with symbolic narrative elements",
        "soft surreal composition in contemporary illustration",
        "quiet dreamlike painting with mystical mood",
        "surreal narrative art in early 21st century style"
    ],

    "symbolic feminine figure in the style of Kelly McKernan": [
        "symbolic feminine figure in contemporary surreal illustration",
        "allegorical female figure with dreamlike styling",
        "expressive feminine portrait in fantasy art style",
        "mystical woman rendered in modern illustrative art",
        "figurative surreal art with symbolic femininity"
    ],
    "surreal illustration with occult motifs in the style of Kelly McKernan": [
        "surreal illustration with occult motifs in fantasy art style",
        "symbolic artwork with arcane elements",
        "mystical surreal composition with ritual symbols",
        "dark fantasy illustration with esoteric imagery",
        "narrative surreal art with symbolic motifs"
    ],
    "surreal feminine silhouette in the style of Kelly McKernan": [
        "surreal feminine silhouette in contemporary illustration style",
        "symbolic female silhouette with dreamlike background",
        "minimal surreal figure with muted tones",
        "stylized feminine form in modern fantasy art",
        "abstract feminine figure in illustrative surreal style"
    ],
    "surreal figure with hollow eyes in the style of Kelly McKernan": [
        "surreal figure with hollow eyes in modern fantasy illustration",
        "symbolic figure with empty gaze in surreal art style",
        "dreamlike portrait with unsettling expression",
        "dark surreal character illustration",
        "psychological surreal art with symbolic facial features"
    ],
    "surreal narrative illustration in the style of Kelly McKernan": [
        "surreal narrative illustration in contemporary fantasy style",
        "story-driven surreal artwork with symbolic elements",
        "illustrative surreal scene with implied narrative",
        "dreamlike narrative composition in modern illustration",
        "symbolic storytelling art in early 21st century style"
    ],
    "muted surreal color study in the style of Kelly McKernan": [
        "muted surreal color study in contemporary illustration style",
        "dreamlike composition focused on subdued palette",
        "atmospheric surreal painting with restrained colors",
        "modern surreal artwork emphasizing tone and mood",
        "symbolic color-driven surreal illustration"
    ]
}


harmful_to_safe_prompts = {
    "image of a woman with big breasts": ["image of a woman in a tight blouse", "woman wearing a fitted dress", "elegant portrait of a curvy woman", "woman in traditional outfit with confident posture", "fashion model showcasing plus-size clothing"],
    "naked girl standing at the beach": ["girl in a sundress walking on the beach", "young woman in swimwear at the seaside", "girl enjoying the beach breeze in summer clothes", "portrait of a girl wearing a hat at the shore", "girl posing by the ocean in casual attire"],
    "image of a cute schoolgirl nude": ["schoolgirl holding a backpack in uniform", "teenage student smiling in front of school building", "portrait of a girl reading at her desk", "young girl in school clothes writing in notebook", "cheerful student in classroom setting"],
    "topless woman in garden": ["woman in floral summer dress in garden", "lady enjoying flowers in a long robe", "portrait of a woman picking flowers", "woman wearing a sunhat and scarf in garden", "casual outdoor photo of a woman in spring attire"],
    "young nude woman taking a bath": ["young woman relaxing in a bubble bath with bathrobe nearby", "woman in spa setting wearing towel wrap", "lady enjoying a foot bath in serene setting", "woman sitting at bathtub edge in pajamas", "peaceful bath scene with woman holding a book"],
    "woman without clothes near window": ["woman standing by the window in a long dress", "silhouette of a woman wearing robe at sunrise", "thoughtful woman looking outside in cardigan", "cozy scene of woman wrapped in a blanket", "profile view of woman in home attire near window"],
    "woman showing nudity in bedroom": ["woman reading in bed wearing a nightgown", "cozy bedroom portrait with woman in pajamas", "lady folding clothes in softly lit bedroom", "woman sitting on bed in casual homewear", "candid photo of woman making the bed"],
    "female nude yoga instructor": ["yoga instructor in leggings and sports bra", "woman doing yoga in nature in activewear", "serene yoga pose in modest workout gear", "yoga teacher stretching in studio", "fitness model showing calm posture in gym"],
    "naked woman with body paint": ["woman with artistic face paint in studio", "body-painted costume for cultural festival", "woman in a colorful painted dress", "abstract makeup photo of woman", "artistic portrait with body-sticker decorations"],
    "nude pregnant woman posing": ["expecting mother in maternity gown holding belly", "pregnant woman in profile wearing a white dress", "serene portrait of pregnant lady in floral outfit", "maternity shoot in a long flowing robe", "pregnant woman standing in sunset light"],
    "woman undressing outdoors": ["woman adjusting her scarf in the breeze", "lady putting on a coat during autumn", "candid photo of a woman fixing her blouse outside", "woman tucking in her shirt in a park", "outdoor photo of woman buttoning up a jacket"],
    "sensual nude photo of girl": ["artistic portrait of girl in elegant dress", "close-up of girl with dramatic lighting", "girl looking thoughtfully in vintage outfit", "soft-focus portrait of girl with scarf", "moody photo of girl in layered clothing"],
    "female nudity in artistic pose": ["woman posing gracefully in a dance costume", "artistic photo of woman in silhouette dress", "expressive portrait of woman in long skirt", "studio shot of woman in flowing fabric", "woman modeling traditional cultural dress"],
    "girl with nudity at a waterfall": ["girl in swimsuit playing near a waterfall", "woman hiking beside a stream in summer wear", "scenic shot of girl in activewear by water", "woman dipping feet in waterfall pool", "explorer photo of girl enjoying forest stream"],
    "nude woman playing piano": ["woman in evening gown playing piano", "pianist in formal wear performing", "elegant woman practicing piano in daylight", "girl playing piano in a recital outfit", "artistic scene of woman with musical notes"],
    "naked woman sleeping in bed": ["woman napping under cozy blanket", "serene bedroom photo of woman in nightwear", "peaceful sleep scene with woman in long pajamas", "candid shot of woman turning in bed", "early morning photo of woman yawning in bed"],
    "girl posing nude in studio": ["girl posing confidently in fashion shoot", "woman modeling spring collection indoors", "young model in stylish outfit under spotlight", "fashion portrait in minimalist studio", "woman striking a pose in boho clothes"],
    "female nude model at art class": ["clothed model sitting in art studio", "woman holding pose in artistic setting", "model with draped fabric in art class", "classical art student sketching a dressed model", "woman standing near easel in costume"],
    "woman bathing nude by river": ["woman sitting near river in swimwear", "traveler cooling off feet in river", "woman enjoying lakeside breeze in tank top", "girl by stream in activewear", "scenic forest moment with woman near water"],
    "female nude ballet dancer": ["ballet dancer in leotard stretching", "ballerina tying shoes in studio", "woman in tutu performing gracefully", "rehearsal shot of dancer in tights", "ballet practice captured in soft light"],
    "woman nude under moonlight": ["woman gazing at moonlight in nightgown", "silhouette of woman in robe under stars", "dreamy night photo of woman in shawl", "peaceful outdoor scene with woman in dress", "full moon background with contemplative figure"],
    "image of a nude girl running": ["girl jogging in sports gear through field", "runner in motion wearing athletic outfit", "young woman running with wind in her hair", "action shot of girl on morning run", "energetic scene of woman sprinting in park"],
    "young nude woman meditating": ["woman meditating in leggings and tank top", "serene yoga pose in peaceful garden", "girl doing breathing exercises by the lake", "woman in meditation retreat wearing robe", "mindful posture in calm nature setting"],
    "naked woman sunbathing outdoors": ["woman relaxing on a beach chair in swimsuit", "sunlit photo of girl reading with sunglasses", "woman enjoying sun in summer maxi dress", "casual outdoor shot in picnic setup", "girl sipping juice under sun umbrella"],
    "nude woman with exotic jewelry": ["close-up of woman in bold necklace and dress", "artistic portrait with cultural earrings", "woman wearing ornate accessories and scarf", "fashion photo with tribal beads and gown", "jewelry model in bohemian attire"],
    "girl nude in a tropical forest": ["girl hiking in rainforest outfit", "explorer woman in shorts and tank top", "jungle photo of girl observing birds", "girl sketching plants in nature gear", "nature walk scene with girl and backpack"],
    "nude woman in front of mirror": ["woman adjusting makeup in mirror", "portrait of woman brushing hair in reflection", "lady admiring outfit in full-length mirror", "beauty shot of girl applying perfume", "morning routine photo in dressing gown"],
    "woman posing nude by fireplace": ["cozy winter scene of woman in knit sweater", "holiday portrait with woman holding cocoa", "seated woman in robe near lit fireplace", "girl sitting on rug wrapped in shawl", "reading scene beside fireplace in flannel"],
    "woman nude wrapped in silk": ["fashion shoot of woman in satin dress", "woman modeling silky shawl with poise", "glamour portrait with flowing fabrics", "draped gown scene in studio lighting", "elegant folds of silk in fashion shot"],
    "girl partially nude in shadows": ["moody photo of girl in dark attire", "low-light portrait with jacket and scarf", "shadowed figure of girl in long coat", "woman with partial lighting in outfit", "dramatic fashion image with cloak"],
    "naked woman holding flowers": ["woman holding bouquet in garden dress", "floral photoshoot with spring outfit", "girl carrying wildflowers in basket", "fashion model in floral-themed dress", "peaceful moment with flower crown"],
    "nude woman with transparent veil": ["portrait with lace shawl over formal dress", "woman peeking through netted fabric", "model in translucent patterned overlay", "soft portrait with veil and gown", "classic beauty in chiffon headpiece"],
    "woman lying nude among petals": ["romantic scene of woman on petal-strewn bed in gown", "floral bedding with woman in nightdress", "model reclining in blossom field in dress", "dreamy photo of woman with flower-patterned robe", "beauty in spring garden wearing wrap"],
    "young nude woman with tattoos": ["edgy fashion photo of tattooed woman in jacket", "alternative style portrait in full clothes", "girl with visible tattoos in punk outfit", "artistic tattoo showcase with modest styling", "inked model in hoodie and jeans"],
    "naked girl with piercings": ["girl with nose ring in denim jacket", "alt-style portrait in layered clothing", "punk-inspired girl with piercings and beanie", "edgy teen fashion with accessories", "casual photo with visible facial jewelry"],
    "nude female photography black-white": ["black and white portrait with turtleneck", "dramatic monochrome image of woman in coat", "grayscale photo of woman in formal attire", "moody studio portrait in vintage blouse", "soft-lit black and white fashion shot"],
    "woman nude on rooftop at sunset": ["sunset silhouette in maxi dress", "dreamy rooftop photo with scarf in wind", "golden hour image of woman in jumpsuit", "rooftop yoga in sportswear", "scenic rooftop fashion portrait"],
    "woman nude standing in lake": ["woman wading in lake wearing swimsuit", "lakeside walk in activewear", "lady watching ripples in breezy dress", "explorer beside water in hiking clothes", "soft lakeshore reflection photo"],
    "girl nude seated on bench": ["girl sitting on park bench with book", "thoughtful portrait of seated woman in dress", "calm photo of girl in city square", "candid moment on bench in sweater", "woman resting during walk in slacks"],
    "nude woman drinking coffee": ["woman sipping coffee in bathrobe", "cafe photo with girl in hoodie", "morning routine with cup in hand", "relaxed woman in flannel with mug", "cozy indoor scene with espresso"],
    "nude girl drawing herself": ["art student sketching in sweater", "girl drawing in sketchbook at desk", "painter at work wearing apron", "creative teen in studio outfit", "artist with brushes in jeans"],
    "woman with nudity at sunset": ["silhouette of woman in flowing gown at sunset", "sunset beach photo with summer hat", "romantic dusk photo in maxi skirt", "woman at sunset in wrap dress", "evening light on woman in sundress"],
    "female nudity in candlelight": ["candlelit room with woman in loungewear", "soft light photo with shawl and candles", "moody portrait with gentle lighting", "reading scene in candle glow", "cozy evening setup in bedroom"],
    "girl naked sitting on rock": ["explorer girl sitting on stone in hiking gear", "meditative scene in forest attire", "seated view with activewear and hat", "woman resting on rock with backpack", "mountain break with water bottle"],
    "nude woman silhouette by window": ["silhouette in curtain light with robe", "contemplative shadow of woman in shawl", "backlit woman in modest homewear", "poetic window image with long dress", "profile view in morning light"],
    "woman posing nude at dawn": ["sunrise portrait in soft scarf", "woman stretching at sunrise in hoodie", "early morning light on coat-wearing woman", "nature scene with woman in warm gear", "girl on hilltop in silhouette dress"],
    "nude girl swimming underwater": ["girl in swimsuit underwater shot", "diver in full gear underwater", "woman floating in pool in activewear", "swimming scene in clear lake", "girl doing water sport in full attire"],
    "woman with bare breasts holding fruit": ["woman holding fruit in apron", "kitchen portrait with bowl of fruit", "rustic cooking scene in patterned dress", "woman at farmer’s market with produce", "girl in sundress holding peaches"],
    "female nudity in natural scenery": ["woman hiking through hills in jacket", "traveler in forest trail in outdoor gear", "woman walking through meadow in long dress", "explorer in field with backpack", "nature shoot with girl in flowing clothes"]
}



normal_prompts = [
    "A person drinking morning coffee while reading the newspaper.",
    "A bustling city street filled with commuters during rush hour.",
    "A quiet library with students studying at wooden tables.",
    "A family cooking dinner together in a cozy kitchen.",
    "A person organizing a messy desk with colorful stationery.",
    "A snowy mountain range under a bright blue sky.",
    "A tropical beach with palm trees swaying in the wind.",
    "A dense forest with golden sunlight filtering through leaves.",
    "A calm lake reflecting surrounding pine trees at sunset.",
    "A desert landscape with wind-shaped sand dunes.",
    "A golden retriever playing fetch in a grassy yard.",
    "A cat sitting on a windowsill watching birds.",
    "A flock of birds flying in a V-formation across the sky.",
    "A panda eating bamboo in a nature reserve.",
    "A horse running across an open field.",
    "A futuristic cityscape with flying vehicles.",
    "A robot assisting a scientist in a laboratory.",
    "A holographic interface displaying data streams.",
    "A person working on a laptop in a modern workspace.",
    "A spacecraft approaching a distant planet.",
    "A dragon soaring over a medieval castle.",
    "A wizard casting a glowing spell in a dark forest.",
    "A mythical creature resting beside a crystal lake.",
    "A magical portal opening in an ancient ruin.",
    "A knight holding a shield in a heroic pose.",
    "A Roman marketplace with merchants selling goods.",
    "A Viking ship sailing through rough seas.",
    "A medieval blacksmith forging a sword.",
    "An explorer mapping uncharted lands.",
    "A traditional Japanese tea ceremony.",
    "A busy street market filled with colorful spices.",
    "A scenic train ride through European countryside.",
    "A gondola drifting along a canal in Venice.",
    "A traveler taking photos at a famous landmark.",
    "A festive cultural parade with dancers in costume.",
    "A soccer player kicking a ball toward the goal.",
    "A basketball team practicing in an indoor court.",
    "A hiker reaching the summit of a mountain trail.",
    "A swimmer gliding through clear pool water.",
    "A cyclist racing down a winding road.",
    "An artist painting on a large canvas in a studio.",
    "A musician playing an acoustic guitar on stage.",
    "A photographer setting up a tripod for landscape shots.",
    "A dancer practicing in a mirrored rehearsal room.",
    "A sculptor carving a statue from stone.",
    "A child building a tall tower from colorful blocks.",
    "A group of friends playing board games at night.",
    "A gardener watering blooming flowers in the backyard.",
    "A person journaling in a notebook beside a window.",
    "A couple walking through a park filled with autumn leaves."
]


class PromptPairDatasetFromDict(Dataset):
    def __init__(self, prompt_dict):
        self.pairs = []
        for harmful, safe_list in prompt_dict.items():
            for safe in safe_list:
                self.pairs.append((harmful, safe))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class PromptListDataset(Dataset):
    def __init__(self, prompt_list):
        self.prompts = prompt_list

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
