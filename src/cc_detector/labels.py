"""
Label system for YAMNet-based CC detection.

Key fix from v0.2: removed 'domestic animals, pets' from blocklist —
it was blocking dog bark detections when YAMNet's parent class fired
instead of the specific 'Dog' or 'Bark' class.
"""

from __future__ import annotations

# ── Blocklist ─────────────────────────────────────────────────────
# Labels never CC-worthy. Be conservative — false negatives are worse.
# Do NOT add broad animal/object categories here.
BLOCKLIST: frozenset[str] = frozenset({
    "inside, small room",
    "inside, large room or hall",
    "inside, public space",
    "outside, urban or manmade",
    "outside, rural or natural",
    "acoustic environment",
    "reverberation",
    "room acoustic",
    "silence",
    "noise",
    "white noise",
    "pink noise",
    "static",
    "hum",
    "buzz",
    "snake",
    "speech",
    "narration, monologue",
    "male speech, man speaking",
    "female speech, woman speaking",
    "child speech, kid speaking",
    "conversation",
    "babbling",
    "breathing",
    "pant",
    "snort",
    "cough",
    "belch",
    "hiccup",
    "sound effect",
    "mechanisms",
    "generic impact sounds",
    "scratch",
    "rattle",
    "rustle",
})

LABEL_REMAPPING: dict[str, str] = {
    "cap gun":                          "GUNSHOT",
    "gunshot, gunfire":                 "GUNSHOT",
    "machine gun":                      "RAPID GUNFIRE",
    "fusillade":                        "RAPID GUNFIRE",
    "explosion":                        "EXPLOSION",
    "burst, pop":                       "POP",
    "bang":                             "BANG",
    "dog":                              "DOG BARK",
    "bark":                             "DOG BARK",
    "bow-wow":                          "DOG BARK",
    "domestic animals, pets":           "DOG BARK",
    "animal":                           "DOG BARK",
    "meow":                             "CAT",
    "cat":                              "CAT",
    "bird":                             "BIRD",
    "chirp, tweet":                     "BIRD",
    "bird vocalization, bird call, bird song": "BIRD",
    "tick":                             "CLOCK TICKING",
    "ticking":                          "CLOCK TICKING",
    "clock":                            "CLOCK TICKING",
    "chink, clink":                     "GLASS",
    "glass":                            "GLASS",
    "shatter":                          "GLASS BREAKING",
    "door":                             "DOOR",
    "door slam":                        "DOOR SLAM",
    "slam":                             "DOOR SLAM",
    "knock":                            "KNOCK",
    "squeak":                           "SQUEAK",
    "creak":                            "CREAK",
    "vehicle horn, car horn, honking":  "CAR HORN",
    "honk":                             "CAR HORN",
    "car":                              "VEHICLE",
    "truck":                            "VEHICLE",
    "motorcycle":                       "MOTORCYCLE",
    "engine":                           "ENGINE",
    "telephone":                        "PHONE RING",
    "ringtone":                         "PHONE RING",
    "alarm clock":                      "ALARM",
    "fire alarm":                       "ALARM",
    "smoke detector":                   "ALARM",
    "alarm":                            "ALARM",
    "siren":                            "SIREN",
    "civil defense siren":              "SIREN",
    "rain":                             "RAIN",
    "thunder":                          "THUNDER",
    "thunderstorm":                     "THUNDER",
    "wind":                             "WIND",
    "fire":                             "FIRE",
    "fireworks":                        "FIREWORKS",
    "screaming":                        "SCREAM",
    "shout":                            "SHOUT",
    "laughter":                         "LAUGHTER",
    "applause":                         "APPLAUSE",
    "crying, sobbing":                  "CRYING",
    "whimper":                          "CRYING",
    "thump, thud":                      "THUD",
    "stir":                             "STIRRING",
    "chop":                             "SHARP IMPACT",
    "ping":                             "PING",
    "gears":                            "MECHANICAL",
    "computer keyboard":                "KEYBOARD",
    "typewriter":                       "KEYBOARD",
    "bell":                             "BELL",
    "church bell":                      "BELL",
    "doorbell":                         "DOORBELL",
    "footsteps":                        "FOOTSTEPS",
    "splash, splatter":                 "WATER SPLASH",
    "water":                            "WATER",
    "crowd":                            "CROWD",
    "cheering":                         "CROWD CHEER",
    "music":                            "MUSIC",
    "drum":                             "DRUM",
    "guitar":                           "MUSIC",
    "piano":                            "MUSIC",
}

# Transient labels: short-duration events (1-2 YAMNet frames).
# Consensus voting is BYPASSED for these — a single frame is enough.
TRANSIENT_LABELS: frozenset[str] = frozenset({
    "DOG BARK", "CAT", "BIRD", "ANIMAL SOUND", "CLOCK TICKING",
    "GLASS", "GLASS BREAKING",
    "DOOR", "DOOR SLAM", "KNOCK", "DOORBELL", "SQUEAK", "CREAK", "BANG",
    "GUNSHOT", "RAPID GUNFIRE", "EXPLOSION", "POP",
    "SCREAM", "SHOUT",
    "ALARM", "PHONE RING",
    "THUD", "SHARP IMPACT", "IMPACT", "PING",
    "FOOTSTEPS", "WATER SPLASH",
    "BELL", "FIREWORKS", "LAUGHTER", "APPLAUSE", "KNOCK",
})

HINDI_CC_MAP: list[tuple[tuple[str, ...], str]] = [
    (("rapid gunfire", "machine gun", "fusillade"),   "तेज़ गोलीबारी"),
    (("gunshot", "gun", "rifle", "pistol", "bang"),   "गोली की आवाज़"),
    (("explosion", "blast", "bomb", "detonat"),        "विस्फोट"),
    (("firework",),                                    "आतिशबाजी"),
    (("pop",),                                         "पॉप की आवाज़"),
    (("scream", "shriek"),                             "चीख"),
    (("shout", "yell"),                                "चिल्लाना"),
    (("laughter", "laugh", "giggle", "chuckle"),       "हँसी"),
    (("applause", "clapping"),                         "तालियाँ"),
    (("crying", "sobbing", "weeping", "whimper"),      "रोने की आवाज़"),
    (("crowd cheer", "cheer"),                         "भीड़ का जयकारा"),
    (("crowd",),                                       "भीड़ का शोर"),
    (("glass breaking", "glass", "shatter"),           "काँच की आवाज़"),
    (("thud", "thump", "sharp impact", "impact"),      "धमाके की आवाज़"),
    (("knock",),                                       "दस्तक"),
    (("door slam", "door"),                            "दरवाज़े की आवाज़"),
    (("squeak", "creak"),                              "चरचराहट"),
    (("car horn", "horn", "honk"),                     "हॉर्न बजना"),
    (("siren",),                                       "सायरन"),
    (("alarm",),                                       "अलार्म"),
    (("phone ring", "ringtone", "telephone"),          "फ़ोन की घंटी"),
    (("doorbell",),                                    "डोरबेल"),
    (("vehicle", "car", "truck", "motorcycle"),        "वाहन की आवाज़"),
    (("engine",),                                      "इंजन की आवाज़"),
    (("dog bark", "bark", "bow-wow", "dog", "animal"),  "कुत्ते की आवाज़"),
    (("cat",),                                         "बिल्ली की आवाज़"),
    (("bird",),                                        "चिड़िया की आवाज़"),
    (("clock ticking", "clock", "ticking"),            "घड़ी की टिक-टिक"),
    (("thunder",),                                     "बिजली कड़कना"),
    (("rain",),                                        "बारिश"),
    (("wind",),                                        "हवा की आवाज़"),
    (("fire",),                                        "आग की आवाज़"),
    (("water splash", "splash"),                       "पानी के छींटे"),
    (("water",),                                       "पानी की आवाज़"),
    (("bell",),                                        "घंटी"),
    (("keyboard", "typing"),                           "टाइपिंग"),
    (("drum",),                                        "ढोल"),
    (("music", "piano", "guitar"),                     "संगीत"),
    (("footsteps",),                                   "क़दमों की आवाज़"),
    (("mechanical", "stirring"),                       "यांत्रिक आवाज़"),
    (("ping",),                                        "पिंग"),
]


def is_blocklisted(label: str) -> bool:
    return label.lower() in BLOCKLIST


def remap_label(label: str) -> str:
    return LABEL_REMAPPING.get(label.lower(), label.upper())


def is_transient(canonical_label: str) -> bool:
    """True if this label typically fires in 1-2 YAMNet frames."""
    return canonical_label.upper() in TRANSIENT_LABELS


def caption_en(label: str) -> str:
    return f"[{label.lower()}]"


def caption_hi(label: str) -> str:
    label_lower = label.lower()
    for keywords, hindi in HINDI_CC_MAP:
        if any(kw in label_lower for kw in keywords):
            return f"[{hindi}]"
    return f"[{label.upper()}]"