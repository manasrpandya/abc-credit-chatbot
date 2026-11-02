"""Reads what people actually type: 35k, 1.2 lakh, rs. 45,000, two wheeler names."""
import difflib
import re

word_numbers = {"zero": 0, "nil": 0, "none": 0, "nothing": 0}
multipliers = {"k": 1e3, "thousand": 1e3, "l": 1e5, "lac": 1e5, "lakh": 1e5, "lakhs": 1e5,
               "lacs": 1e5, "cr": 1e7, "crore": 1e7}
amount_rx = re.compile(r"(\d+(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?|l|cr|crore)?\b")


def parse_amount(text):
    s = str(text).lower().replace(",", "").replace("₹", " ")
    s = re.sub(r"\brs\.?", " ", s).replace("/-", " ")
    for w, v in word_numbers.items():
        if re.fullmatch(rf"\s*{w}\s*", s):
            return float(v)
    hits = amount_rx.findall(s)
    if len(hits) != 1:
        return None
    number, unit = hits[0]
    return float(number) * multipliers.get(unit, 1.0)


def parse_percent(text):
    found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(%|percent)\s*", str(text).lower())
    return float(found.group(1)) if found else None


def parse_age(text):
    hits = re.findall(r"\b(\d{1,3})\b", str(text).lower())
    return int(hits[0]) if len(hits) == 1 else None


def parse_pincode(text):
    s = re.sub(r"\s+", "", str(text))
    hits = re.findall(r"(?<!\d)\d{6}(?!\d)", s)
    return hits[0] if len(hits) == 1 else None


yes_words = {"yes", "y", "yeah", "yep", "haan", "ha", "ok", "okay", "sure", "correct", "right", "active", "true"}
no_words = {"no", "n", "nope", "nahi", "nahin", "none", "never", "false", "nil"}


def parse_yes_no(text):
    words = re.sub(r"[^a-z ]", " ", str(text).lower()).split()
    if not words:
        return None
    if words[0] in no_words:
        return False
    if words[0] in yes_words:
        return True
    return None


employment_words = {
    "SAL": ["salaried", "salary", "job", "employee", "employed", "service", "naukri"],
    "SEP": ["self employed", "self-employed", "business", "shop", "freelance", "contractor"],
    "AGR": ["farmer", "farming", "agriculture", "agri", "kisan"],
    "STU": ["student", "studying", "college"],
    "PEN": ["pension", "pensioner", "retired"],
    "NREGI": ["irregular", "daily wage", "gig", "labour", "labor", "non regular", "seasonal"],
    "NPP": ["probation", "contract job", "temporary", "non permanent"],
    "NONEARNMEM": ["homemaker", "housewife", "not earning", "non earning", "unemployed", "dependent"],
}


def parse_employment(text):
    s = str(text).lower().strip()
    if s.upper() in employment_words:
        return s.upper()
    best = None
    for code, words in employment_words.items():
        for w in words:
            if w in s and (best is None or len(w) > best[1]):
                best = (code, len(w))
    return best[0] if best else None


def clean_model_name(name):
    s = name.replace("BSVI", "").replace(" - ", " ")
    s = re.sub(r"\bU\d{3}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    out = s.title()
    for a, b in (("Tvs", "TVS"), ("Rtr", "RTR"), ("Rr ", "RR "), ("Iqube", "iQube"), ("Xl ", "XL "),
                 (" Ev", " EV"), ("Sxc", "SXC"), ("Abs", "ABS"), (" Xt", " XT"), (" Xp", " XP"), ("Zx", "ZX")):
        out = out.replace(a, b)
    return out


filler = r"\b(tvs|bike|scooter|the|i|want|to|buy|a|an|new|model|looking|at|for|is|it|its)\b"


def match_vehicle(text, catalogue):
    """catalogue: canonical name -> display name. Best matches first, empty if nothing is close."""
    q = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    words = re.sub(filler, " ", q).split()
    if not words:
        return []
    scored = []
    for canon, display in catalogue.items():
        have = re.sub(r"[^a-z0-9 ]", " ", (canon + " " + display).lower()).split()
        hits = sum(1 for w in words if any(w == h or (len(w) > 3 and h.startswith(w)) for h in have))
        ratio = difflib.SequenceMatcher(None, " ".join(words), display.lower()).ratio()
        scored.append((hits / len(words) + ratio / 2, canon))
    scored.sort(reverse=True)
    return [c for s, c in scored[:6] if s >= 0.75]
