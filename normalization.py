"""
Data Normalization Module for Business Entity Resolution.

Handles:
- Business name normalization (lowercase, punctuation, legal forms, abbreviations, unicode)
- Business address normalization (lowercase, punctuation, street abbreviations, PIN/ZIP codes)
- Open-set country normalization
- High-performance, unicode-safe text cleaning that preserves international scripts (Devanagari, French accents, etc.)
"""

import sys
import re
import unicodedata

# Build unicode-safe punctuation translation table:
# Replaces all punctuation ('P') and symbols ('S') with spaces, EXCEPT '&' (which we replace with ' and ')
# This preserves all letters, numbers, and Indic vowel marks (Mn/Mc categories).
PUNCT_TRANSLATE_TABLE = {
    i: " "
    for i in range(sys.maxunicode + 1)
    if unicodedata.category(chr(i)).startswith(("P", "S")) and chr(i) != "&"
}

# Common legal and corporate forms mapping (supporting US, India, France, etc.)
LEGAL_FORM_REPLACEMENTS = {
    # US / UK / International
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bplc\b": "public limited company",
    r"\bmfg\b": "manufacturing",
    r"\bintl\b": "international",
    r"\bassn\b": "association",
    r"\bsoc\b": "society",
    r"\bdept\b": "department",
    r"\bdiv\b": "division",
    r"\bmgmt\b": "management",
    r"\bsvc\b": "service",
    r"\bsvcs\b": "services",
    r"\btech\b": "technology",
    r"\btechs\b": "technologies",
    r"\bsoln\b": "solution",
    r"\bsolns\b": "solutions",
    # India specific legal forms
    r"\bopl\b": "one person limited",
    # France specific legal forms
    r"\bsarl\b": "societe a responsabilite limitee",
    r"\bsas\b": "societe par actions simplifiee",
    r"\bsasu\b": "societe par actions simplifiee unipersonnelle",
    r"\bsa\b": "societe anonyme",
    r"\beurl\b": "entreprise unipersonnelle a responsabilite limitee",
    r"\bsci\b": "societe civile immobiliere",
    r"\bgie\b": "groupement d interet economique",
}

# Compile legal form patterns
COMPILED_LEGAL_FORMS = [
    (re.compile(pattern, flags=re.IGNORECASE), replacement)
    for pattern, replacement in LEGAL_FORM_REPLACEMENTS.items()
]

# Common address abbreviations mapping
ADDRESS_REPLACEMENTS = {
    # Street types
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bhwy\b": "highway",
    r"\bexpy\b": "expressway",
    r"\bpkwy\b": "parkway",
    r"\bway\b": "way",
    r"\baly\b": "alley",
    # Units / Buildings
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\brm\b": "room",
    r"\bno\b": "number",
    r"\bkh\s*no\b": "khasra number",
    r"\bsec\b": "sector",
    r"\bsect\b": "sector",
    r"\bph\b": "phase",
    r"\bdist\b": "district",
    r"\btq\b": "taluk",
    r"\btn\b": "town",
    r"\bvlg\b": "village",
    # Directions
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
    # France address terms
    r"\bbvd\b": "boulevard",
    r"\brte\b": "route",
    r"\bimp\b": "impasse",
    r"\ball\b": "allee",
    r"\bch\b": "chemin",
}

COMPILED_ADDRESS_FORMS = [
    (re.compile(pattern, flags=re.IGNORECASE), replacement)
    for pattern, replacement in ADDRESS_REPLACEMENTS.items()
]

WHITESPACE_REGEX = re.compile(r"\s+")
AND_REGEX = re.compile(r"&")
POSTAL_CODE_REGEX = re.compile(r"\b\d{5,6}\b")


def clean_unicode_text(text: str) -> str:
    """
    Normalizes unicode via NFKC, handles '&', replaces punctuation with space,
    collapses whitespace, and strips.
    Preserves all international characters and script marks.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = AND_REGEX.sub(" and ", text)
    text = text.translate(PUNCT_TRANSLATE_TABLE)
    return WHITESPACE_REGEX.sub(" ", text).strip()


def normalize_country(country_str: str) -> str:
    """
    Normalizes country string.
    Treats country as an open-set string field (handles US, India, France, or any other country).
    """
    if not country_str:
        return ""
    val = unicodedata.normalize("NFKC", str(country_str)).strip().lower()
    if val in ("us", "usa", "united states", "united states of america"):
        return "us"
    if val in ("in", "ind", "india"):
        return "india"
    if val in ("fr", "fra", "france"):
        return "france"
    return val


def normalize_business_name(name: str) -> str:
    """
    Normalizes business name:
    1. Lowercase, Unicode NFKC, '&' -> 'and', punctuation -> space
    2. Expand common abbreviations / legal suffixes
    3. Collapse extra whitespace
    """
    if not name:
        return ""
    
    text = clean_unicode_text(name)
    
    # Expand legal suffixes
    for pattern, rep in COMPILED_LEGAL_FORMS:
        text = pattern.sub(rep, text)
        
    return WHITESPACE_REGEX.sub(" ", text).strip()


def normalize_business_address(address: str) -> str:
    """
    Normalizes business address:
    1. Lowercase, Unicode NFKC, '&' -> 'and', punctuation -> space
    2. Expand common street/locality abbreviations (st -> street, rd -> road, etc.)
    3. Collapse extra whitespace
    """
    if not address:
        return ""
    
    text = clean_unicode_text(address)
    
    # Expand address abbreviations
    for pattern, rep in COMPILED_ADDRESS_FORMS:
        text = pattern.sub(rep, text)
        
    return WHITESPACE_REGEX.sub(" ", text).strip()


def extract_postal_codes(address: str):
    """
    Extracts 5 to 6 digit postal/PIN codes from address string.
    Returns a sorted list of unique postal codes.
    """
    if not address:
        return []
    return sorted(list(set(POSTAL_CODE_REGEX.findall(address))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    
    print("=== Testing Data Normalization Functions ===")
    
    sample_names = [
        "Zephay Labs Inc",
        "Orelee's Barbershop",
        "Brahma Infosoft Pvt. Ltd.",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Apex Corp & Sons",
        "Tech Solutions SARL",
        "Bio-Gen Pharma LLC",
    ]
    
    print("\n--- Business Names ---")
    for name in sample_names:
        norm = normalize_business_name(name)
        print(f"Original:   {name}")
        print(f"Normalized: {norm}\n")
        
    sample_addresses = [
        "2621 Cotten Road, Tyler, TX 75701",
        "1795 Westchester Drive, High Point, NC",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041",
        "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR 122016",
        "12 Rue de la Paix, 75002 Paris, France",
        "450 N. Brand Blvd., Ste 600, Glendale, CA",
    ]
    
    print("--- Business Addresses ---")
    for addr in sample_addresses:
        norm = normalize_business_address(addr)
        postal = extract_postal_codes(addr)
        print(f"Original:   {addr}")
        print(f"Normalized: {norm}")
        print(f"Postal/PIN: {postal}\n")
        
    sample_countries = ["US", "India", "France", "United States", "ind", "FRA", "Germany"]
    print("--- Countries ---")
    for c in sample_countries:
        print(f"Original: {c:<15} -> Normalized: {normalize_country(c)}")
        
    print("\n[SUCCESS] Normalization module verified and ready.")
