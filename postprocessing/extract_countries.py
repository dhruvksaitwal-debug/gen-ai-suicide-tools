"""Extracts distinct countries mentioned in the pipeline's free-text 'location' field
across the full corpus, using pycountry's authoritative name database plus a manual alias
map for common informal variants (USA, UK, South Korea, etc.) that don't exact-match."""
import glob
import re
import pandas as pd
import pycountry

ALIASES = {
    "usa": "United States", "us": "United States", "u.s.": "United States",
    "u.s.a.": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "south korea": "Korea, Republic of", "republic of korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "mainland china": "China", "china": "China", "hong kong": "Hong Kong",
    "taiwan": "Taiwan, Province of China",
    "russia": "Russian Federation",
    "iran": "Iran, Islamic Republic of",
    "vietnam": "Viet Nam",
    "south africa": "South Africa",
    "czech republic": "Czechia", "czechia": "Czechia",
    "netherlands": "Netherlands", "the netherlands": "Netherlands", "holland": "Netherlands",
    "ivory coast": "Cote d'Ivoire",
    "bolivia": "Bolivia, Plurinational State of",
    "tanzania": "Tanzania, United Republic of",
    "syria": "Syrian Arab Republic",
    "laos": "Lao People's Democratic Republic",
    "moldova": "Moldova, Republic of",
    "brunei": "Brunei Darussalam",
    "palestine": "Palestine, State of",
    "cape verde": "Cabo Verde",
    "swaziland": "Eswatini",
    "burma": "Myanmar",
    "macedonia": "North Macedonia",
    "uae": "United Arab Emirates", "u.a.e.": "United Arab Emirates",
    "venezuela": "Venezuela, Bolivarian Republic of",
    "micronesia": "Micronesia, Federated States of",
    # ISO renamed Turkey -> Turkiye in 2022; pycountry's default name no longer matches
    # the word actually used throughout the suicide-research literature.
    "turkey": "Turkey",
    "turkiye": "Turkey",
    # Bare "Korea" (no North/South qualifier) is South Korea in essentially all suicide
    # research literature -- North Korea publishes negligible biomedical suicide research.
    "korea": "Korea, Republic of",
    "seoul": "Korea, Republic of",
    "samsung medical center": "Korea, Republic of",
    "trondheim": "Norway",
    "university of alberta": "Canada",
    "kent": "United Kingdom", "south-east london": "United Kingdom",
    "new delhi": "India", "vijayawada": "India",
}

# US states/territories and a handful of major cities mentioned without an explicit
# "USA"/"United States" qualifier -- the article's institutional affiliation makes the
# country unambiguous even though the word "country" is never used.
US_STATE_OR_CITY = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "chicago", "pittsburgh", "st. louis", "saint louis", "san diego", "miami",
    "new york city", "new york state",
}


def build_country_lookup():
    lookup = {}
    for c in pycountry.countries:
        names = {c.name}
        if hasattr(c, "official_name"):
            names.add(c.official_name)
        if hasattr(c, "common_name"):
            names.add(c.common_name)
        for n in names:
            lookup[n.lower()] = c.name
    for alias, canonical in ALIASES.items():
        lookup[alias] = canonical
    return lookup


LOOKUP = build_country_lookup()
# Sort longer names first so "United States" matches before a shorter substring collision.
SORTED_NAMES = sorted(LOOKUP.keys(), key=len, reverse=True)


def extract_countries(text: str) -> set:
    text_l = text.lower()
    # Normalize punctuation that separates a country list.
    text_l = re.sub(r"[;/]", ",", text_l)
    found = set()
    for name in SORTED_NAMES:
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text_l):
            found.add(LOOKUP[name])
    for us_place in US_STATE_OR_CITY:
        if re.search(r"\b" + re.escape(us_place) + r"\b", text_l):
            found.add("United States")
            break
    return found


def main():
    all_files = glob.glob("test_results/Done/Batch-1/*.csv") + glob.glob("test_results/Done/Batch-2/*.csv")
    locations = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if "question" not in df.columns:
                continue
            row = df[df["question"] == "location"]
            if not row.empty:
                val = str(row.iloc[0]["answer"]).strip()
                if val and val.lower() != "nan":
                    locations.append(val)
        except Exception:
            pass

    all_countries = set()
    unmatched = []
    for loc in locations:
        countries = extract_countries(loc)
        if countries:
            all_countries.update(countries)
        else:
            unmatched.append(loc)

    print(f"Total location field values: {len(locations)}")
    print(f"Distinct countries identified: {len(all_countries)}")
    print(f"Location values with no country match: {len(unmatched)}")

    with open("postprocessing/_countries_found.txt", "w", encoding="utf-8") as f:
        for c in sorted(all_countries):
            f.write(c + "\n")

    with open("postprocessing/_locations_unmatched.txt", "w", encoding="utf-8") as f:
        for u in unmatched:
            f.write(u + "\n")

    print("Wrote postprocessing/_countries_found.txt and _locations_unmatched.txt")


if __name__ == "__main__":
    main()
