
"""AWAAZ: Context-aware scam text analysis and risk scoring."""

import re
import difflib

FUZZY_THRESHOLD = 0.88
FUZZY_MIN_CHARS = 8
FUZZY_TOKEN_THRESHOLD = 0.80

SCAM_TEXT_THRESHOLD = 20


# --------------------------------------------------
# 1. WEAK INDICATORS
# --------------------------------------------------

WEAK_RULES = {
    "Paisa maangna": (
        5,
        [
            "paisa bhejo", "paise bhejo", "paise bhej dein",
            "paisa bhej dein", "send money", "send me money",
            "need money", "need some money", "payment karo",
            "amount transfer", "transfer karo",
            "processing fee", "registration fee",
            "mujhe paise chahiye", "paise ki zaroorat",
            "money transfer", "bank transfer",
        ],
    ),

    "Urgency": (
        5,
        [
            "urgent", "urgently", "immediately", "right now",
            "jaldi karo", "abhi karo", "foran", "aaj hi",
            "last chance",
        ],
    ),

    "Fear or emergency": (
        5,
        [
            "emergency", "hospital", "accident", "police",
            "problem ho gayi", "mushkil mein hoon",
            "stuck", "urgent help",
        ],
    ),

    "Relative impersonation": (
        5,
        [
            "your son", "your daughter", "main tumhara beta",
            "main tumhari beti", "main tumhara bhai",
            "main tumhari behen", "beta bol raha hoon",
            "beti bol rahi hoon",
        ],
    ),
}


# --------------------------------------------------
# 2. STRONG INDICATORS
# --------------------------------------------------

STRONG_RULES = {
    "Secrecy": (
        20,
        [
            "kisi ko mat batana", "kisi ko nahi batana",
            "secret rakho", "dont tell anyone",
            "don't tell anyone", "keep it secret",
            "chup raho",
        ],
    ),

    "New number or account": (
        15,
        [
            "new number",
            "naya number",
            "naye number",
            "mera number change ho gaya",
            "number change ho gaya hai",
            "number badal gaya hai",
            "new account",
            "naya account",
            "naye account mein",
            "different account",
            "dusre account",
            "is number par",
            "send to this account",
            "account change ho gaya",
        ],
    ),

    "Prize or fee bait": (
        30,
        [
            "processing fee", "registration fee",
            "claim your prize", "pay to receive prize",
            "prize jeet gaye", "inaam jeet gaye",
            "lottery winner", "you have won",
            "you won a prize",
        ],
    ),

    "OTP or verification demand": (
        40,
        [
            "share otp",
            "send otp",
            "tell me otp",
            "otp share karo",
            "otp share karein",
            "otp share kar dein",
            "otp bhejo",
            "otp bhejein",
            "otp batao",
            "otp batayein",
            "otp mujhe bata dein",
            "apna otp share karein",
            "apna otp bhejein",
            "apna otp bata dein",
            "share your otp",
            "send your otp",
            "tell me your otp",
            "please share your otp",
            "send otp now",
            "verification code share",
            "share the code",
            "share verification code",
            "confirmation code bhejo",
            "account details share",
        ],
    ),
}


# --------------------------------------------------
# 3. OTP PROTECTION WARNINGS
# --------------------------------------------------

OTP_WARNING_PATTERNS = [
    r"\botp\b.{0,35}\b(?:mat|nahi|na|never|not|don't|dont)\b",
    r"\b(?:mat|nahi|na|never|not|don't|dont)\b.{0,35}\botp\b",
    r"\botp\b.{0,35}\b(?:batana|share karna|bhejna)\b",
]


# --------------------------------------------------
# 4. URDU SCRIPT SUPPORT
# --------------------------------------------------

URDU_PATTERNS = {
    "Paisa maangna": [
        "مجھے پیسے چاہئیں",
        "مجھے پیسے چاہیے",
        "مجھے پیسوں کی ضرورت ہے",
        "پیسے بھیج دیں",
        "پیسے بھیج دیجیے",
        "پیسے بھیج دیجئے",
        "رقم بھیج دیں",
        "رقم بھیج دیجیے",
        "مجھے رقم چاہیے",
    ],

    "Urgency": [
        "فوری",
        "فوراً",
        "جلدی کریں",
        "ابھی کریں",
        "ابھی",
    ],

    "Fear or emergency": [
        "ہسپتال",
        "حادثہ",
        "ایمرجنسی",
        "پولیس",
        "مشکل میں ہوں",
    ],

    "OTP or verification demand": [
        "او ٹی پی بھیجیں",
        "او ٹی پی بتائیں",
        "او ٹی پی شیئر کریں",
    ],

    "Prize or fee bait": [
        "انعام جیت گئے",
        "پروسیسنگ فیس",
        "رجسٹریشن فیس",
    ],
}


# --------------------------------------------------
# 5. BANK AUTHORITY PATTERNS
# --------------------------------------------------

BANK_AUTHORITY_PATTERNS = [
    "bank security",
    "bank security department",
    "bank security team",
    "banking security department",
    "bank fraud department",
    "bank representative",
    "bank support department",
    "bank customer support",
    "calling from bank",
    "calling from the bank",
    "i am from bank security",
    "i am calling from bank",
    "this is bank security",

    "bank ki taraf se",
    "bank ki security",
    "bank ki security department",
    "bank security department se",
    "bank se call hai",
    "bank se baat kar raha",
    "bank se baat kar rahi",
    "bank ka security department",

    "بینک کی طرف سے",
    "بینک کی جانب سے",
    "بینک سکیورٹی",
    "بینک سیکیورٹی",
    "بینکی سکیوریٹی",
    "بینکنگ سکیورٹی ڈیپارٹمنٹ",
    "بینک سکیورٹی ڈیپارٹمنٹ",
    "بینک کے سکیورٹی ڈیپارٹمنٹ",
    "بینک کی سیکیورٹی ٹیم",

    "بینک کے سپورٹ ڈیپارٹمنٹ",
    "بینک کے سپوڑ ڈیپارٹمنٹ",
    "بینک سپورٹ ڈیپارٹمنٹ",

    "بینک کے سپوڑ",
]


# --------------------------------------------------
# 6. SUSPICIOUS BANK ACTIVITY PATTERNS
# --------------------------------------------------

BANK_ACTIVITY_PATTERNS = [
    "unusual activity",
    "suspicious activity",
    "unusual transaction",
    "suspicious transaction",
    "unauthorized activity",
    "unauthorised activity",
    "unauthorized transaction",
    "unrecognised transaction",
    "unrecognized transaction",
    "unknown transaction",
    "suspicious login",
    "unusual login",
    "suspicious activity detected",
    "unusual activity detected",

    "account mein suspicious activity",
    "account mein unusual activity",
    "account mein koi problem",
    "suspicious activity notice",

    "مشکوک سرگرمی",
    "غیر معمولی سرگرمی",
    "مشکوک ٹرانزیکشن",
    "غیر معمولی ٹرانزیکشن",
    "اکاؤنٹ میں مشکوک سرگرمی",
    "اکاؤنٹ میں غیر معمولی سرگرمی",
    "مشکوک ایکٹیویٹی",
    "سسپیشس ایکٹیویٹی",
    "غیر معمولی ایکٹیویٹی",
    "ایکٹیویٹی دیکھی گئی",
    "ایکٹیویٹی نوٹس کی گئی",
    "ایکٹیویٹی ڈیٹیکٹ ہوئی",
    # Targeted Urdu ASR variations
    "سپیشی سکٹی بیٹی سنوٹس",
    "سپیشی سکٹی",
    "سنوٹس کی گئی",

]


# --------------------------------------------------
# 7. BANK ACCOUNT THREAT PATTERNS
# --------------------------------------------------

BANK_THREAT_PATTERNS = [
    "account will be blocked",
    "account may be blocked",
    "account could be blocked",
    "account will be suspended",
    "account may be suspended",
    "account will be closed",
    "account may be closed",
    "account will be deactivated",

    "account band ho sakta hai",
    "account block ho sakta hai",
    "account band ho jayega",
    "account block ho jayega",
    "account suspend ho jayega",

    "اکاؤنٹ بند ہو سکتا ہے",
    "اکاؤنٹ بلاک ہو سکتا ہے",
    "اکاؤنٹ معطل ہو سکتا ہے",
    "اکاؤنٹ بند کر دیا جائے گا",
    "اکاؤنٹ بلاک کر دیا جائے گا",
]


# --------------------------------------------------
# 8. BANK VERIFICATION PATTERNS
# --------------------------------------------------

BANK_VERIFICATION_PATTERNS = [
    "verify your details",
    "verify details",
    "confirm your details",
    "confirm details",
    "verify your account",
    "confirm your account",
    "verify your information",
    "confirm your information",
    "update your details",
    "provide your details",
    "share your details",
    "reverify",
    "re-verify",
    "verify your identity",
    "confirm your identity",

    "details verify",
    "details ko verify",
    "details confirm",
    "details ko confirm",
    "details reverify",
    "verify karein",
    "verify karen",
    "confirm karein",
    "confirm karen",
    "tasdeeq karein",
    "tasdeeq karen",
    "account verify karne ke liye",
    "account ko verify karne ke liye",

    "تفصیلات کی تصدیق",
    "معلومات کی تصدیق",
    "اپنی تفصیلات کی تصدیق",
    "تفصیلات کنفرم کریں",
    "تفصیلات ویریفائی کریں",
    "ڈیٹیلز کو ویریفائی",
    "ڈیٹیلز کو ریویریفائی",
    "ڈیٹیلز کنفرم",
    "تفصیلات کو کنفرم",
    "تصدیق کریں",
    "ری ویریفائی کریں",
    "ڈیٹیلز کو ریبیگر",
    "ویڈیفائ کرنے کے لیے",
    "ویڈیفائ کرنے کے لئے",
    "ویریفائی کرنے کے لیے",
    "ویریفائی کرنے کے لئے",
    "اکاؤنٹ کو ویریفائی کرنے کے لیے",
    "اکاؤنٹ کو ویریفائی کرنے کے لئے",
    "اگاؤن کو ویڈیفائ کرنے کے لئے",
        # Targeted Urdu ASR variations
    "ڈیٹیلز کو ریبیگر",
    "ریبیگر کریں",
]


# --------------------------------------------------
# 9. ADDITIONAL INFORMATION REQUEST PATTERNS
# --------------------------------------------------

BANK_ADDITIONAL_INFO_PATTERNS = [
    "need more information",
    "need some more information",
    "we need more information",
    "we need some more information",
    "require more information",
    "requires more information",
    "additional information required",
    "need additional information",
    "provide more information",
    "more details required",
    "need more details",
    "some more details",
    "additional details needed",

    "mazeed maloomat darkar",
    "mazeed maloomat chahiye",
    "kuch mazeed maloomat",
    "mazeed tafseel darkar",

    "مزید معلومات",
    "کچھ مزید معلومات",
    "مزید معلومات درکار",
    "ہمیں مزید معلومات درکار",
    "ہمیں کچھ مزید معلومات درکار",
    "کچھ مزید معلومات درکار",
    "مزید معلومات چاہیے",
    "مزید معلومات چاہئیں",
    "مزید تفصیلات درکار",
    "کچھ مزید تفصیلات",
    "مزید مالوماد",
    "کچھ مزید مالوماد",
    "ہمیں کچھ مزید مالوماد درکار",
]


# --------------------------------------------------
# 10. NORMALIZATION HELPERS
# --------------------------------------------------

def normalize(text):
    """Normalize text for matching."""

    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _clean(text):
    """Remove punctuation while preserving Unicode letters."""

    text = normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def _contains(text, phrase):
    """Check a phrase using word boundaries."""

    phrase = _clean(phrase)

    if not phrase:
        return False

    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

    return re.search(pattern, text) is not None


def _is_otp_warning(text):
    """Identify protective warnings about not sharing OTP."""

    return any(
        re.search(pattern, text)
        for pattern in OTP_WARNING_PATTERNS
    )


# --------------------------------------------------
# 11. IMPROVED FUZZY MATCHING
# --------------------------------------------------

def _find_matches(text, phrases):
    """
    Find exact matches first.

    Fuzzy matching also checks individual word similarity
    to reduce false positives.
    """

    matches = []
    text_words = text.split()

    for phrase in phrases:

        clean_phrase = _clean(phrase)

        if not clean_phrase:
            continue

        if _contains(text, clean_phrase):

            matches.append((phrase, clean_phrase, False))
            continue

        if len(clean_phrase.replace(" ", "")) < FUZZY_MIN_CHARS:
            continue

        phrase_words = clean_phrase.split()
        target_size = len(phrase_words)

        found = False

        for size in range(max(1, target_size - 1), target_size + 2):

            for i in range(len(text_words) - size + 1):

                segment_words = text_words[i:i + size]
                segment = " ".join(segment_words)

                similarity = difflib.SequenceMatcher(
                    None,
                    segment,
                    clean_phrase
                ).ratio()

                if similarity < FUZZY_THRESHOLD:
                    continue

                token_support = all(
                    max(
                        difflib.SequenceMatcher(
                            None,
                            phrase_word,
                            segment_word
                        ).ratio()
                        for segment_word in segment_words
                    ) >= FUZZY_TOKEN_THRESHOLD

                    for phrase_word in phrase_words
                )

                if token_support:

                    matches.append((phrase, segment, True))
                    found = True
                    break

            if found:
                break

    return matches


# --------------------------------------------------
# 12. MAIN TEXT DETECTION ENGINE
# --------------------------------------------------

def text_check(text):
    """
    Analyze a transcript for suspicious content.

    Returns:
        score: 0-100
        hits: category, weight, keywords, matched, fuzzy
    """

    cleaned = _clean(text)

    if not cleaned:
        return 0, []

    found = {}

    otp_warning = _is_otp_warning(cleaned)

    all_rules = [
        (name, weight, phrases)
        for name, (weight, phrases) in WEAK_RULES.items()
    ] + [
        (name, weight, phrases)
        for name, (weight, phrases) in STRONG_RULES.items()
    ]

    # A. Existing rule-based detection

    for name, weight, phrases in all_rules:

        if name == "OTP or verification demand" and otp_warning:
            continue

        if name == "Secrecy" and otp_warning:
            continue

        matches = _find_matches(cleaned, phrases)

        if name == "Paisa maangna":

            words = cleaned.split()

            request_words = {
                "چاہیے",
                "چاہئیں",
                "چاہیں",
                "ضرورت",
                "بھیج",
                "بھیجیں",
                "دیں",
                "دیجیے",
                "دیجئے",
                "درکار",
            }

            for i, word in enumerate(words):

                if word.startswith("پیس") or word.startswith("رقم"):

                    nearby_words = words[i + 1:i + 6]

                    if any(w in request_words for w in nearby_words):

                        matches.append(
                            ("Urdu money request context", word, False)
                        )

                        break

        for urdu_phrase in URDU_PATTERNS.get(name, []):

            if _contains(cleaned, _clean(urdu_phrase)):

                matches.append(
                    (urdu_phrase, urdu_phrase, False)
                )

        if not matches:
            continue

        entry = {
            "category": name,
            "weight": weight,
            "keywords": [],
            "matched": [],
            "fuzzy": True,
        }

        for keyword, matched, fuzzy in matches:

            if keyword not in entry["keywords"]:

                entry["keywords"].append(keyword)
                entry["matched"].append(matched)

            entry["fuzzy"] = entry["fuzzy"] and fuzzy

        found[name] = entry

    # B. Bank contextual detection

    def has_any(phrases):

        return any(
            _contains(cleaned, _clean(phrase))
            for phrase in phrases
        )

    has_bank_authority = has_any(BANK_AUTHORITY_PATTERNS)
    has_suspicious_activity = has_any(BANK_ACTIVITY_PATTERNS)
    has_account_threat = has_any(BANK_THREAT_PATTERNS)

    has_verification_request = has_any(
        BANK_VERIFICATION_PATTERNS
    )

    has_additional_info_request = has_any(
        BANK_ADDITIONAL_INFO_PATTERNS
    )

    bank_impersonation_detected = (
        has_bank_authority
        and has_verification_request
        and (
            has_suspicious_activity
            or has_account_threat
        )
    )

    if bank_impersonation_detected:

        bank_evidence = [
            "Bank/security authority claim",
            "Suspicious activity or account threat",
            "Request to verify or confirm details",
        ]

        found["Bank impersonation / verification pressure"] = {
            "category": "Bank impersonation / verification pressure",
            "weight": 25,
            "keywords": bank_evidence,
            "matched": bank_evidence,
            "fuzzy": False,
        }

    elif (
        has_bank_authority
        and has_verification_request
        and has_additional_info_request
    ):

        bank_info_evidence = [
            "Bank/support department identity claim",
            "Account verification request",
            "Additional information requested",
        ]

        found["Bank account verification / information request"] = {
            "category": "Bank account verification / information request",
            "weight": 20,
            "keywords": bank_info_evidence,
            "matched": bank_info_evidence,
            "fuzzy": False,
        }

    # B2. Urdu ASR prize + payment context detection

    words = set(cleaned.split())

    prize_indicators = {
        "لاٹری",
        "لطری",
        "انعام",
        "انامتہ",
        "انعامتہ",
        "lottery",
        "prize",
        "inaam",
    }

    payment_indicators = {
        "فیس",
        "فییدہ",
        "پیمینٹ",
        "پیمنٹ",
        "ادائیگی",
        "payment",
        "fee",
        "fees",
    }

    has_prize_context = bool(
        words.intersection(prize_indicators)
    )

    has_payment_context = bool(
        words.intersection(payment_indicators)
    )

    if has_prize_context and has_payment_context:

        prize_evidence = [
            "Prize or lottery indication detected",
            "Payment or fee demand detected",
        ]

        found["Prize or fee bait"] = {
            "category": "Prize or fee bait",
            "weight": 30,
            "keywords": prize_evidence,
            "matched": prize_evidence,
            "fuzzy": False,
        }

    # C. Final text score

    hits = list(found.values())

    score = min(
        sum(item["weight"] for item in hits),
        100
    )

    return score, hits


# --------------------------------------------------
# 13. COMBINED CAUTION SCORE
# --------------------------------------------------

def combined_risk(fake_score, text_score):

    fake_score = max(
        0.0,
        min(1.0, float(fake_score))
    )

    text_score = max(
        0.0,
        min(100.0, float(text_score))
    )

    # This is a combined caution score, NOT a scam probability.
    risk = (
        0.35 * fake_score * 100
        + 0.65 * text_score
    )

    return round(max(0.0, min(100.0, risk)))


# --------------------------------------------------
# 14. FINAL ALERT LEVEL
# --------------------------------------------------

def final_level(fake_score, text_score):

    fake_score = max(
        0.0,
        min(1.0, float(fake_score))
    )

    text_score = max(
        0.0,
        min(100.0, float(text_score))
    )

    # RED: Strong scam-content indicators.
    if text_score >= 60:
        return "RED"

    # YELLOW: Synthetic voice indicators OR content caution.
    if fake_score >= 0.5 or text_score >= 5:
        return "YELLOW"

    return "GREEN"


# --------------------------------------------------
# 15. SEPARATE SECURITY ASSESSMENT
# --------------------------------------------------

def security_assessment(fake_score, text_score):

    fake_score = max(
        0.0,
        min(1.0, float(fake_score))
    )

    text_score = max(
        0.0,
        min(100.0, float(text_score))
    )

    ai_voice_alert = fake_score >= 0.5
    scam_warning = text_score >= SCAM_TEXT_THRESHOLD
    verification_advice = text_score >= 5

    if ai_voice_alert and scam_warning:

        message = (
            "AI-generated voice indicators and suspicious "
            "conversation patterns detected. Verify both "
            "the speaker's identity and the request independently."
        )

    elif ai_voice_alert:

        message = (
            "AI-generated or cloned-voice indicators detected. "
            "The conversation may be normal, but independently "
            "verify the speaker's identity."
        )

    elif scam_warning:

        message = (
            "Suspicious conversation patterns detected. "
            "Verify the person's identity and request through "
            "an independent trusted channel before acting."
        )

    elif verification_advice:

        message = (
            "A potentially sensitive request was detected. "
            "Verify the request independently before taking action."
        )

    else:

        message = (
            "No strong AI-voice or scam-content indicators detected. "
            "This does not guarantee that the audio is safe."
        )

    return {
        "ai_voice_alert": ai_voice_alert,
        "scam_warning": scam_warning,
        "verification_advice": verification_advice,
        "message": message,
    }
