"""Code-mixed lexicons for Hinglish, Kanglish (romanised Kannada) and English.

Patterns are intentionally keyword-level rather than full-phrase so they stay
robust to ASR word drops (for example "link mat  main khud kar dunga"). Each
category maps to a business concept in the problem statement:

* HEDGE / CONDITIONAL / VAGUE_DATE / SPECIFIC_DATE  -> "what was said"
* PRESSURE / AGENT_PROPOSES_DATE / PART_PAYMENT_ASK -> "how the conversation went"
* HARDSHIP_*                                         -> infeasibility, never escape
* THIRD_PARTY                                        -> promise on borrower's behalf
* PRIOR_BREAK_EXCUSE / AGENT_MENTIONS_PRIOR_BREAK    -> repeat promiser

The lexicon is versioned so that every score can be traced to the exact rules
that produced it (auditability requirement).
"""
from __future__ import annotations

import re
from typing import Dict, Pattern

LEXICON_VERSION = "2026.10.1"


def _c(*alts: str) -> Pattern:
    return re.compile(r"(?:" + "|".join(alts) + r")", re.IGNORECASE)


# --------------------------------------------------------------------------- customer
CUSTOMER: Dict[str, Pattern] = {
    "self_identify": _c(r"\bmain hi\b", r"\bbol raha\b", r"\bspeaking\b", r"\bnaane\b", r"\bhaudu\b",
                        r"\bkya baat hai\b", r"\benu vishaya\b", r"\bwhat is it about\b", r"\btell me\b",
                        r"\bboliye\b", r"\bbolo\b", r"\bheli\b"),
    "third_party": _c(r"\bwife\b", r"\bmother\b", r"\bmummy\b", r"\bbhai\b", r"\bbeta\b", r"\bbrother\b",
                      r"\bhusband\b", r"\bhendathi\b", r"\bamma\b(?!.*hospital)", r"\bthamma\b", r"\bappa\b",
                      r"\bfather\b", r"\bpapa\b", r"\bunki\b", r"\bunka\b", r"\bunko\b", r"\bavra\b",
                      r"\bavrige\b", r"\bavnu\b", r"\bavru\b", r"\bhe is not\b", r"\bshe is\b", r"\btell him\b",
                      r"\blet (?:her|him) know\b", r"\bbata dung", r"\bbol dung", r"\bkeh dunga\b",
                      r"\bheltini\b", r"\bmaadtaare\b", r"\bkar denge\b", r"\bghar pe nahi\b", r"\bmaneli illa\b",
                      r"\bkaam pe\b", r"\bat work\b", r"\bhorage hogid", r"\bkelsakke\b", r"\bbahar gaye\b",
                      r"\bso rahe\b", r"\bgone out\b"),
    "wrong_party": _c(r"wrong number", r"koi nahi hai", r"naya number", r"nahi jaanta", r"no such person"),
    "hardship_salary": _c(r"salary.{0,25}(?:nahi|late|hold|bandilla|not come|aagide)", r"salary is late",
                          r"salary has not", r"salary late", r"salary hold"),
    "hardship_no_funds": _c(r"paise nahi", r"duddu illa", r"don'?t have money", r"no money",
                            r"balance nahi", r"balance irlilla", r"no balance"),
    "hardship_business": _c(r"business.{0,20}(?:down|slow)", r"sale hi nahi", r"vyapara illa", r"no sales",
                            r"tumba down", r"party ke paas", r"party inda payment", r"client has not paid",
                            r"phasa hua"),
    "hardship_medical": _c(r"hospital", r"medical", r"admit"),
    "hardship_job_loss": _c(r"job chali", r"company band", r"kelsa illa", r"company close", r"lost my job",
                            r"kaam nahi hai", r"no work"),
    "hedge": _c(r"dekhta", r"dekhenge", r"\btry\b", r"koshish", r"shayad", r"\bmaybe\b", r"let me see",
                r"nodtini", r"nodona", r"prayatna", r"kuch karta", r"i will see", r"\bnodi\b", r"\bperhaps\b"),
    "conditional": _c(r"\btoh\b", r"\bagar\b", r"\bif\b", r"kottre", r"bandre", r"aane ke baad", r"aayi toh",
                      r"aaye toh", r"\bonce\b"),
    "firm_commitment": _c(r"pakka", r"definitely", r"for sure", r"\bconfirm", r"\bsure\b(?! i will let)",
                          r"guarantee", r"\bkhandita\b", r"poora .{0,15}clear", r"clear the whole",
                          r"full rs", r"whole rs"),
    "specific_date": _c(r"\b\d{1,2}\s*(?:tareekh|tarikh|ne tareekhu|th\b|st\b|nd\b|rd\b)",
                        r"\b\d{1,2}\s*(?:april|may|june|july)\b", r"\bkal\b", r"\bnaale\b", r"tomorrow",
                        r"parson", r"day after", r"\baaj\b", r"\bivattu\b", r"\btoday\b"),
    "vague_date": _c(r"agle hafte", r"next week", r"mundina vaara", r"month end", r"mahine ke end",
                     r"konege", r"end of (?:the )?month", r"salary aane ke baad", r"kuch din", r"few days",
                     r"jaldi", r"soon"),
    "salary_day_info": _c(r"har mahine", r"every month", r"prati tingalu", r"tareekh ko aati",
                          r"comes on the", r"tareekhu barutte", r"tak aati hai"),
    "partial_offer": _c(r"baaki", r"not the full", r"poora abhi nahi", r"poora eega aagalla", r"rest later",
                        r"the rest", r"amele"),
    "full_amount": _c(r"\bpoora\b", r"\bfull\b", r"\bwhole\b", r"clear kar"),
    "closing_escape": _c(r"baad mein", r"\bbusy\b", r"meeting", r"rakhta", r"phone idtini", r"have to go",
                         r"call later", r"call me later", r"amele (?:call|maataad)", r"shaam ko call",
                         r"call tomorrow", r"\boutside\b", r"horage idini", r"naale call"),
    "passive_agree": _c(r"^\W*(?:ok|okay|haan|han|theek|thik|sari|aytu|yes|ji|yeah|fine|alright|hmm|achha)"
                        r"(?:\W+(?:ok|okay|haan|theek hai|sari|aytu|yes|ji|bye|thank you|thanks|hai))*\W*$",
                        r"ok ok kar", r"ho jayega\W*$", r"aagutte\W*$", r"theek hai theek hai", r"sari sari",
                        r"yes yes", r"okay okay"),
    "link_accept": _c(r"bhej do", r"bhej dijiye", r"link kalsi", r"send the link", r"link dijiye", r"bhej\b"),
    "link_refuse": _c(r"mat bhejo", r"link beda", r"don'?t send", r"branch", r"khud kar dunga\W*$",
                      r"do it myself"),
    "prior_break_excuse": _c(r"alag baat", r"that was different", r"adu bere", r"dhyan se nikal", r"forgot",
                             r"marethu", r"problem ho gayi", r"problem aagittu", r"there was a problem",
                             r"balance nahi tha", r"balance irlilla"),
    "refusal": _c(r"abhi nahi ho payega", r"not possible", r"eega aagalla", r"nahi ho payega",
                  r"nahi kar sakta", r"cannot pay", r"can'?t pay", r"won'?t pay"),
    "dispute": _c(r"already paid", r"pay kar diya", r"kattidini", r"dispute", r"galat"),
    "pay_now": _c(r"abhi (?:\S+ )?(?:rupaye )?kar sakta", r"eega .{0,12}kodbahudu", r"pay rs .{0,10}now",
                  r"i can pay rs", r"abhi paise hain"),
    "self_agency": _c(r"main khud", r"naane maadtini", r"i will do it myself on", r"myself on"),
    "verbal_commit": _c(r"kar dunga", r"kar denge", r"de dunga", r"ho jayega", r"maadtini", r"kodtini",
                        r"aagutte", r"i will (?:pay|do)", r"will be done", r"kattini"),
}

# --------------------------------------------------------------------------- agent / bot
AGENT: Dict[str, Pattern] = {
    "open_date_question": _c(r"kab tak", r"yaavaga", r"by when", r"when will", r"kis tareekh",
                             r"which date", r"dinaankadolage", r"payment kab"),
    "proposes_date": _c(r"ho jayega na", r"will \d{1,2}\s*\w+ work", r"can you pay by", r"tak kar denge",
                        r"ge aagutta", r"olage maadtira", r"\bpossible\?"),
    "pressure": _c(r"cibil", r"credit score", r"escalat", r"last reminder", r"case aage", r"case munde",
                   r"case moves", r"karna padega", r"maadle beku", r"need to pay today", r"legal",
                   r"haalaagtide", r"warna", r"otherwise"),
    "part_payment_ask": _c(r"kam se kam", r"at least", r"kanishta", r"kar sakte hain kya",
                           r"can you do rs", r"today and the rest", r"kar dijiye\W+baaki",
                           r"maadi\W+baaki", r"maadtira\?"),
    "salary_question": _c(r"salary"),
    "mentions_prior_break": _c(r"pichli baar", r"pehle bhi", r"previously", r"last time",
                               r"hinde kooda", r"miss ho gaya"),
    "confirm_ptp": _c(r"confirm", r"noting a promise", r"record kar", r"ptp note", r"recording it",
                      r"note kar", r"make a note", r"note maadkol", r"ptp note kar"),
    "third_party_message": _c(r"message de", r"bata dijiye", r"pass a message", r"tell him", r"tell her",
                              r"message kodtira", r"avrige heli", r"call aaya tha", r"call came from"),
    "link_offer": _c(r"link"),
    "today_ask": _c(r"aaj payment", r"ivattu payment", r"can you pay today", r"aaj hi"),
    "empathy": _c(r"samajh sakta", r"understand", r"sorry to hear", r"restructur", r"option",
                  r"arthavaagutte"),
}

FILLER = re.compile(r"\b(?:umm+|uh+|hmm+|matlab|achha|like|so)\b", re.IGNORECASE)
INAUDIBLE = re.compile(r"\[inaudible\]", re.IGNORECASE)
AMOUNT = re.compile(r"(?:rs\.?\s*|₹\s*)?(\d{1,3}(?:,\d{2,3})+|\d{3,7})(?:\s*(?:rupees|rupaye|rs))?", re.IGNORECASE)
DAY_NUM = re.compile(r"\b(\d{1,2})\s*(?:tareekh|tarikh|ne tareekhu|th\b|st\b|nd\b|rd\b|april|may|june|july)",
                     re.IGNORECASE)
MONTHS = {"april": 4, "may": 5, "june": 6, "july": 7}
DATE_WITH_MONTH = re.compile(r"\b(\d{1,2})\s*(april|may|june|july)\b", re.IGNORECASE)


def extract_amounts(text: str) -> list:
    vals = []
    for m in AMOUNT.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if v >= 50:
            vals.append(v)
    return vals


def extract_day_numbers(text: str) -> list:
    return [int(m.group(1)) for m in DAY_NUM.finditer(text) if 1 <= int(m.group(1)) <= 31]
