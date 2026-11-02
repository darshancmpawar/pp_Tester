"""
Client Logic — every client's tunable configuration in one file.

ARCHITECTURE:
  BaseLogic           — the contract + the default (Tekion-like) values
  Per-client classes  — override only what differs
  LOGIC_REGISTRY      — maps client key → logic class

TO ADD A CLIENT WITH THE SAME LOGIC:   add 1 line to LOGIC_REGISTRY
TO ADD A CLIENT WITH DIFFERENT LOGIC:  subclass, override, add to LOGIC_REGISTRY


═══ UNITS AND VOCABULARY (read this before touching any number) ═══

  MG   Meal Group — headcount. How many people eat that meal.
  PP   Per-Pax quantity — kilograms of one dish per person.
       Typical range 0.03 (rasam) to 0.22 (biryani).
  Total Qty = Client PP × Client MG     — what the client is billed for (kg)
  Vendor PP = Client PP + bump          — what the kitchen cooks per head (kg)
  Vendor MG = Total Qty / Vendor PP     — headcount implied by the vendor's PP
  Ordered Qty = Vendor PP × Vendor MG   — what the kitchen actually prepares (kg)

  The vendor plan exists because the kitchen cooks a little more per head than
  the client is billed for, and therefore serves a smaller "effective" MG for
  the same total weight. Every tunable below moves one of two dials:
    • how much extra goes into each portion   (the *bump* family)
    • how far the vendor headcount is trimmed (the *adjust_vendor_mg* family)


═══ WHERE THE NUMBERS COME FROM ═══

  These are operational constants fitted to each site's observed wastage, not
  model outputs. They were derived from each client's historical dataset
  (the same .xlsx the model trains on) and are reviewed when a site's volume
  shifts. Raising a bump or lifting a floor means more food cooked and more
  wastage; lowering it risks running out. Change one dial at a time and
  compare against that client's wastage sheet for the following week.
"""
from abc import ABC, abstractmethod
from ml_core import norm

# ═══════════════════════════════════════════════════════════════════════
# Base contract — also the default (Tekion-shaped) parameter set
# ═══════════════════════════════════════════════════════════════════════

class BaseLogic(ABC):

    # ── Schema: what the model trains on and what the UI shows ──────────

    @property
    @abstractmethod
    def encoder_columns(self):
        """Dataset columns that become model features. One LabelEncoder and one
        Embedding layer per entry, so changing this list changes the model shape
        and REQUIRES retraining (scripts/train.py --force)."""

    @property
    @abstractmethod
    def fixed_categories(self):
        """The menu-item input boxes the UI renders, in order. Display spelling —
        canonicalize_category() maps these to the model's internal names."""

    @property
    @abstractmethod
    def star_categories(self):
        """Premium categories (canonical lowercase). They carry the menu: vendor
        MG is averaged per group with star split out from the rest, and the
        aggressive plan bumps only these (plus non-veg). Moving a category in or
        out of this set shifts vendor MG for the whole day, not just that row."""

    # ── Feature switches ────────────────────────────────────────────────

    # True  → user chooses veg/non-veg day with a toggle (Tekion).
    # False → the non-veg section is always shown, sharing the client MG.
    @property
    def has_nonveg_toggle(self): return True

    # How many non-veg input rows to render. Only Stripe uses 2.
    @property
    def nonveg_item_count(self): return 1

    @property
    def has_vendor_plan(self): return True
    @property
    def has_aggressive_plan(self): return True
    @property
    def has_special_day(self): return True

    # False = multiplier-only client with no ML artifacts (Toasttab). Must match
    # has_embeddings in client_database.py — app.py errors out if they disagree.
    @property
    def has_embeddings(self): return True

    # ── TUNING: portion sizes ───────────────────────────────────────────

    # kg/person. Salad is a fixed serving, never predicted — a standard 60 g
    # side. Model predictions for salad were noisy because portioning is by
    # hand, so this bypasses the model entirely.
    @property
    def salad_per_pax(self): return 0.06

    # Headcount pre-filled in the MG boxes. Set to the client's typical daily
    # footfall so the operator usually only tweaks it. Cosmetic — it never
    # enters a formula, it is only the starting value in the UI.
    @property
    def default_mg(self): return 300

    # ── TUNING: vendor bumps (extra kg per head above the client PP) ─────

    # kg/person added to every dish's Client PP to get Vendor PP. 10 g/head
    # covers serving spillage and the last-in-queue margin.
    # Raise → more cooked, more wastage. Lower → risk of shortfall.
    @property
    def default_bump(self): return 0.01

    # kg/person for non-veg biryani on a non-veg day (replaces default_bump).
    # 3.5× the normal bump because biryani is the day's hero dish: uptake runs
    # well above forecast and a shortfall is the most visible failure.
    @property
    def biryani_bump(self): return 0.035

    # Safety floor, as a fraction of Client MG. The combined vendor MG
    # (veg + non-veg) may not fall below 82.5% of what the client is paying
    # for, whatever the averages say — this catches a menu of unusually heavy
    # dishes dragging the implied headcount down. 0.0 disables the floor
    # (used by the 3-group and day-based clients, whose formulas trim less).
    @property
    def vendor_floor_ratio(self): return 0.825

    # Which formula planner.vendor_plan() dispatches to:
    #   "tekion_2group"    veg MG and non-veg MG tracked separately
    #   "3group"           one blended MG across non-veg / star / rest
    #   "day_based"        MG derived from the weekday reduction table
    #   "toasttab_formula" no plan; MG arithmetic only
    #   None               no vendor plan
    @property
    def vendor_mg_method(self): return "tekion_2group"

    # True → veg and non-veg headcounts are split and shown as two chips
    # (Tekion bills them separately). False → one headcount for everyone.
    @property
    def has_separate_nonveg_mg(self): return True

    # Optional {canonical category: kg/person}. When set, the CLIENT plan
    # ignores model predictions and bills these contracted portions instead.
    # The vendor plan still uses the model — only billing is fixed.
    @property
    def fixed_pp_map(self): return None

    # UI label → model category, for sites that split one category across
    # counters, e.g. {"North Veg dry": "Veg dry"}.
    @property
    def category_display_map(self): return {}

    # ── Canonical spelling maps ─────────────────────────────────────────
    # Datasets are hand-maintained, so the same category arrives spelled many
    # ways. These collapse the variants to the one token the encoders were
    # fitted on. A missing alias silently becomes an unseen value at predict
    # time and falls into the encoder's "other" bucket — add aliases freely.

    @property
    def category_map(self): return {}
    @property
    def day_type_map(self): return {"regular":"regular","previous day of holiday":"previous day of holiday","next day of holiday":"next day of holiday","holiday":"holiday"}
    @property
    def holiday_type_map(self): return {"n/a":"not applicable","na":"not applicable","not applicable":"not applicable","non-important holiday":"non-important holiday","important holiday":"important holiday","compulsory holiday":"compulsory holiday"}
    @property
    def meal_day_map(self): return {"veg":"veg","nonveg":"nonveg","non-veg":"nonveg","non veg":"nonveg"}

    def canonicalize_category(self,s): n=norm(s); return self.category_map.get(n,n)
    def canonicalize_day_type(self,s): n=norm(s); return self.day_type_map.get(n,n)
    def canonicalize_holiday_type(self,s): n=norm(s); return self.holiday_type_map.get(n,n)
    def canonicalize_meal_day(self,s): n=norm(s); return self.meal_day_map.get(n,n)

    # ── TUNING: vendor MG tier adjustment ───────────────────────────────
    # Trim the raw averaged vendor MG down a tier. Small kitchens are trimmed
    # by a PERCENTAGE (their absolute numbers are small, so a flat cut would
    # gut them); larger kitchens by a FLAT headcount (the percentage would be
    # a bigger trim than the wastage justifies). The thresholds are each
    # client's own volume bands — see the per-client overrides below.
    # Default: no adjustment, for clients with no tiering rule.

    def adjust_vendor_mg(self, mg): return mg
    def adjust_nonveg_vendor_mg(self, mg): return mg

    # ── TUNING: aggressive plan bump ────────────────────────────────────

    def aggressive_bump(self, w):
        """
        Add a sliding safety margin to one Ordered Qty (kg) for a star/non-veg
        dish. `sp` is a flat 10% "spare" of the row, of which we actually add
        back a fraction `f` that SHRINKS as the dish gets bigger:

            spare ≤ 2 kg   → +100% of it  (tiny dishes: 10% is noise, take it all)
            ≤ 4 kg         → +35%
            ≤ 6 kg         → +25%
            ≤ 8 kg         → +15%
            > 8 kg         → +10%         (big dishes: 10% is already a lot of food)

        Net effect ranges from +10% on a small dish down to about +1% on a
        large one. Raising any `f` means more cooked for the popular dishes.
        """
        sp=0.10*w
        if sp<=2:f=1.0
        elif sp<=4:f=0.35
        elif sp<=6:f=0.25
        elif sp<=8:f=0.15
        else:f=0.10
        return round(w+f*sp,1)

    # ── TUNING: special-day MG reduction matrix ─────────────────────────
    # Percentage of the client MG to REMOVE when a day sits next to a holiday
    # — people take the bridge day off, so fewer eat. Keyed by
    # (day_type, holiday_type, weekday); values are whole percents.
    #
    # Read a row as: "on this kind of day, attendance drops by N%".
    # Mondays and Fridays drop hardest (long weekends); midweek drops least.
    # A holiday itself cuts most (8–13%); the days either side cut 6–11%.
    # Any combination not listed falls back to DEFAULT_REDUCTION_PCT below.
    # Derived from attendance logs; re-fit when a site's holiday calendar changes.

    DEFAULT_REDUCTION_PCT = 10

    _REDUCTION = {
        ("holiday","important holiday","monday"):12,("holiday","important holiday","tuesday"):10,("holiday","important holiday","wednesday"):10,("holiday","important holiday","thursday"):10,("holiday","important holiday","friday"):12,
        ("holiday","non-important holiday","monday"):13,("holiday","non-important holiday","tuesday"):10,("holiday","non-important holiday","wednesday"):10,("holiday","non-important holiday","thursday"):9,("holiday","non-important holiday","friday"):8,
        ("next day of holiday","compulsory holiday","monday"):10,("next day of holiday","compulsory holiday","tuesday"):8,("next day of holiday","compulsory holiday","wednesday"):9,("next day of holiday","compulsory holiday","thursday"):10,("next day of holiday","compulsory holiday","friday"):11,
        ("next day of holiday","important holiday","monday"):9,("next day of holiday","important holiday","tuesday"):8,("next day of holiday","important holiday","wednesday"):7,("next day of holiday","important holiday","thursday"):8,("next day of holiday","important holiday","friday"):9,
        ("next day of holiday","non-important holiday","monday"):9,("next day of holiday","non-important holiday","tuesday"):9,("next day of holiday","non-important holiday","wednesday"):8,("next day of holiday","non-important holiday","thursday"):8,("next day of holiday","non-important holiday","friday"):9,
        ("previous day of holiday","compulsory holiday","monday"):10,("previous day of holiday","compulsory holiday","tuesday"):9,("previous day of holiday","compulsory holiday","wednesday"):9,("previous day of holiday","compulsory holiday","thursday"):8,("previous day of holiday","compulsory holiday","friday"):10,
        ("previous day of holiday","important holiday","monday"):10,("previous day of holiday","important holiday","tuesday"):6,("previous day of holiday","important holiday","wednesday"):6,("previous day of holiday","important holiday","thursday"):7,("previous day of holiday","important holiday","friday"):9,
        ("previous day of holiday","non-important holiday","monday"):9,("previous day of holiday","non-important holiday","tuesday"):8,("previous day of holiday","non-important holiday","wednesday"):11,("previous day of holiday","non-important holiday","thursday"):10,("previous day of holiday","non-important holiday","friday"):10,
    }

    def get_reduction_pct(self,dt,ht,wd):
        """Look up the attendance drop % for (day_type, holiday_type, weekday)."""
        return self._REDUCTION.get((dt.strip().lower(),ht.strip().lower(),wd.strip().lower()),
                                   self.DEFAULT_REDUCTION_PCT)


# ═══════════════════════════════════════════════════════════════════════
# TEKION — 7-feature model, 2-group vendor MG, separate non-veg track
# Volume band: roughly 150–900 MG, median ~300.
# ═══════════════════════════════════════════════════════════════════════
class TekionLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type","meal_day"]
    @property
    def fixed_categories(self): return ["Flavour Rice","Indian Bread","White Rice","Veg Dry","Veg Curry","Dal","Sambar","Rasam","Salad"]
    @property
    def star_categories(self): return {"flavoured rice","veg curry"}
    @property
    def category_map(self): return {"indian bread":"indian bread","indian breads":"indian bread","veg dry":"veg dry","veg curry":"veg curry","gravy veg":"veg curry","flavour rice":"flavoured rice","flavoured rice":"flavoured rice","white rice":"white rice","steamed rice":"white rice","dal":"dal","sambar":"sambar","rasam":"rasam","salad":"salad","non veg biryani":"non veg biryani","non veg curry":"non veg curry"}

    # Veg tiers: below 350 heads trim 6%; mid-size trim a flat 10 heads;
    # above 700 trim 20 — large kitchens absorb a bigger flat cut.
    def adjust_vendor_mg(self,mg):
        if mg<350: return mg*0.94
        elif 350<=mg<=700: return mg-10
        else: return mg-20

    # Non-veg is trimmed more gently (3% / 15 heads): non-veg uptake is
    # steadier than veg, and running out of the day's meat dish is costly.
    def adjust_nonveg_vendor_mg(self,mg):
        if mg<400: return mg*0.97
        else: return mg-15

# ═══════════════════════════════════════════════════════════════════════
# CLARIO — 5-feature model (no day_type/holiday_type in its dataset),
# 2-group vendor MG, client plan only.
# Volume band: 40–240 MG, median ~130 — the smallest site.
# ═══════════════════════════════════════════════════════════════════════
class ClarioLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","meal_day"]
    @property
    def fixed_categories(self): return ["Flavoured Rice","Indian Bread","White Rice","Healthy Rice","Veg Dry","Veg Gravy","Dal","Sambar/Rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg gravy"}
    @property
    def category_map(self):
        return {
            "indian bread":"indian bread",
            "indian breads":"indian bread",
            "veg dry":"veg dry",
            "veg curry":"veg gravy",
            "gravy veg":"veg gravy",
            "veg gravy":"veg gravy",
            "healty rice":"healthy rice",   # typo alias
            "healthy rice":"healthy rice",  # correct form
            "flavour rice":"flavoured rice",
            "flavoured rice":"flavoured rice",
            "white rice":"white rice",
            "steamed rice":"white rice",
            "dal":"dal",
            "sambar":"sambar/rasam",
            "sambar/rasam":"sambar/rasam",
            "rasam":"sambar/rasam",
        }

    @property
    def has_special_day(self): return False   # no day_type data to drive it
    @property
    def vendor_mg_method(self): return "tekion_2group"
    @property
    def default_mg(self): return 130          # site median

    # Same shape as Tekion, thresholds scaled to Clario's much smaller band:
    # under 80 heads trim 6%, mid trim 5 heads, above 160 trim 8.
    def adjust_vendor_mg(self, mg):
        if mg < 80:          return mg * 0.94
        elif 80 <= mg <= 160: return mg - 5
        else:                 return mg - 8

    def adjust_nonveg_vendor_mg(self, mg):
        if mg < 80: return mg * 0.97
        else:       return mg - 5

# ═══════════════════════════════════════════════════════════════════════
# ODESSIA — 6-feature model, 3-group vendor MG, contracted PP billing
# Volume band: roughly 80–500 MG.
# ═══════════════════════════════════════════════════════════════════════
class OdessiaLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type"]
    @property
    def fixed_categories(self): return ["Dal/Sambar","Flavoured rice","Indian Bread","Rasam","Veg dry","Veg Gravy","White Rice"]
    @property
    def star_categories(self): return {"flavoured rice","veg curry"}
    @property
    def category_map(self): return {"indian breads":"indian bread","indian bread":"indian bread","veg dry":"veg dry","veg gravy":"veg curry","veg curry":"veg curry","flavoured rice":"flavoured rice","flavour rice":"flavoured rice","white rice":"white rice","steamed rice":"white rice","dal":"dal","sambar":"sambar","rasam":"rasam","dal/sambar":"dal/sambar"}
    @property
    def vendor_mg_method(self): return "3group"
    @property
    def has_separate_nonveg_mg(self): return False   # one headcount for everyone
    @property
    def has_nonveg_toggle(self): return False        # non-veg section always shown
    @property
    def vendor_floor_ratio(self): return 0.0         # 3-group already trims lightly

    # Contracted per-pax portions (kg/person) — the client is billed on these
    # regardless of what the model predicts. Unlisted categories fall back to
    # 0.10 kg in planner.fixed_pp_client_plan().
    @property
    def fixed_pp_map(self):
        return {"dal/sambar":0.066,"flavoured rice":0.102,"indian bread":0.05,"rasam":0.033,"veg dry":0.075,"veg curry":0.102,"white rice":0.088,"non veg gravy":0.122,"non veg biryani":0.22,"non veg dry":0.099}

    # Four tiers rather than three: 6% under 120 heads, then flat 5 / 10 / 15.
    def adjust_vendor_mg(self,mg):
        if mg<120: return mg*0.94
        elif 121<=mg<=230: return mg-5
        elif 231<=mg<=400: return mg-10
        else: return mg-15

# ═══════════════════════════════════════════════════════════════════════
# RIPPLING — 6-feature model, 3-group vendor MG.
# Serves one veg-dry recipe at two counters (North/South), so the UI shows
# both but only North is predicted — South mirrors it (see app.py).
# ═══════════════════════════════════════════════════════════════════════
class RipplingLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type"]
    @property
    def fixed_categories(self): return ["Flavoured rice","White Rice","North Veg dry","South Veg dry","Veg gravy","Dal","Sambar/Rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg gravy"}
    @property
    def category_display_map(self): return {"North Veg dry":"Veg dry","South Veg dry":"Veg dry"}
    @property
    def vendor_mg_method(self): return "3group"
    @property
    def has_separate_nonveg_mg(self): return False
    @property
    def has_nonveg_toggle(self): return False
    @property
    def vendor_floor_ratio(self): return 0.0

    # Two tiers only: 6% under 120 heads, flat 10 above.
    def adjust_vendor_mg(self,mg):
        if mg<120: return mg*0.94
        else: return mg-10

# ═══════════════════════════════════════════════════════════════════════
# STRIPE — 6-feature model, day-based vendor MG, always non-veg (2 items)
# ═══════════════════════════════════════════════════════════════════════
class StripeLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type"]
    @property
    def fixed_categories(self): return ["indian bread","veg dry","veg curry","flavoured rice","white rice","dal","sambar","rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg curry"}
    @property
    def category_map(self): return {"indian breads":"indian bread","indian bread":"indian bread","veg dry":"veg dry","gravy veg":"veg curry","veg curry":"veg curry","flavoured rice":"flavoured rice","steamed rice":"white rice","white rice":"white rice","dal":"dal","sambar":"sambar","rasam":"rasam","non veg dry":"non veg dry","non veg gravy":"non veg gravy","non veg biryani":"non veg biryani","non veg curry":"non veg curry"}
    @property
    def custom_nonveg_mode(self): return "Required"   # every day is a non-veg day
    @property
    def nonveg_cats(self): return ["non veg dry","non veg gravy","non veg biryani","non veg curry"]
    @property
    def vendor_mg_method(self): return "day_based"
    @property
    def nonveg_item_count(self): return 2
    @property
    def has_nonveg_toggle(self): return False
    @property
    def has_separate_nonveg_mg(self): return False
    @property
    def vendor_floor_ratio(self): return 0.0

    # TUNING: fraction of Client MG removed per weekday to get Vendor MG.
    # Stripe's vendor MG comes straight off the calendar rather than off the
    # menu — attendance here is a stable weekly shape. Mondays and Fridays
    # drop most (23%, WFH bookends), midweek least (21%). Raising a value
    # orders LESS food that day. Re-fit from swipe data, not from wastage.
    @property
    def day_reductions(self): return {"monday":0.23,"tuesday":0.21,"wednesday":0.21,"thursday":0.22,"friday":0.23}

# ═══════════════════════════════════════════════════════════════════════
# TESSOLVE — 4-feature model (dataset has no day_type/holiday_type/meal_day),
# veg only, 2-group vendor MG, client plan only.
# Volume band: 75–650 MG, median ~570 — the largest site.
# ═══════════════════════════════════════════════════════════════════════
class TessolveLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category"]
    @property
    def fixed_categories(self): return ["Healthy rice","Indian Bread","Flavoured Rice","Veg gravy","Veg Dry","Dal/Sambar","white rice","Rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg gravy"}
    @property
    def has_nonveg_toggle(self): return False
    @property
    def has_separate_nonveg_mg(self): return False
    @property
    def has_special_day(self): return False
    @property
    def vendor_mg_method(self): return "tekion_2group"
    @property
    def default_mg(self): return 570          # site median

    # Tekion's shape with thresholds scaled up to Tessolve's band:
    # 6% under 300 heads, flat 10 mid, flat 20 above 600.
    def adjust_vendor_mg(self, mg):
        if mg < 300:           return mg * 0.94
        elif 300 <= mg <= 600: return mg - 10
        else:                  return mg - 20

# ═══════════════════════════════════════════════════════════════════════
# TOASTTAB — multiplier only. No dataset, no model, no menu input:
# the operator enters a headcount and gets a vendor headcount back.
# ═══════════════════════════════════════════════════════════════════════
class ToasttabLogic(BaseLogic):
    @property
    def has_embeddings(self): return False
    @property
    def encoder_columns(self): return []
    @property
    def fixed_categories(self): return []
    @property
    def star_categories(self): return set()
    @property
    def has_nonveg_toggle(self): return False
    @property
    def has_vendor_plan(self): return False
    @property
    def has_aggressive_plan(self): return False
    @property
    def has_special_day(self): return False
    @property
    def vendor_mg_method(self): return "toasttab_formula"
    @property
    def default_mg(self): return 100

    def toasttab_adjust(self, c):
        """
        TUNING: flat headcount trim by band (c = Client MG, heads).
          ≥135 → −15      95–134 → −10      55–94 → −5      <55 → no trim
        Flat rather than percentage all the way up, and nothing at all below
        55 heads where a trim would be larger than the day's actual no-shows.
        """
        if c>=135: return c-15
        elif c>=95: return c-10
        elif c<55: return c
        else: return c-5



# ═══════════════════════════════════════════════════════════════════════
# REGISTRY — map client keys to logic classes
# Same logic? Just point to the same class.
# ═══════════════════════════════════════════════════════════════════════
LOGIC_REGISTRY = {
    "tekion":    TekionLogic,
    "clario":    ClarioLogic,
    "odessia":   OdessiaLogic,
    "rippling":  RipplingLogic,
    "stripe":    StripeLogic,
    "tessolve":  TessolveLogic,
    "toasttab":  ToasttabLogic,
}

def get_logic(ck):
    key = ck.strip().lower()
    cls = LOGIC_REGISTRY.get(key)
    if not cls: raise ValueError(f"No logic for '{ck}'. Available: {sorted(LOGIC_REGISTRY)}")
    return cls()
