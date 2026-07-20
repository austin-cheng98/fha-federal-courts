# FHA opinion coding codebook

Read the full district-court opinion. Code what the OPINION ITSELF ADJUDICATES OR
SUBSTANTIVELY DISCUSSES as an asserted Fair Housing Act theory. Do NOT code a claim
merely because a phrase appears (e.g. a string-cited case name, a quoted statute, or
a passing reference in procedural history). Code 1 only if the claim is actually at
issue in this opinion.

## Claims (each 0 or 1, NOT mutually exclusive)

- **disparate_treatment** — intentional discrimination because of a protected
  characteristic: differential terms, conditions, or treatment; pretext analysis;
  discriminatory intent/animus. Often (not always) analyzed under McDonnell Douglas.

- **disparate_impact** — a facially neutral policy or practice challenged for its
  discriminatory EFFECT on a protected group. Requires effects-based reasoning
  (statistics, robust causality, HUD three-step burden shifting, Inclusive
  Communities). Do NOT code 1 merely because "disparate impact" is named in a
  citation, a legal-standard recitation, or a claim the court declines to reach.

- **refusal_rent_sell** — refusal to rent, sell, negotiate, or otherwise make housing
  available; steering; misrepresenting availability. §3604(a)/(d) conduct.

- **reasonable_accommodation** — disability-based duty to make reasonable
  accommodations in rules/policies/services, or to permit reasonable modifications.
  §3604(f)(3)(A)-(B). Includes failure-to-accommodate and interactive-process disputes.

- **zoning_exclusionary** — municipal land use, zoning, permitting, occupancy limits,
  or siting decisions challenged as exclusionary or discriminatory.

## framework (exactly one)
- `mcdonnell` — McDonnell Douglas burden shifting is explicitly applied/invoked
- `hud` — HUD three-step / Inclusive Communities burden-shifting framework applied
- `both` — both explicitly applied
- `none` — no explicit burden-shifting framework applied

## Output
Return ONLY the structured object. `evidence` = a short verbatim quote from the
opinion supporting each claim you code 1 (empty string if you coded 0).
Base every judgment solely on the opinion text provided.
