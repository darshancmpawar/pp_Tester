# Clario kitchen data request

12 Nov 2026 · live version: https://claude.ai/artifact/7cecd6Bm7mJrYVvKzSR5La

## Why we need this

The app's quantity suggestion for Clario lunch is now within 5 g per person of what was eaten for about 3 of every 4 dishes, on days it had never seen. To beat the kitchen's own plan, it needs four things only the kitchen can give.

On 13–29 Oct 2026 the app was within 5 g on 61 of 84 dishes; the kitchen's own cooked amount was within 5 g on 70 of 84. Biryani days are the weakest: 66% of dishes within 5 g, against 86% on veg days.

- **Biryani-day portions** are our estimate from past data, not the kitchen's standard.
- **People served** is assumed to equal client MG; we do not know how many actually ate.
- **Run-outs are hidden.** When a dish runs out, the log shows eaten = cooked, so the app learns to cook too little.
- **No fair test is left.** All past test days have been used, so only new days can show how the app really does.

## 1. Confirm the biryani-day portions

The kitchen answered on 13 Nov 2026, and the app now uses its figures (last column). The middle column was our estimate, from what was served on biryani days from Apr to Sep 2026.

| Category | Every day (g per person) | Biryani day, our estimate (g per person) | Kitchen's biryani-day standard (g per person) |
| --- | --- | --- | --- |
| Sambar/Rasam | 60 | 100 | 100 (rasam and sambar) |
| White Rice | 100 | 70 | 70 |
| Healthy Rice | 60 (Wednesday 120) | 120 | 120 (curd rice and the other dishes) |
| Dal | 70 | 60 | 60 |
| Veg Gravy | 120 | 100 | 120 |
| Veg Dry | 100 | 110 | 120 |
| Indian Bread | 66 | 60 | 60 or 66, depending on the other dishes |
| Flavoured Rice | 150 | 160 | 150 |

For bread, the app keeps 60 g as the usual biryani-day figure and learns the 66 g days from the other dishes on the menu.

Two everyday portions also still need a decision:

| Category | Kitchen's stated standard (g) | Served since 2026 (g) | Use from now (g) |
| --- | --- | --- | --- |
| Indian Bread | 60 | 66 |  |
| Healthy Rice, Wednesday | 100 | 120 |  |

## 2. Record how many people actually ate, every day

Every lunch, record the number of people served, counted at the counter (plates or coupons), not the client MG. Today the app divides what was eaten by the client MG; if 110 people ate on a 120 MG day, every portion looks 8% smaller than it was.

- **People served:** the total for lunch.
- **Non-veg takers:** how many took the non-veg dish, if the counter can count it.
- **Client MG:** as now, so we can compare the two.

## 3. When a dish runs out, record the headcount at that moment

This is the most important ask: shortages upset the client most, and today the log cannot show them. For every dish that runs out, record:

1. **People served when it ran out**, read from the counter at that moment.
2. **Time** it ran out.
3. **Top-up:** whether more was cooked, and how many kg.
4. **Ran out again?** If the top-up also finished, the people served at that point.

Why it matters, with an example: Dal, 7 kg cooked, 120 people at lunch, ran out after 90 people.

|  | What the log shows today | What really happened |
| --- | --- | --- |
| Dal per person | 7 kg ÷ 120 = 58 g | 7 kg ÷ 90 = 78 g |
| Dal needed for 120 people | 7.0 kg | 9.3 kg |

Today the app learns 58 g and cooks short again. With the headcount at run-out it learns 78 g. A day with a small leftover needs nothing extra: it already shows the amount was enough.

## Daily log sheet

One row per day in the first table and one row per dish in the second. The rows below are examples; add your own under them.

| Date | People served | Non-veg takers | Client MG |
| --- | --- | --- | --- |
| 16 Nov 2026 (example) | 118 | 52 | 120 |

| Date | Dish | App suggestion (kg) | Cooked (kg) | Top-up (kg) | Left over (kg) | People served at run-out |
| --- | --- | --- | --- | --- | --- | --- |
| 16 Nov 2026 (example) | Dal Butter Fry | 7.2 | 7.0 | 1.5 | 0.3 | 92 |
| 16 Nov 2026 (example) | Steamed White Rice | 11.8 | 12.0 | 0 | 0.4 |  |

Leave "People served at run-out" empty when the dish did not run out. Weigh leftovers to 0.1 kg, as now.

## 4. Run a 2–4 week shadow trial

For 2–4 weeks, note the app's number every morning but keep cooking to the kitchen's own plan. These are days the app has never seen, so this is the fair test of whether it is ready.

1. Before cooking, enter the day's menu and client MG in the app (Clario) and copy each dish's suggested kg into the log.
2. Cook to the kitchen's own plan as usual. Do not change the cooking because of the app.
3. After lunch, fill in the rest of the log: people served, top-ups, leftovers, run-outs.
4. Send the log every week.

The rule, fixed before the trial starts: the app's number replaces the kitchen's plan for a category only if it lands within 5 g per person of what was eaten on more dishes than the kitchen's plan does, with no more run-outs.

Still to agree: the start date, and who fills the log each day.

## What happens with the data

| When | What we do with it |
| --- | --- |
| The week the portions come back | Update the app's standard portions and retrain it |
| Start of every month | Retrain the app on the month's log, run-outs corrected with the headcount at run-out |
| End of the trial | Compare the app and the kitchen's plan, category by category, against the rule in section 4 |
