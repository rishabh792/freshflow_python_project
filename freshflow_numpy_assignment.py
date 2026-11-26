"""
FRESHFLOW - NumPy Data Analyst End-to-End Assignment
Only NumPy is used (no Pandas). Covers Phases 1-12 and Tasks A-L of the spec.
Loads data directly from freshflow_sales.csv.
"""
import numpy as np
from numpy import testing as npt

CSV_FILE = "freshflow_sales.csv"
COLS = ["transaction_id", "day_index", "outlet_id", "product_code", "category_code",
        "quantity", "unit_price", "unit_cost", "discount_rate",
        "stock_before_units", "payment_code", "customer_rating"]
C = {name: i for i, name in enumerate(COLS)}          # column name -> column index

# ---- 4.2 Code dictionaries ----
OUTLETS = {1: "Outlet A", 2: "Outlet B", 3: "Outlet C"}
PRODUCTS = {1: "Orange Juice", 2: "Mango Juice", 3: "Apple Juice", 4: "Watermelon Juice",
            5: "Banana Smoothie", 6: "Mixed Fruit Smoothie", 7: "Lemonade", 8: "Iced Tea"}
CATEGORIES = {1: "Juice", 2: "Smoothie", 3: "Lemonade", 4: "Tea"}
PAYMENTS = {1: "Cash", 2: "UPI", 3: "Card", 4: "Other digital payment"}
# index = product code -> category code (index 0 unused)
CATEGORY_OF_PRODUCT = np.array([0, 1, 1, 1, 1, 2, 2, 3, 4])


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# =====================================================================
# HELPER FUNCTIONS (Phase 10 - reusable functions)
# =====================================================================
def safe_divide(num, den):
    """Division that returns NaN where denominator is zero (guard for zero denominators)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def record_metrics(q, p, c, d):
    """Section 7 formulas (vectorised)."""
    gross = q * p                       # Gross sales = quantity x unit_price
    disc_amt = gross * d                # Discount amount = gross_sales x discount_rate
    net = gross - disc_amt              # Net revenue = gross_sales - discount_amount
    cogs = q * c                        # Cost of goods sold = quantity x unit_cost
    profit = net - cogs                 # Gross profit = net_revenue - cost_of_goods_sold
    return gross, disc_amt, net, cogs, profit


def summary_stats(x):
    """Phase 5 - descriptive statistics."""
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(mean=np.mean(x), median=np.median(x), std_pop=np.std(x), std_sample=np.std(x, ddof=1),
                var=np.var(x), min=np.min(x), max=np.max(x), ptp=np.ptp(x), q1=q1, q2=med, q3=q3)


# =====================================================================
# PHASE 1 : REPRESENT AND INSPECT DATA
# =====================================================================
def phase1_demo(data):
    header("PHASE 1 - Represent and inspect data")
    a = np.array([1, 2, 3])
    b = np.asarray(data)                               # no copy if already ndarray
    print("np.array:", a, "| np.arange:", np.arange(0, 10, 2), "| np.linspace:", np.linspace(0, 1, 5))
    print("np.zeros:", np.zeros((2, 2)).tolist(), "| np.ones:", np.ones(3), "| np.full:", np.full(3, 7))
    print("np.eye:", np.eye(2).tolist(), "| np.ndim:", b.ndim, "| size:", b.size)
    print("shape:", b.shape, "| dtype:", b.dtype, "| itemsize:", b.itemsize, "| nbytes:", b.nbytes)
    print("astype(int32) of first row:", data[0].astype(np.int32))


# =====================================================================
# MAIN ANALYSIS
# =====================================================================
# ---- 4.3 CSV loading constraints ----
data = np.genfromtxt(CSV_FILE, delimiter=",", skip_header=1)   # blanks/'nan' -> NaN
if data.ndim == 1:
    data = data.reshape(1, -1)                         # keep two-dimensional for a single row
raw = data.copy()                                      # keep raw unchanged; never overwrite source
phase1_demo(raw)

# column vectors (views of raw -> we only read them, never write)
tid = raw[:, C["transaction_id"]];  day = raw[:, C["day_index"]]
outlet = raw[:, C["outlet_id"]];    product = raw[:, C["product_code"]]
category = raw[:, C["category_code"]]; qty = raw[:, C["quantity"]]
price = raw[:, C["unit_price"]];    cost = raw[:, C["unit_cost"]]
disc = raw[:, C["discount_rate"]];  stock = raw[:, C["stock_before_units"]]
pay = raw[:, C["payment_code"]];    rating = raw[:, C["customer_rating"]]

# =====================================================================
# TASK A - DATA PROFILE
# =====================================================================
header("TASK A - Data profile")
print("Rows, columns:", raw.shape, "| dtype:", raw.dtype)
assert raw.shape[0] == 5000, "Exactly 5000 data rows must be loaded"
print(f"{'column':22s}{'min':>10s}{'max':>10s}{'missing':>10s}")
for name, i in C.items():
    col = raw[:, i]
    print(f"{name:22s}{np.nanmin(col):10.2f}{np.nanmax(col):10.2f}{np.isnan(col).sum():10d}")
print("All finite values per column:", np.isfinite(raw).sum(axis=0))
print("Infinite values:", np.isinf(raw).sum())

# =====================================================================
# TASK B - INTEGRITY AUDIT  (Phase 2 masks, Phase 3 validation)
# =====================================================================
header("TASK B - Integrity audit")
ids, inv, counts = np.unique(tid, return_inverse=True, return_counts=True)
dup_mask = counts[inv] > 1                                         # all records sharing an ID
valid_outlet = (outlet >= 1) & (outlet <= 3) & (outlet == np.round(outlet))
valid_product = (product >= 1) & (product <= 8) & (product == np.round(product))
safe_prod = np.where(valid_product, product, 0).astype(int)       # safe index into mapping
cat_mismatch = valid_product & (category != CATEGORY_OF_PRODUCT[safe_prod])
bad_qty = np.isfinite(qty) & (qty <= 0)
miss_price = np.isnan(price)
bad_price = np.isfinite(price) & (price <= 0)
miss_cost = np.isnan(cost)
bad_cost = np.isfinite(cost) & (cost < 0)
bad_disc = np.isfinite(disc) & ((disc < 0) | (disc > 1))
bad_pay = ~((pay >= 1) & (pay <= 4))
miss_rating = np.isnan(rating)
bad_rating = np.isfinite(rating) & ((rating < 1) | (rating > 5))
zero_stock = (stock == 0) & (qty > 0)
high_disc = np.isfinite(disc) & (disc > 0.30) & (disc <= 1)

# unusually high price = more than 3x the median valid price of the same product
price_ok = np.where(price > 0, price, np.nan)
med_by_prod = np.array([np.nanmedian(price_ok[product == p]) for p in range(1, 9)])
with np.errstate(invalid="ignore"):
    high_price = valid_product & (price > 3 * med_by_prod[np.maximum(safe_prod - 1, 0)])

checks = {
    "duplicate transaction IDs (records)": dup_mask, "unknown outlet code": ~valid_outlet,
    "unknown product code": ~valid_product, "category mismatch": cat_mismatch,
    "invalid quantity (<=0)": bad_qty, "missing unit_price": miss_price,
    "invalid unit_price (<=0)": bad_price, "missing unit_cost": miss_cost,
    "invalid unit_cost (<0)": bad_cost, "discount outside 0-1": bad_disc,
    "invalid payment code": bad_pay, "missing rating": miss_rating,
    "invalid rating (not 1-5)": bad_rating, "zero stock before sale": zero_stock,
    "unusually high discount": high_disc, "unusually high price": high_price,
}
for name, m in checks.items():
    print(f"{name:40s}{int(m.sum()):6d}")
print("Duplicate IDs (unique):", ids[counts > 1])
bad_ids = dup_mask | ~valid_outlet
print("Impossible discounts (>1):", int(bad_disc.sum()))

# =====================================================================
# TASK C - CLEANING STRATEGY  (raw unchanged, masks + cleaned view)
# =====================================================================
header("TASK C - Cleaning strategy")
exclude = (dup_mask | ~valid_outlet | ~valid_product | cat_mismatch |
           ~(qty > 0) | ~(price > 0) | ~(cost >= 0) | ~((disc >= 0) & (disc <= 1)))
valid = ~exclude                                       # valid-record mask used for ALL metrics
cleaned = raw.copy()                                   # view for analysis; raw stays untouched
cleaned[bad_rating, C["customer_rating"]] = np.nan     # invalid rating -> missing (not zero)
rules = [("Duplicate IDs", dup_mask, "excluded (flagged, not blindly deduplicated)"),
         ("Unknown outlet/product, category mismatch", ~valid_outlet | ~valid_product | cat_mismatch, "excluded"),
         ("Qty <= 0", bad_qty, "excluded (no refund semantics assumed)"),
         ("Missing/invalid price", miss_price | bad_price, "excluded (price not imputed)"),
         ("Missing/invalid cost", miss_cost | bad_cost, "excluded (profit not calculable)"),
         ("Discount > 1 or < 0", bad_disc, "excluded (impossible)"),
         ("Missing rating", miss_rating, "kept; ignored in rating mean (not set to 0)"),
         ("Invalid rating", bad_rating, "set to NaN in cleaned view"),
         ("High price / high discount / zero stock", high_price | high_disc | zero_stock,
          "kept; flagged for review (not auto-deleted)")]
for name, m, action in rules:
    print(f"{name:42s}flagged={int(m.sum()):4d}  -> {action}")
print(f"Retained valid records: {int(valid.sum())} | Excluded: {int(exclude.sum())} | Total: {raw.shape[0]}")

# valid-record arrays used below
v_day = cleaned[valid, C["day_index"]].astype(int)
v_out = cleaned[valid, C["outlet_id"]].astype(int)
v_prod = cleaned[valid, C["product_code"]].astype(int)
v_q = cleaned[valid, C["quantity"]]
v_p = cleaned[valid, C["unit_price"]]
v_c = cleaned[valid, C["unit_cost"]]
v_d = cleaned[valid, C["discount_rate"]]
v_rate = cleaned[valid, C["customer_rating"]]

# =====================================================================
# TASK D - METRIC ENGINE
# =====================================================================
header("TASK D - Metric engine")
gross, disc_amt, net, cogs, profit = record_metrics(v_q, v_p, v_c, v_d)
tot_gross, tot_disc, tot_net = gross.sum(), disc_amt.sum(), net.sum()
tot_cogs, tot_profit = cogs.sum(), profit.sum()
n_valid = valid.sum()
print(f"Gross sales      : {tot_gross:,.2f} INR")
print(f"Discount amount  : {tot_disc:,.2f} INR")
print(f"Net revenue      : {tot_net:,.2f} INR")
print(f"COGS             : {tot_cogs:,.2f} INR")
print(f"Gross profit     : {tot_profit:,.2f} INR")
print(f"Gross margin %   : {safe_divide(tot_profit, tot_net) * 100:.2f}   (total profit / total net revenue x 100)")
print(f"Avg selling price: {safe_divide(tot_net, v_q.sum()):.2f}   (net revenue / quantity)")
print(f"Avg transaction value: {safe_divide(tot_net, n_valid):.2f}")
print(f"Weighted discount %  : {safe_divide(tot_disc, tot_gross) * 100:.2f}")
print(f"Mean valid rating    : {np.nanmean(v_rate):.2f} (valid ratings = {np.isfinite(v_rate).sum()})")

# =====================================================================
# TASK E - PRODUCT AND OUTLET COMPARISON  (Phase 6 groups)
# =====================================================================
header("TASK E - Product and outlet comparison")
o_net = np.bincount(v_out, weights=net, minlength=4)[1:]
o_profit = np.bincount(v_out, weights=profit, minlength=4)[1:]
o_units = np.bincount(v_out, weights=v_q, minlength=4)[1:]
p_net = np.bincount(v_prod, weights=net, minlength=9)[1:]
p_profit = np.bincount(v_prod, weights=profit, minlength=9)[1:]
p_units = np.bincount(v_prod, weights=v_q, minlength=9)[1:]

print("Outlet ranking by net revenue (high -> low):")
for r, i in enumerate(np.argsort(o_net)[::-1], 1):
    print(f" {r}. {OUTLETS[i + 1]:9s} net={o_net[i]:12,.2f} profit={o_profit[i]:12,.2f} units={o_units[i]:6.0f}")
print("Product ranking by gross profit (high -> low):")
for r, i in enumerate(np.argsort(p_profit)[::-1], 1):
    m = safe_divide(p_profit[i], p_net[i]) * 100
    print(f" {r}. {PRODUCTS[i + 1]:22s} net={p_net[i]:11,.2f} profit={p_profit[i]:11,.2f} units={p_units[i]:5.0f} margin={m:5.1f}%")
print("Highest-volume product :", PRODUCTS[np.argmax(p_units) + 1])
print("Most profitable product:", PRODUCTS[np.argmax(p_profit) + 1])
print("Lowest-volume product  :", PRODUCTS[np.argmin(p_units) + 1])
print("Highest-volume product need not be most profitable: unit margin, price and discounts differ by product.")

# =====================================================================
# TASK F - DESCRIPTIVE STATISTICS  (Phase 5)
# =====================================================================
header("TASK F - Descriptive statistics")
for label, x in [("Net revenue", net), ("Gross profit", profit), ("Unit price", v_p)]:
    s = summary_stats(x)
    print(f"{label}: " + ", ".join(f"{k}={val:.2f}" for k, val in s.items()))
    print(f"   quantile(0.9)={np.quantile(x, 0.9):.2f} | mean - median = {s['mean'] - s['median']:.2f} "
          f"({'right-skewed' if s['mean'] > s['median'] else 'left/symmetric'})")
counts_h, edges = np.histogram(net, bins=8)
print("Histogram of net revenue (counts):", counts_h)
print("Bin edges:", np.round(edges, 1))
z_net = (net - net.mean()) / net.std()
print("Extreme values (|z|>3):", int((np.abs(z_net) > 3).sum()), "| max net:", net.max().round(2))
print("Sample std (ddof=1) vs population std (ddof=0) shown above; sample is used for inference.")
weighted_avg_price = np.average(v_p, weights=v_q)
print(f"Unweighted avg price={v_p.mean():.2f} vs quantity-weighted avg price={weighted_avg_price:.2f}")

# =====================================================================
# TASK G - DAILY TREND  (Phase 7)
# =====================================================================
header("TASK G - Daily trend (days 1-30)")
daily = np.bincount(v_day, weights=net, minlength=31)[1:]
change = np.diff(daily)                                 # absolute change
pct = safe_divide(change, daily[:-1]) * 100             # % change, safe for zero previous day
cs = np.cumsum(np.concatenate(([0.0], daily)))
ma7 = (cs[7:] - cs[:-7]) / 7                            # 7-day moving average
z_day = (daily - daily.mean()) / daily.std()
unusual_days = np.nonzero(np.abs(z_day) > 2)[0] + 1
print("Day  Revenue   Change   %Change")
for i in range(30):
    c_txt = f"{change[i - 1]:10.2f}{pct[i - 1]:9.2f}" if i > 0 else f"{chr(45):>10s}{chr(45):>9s}"
    print(f"{i + 1:3d}{daily[i]:10.2f}{c_txt}")
print("7-day moving average (first 5):", np.round(ma7[:5], 2))
print("Cumulative revenue (day 30):", round(np.cumsum(daily)[-1], 2))
print("Highest day:", np.argmax(daily) + 1, "| Lowest day:", np.argmin(daily) + 1)
print("Unusual days (|z|>2) for review:", unusual_days if unusual_days.size else "none")
print("Limitation: only 30 days -> no annual seasonality claims.")

# =====================================================================
# TASK H - DISCOUNT INVESTIGATION
# =====================================================================
header("TASK H - Discount investigation")
bands = [("0% (none)", v_d == 0), ("0-10%", (v_d > 0) & (v_d <= 0.10)),
         ("10-20%", (v_d > 0.10) & (v_d <= 0.20)), (">20%", v_d > 0.20)]
print(f"{'band':12s}{'txns':>6s}{'units':>8s}{'avg qty':>9s}{'net rev':>13s}{'profit':>12s}{'margin%':>9s}")
for name, m in bands:
    n_b = m.sum()
    print(f"{name:12s}{n_b:6d}{v_q[m].sum():8.0f}{safe_divide(v_q[m].sum(), n_b):9.2f}"
          f"{net[m].sum():13,.2f}{profit[m].sum():12,.2f}{safe_divide(profit[m].sum(), net[m].sum()) * 100:9.2f}")
print("Discussion: higher discount bands lower margin because price falls while cost stays fixed.")
print("Selection bias: discounts may not be randomly assigned; confounding (product, outlet, day)")
print("can drive quantity differences, so we do not claim discounts CAUSE higher quantity.")

# =====================================================================
# TASK I - ANOMALY REGISTER  (Phase 8: IQR + z-score + combined masks)
# =====================================================================
header("TASK I - Anomaly register")
net_all = np.full(raw.shape[0], np.nan)
net_all[valid] = net
q1, q3 = np.percentile(net, [25, 75])
iqr = q3 - q1
with np.errstate(invalid="ignore"):
    iqr_flag = (net_all < q1 - 1.5 * iqr) | (net_all > q3 + 1.5 * iqr)
    z_all = (net_all - np.nanmean(net_all)) / np.nanstd(net_all)
    z_flag = np.abs(z_all) > 3
reason_masks = {"DUP_ID": dup_mask, "BAD_OUTLET": ~valid_outlet, "CAT_MISMATCH": cat_mismatch,
                "BAD_QTY": bad_qty, "MISSING_PRICE": miss_price, "BAD_PRICE": bad_price,
                "MISSING_COST": miss_cost, "BAD_COST": bad_cost, "BAD_DISCOUNT": bad_disc,
                "HIGH_DISCOUNT": high_disc, "HIGH_PRICE": high_price, "ZERO_STOCK": zero_stock,
                "BAD_RATING": bad_rating, "NET_IQR": iqr_flag, "NET_ZSCORE": z_flag}
any_flag = np.zeros(raw.shape[0], dtype=bool)
for m in reason_masks.values():
    any_flag |= m
register = []
for i in np.nonzero(any_flag)[0]:
    reasons = [k for k, m in reason_masks.items() if m[i]]
    impact = net_all[i] if np.isfinite(net_all[i]) else np.nan   # impact only when calculable
    register.append((int(tid[i]), ",".join(reasons), impact))
register.sort(key=lambda r: (np.isnan(r[2]), -np.nan_to_num(r[2])))
print(f"Total flagged records: {len(register)} (IQR bounds: {q1 - 1.5 * iqr:.1f} to {q3 + 1.5 * iqr:.1f})")
print(f"{'txn_id':>7s}  {'reason codes':45s}{'net revenue impact':>20s}")
for t, r, imp in register[:20]:
    print(f"{t:7d}  {r[:44]:45s}{('not calculable' if np.isnan(imp) else f'{imp:,.2f}'):>20s}")

# =====================================================================
# PHASE 9 - RELATIONSHIPS
# =====================================================================
header("PHASE 9 - Relationships (corrcoef / cov)")
print("corr(discount, quantity):", round(np.corrcoef(v_d, v_q)[0, 1], 4))
print("corr(price, quantity)   :", round(np.corrcoef(v_p, v_q)[0, 1], 4))
print("cov(discount, quantity) :", round(np.cov(v_d, v_q)[0, 1], 6))
print("Correlation does not prove causation.")

# =====================================================================
# TASK J - SCENARIO ANALYSIS  (scenario, not forecast)
# =====================================================================
header("TASK J - Scenario analysis (same valid records in every scenario)")
def scenario(new_d, q_factor=1.0):
    q_new = v_q * q_factor
    _, _, n_s, _, p_s = record_metrics(q_new, v_p, v_c, new_d)
    return n_s.sum(), p_s.sum()

base_net, base_profit = scenario(v_d)
scenarios = {
    "S1: discounts capped at 10% (qty constant)": (np.minimum(v_d, 0.10), 1.0),
    "S2: all discounts halved (qty constant)": (v_d * 0.5, 1.0),
    "S3: discounts +5pp (assume qty +3% - ASSUMPTION)": (np.minimum(v_d + 0.05, 1.0), 1.03),
}
print(f"Baseline: net={base_net:,.2f} profit={base_profit:,.2f}")
for name, (nd, qf) in scenarios.items():
    sn, sp = scenario(nd, qf)
    print(f"{name}\n   net={sn:,.2f} ({sn - base_net:+,.2f}, {safe_divide(sn - base_net, base_net) * 100:+.2f}%)"
          f" | profit={sp:,.2f} ({sp - base_profit:+,.2f}, {safe_divide(sp - base_profit, base_profit) * 100:+.2f}%)")
print("These are scenarios, not predictions. Holding quantity constant ignores demand response.")

# =====================================================================
# PHASE 10 - BUILD A PIPELINE (combine / split / cumulative)
# =====================================================================
header("PHASE 10 - Combine / split / cumulative")
stacked = np.vstack([o_net, o_profit])                  # rows: net, profit; cols: outlets
print("vstack shape:", stacked.shape, "| hstack:", np.hstack([o_net, o_profit]).shape,
      "| stack(axis=0):", np.stack([o_net, o_profit]).shape, "| concatenate:", np.concatenate([o_net, o_profit]).shape)
weeks = np.split(daily[:28], 4)                         # equal 7-day weeks
print("Weekly revenue (split):", [round(float(w.sum()), 2) for w in weeks])
print("array_split of 30 days into 4 parts:", [len(p) for p in np.array_split(daily, 4)])
print("Growth index (cumprod of 1+pct/100, first 5):", np.round(np.cumprod(1 + pct[:5] / 100), 4))
print("vstack(row-wise) / hstack(col-wise) / vsplit / hsplit demo:",
      np.vsplit(stacked, 2)[0].shape, np.hsplit(stacked, 3)[0].shape)
print("Optional einsum check - gross via einsum:", round(np.einsum("i,i->", v_q, v_p), 2),
      "| matmul:", round(v_q @ v_p, 2))

# =====================================================================
# PHASE 11 - DEBUG AND OPTIMISE
# =====================================================================
header("PHASE 11 - Debugging demonstrations")
try:                                                    # shape mismatch / broadcasting
    np.array([1, 2, 3]) + np.array([1, 2])
except ValueError as e:
    print("Shape mismatch ->", e)
print("Broadcast (3,1)+(1,3) shape:", (np.ones((3, 1)) + np.ones((1, 3))).shape)
print("dtype overflow int8 (127+1):", np.array([127], dtype=np.int8) + np.int8(1))
print("NaN propagation: np.mean ->", np.mean(price), "| np.nanmean ->", round(np.nanmean(price), 2))
a = np.arange(5); view = a[1:3]; copy = a[1:3].copy(); view[0] = 99
print("View changed original:", a, "| copy unchanged:", copy)
# Documented bug fix: np.mean on a column with NaN returned nan -> fixed by np.nanmean / valid mask.

# =====================================================================
# TASK K - VALIDATION TESTS
# =====================================================================
header("TASK K - Validation tests")
rng = np.random.default_rng(7)
valid_idx = np.nonzero(valid)[0]
sample = rng.choice(valid_idx, size=5, replace=False)    # 5 sample rows, verified by hand
for i in sample:
    q_, p_, c_, d_ = float(qty[i]), float(price[i]), float(cost[i]), float(disc[i])
    g_ = q_ * p_; n_ = g_ - g_ * d_; pr_ = n_ - q_ * c_   # hand/scalar calculation
    j = np.nonzero(valid_idx == i)[0][0]
    npt.assert_allclose([g_, n_, pr_], [gross[j], net[j], profit[j]])
    print(f"txn {int(tid[i]):5d}: gross={g_:8.2f} net={n_:8.2f} profit={pr_:8.2f}  -> matches NumPy")
npt.assert_allclose(np.nanmean(np.array([1.0, np.nan, 3.0])), 2.0)       # NaN test
assert np.isnan(safe_divide(5, 0))                                         # zero-denominator test
try:                                                                       # invalid shape test
    np.arange(10).reshape(3, 4)
except ValueError as e:
    print("Invalid reshape caught ->", e)
npt.assert_array_equal(np.unique(v_out), np.array([1, 2, 3]))              # valid outlets only
npt.assert_allclose(tot_net, tot_gross - tot_disc)                         # identity check
npt.assert_allclose(tot_profit, tot_net - tot_cogs)
print("All validation assertions passed.")

# =====================================================================
# TASK L - EXECUTIVE RECOMMENDATION
# =====================================================================
header("TASK L - Executive recommendation (three evidence-supported findings)")
best_o = np.argmax(o_profit) + 1
top_vol = np.argmax(p_units) + 1
top_prof = np.argmax(p_profit) + 1
m_none = bands[0][1]; m_high = bands[2][1] | bands[3][1]
mg_none = safe_divide(profit[m_none].sum(), net[m_none].sum()) * 100
mg_high = safe_divide(profit[m_high].sum(), net[m_high].sum()) * 100
print(f"1. Finding : {OUTLETS[best_o]} earns the highest gross profit ({o_profit[best_o - 1]:,.0f} INR).\n"
      f"   Action  : study and replicate its practices at other outlets.\n"
      f"   Monitor : gross profit per outlet per week. Limitation: no operating expenses included.\n"
      f"   Next check: compare staffing/footfall data.")
same = "also" if top_vol == top_prof else "but " + PRODUCTS[top_prof] + " gives the most profit, not"
print(f"2. Finding : {PRODUCTS[top_vol]} sells the most units and {same if top_vol == top_prof else same} "
      f"{'gives the most profit' if top_vol == top_prof else PRODUCTS[top_vol]}.\n"
      f"   Action  : prioritise promotion by profit contribution, not volume alone.\n"
      f"   Monitor : gross margin % by product. Limitation: 30 days only.\n"
      f"   Next check: repeat with next month's data.")
print(f"3. Finding : margin with no discount = {mg_none:.1f}% vs {mg_high:.1f}% with discounts above 10%.\n"
      f"   Action  : cap deep discounts and review flagged high-discount transactions.\n"
      f"   Monitor : weighted discount % and margin %. Limitation: correlation is not causation.\n"
      f"   Next check: run a controlled discount test at one outlet.")
print("\nGeneral limitations: 5000-row dataset, 30-day window, no operating expenses,"
      " no inventory-waste conclusions from the transaction CSV alone.")
