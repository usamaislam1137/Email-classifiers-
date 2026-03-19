# Dummy email data for manual testing (alternate set)

These examples are **not** the same as the three quick-fill samples in `rails_app/app/controllers/classifications_controller.rb` (`new` action). Use them for extra UI/API checks.

**Fields:** `sender`, `recipients`, `subject`, `body`, optional `date` — same as the classify form and `POST /predict`.

**Expected priority** is a *design intent* from `ml/config.py` keywords (urgent / important / routine / fyi-automated). Your model’s scores may differ.

---

## Copy-paste table (web form)

| # | sender | recipients | subject | body | date | Expected |
|---|--------|------------|---------|------|------|----------|
| 1 | `incident@aurora-hospital.org` | `oncall@aurora-hospital.org` | `CODE BLUE — immediate attention required` | `Emergency on Ward 4. All available staff report now. This is critical and time sensitive. Respond immediately — lives at stake. ASAP.` | `2026-04-10T02:15` | **critical** |
| 2 | `cfo@veloxfinance.io` | `audit@veloxfinance.io` | `Regulator deadline TODAY — sign-off before 17:00` | `Urgent action required: we must file by EOD today. Top priority. Please review and respond now. Time-sensitive compliance window.` | `2026-04-10T09:00` | **critical** |
| 3 | `sre@northgrid.cloud` | `pagerduty@northgrid.cloud` | `SEV-1: production API down — respond now` | `Customer impact is severe. Immediate attention needed. Right away — we need owners on bridge. Critical incident.` | `2026-04-10T03:40` | **critical** |
| 4 | `procurement@cityworks.gov` | `vendors@example-supply.co` | `RFP responses due Monday — please confirm receipt` | `Important: all bids must arrive by Monday close of business. Please review the attached scope. Response needed to participate. Mandatory for shortlisted vendors.` | `2026-04-08T14:00` | **high** |
| 5 | `advisor@hudsonlaw.com` | `client@startup.io` | `Contract review — please advise by Wednesday` | `Following up on the draft agreement. Deadline for comments is Wednesday. Meeting scheduled Tuesday if you need to walk through clauses. Please respond with redlines.` | `2026-04-08T11:30` | **high** |
| 6 | `registrar@metro-uni.edu` | `students@metro-uni.edu` | `Thesis submission portal opens next week` | `Please confirm you have uploaded your advisor approval. Required step before final deadline. Follow-up reminders will be sent.` | `2026-04-09T10:00` | **high** |
| 7 | `jamie@studio-knot.design` | `team@studio-knot.design` | `Font files for the brand refresh (v3)` | `Sharing the updated OTFs in the drive. No rush — use whichever weight you prefer for mockups. Ping me if links break.` | `2026-04-07T16:45` | **normal** |
| 8 | `facilities@oak-campus.edu` | `staff@oak-campus.edu` | `Elevator B back in service` | `Routine notice: Elevator B maintenance finished. Standard announcement for the west wing. Thanks for patience this week.` | `2026-04-07T08:00` | **normal** |
| 9 | `digest@notifications.linkedin.com` | | `People you may know — weekly roundup` | `You have 3 new connection suggestions. This is an automated notification. Unsubscribe from digests in settings. No reply needed.` | `2026-04-06T06:00` | **low** |
| 10 | `promo@freshcart.market` | | `Monthly promotional mailing list` | `FYI: weekly deals inside. Newsletter for subscribers only. Click unsubscribe if you no longer want promotional emails. Mailing list message.` | `2026-04-06T07:30` | **low** |

---

## JSON + illustrative API result (different from above table)

Use your ML base URL (e.g. `http://127.0.0.1:5001`).

### Critical — hospital / emergency tone

**Request**

```json
{
  "sender": "incident@aurora-hospital.org",
  "recipients": "oncall@aurora-hospital.org",
  "subject": "CODE BLUE — immediate attention required",
  "body": "Emergency on Ward 4. All available staff report now. This is critical and time sensitive. Respond immediately — lives at stake. ASAP.",
  "date": "2026-04-10T02:15:00"
}
```

**Illustrative response** (yours will vary)

```json
{
  "priority": "critical",
  "priority_index": 0,
  "confidence": 0.78,
  "confidence_scores": {
    "critical": 0.78,
    "high": 0.12,
    "normal": 0.06,
    "low": 0.04
  },
  "model_used": "random_forest",
  "processing_time_ms": 41
}
```

### High — procurement / legal

**Request**

```json
{
  "sender": "advisor@hudsonlaw.com",
  "recipients": "client@startup.io",
  "subject": "Contract review — please advise by Wednesday",
  "body": "Following up on the draft agreement. Deadline for comments is Wednesday. Meeting scheduled Tuesday if you need to walk through clauses. Please respond with redlines.",
  "date": "2026-04-08T11:30:00"
}
```

**Illustrative response**

```json
{
  "priority": "high",
  "priority_index": 1,
  "confidence": 0.62,
  "confidence_scores": {
    "critical": 0.08,
    "high": 0.62,
    "normal": 0.24,
    "low": 0.06
  },
  "model_used": "xgboost",
  "processing_time_ms": 36
}
```

### Normal — casual internal

**Request**

```json
{
  "sender": "jamie@studio-knot.design",
  "recipients": "team@studio-knot.design",
  "subject": "Font files for the brand refresh (v3)",
  "body": "Sharing the updated OTFs in the drive. No rush — use whichever weight you prefer for mockups. Ping me if links break.",
  "date": "2026-04-07T16:45:00"
}
```

**Illustrative response**

```json
{
  "priority": "normal",
  "priority_index": 2,
  "confidence": 0.58,
  "confidence_scores": {
    "critical": 0.03,
    "high": 0.12,
    "normal": 0.58,
    "low": 0.27
  },
  "model_used": "random_forest",
  "processing_time_ms": 34
}
```

### Low — social / promo newsletter

**Request**

```json
{
  "sender": "digest@notifications.linkedin.com",
  "recipients": "",
  "subject": "People you may know — weekly roundup",
  "body": "You have 3 new connection suggestions. This is an automated notification. Unsubscribe from digests in settings. No reply needed.",
  "date": "2026-04-06T06:00:00"
}
```

**Illustrative response**

```json
{
  "priority": "low",
  "priority_index": 3,
  "confidence": 0.71,
  "confidence_scores": {
    "critical": 0.02,
    "high": 0.04,
    "normal": 0.23,
    "low": 0.71
  },
  "model_used": "random_forest",
  "processing_time_ms": 31
}
```

---

## One-line curl (unique payload)

```bash
curl -s -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"sender":"sre@northgrid.cloud","recipients":"ops@northgrid.cloud","subject":"SEV-1 outage — immediate","body":"Customer impact severe. Immediate attention. Respond now. Critical incident bridge.","date":"2026-04-10T03:40:00"}'
```

---

*Alternate dummy set for `email_priority_system/`. Does not duplicate the built-in demo buttons in the Rails “new classification” view.*
