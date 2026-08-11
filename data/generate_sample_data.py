"""
generate_sample_data.py

Run:
    python data/generate_sample_data.py
Produces:
    data/support_tickets.csv
    data/knowledge_base/*.txt   (documents for the RAG pipeline to index)
"""

import random
import pandas as pd
from pathlib import Path

random.seed(42)

CATEGORIES = ["billing", "technical", "account", "general"]
URGENCY = ["low", "medium", "high"]

TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month.",
        "My invoice for {month} shows the wrong amount, can you check?",
        "I want a refund for the {plan} plan, it was charged by mistake.",
        "The payment failed but money was still deducted from my card.",
        "Can you explain the extra charge on my last bill?",
    ],
    "technical": [
        "The app keeps crashing when I try to upload a file on {platform}.",
        "I can't log in, it says 'server error' every time on {platform}.",
        "The API is returning a 500 error when I call the /users endpoint.",
        "Sync between devices is not working since the last update.",
        "The dashboard is loading extremely slowly today.",
    ],
    "account": [
        "I forgot my password and the reset email never arrives.",
        "How do I change the email linked to my account?",
        "I want to delete my account and all my data permanently.",
        "My account got locked after too many login attempts.",
        "Can I merge two accounts under one email?",
    ],
    "general": [
        "Does your {plan} plan include priority support?",
        "What are your business hours for support on {platform}?",
        "I love the product, just wanted to say thanks!",
        "Do you have a public roadmap for upcoming features?",
        "Is there a student discount available?",
    ],
}

PLANS = ["Basic", "Pro", "Enterprise", "Team"]
MONTHS = ["January", "February", "March", "April", "May", "June"]
PLATFORMS = ["iOS", "Android", "Web", "Desktop"]


def make_ticket(ticket_id: int) -> dict:
    category = random.choice(CATEGORIES)
    template = random.choice(TEMPLATES[category])
    text = template.format(
        plan=random.choice(PLANS),
        month=random.choice(MONTHS),
        platform=random.choice(PLATFORMS),
    )

    # Urgency correlates loosely with category (billing/technical skew higher)
    if category in ("billing", "technical"):
        urgency = random.choices(URGENCY, weights=[0.2, 0.4, 0.4])[0]
    else:
        urgency = random.choices(URGENCY, weights=[0.5, 0.35, 0.15])[0]

    return {
        "ticket_id": ticket_id,
        "text": text,
        "category": category,
        "urgency": urgency,
    }


def main():
    out_dir = Path(__file__).parent
    n = 600
    rows = [make_ticket(i) for i in range(1, n + 1)]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "support_tickets.csv", index=False)
    print(f"Wrote {len(df)} rows to {out_dir / 'support_tickets.csv'}")

    # Knowledge base documents for the RAG pipeline (simulates a help center)
    kb_dir = out_dir / "knowledge_base"
    kb_dir.mkdir(exist_ok=True)

    docs = {
        "billing_faq.txt": (
            "Billing FAQ\n"
            "Refunds are processed within 5-7 business days to the original "
            "payment method. Duplicate charges are automatically detected "
            "within 24 hours and reversed; if not, contact support with your "
            "invoice ID. You can view and download invoices from "
            "Account > Billing > Invoices. Plan changes are prorated."
        ),
        "technical_faq.txt": (
            "Technical Support FAQ\n"
            "If the app crashes on upload, first check the file size limit "
            "(500MB). Server error 500 on login usually indicates a session "
            "token issue; clearing cache resolves it in most cases. Sync "
            "issues after an update are resolved by force-closing the app "
            "and logging back in. API 500 errors on /users should be "
            "reported with the request ID from response headers."
        ),
        "account_faq.txt": (
            "Account Management FAQ\n"
            "Password reset emails can take up to 10 minutes; check spam "
            "folder first. To change your account email, go to "
            "Settings > Profile > Email and verify the new address. "
            "Account deletion is permanent after a 30-day grace period. "
            "Accounts cannot currently be merged automatically; contact "
            "support for a manual merge request."
        ),
        "general_faq.txt": (
            "General Information\n"
            "Support hours are Monday-Friday, 9am-6pm in your local "
            "timezone, with 24/7 support for Enterprise plans. Pro and "
            "Enterprise plans include priority support with under-2-hour "
            "response times. Students with a valid .edu email get 50% off "
            "any paid plan. The product roadmap is public and updated "
            "quarterly."
        ),
    }
    for filename, content in docs.items():
        (kb_dir / filename).write_text(content)
    print(f"Wrote {len(docs)} knowledge-base docs to {kb_dir}")


if __name__ == "__main__":
    main()
