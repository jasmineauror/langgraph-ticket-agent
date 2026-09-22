# What happens when a payment fails

If a scheduled charge fails, Meridian retries it automatically three times: after
3 days, 5 days, and 10 days. The billing contact receives an email on each failure.

During this retry window your workspace stays fully active. There is no immediate
loss of access.

If all three retries fail, the workspace moves to a **read-only** state on day 14.
In read-only mode existing dashboards and data remain visible and exportable, but
new data ingestion is paused and you cannot invite members or create new
dashboards.

Updating the payment method and running **Retry payment** restores full access
within a few minutes. Data is not deleted when a workspace goes read-only; ingestion
resumes from the point it paused, though data that arrived during the paused window
is not backfilled.
