# Webhooks and delivery retries

Configure webhook endpoints under **Settings → Developers → Webhooks**. Each endpoint
subscribes to one or more event types and must respond with a 2xx status within
10 seconds.

Every request carries an `X-Meridian-Signature` header: an HMAC-SHA256 of the raw
request body using your endpoint's signing secret. Always verify this signature
against the raw body before trusting a payload, and compare digests in constant time.

**Retries.** A non-2xx response or a timeout triggers retries with exponential
backoff over roughly 24 hours: at 1 minute, 5 minutes, 30 minutes, 2 hours, and
12 hours. After the final attempt the delivery is marked failed and is not retried.

An endpoint returning errors for 24 hours straight is automatically disabled and the
workspace Owners are emailed. Failed deliveries are visible for 7 days under the
endpoint's **Recent deliveries** tab, where each can be replayed individually.

Webhook deliveries are at-least-once, so make your handlers idempotent using the
`event_id` field.
