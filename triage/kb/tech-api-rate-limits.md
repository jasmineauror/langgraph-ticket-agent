# API rate limits

The Meridian REST API is rate limited per API key:

- **Free and Starter plans:** 60 requests per minute
- **Growth plan:** 600 requests per minute
- **Enterprise plan:** 3,000 requests per minute

Limits are enforced using a sliding one-minute window. Every response includes
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers so you
can pace your client.

Exceeding the limit returns HTTP **429** with a `Retry-After` header giving the
seconds to wait. Clients should honor `Retry-After` and apply exponential backoff
with jitter rather than retrying on a fixed interval.

The bulk ingestion endpoint has a separate, higher limit and accepts up to 1,000
events per request; batching into it is the recommended approach for high-volume
writes instead of requesting a limit increase.
