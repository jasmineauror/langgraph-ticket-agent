# Available integrations

Meridian connects to other tools under **Settings → Integrations**.

**Data sources (inbound):** Postgres, MySQL, Snowflake, BigQuery, Redshift, Stripe,
Segment, and generic webhook ingestion. Warehouse sources sync on a schedule you set,
with a minimum interval of 15 minutes.

**Destinations (outbound):** Slack, Microsoft Teams, PagerDuty, and email for alert
delivery; S3 and GCS for scheduled data exports.

**Reverse ETL** to push computed Meridian metrics back into Salesforce or HubSpot is
available on the Enterprise plan.

Each integration is authorized per workspace by an Owner or Admin. Connecting a
source does not begin a sync until you select which tables or streams to include, so
you can connect and configure before any data moves.

If a sync fails, the integration's detail page shows the last error and the last
successful sync timestamp.
