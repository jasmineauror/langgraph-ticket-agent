# Exporting your data

There are three ways to get data out of Meridian:

**Dashboard CSV export.** Any chart or table has an **Export → CSV** option in its
overflow menu. This exports the data currently in view, respecting active filters,
up to 50,000 rows.

**Scheduled exports.** Under **Settings → Exports** you can schedule a daily or
weekly CSV or Parquet drop to an S3 bucket or Google Cloud Storage bucket you own.
Available on Growth and Enterprise plans.

**Full workspace export.** Under **Settings → Data → Request full export** you can
request a complete archive of your workspace's raw event data. The archive is
prepared asynchronously and you receive a download link by email, typically within a
few hours. The link expires after 7 days.

Exports include raw events and dashboard definitions. They do not include audit logs.
