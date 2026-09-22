# API keys and authentication

Create API keys under **Settings → Developers → API keys**. Pass the key as a bearer
token on every request:

```
Authorization: Bearer mk_live_xxxxxxxx
```

Keys are shown **only once** at creation time. If you lose a key you must revoke it
and create a new one; it cannot be recovered.

Each key carries a scope, either `read` or `read_write`, fixed at creation. To change
a key's scope, create a new key with the scope you need and revoke the old one.

Keys are workspace-scoped, not user-scoped, so a key keeps working after the person
who created it leaves the workspace. For that reason we recommend naming keys after
the system that uses them rather than after a person.

Revoking a key takes effect immediately and cannot be undone.
