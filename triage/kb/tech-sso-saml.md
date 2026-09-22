# SSO and SAML setup

SAML 2.0 single sign-on is available on the **Enterprise** plan. Configure it under
**Settings → Security → SAML SSO**.

You will need to exchange metadata with your identity provider: Meridian shows its
ACS URL and Entity ID on that page, and you supply your IdP's sign-on URL and X.509
signing certificate. We have verified setup guides for Okta, Microsoft Entra ID,
Google Workspace, and OneLogin.

**Just-in-time provisioning** is supported. A user who authenticates successfully but
has no Meridian account gets one created automatically with the default role you
choose. SCIM provisioning for automated deprovisioning is also available.

Once SSO is **enforced**, password login is disabled for all members except Owners,
who retain a password fallback so the workspace cannot be locked out by an IdP
misconfiguration.

Test SSO with the **Test connection** button before enforcing it.
