# Secrets Directory

This directory should contain sensitive files that are required for the project but should **NOT** be committed to version control.

## Required Files

Place the following files in this directory:

### `client_secrets.json`
OAuth client secrets file for Google API authentication (if using Google Drive integration).

### `ssh_key.ssh`
Private SSH key for secure connections.

### `ssh_key.ssh.pub`
Public SSH key corresponding to the private key.

### `zenodo_token.txt`
API token for Zenodo uploads (get from https://zenodo.org/account/settings/applications/tokens/).

## Important Notes

- **DO NOT** commit these files to Git
- These files are excluded via `.gitignore`
- If you need to share these files with collaborators, use secure methods (encrypted file sharing, password managers, etc.)
- For production deployments, use environment variables or secure secret management systems instead of storing files

## Setup

1. Copy your secret files to this directory
2. Ensure proper file permissions (e.g., `chmod 600` for SSH keys)
3. Verify that `.gitignore` is properly configured to exclude these files

