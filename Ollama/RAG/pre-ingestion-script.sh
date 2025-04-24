#!/bin/bash

# Pull docs from GitHub
git clone https://github.com/mycompany/docs.git /mnt/500GB/docs/user_guide

# Check out source from SVN
svn checkout https://svn.mycompany.com/repo/trunk/src /mnt/500GB/docs/source_code

# Grab release notes
curl https://internal.portal/releases/v1.4.0.md -o /mnt/500GB/docs/release_notes/v1.4.0.md
