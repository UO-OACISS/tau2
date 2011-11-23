#! /bin/sh

# default repositories
repos="silc otf2 opari2 utility"

# take repos from the command line, if any
if [ $# -gt 0 ]; then
    repos="$*"
fi

# use absolute path to svnlook
SVNLOOK=/usr/bin/svnlook

PRE_COMMIT="$(mktemp)"

$SVNLOOK cat /svn-base/common-root trunk/beautifier/pre-commit > "$PRE_COMMIT" &&
    chmod 0755 "$PRE_COMMIT" &&
    chown apache:svnsilc "$PRE_COMMIT"
if [ $? -ne 0 ]; then
    echo "Can't generate pre-commit hook." >&2
    exit 1
fi

for repo in $repos
do
    if [ ! -d /svn-base/$repo-root/hooks ]; then
        echo >&2 "No repository for $repo."
        continue
    fi

    if cp --archive --backup=numbered "$PRE_COMMIT" /svn-base/$repo-root/hooks/pre-commit; then
        echo "pre-commit hook installed for repository $repo."
    else
        echo >&2 "Installation failed for repository $repo."
    fi
done

rm -f "$PRE_COMMIT"
