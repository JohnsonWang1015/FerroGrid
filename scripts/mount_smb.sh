#!/usr/bin/env bash
# Mount a Samba/CIFS share on a GPU server so FerroGrid jobs can --mount it.
#
#   ./scripts/mount_smb.sh <ssh-host> //server/share /mnt/point <smb-user> [port]
#
# FerroGrid itself needs no Samba support: `--mount` bind-mounts any host path,
# and the container does not care whether it is ext4, NFS or CIFS. What does
# need care is the host-side mount, because CIFS fixes file ownership at mount
# time -- get uid/gid wrong and jobs (which run as your uid, not root) cannot
# write to the share.
#
# Requires sudo on the target host; you will be prompted.
set -euo pipefail

HOST="${1:?usage: mount_smb.sh <ssh-host> //server/share /mnt/point <smb-user> [port]}"
SHARE="${2:?missing share, e.g. //140.123.105.254/esl}"
POINT="${3:?missing mount point, e.g. /mnt/share}"
SMBUSER="${4:?missing SMB username}"
PORT="${5:-445}"

[[ "$SHARE" == //* ]] || { echo "share must look like //server/name"; exit 1; }

read -rsp "SMB password for $SMBUSER: " SMBPASS; echo

# The remote uid/gid decide who owns the files once mounted. Match the account
# the agent runs as, which is the account that runs the training containers.
read -r RUID RGID <<<"$(ssh "$HOST" 'id -u; id -g' | tr '\n' ' ')"
echo "==> [$HOST] mounting as uid=$RUID gid=$RGID"

# Password goes over the SSH channel into a 0600 root-owned credentials file,
# never onto the command line where `ps` would show it.
SMBPASS="$SMBPASS" ssh -t "$HOST" "
set -euo pipefail
sudo -v

if ! command -v mount.cifs >/dev/null; then
    echo '==> installing cifs-utils'
    sudo apt-get update -qq && sudo apt-get install -y -qq cifs-utils
fi

CRED=/etc/ferrogrid-smb-$(echo '$SHARE' | tr -c 'A-Za-z0-9' '-').cred
sudo install -m 600 /dev/null \"\$CRED\"
printf 'username=%s\npassword=%s\n' '$SMBUSER' \"\$SMB_PASSWORD\" | sudo tee \"\$CRED\" >/dev/null

sudo mkdir -p '$POINT'

OPTS=\"credentials=\$CRED,uid=$RUID,gid=$RGID,file_mode=0664,dir_mode=0775,iocharset=utf8,vers=3.0,_netdev,nofail\"
[[ '$PORT' != 445 ]] && OPTS=\"\$OPTS,port=$PORT\"

# Replace any existing fstab line for this mount point, then mount from fstab
# so a reboot brings it back the same way.
sudo sed -i \"\\|[[:space:]]$POINT[[:space:]]|d\" /etc/fstab
echo \"$SHARE $POINT cifs \$OPTS 0 0\" | sudo tee -a /etc/fstab >/dev/null

mountpoint -q '$POINT' && sudo umount '$POINT'
sudo mount '$POINT'
echo '==> mounted:'
mount | grep \" on $POINT \"
touch '$POINT/.ferrogrid-write-test' && rm -f '$POINT/.ferrogrid-write-test' \
    && echo '==> writable by this user' \
    || echo '!! NOT writable -- check the uid/gid and the share permissions'
" SMB_PASSWORD="$SMBPASS"

echo
echo "==> done. Use it in a job with:"
echo "      ferro train ... --mount $POINT ..."
