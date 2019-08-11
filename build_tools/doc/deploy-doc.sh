#!/bin/bash
# Deploys documentation to GitHub repo to be served via GitHub Pages
set -e

PWD=`pwd`
if [[ $PWD != */fat-forensics ]]; then
  echo "This script must be run from the root directory of this repository!"
  exit 1
fi

KEY_ENC="build_tools/doc/fat-forensics-deploy-key.enc"
KEY="build_tools/doc/fat-forensics-deploy-key"

DOC_REPO="So-Cool/fat-forensics-doc"
DOC_SOURCE="doc/_build/html/*"
DOC_TARGET="$HOME/fat-forensics-doc"

DEPLOY_MD5=`git rev-parse --short HEAD`
DEPLOY_BRANCH=`git branch --show-current`

openssl aes-256-cbc \
  -K $encrypted_6e9d23dcb3fc_key \
  -iv $encrypted_6e9d23dcb3fc_iv \
  -in $KEY_ENC \
  -out $KEY \
  -d

chmod 600 $KEY
eval `ssh-agent -s`
ssh-add $KEY
git config user.name "Travis Deploy Bot"
git config user.email "travis-deploy@fat-forensics.org"

git clone git@github.com:$DOC_REPO.git --depth 1 $DOC_TARGET

echo "Pushing the Documentation to GitHub: $DOC_REPO"
cp -r $DOC_SOURCE $DOC_TARGET
cd $DOC_TARGET
git add *
git commit -am "Deploying documentation from $DEPLOY_BRANCH:$DEPLOY_MD5"
git push origin master
