#!/bin/bash
#title       :deploy-doc.sh
#description :Deploys documentation to a GitHub repo (served via GitHub Pages)
#author      :Kacper Sokol <k.sokol@bristol.ac.uk>
#license     :new BSD
#==============================================================================

set -e

PWD=`pwd`
if [[ $PWD != */fat-forensics ]]; then
  echo "This script must be run from the root directory of this repository!"
  exit 1
fi

KEY_ENC="build_tools/doc/fat-forensics-deploy-key.enc"
KEY="build_tools/doc/fat-forensics-deploy-key"

DOC_REPO="fat-forensics/fat-forensics-doc"
DOC_SOURCE="doc/_build/html/*"
DOC_TARGET="$HOME/fat-forensics-doc"

DEPLOY_MD5=`git rev-parse --short HEAD`
if [ -z "$1" ]; then
  DEPLOY_BRANCH=`git rev-parse --abbrev-ref HEAD`  # git branch --show-current
else
  DEPLOY_BRANCH="$1"
fi

git config user.name "Travis Deploy Bot"
git config user.email "travis-deploy@fat-forensics.org"

git clone https://fat-forensics-automator:$FAT_FORENSICS_AUTOMATOR_GITHUB_TOKEN@github.com/$DOC_REPO.git --depth 1 $DOC_TARGET

echo "Pushing the Documentation to GitHub: $DOC_REPO"
cp -r $DOC_SOURCE $DOC_TARGET
cd $DOC_TARGET
git add -f *
if [ -f ".nojekyll" ]; then git add -f .nojekyll; fi

# Check if anything has actually changed
GIT_CHANGES=`git status --porcelain`

if [ -z "$GIT_CHANGES" ]; then
  echo "Nothing to commit. No documentation deployment necessary."
else
  git commit -am "Deploying documentation from $DEPLOY_BRANCH:$DEPLOY_MD5"
  git push origin master
fi
