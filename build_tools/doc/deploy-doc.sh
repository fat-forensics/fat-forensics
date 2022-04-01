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

DOC_REPO="fat-forensics/fat-forensics-doc"
DOC_SOURCE="doc/_build/html/"
DOC_TARGET="$HOME/fat-forensics-doc"

DOC_DEPLOY_USER="fat-forensics-automator"
DOC_DEPLOY_URL="https://$DOC_DEPLOY_USER:$FAT_FORENSICS_AUTOMATOR_GITHUB_TOKEN"

DEPLOY_MD5=`git rev-parse --short HEAD`
if [ -z "$1" ]; then
  DEPLOY_BRANCH=`git rev-parse --abbrev-ref HEAD`  # git branch --show-current
else
  DEPLOY_BRANCH="$1"
fi

git config user.name "GitHub Action Deploy Bot"
git config user.email "github-action-deploy@fat-forensics.org"

git clone $DOC_DEPLOY_URL@github.com/$DOC_REPO.git --depth 1 $DOC_TARGET

echo "Pushing the Documentation to GitHub: $DOC_REPO"
rsync -rclpDv --exclude=".git" --delete-before $DOC_SOURCE $DOC_TARGET
cd $DOC_TARGET
git add -f *
if [ -f ".nojekyll" ]; then git add -f .nojekyll; fi
if [ -f ".buildinfo" ]; then git add -f .buildinfo; fi

# Check if anything has actually changed
GIT_CHANGES=`git status --porcelain`

if [ -z "$GIT_CHANGES" ]; then
  echo "Nothing to commit. No documentation deployment necessary."
else
  git status
  git commit -am "Deploying documentation from $DEPLOY_BRANCH:$DEPLOY_MD5"
  git push origin master
  git status
fi
