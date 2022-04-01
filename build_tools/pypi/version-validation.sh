#!/bin/bash
#title       :version-validation.sh
#description :Checks for match between git tag and FAT Forensics version
#author      :Kacper Sokol <k.sokol@bristol.ac.uk>
#license     :new BSD
#==============================================================================

FATF_VERSION=$(python -c "import fatf; print(fatf.__version__)")

# git describe --tags
GIT_TAG=$(git tag --points-at HEAD)

if [ -z "$GITHUB_TAG" ]; then
  if [ -z "$GIT_TAG" ]; then
    echo "This git commit is not tagged. Cannot create a release."
    exit 1
  else
    if [ "$GIT_TAG" == "$FATF_VERSION" ]; then
      echo "Safe to deploy FAT Forensics version $FATF_VERSION."
    else
      echo "The fatf.__varsion__ ($FATF_VERSION) and git tag ($GIT_TAG) do" \
        "not agree."
      exit 1
    fi
  fi
else
  if [ "$GIT_TAG" == "$GITHUB_TAG" ]; then
    if [ "$GITHUB_TAG" == "$FATF_VERSION" ]; then
      echo "Safe to deploy FAT Forensics version $FATF_VERSION."
    else
      echo "The fatf.__varsion__ ($FATF_VERSION) and git tag ($GIT_TAG) do" \
        "not agree."
      exit 1
    fi
  else
    echo "Internal error: the GitHub tag ($GITHUB_TAG) and git tag" \
      "($GIT_TAG) do not agree."
    exit 1
  fi
fi
