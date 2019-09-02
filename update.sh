#!/bin/bash -e

NAME="CondensedSummaries.pdf"
cp "${APPLE_WORK}/Notes/DeepLearningNotes/${NAME}" "assets/pdf/${NAME}"
git add assets/pdf/${NAME}
git commit -m "Updated DL notes."
git push origin master

