#!/bin/bash -e

#NAME="CondensedSummaries.pdf"
#cp "${NOTES}/DeepLearningNotes/${NAME}" "assets/pdf/${NAME}"

# NAME="CS236.pdf"
# cp "${NOTES}/CS236_DeepGenerativeModels/${NAME}" "assets/pdf/${NAME}"

## declare an array variable
baseDir="${NOTES}/DeepLearningNotes"
declare -a arr=("DeepLearningBook" "PapersAndTutorials" "MiscStanford" "Textbooks" "BlogsAndAppendix")

## now loop through the above array
for name in "${arr[@]}"; do
    pdfName="${name}.pdf"
    pdfPath="${baseDir}/${pdfName}"
    cp ${pdfPath} "assets/pdf/${pdfName}"
done

git add assets/pdf
git commit -m "Updated notes."
git status
git push origin master

