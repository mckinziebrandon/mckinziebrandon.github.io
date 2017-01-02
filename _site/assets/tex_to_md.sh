# Convert my custom LaTeX commands into suitable markdown.

# Sed syntax:
#   sed -i 's/original/new/g' file.txt
#   --> '-i' = in-place (i.e. save back to original file).

# Vim in Ex mode:
#   ex -sc '%s/OLD/NEW/g|x' file
#   --> 'x' means 'save and close'.

# First, make copies of all *.tex with *.md extension.
for file in `ls *.tex`; do
    cp "${file}" `echo "${file}" | sed 's/\.tex$/\.md/'`
done


for file in `ls *.md`; do

    echo "Looking in ${file} . . . "

    # Naive approach 1: just delete the stuff at front.
    # sed 's/^\\p \\blue{/__/g' "${file}"

    # Naive approach 2: destroy title but at least get correct format.
    sed -i 's/^\\p \\blue{[^}]*}/__EMPTYTITLE__/g' "${file}"

done
