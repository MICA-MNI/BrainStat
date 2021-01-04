DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
[[ ! -d $DIR/python_doc/autodoc ]] && mkdir $DIR/python_doc/autodoc
echo "Deleting old autodoc files."
find $DIR/python_doc/autodoc/ -maxdepth 1 -name "*.rst" -type f -delete -print
echo ""
echo "Creating new autodoc files."
sphinx-apidoc -f -o $DIR/python_doc/autodoc $DIR/../brainstat
