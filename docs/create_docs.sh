#!/bin/bash
for file in src/gpytoolbox/*.py; do
	name=$(basename "$file" .py)
	if ! [ "$name" = "__init__" ]; then
		path="docs/${name}.md"
		echo "---" > $path
		echo "title: \"${name}\"" >> $path
		echo "---" >> $path
		echo >> $path
		echo "::: src.gpytoolbox.${name}" >> $path
	fi
done
