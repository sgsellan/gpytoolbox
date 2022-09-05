#!/bin/bash
#Delete everything in docs except for index.md
shopt -s nullglob
for file in docs/*.md; do
	name=$(basename "$file")
	if ! [ "$name" = "index.md" ]; then
		rm "$file"
	fi
done
#Delete everything in docs subdirectories except for index.md
shopt -s nullglob
for file in docs/*/*.md; do
	name=$(basename "$file")
	if ! [ "$name" = "index.md" ]; then
		rm "$file"
	fi
done
#For all files not in a module
shopt -s nullglob
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
#For all files in a module
shopt -s nullglob
for file in src/gpytoolbox/*/*.py; do
	name=$(basename "$file" .py)
	modulename=$(basename "$(dirname $file)")
	if ! [ "$name" = "__init__" ]; then
		mkdir -p "docs/${modulename}"
		path="docs/${modulename}/${name}.md"
		echo "---" > $path
		echo "title: \"${modulename}.${name}\"" >> $path
		echo "---" >> $path
		echo >> $path
		echo "::: src.gpytoolbox.${modulename}.${name}" >> $path
	fi
done