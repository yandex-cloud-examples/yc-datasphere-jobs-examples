ls "$1" | wc -l > files_in_texts_directory.txt
du -hs "$2" > dataset_size.txt
printenv MY-SECRET > secret.txt
