DIRS="include src tools test vendor/otf2/src vendor/otf2/include vendor/otf2/test vendor/opari2 vendor/otf2/vendor/utility"

for i in $DIRS; do
  grep -R -n -i "todo" $i | grep -v .svn | grep -v "backup" | grep -v "GENERATE_TODOLIST" | grep -v "gcc-mpich2" | grep -v "build-config/ltmain.sh" | grep -v doxygen | grep -v count_todo_comments.sh | grep -v "vendor/otf2/src/stuff/Prototype.tar" 
done