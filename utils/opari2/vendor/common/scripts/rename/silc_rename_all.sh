#!/bin/sh


WHAT="common"

echo "\n"
echo " ### checkout + rename ### "$WHAT
read -p "[press enter]" enter
sh ./silc_rename_script.sh $WHAT"-root/trunk/" $WHAT"_src"

echo "\n"
echo " ### done ### "$WHAT
read -p "[press enter]" enter



for WHAT in "utility" "otf2" "opari2" "silc"; do

    echo "\n"
    echo " ### checkout + rename ### "$WHAT
    read -p "[press enter]" enter
    sh ./silc_rename_script.sh $WHAT"-root/trunk/" $WHAT"_src"


    echo "\n"
    echo " ### bootstrap ### "$WHAT
    read -p "[press enter]" enter
    cd $WHAT"_src"
    ./bootstrap
    cd ../

    echo "\n"
    echo " ### configure ### "$WHAT
    read -p "[press enter]" enter
    mkdir $WHAT"_build"
    cd $WHAT"_build"
    ../$WHAT"_src"/configure --prefix=/tmp/ssssssss

    echo "\n"
    echo " ### make ### "$WHAT
    read -p "[press enter]" enter
    make

    echo "\n"
    echo " ### make check ### "$WHAT
    read -p "[press enter]" enter
    make check

    echo "\n"
    echo " ### make distcheck ### "$WHAT
    read -p "[press enter]" enter
    make distcheck

    echo "\n"
    echo " ### done ### "$WHAT
    read -p "[press enter]" enter
    cd ../

done
