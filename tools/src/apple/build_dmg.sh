#!/bin/bash

# Waring this will override any old TAU distribution and remove any files in
# "/Applications/TAU"
#
# argument 1: new tau2 tarball (or "skip" if TAU distro is already in
# /Applications/TAU)
# argument 2: tau version number
if [ $# -ne 2 ] ; then
	echo "Usage: build_dmg.sh <TAU tarball> <version>"
	echo "If the correct version of TAU is already been placed in
/Applications/TAU let the first argument be \"skip\"."
	exit
fi

if `test "$1" != "skip"`; then
	TAR_BALL=$1
	rm -rf ../TAU
	mkdir ../TAU
	cd ../TAU
	cp $1 .
	tar xzf $1
	mv tau-${2} tau
	cd tau
	echo "TAU source code"
	ls -l

	#configure TAU with GNU compilers targeting both Intel and PowerPC
	#architectures (universal binary).
	./configure -cc=gcc -c++=g++ -useropt=-arch\ i386\ -arch\ ppc
	make install
	cd ..
	cp -r tau ../TAU_new_upgrade/TAU
	cd ../TAU_new_upgrade
else
	echo "Skipping TAU build"
fi

DMG_NAME="tau-${2}.tmp.dmg"
DMG_FINAL_NAME="tau-${2}.dmg"
DMG_TITLE="TAU"

rm *.dmg

#cp disk_template/template.dmg $DMG_NAME
rm -rf work_dir
#mkdir work_dir

#mount template dmg.
hdiutil attach disk_template/new_template.dmg -noautoopen -mountpoint work_dir

#chmod u+wX work_dir/
#hdiutil create -fs HFSX -layout SPUD -size 600m $DMG_NAME -srcfolder template -format UDRW -volname $DMG_TITLE
#hdiutil attach $DMG_NAME -noautoopen -mountpoint work_dir
echo "Attached empty image."

#copy TAU into template.
cp -r /Applications/TAU/tau work_dir/TAU/.

#cp -r .background work_dir/.
#cp .VolumeIcon.icns work_dir/.
#cd work_dir
#ln -s /Applications "Applications"
#cd ..
echo `ls -l work_dir`
echo "populated image."

# detach template image.
WC_DEV=`hdiutil info | grep "work_dir" | grep "Apple_HFS" | awk '{print $1}'`
echo "disk image is: $WC_DEV"
echo "hdiutil detach $WC_DEV"
hdiutil detach $WC_DEV
echo "Detached image."

# compress dmg.
hdiutil convert disk_template/new_template.dmg -format UDZO -imagekey zlib-level=9 -o $DMG_FINAL_NAME
echo "converted image."
rm -rf $DMG_NAME
