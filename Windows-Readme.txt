
Windows Readme.
Author: Robert Bell
Contact: bertie@cs.uoregon.edu
Date: 20th December 2001

Supported Systems: Windows9x/NT/2000/XP.
Compiler:  Microsoft Visual C++ Version 6.0.

Section1.

The following steps detail how to build TAU libraries on Windows9x/NT/2000/XP.

For illustrative purposes, we assmue that the TAU root directory is:
"C:\TAU-SOURCE-DIR". 

1) Download TAU.  TAU is distributed as source and prebuilt libraries forWindows.
    If you wish to use the prebuilt libraries, skip to steps 25 and 26.

2) Open Microsoft Visual C++ ... henceforth referred to as VC++.

3)i) If you wish to create a dynamic library proceed to step 4.
   ii) If you wish to create a static library proceed to step 12.

4) Creating a dynamic library allows you to profile Java code using Sun's JDK1.2+.

5) From the "File" menu in VC++, select "New".

6) Click on the "Projects" tab.

7) Select "Win32 Dynamic-Linked Library".

8) Type in a name for your new library.

9) Make sure that the radio button on the right of the new project window is set to 
    "Create a new workspace".

10) Click "OK"

11)  Please skip to step 18 below. 

12) From the "File" menu in VC++, select "New".

13) Click on the "Projects" tab.

14) Select "Win32 Static Library".

15) Type in a name for your new library.

16) Make sure that the radio button on the right of the new project window is set to
      "Create a new workspace".

17) Click "OK"

18) Open Windows Explorer, and, from the TAU source you downloaded, 
       copy the C:\TAU-SOURCE-DIR\include\Profile and C:\TAU-SOURCE-DIR\src\Profile
       directories to your new project directory.  For example, if you new project 
       was located in "C:\Program Files\DevStudio\MyProjects\NewTauLib", you would
       now have two new subdirectories of "C:\Program Files\DevStudio\MyProject\NewTauLib"
       named, "include\Profile" and "src\Profile".

19) Now, back in VC++, from the "Project" menu, select "Add To Project" and click 
      on "Files".  Move to your new "src\Profile" directory and select the following list of
      files: (holding down the control key whilst clicking so that you can select more than one file)

		FunctionInfo.cpp
		JavaThreadLayer.cpp
		Profiler.cpp
		RtsLayer.cpp
		RtsThread.cpp
		TauJava.cpp
		TauMapping.cpp
		UserEvent.cpp
		WindowsThreadLayer.cpp

    Now click OK.

20) From the "Project" again, select "Settings" and then click on the "C/C++" tab.

8) Make sure the "Catagory" is "General".

9) Set the warning level to none.

21) Under "Preprocessor definitions" add: (separated by commas)

		TAU_WINDOWS
		TAU_DOT_H_LESS_HEADERS
		PROFILING_ON

	    If you want to profile a Java application, also add:

		JAVA

	   Click "OK"

22) Under "Additional include directories", set the include directory from
    the copied over include directory, ie .\include. Also add the include directories for jvmpi.h
    and jni_md.h.  These are typically in "C:\JAVA_ROOT_DIR\include" and "C:\JAVA_ROOT_DIR\include\win32".
    This is a comma separated list.
    Thus, when done, you should have three new include directories listed.
    For example: .\include,C:\jdk1.3.1_01\include,C:\jdk1.3.1_01\include\win32
    Now click "OK".


23)  Now, from the "Build" menu, select "Build PROJECT_NAME.dll (or .lib)"

24)  Ignoring warnings, you should now have a library file in your project debug directory.

25)  If you created a dll for use with Java, you only need to make sure that the dll is
     in a location that can be found by Java when it is running.  The command to profile
     your Java application is: java -Xrunlibtau "Java Application Name" "Application parameters".
     The default libtau.dll for use with a Java app. is provided in: "C:\TAU-SOURCE-DIR\windows\lib".
     If, when building your dll from the source, you named it something other than libtau.dll, you can
     either rename it, or replace "libtau" in "java -Xrunlibtau" with your dll name.

26) If you created a static library, you will need to include a reference to it in when you
      build your application.  You can do this by adding the library file to you list of
      libraries in "Project -> Settings -> Link" inside VC++.  You must then make sure
      that the library is in a location know to VC++.  You can do this in your 
      "Tools ->Options->Directories->Library files" section of VC++


Section 2.

The Windows port ships with a prebuilt version of pprof which can be used to view your profiling data
(See the TAU documentation for more details).  Make sure that pprof.exe is in your current path.  It can
be found in C:\TAU-SOURCE-DIR\windows\bin.  In addition, there is also a java version of Racy called jRacy
which can also be found in C:\TAU-SOURCE-DIR\windows\bin.  For instructions as to how to run jRacy, please
see the Readme-jRacy.txt in said directory.

For information on how to profile your C/C++ and Java code, please see the TAU documentation.

NOTE:  Please read the README.JAVA file for important information regarding the use of java -Xrun.
