package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Vector;

public class GprofDataSource extends DataSource {

    //    private int indexStart = 0;
    //    private int percentStart = 0;
    //    private int selfStart = 0;
    //    private int descendantsStart = 0;
    //    private int calledStart = 0;
    //    private int nameStart = 0;
    private boolean fixNames = false;
    private int linenumber = 0;
    private boolean fixLengths = true;

    private String currentFile;

    public GprofDataSource(File[] files, boolean fixNames) {
        super();
        this.fixNames = fixNames;
        this.files = files;
    }

    private File files[];

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        return 0;
    }

    public void load() throws FileNotFoundException, IOException {
        //Record time.
        long time = System.currentTimeMillis();

        //######
        //Frequently used items.
        //######
        Function function = null;
        FunctionProfile functionProfile = null;

        Node node = null;
        Context context = null;
        edu.uoregon.tau.perfdmf.Thread thread = null;
        int nodeID = -1;

        String inputString = null;

        Function callPathFunction = null;

        //######
        //End - Frequently used items.
        //######

        for (int fIndex = 0; fIndex < files.length; fIndex++) {
            File file = files[fIndex];
            currentFile = files[fIndex].toString();

            //System.out.println("Processing " + file + ", please wait ......");
            FileInputStream fileIn = new FileInputStream(file);
            InputStreamReader inReader = new InputStreamReader(fileIn);
            BufferedReader br = new BufferedReader(inReader);

            // Since this is gprof output, there will only be one node, context, and thread per file
            node = this.addNode(++nodeID);
            context = node.addContext(0);
            thread = context.addThread(0);

            // Time is the only metric tracked with gprof.
            this.addMetric("Time");

            boolean callPathSection = false;
            boolean parent = true;
            Vector<LineData> parents = new Vector<LineData>();
            LineData self = null;
            Vector<LineData> children = new Vector<LineData>();

            fixLengths = true;
            linenumber = 1;
            while ((inputString = br.readLine()) != null) {
                if (inputString.startsWith("gmon") && linenumber == 1) {
                    throw new DataSourceException(
                            "Error: this appears to be a gmon.out file, please run gprof > gprof.out and load gprof.out instead");
                }
                int length = inputString.length();
                if (length != 0) {
                    // The first time we see g, set the call path section to be true,
                    // and the second time, set it to be false.
                    // check for "granularity: " or "ngranularity: "
                    int idx = inputString.indexOf("granularity: ");
                    if (idx == 0 || idx == 1) {
                        if (!callPathSection) {
                            // System.out.println("###### Call path section ######");
                            callPathSection = true;
                        } else {
                            // System.out.println("###### Summary section ######");
                            callPathSection = false;
                        }
                    }

                    if (callPathSection) {
                        if ((inputString.indexOf("index") == 0) && (inputString.indexOf("time") >= 0)
                                && (inputString.indexOf("self") >= 0) && (inputString.indexOf("called") >= 0)
                                && (inputString.indexOf("name") >= 0)) {
                        } else if (inputString.charAt(0) == '[') {
                            self = getSelfLineData(inputString);
                            parent = false;
                        } else if (inputString.charAt(0) == '-') {

                            function = this.addFunction(self.s0, 1);
                            function.addGroup(addGroup("TAU_DEFAULT"));

                            functionProfile = new FunctionProfile(function);
                            thread.addFunctionProfile(functionProfile);

                            functionProfile.setInclusive(0, self.d1 + self.d2);
                            functionProfile.setExclusive(0, self.d1);
                            functionProfile.setNumCalls(self.i0);

                            int numSubr = 0;
                            for (int i = 0; i < children.size(); i++) {
                                LineData lineDataChild = children.get(i);
                                numSubr += lineDataChild.i0;
                            }

                            functionProfile.setNumSubr(numSubr);
                            //functionProfile.setInclusivePerCall(0, (self.d1 + self.d2) / self.i0);

                            for (int i = 0; i < parents.size(); i++) {
                                LineData lineDataParent = parents.elementAt(i);
                                function = this.addFunction(lineDataParent.s0, 1);
                                function.addGroup(addGroup("TAU_DEFAULT"));
                                //String s = lineDataParent.s0 + " => " + self.s0 + "  ";
                                callPathFunction = this.addFunction(lineDataParent.s0 + " => " + self.s0 + "  ", 1);
                                callPathFunction.addGroup(addGroup("TAU_DEFAULT"));
                                callPathFunction.addGroup(addGroup("TAU_CALLPATH"));

                                functionProfile = new FunctionProfile(callPathFunction);
                                thread.addFunctionProfile(functionProfile);
                                functionProfile.setInclusive(0, lineDataParent.d0 + lineDataParent.d1);
                                functionProfile.setExclusive(0, lineDataParent.d0);
                                functionProfile.setNumCalls(lineDataParent.i0);

                                //  functionProfile.setInclusivePerCall(0,
                                //         (lineDataParent.d0 + lineDataParent.d1) / lineDataParent.i0);

                            }
                            parents.clear();

                            for (int i = 0; i < children.size(); i++) {
                                LineData lineDataChild = children.elementAt(i);
                                function = this.addFunction(lineDataChild.s0, 1);
                                function.addGroup(addGroup("TAU_DEFAULT"));
                                //String s = self.s0 + " => " + lineDataChild.s0 + "  ";
                                callPathFunction = this.addFunction(self.s0 + " => " + lineDataChild.s0 + "  ", 1);
                                callPathFunction.addGroup(addGroup("TAU_DEFAULT"));
                                callPathFunction.addGroup(addGroup("TAU_CALLPATH"));

                                functionProfile = new FunctionProfile(callPathFunction);
                                thread.addFunctionProfile(functionProfile);
                                functionProfile.setInclusive(0, lineDataChild.d0 + lineDataChild.d1);
                                functionProfile.setExclusive(0, lineDataChild.d0);
                                functionProfile.setNumCalls(lineDataChild.i0);
                            }
                            children.clear();
                            parent = true;
                        } else if (inputString.charAt(length - 1) == ']') {
                            // check for cycle line
                            if (inputString.indexOf("<cycle") >= 0) {
                                if (parent) {
                                    parents.add(getParentChildLineData(inputString));
                                } else {
                                    children.add(getParentChildLineData(inputString));
                                }
                            } else {
                                if (parent) {
                                    parents.add(getParentChildLineData(inputString));
                                } else {
                                    children.add(getParentChildLineData(inputString));
                                }
                            }
                        }
                    } else if (inputString.charAt(length - 1) == ']') {
                        // System.out.println(getSummaryLineData(inputString).s0);
                    }
                }
                linenumber++;
            } // while lines in file
        } // for elements in vector v

        this.generateDerivedData();

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Done processing data!");
        //System.out.println("Time to process (in milliseconds): " + time);
    }

    //####################################
    //Private Section.
    //####################################

    //######
    //Gprof.dat string processing methods.
    //######

    //    private void getFieldLengths(String string) {
    //
    //        System.out.println("FIELD LENGTHS@!!!!");
    //
    //        /*
    //         * parse a line that looks like: index %time self childen
    //         * called+self name index ...or... index % time self children called
    //         * name index [xxxx] 100.0 xxxx.xx xxxxxxxx.xx xxxxxxx+xxxxxxx ssssss...
    //         */
    //        StringTokenizer st = new StringTokenizer(string, " \t\n\r");
    //        String index = st.nextToken();
    //        String percent = st.nextToken();
    //        if (percent.compareTo("%") == 0)
    //            percent += " " + st.nextToken();
    //        String self = st.nextToken();
    //        String descendants = st.nextToken();
    //        String called = st.nextToken();
    //        String name = st.nextToken();
    //        // this should be 0, left justified
    //        indexStart = string.indexOf(index);
    //        // this should be about 7, right justified
    //        percentStart = string.indexOf(percent);
    //        // this should be about 13, right justified
    //        selfStart = string.indexOf(percent) + percent.length() + 1;
    //        // this should be about 21, right justified
    //        descendantsStart = string.indexOf(self) + self.length() + 1;
    //        // this should be about 33, left justified
    //        calledStart = string.indexOf(descendants) + descendants.length() + 1;
    //        // this should be about 49, left justified
    //        nameStart = string.indexOf(name);
    //        return;
    //    }

    private LineData getSelfLineData(String string) {
        LineData lineData = new LineData();
        StringTokenizer st = new StringTokenizer(string, " \t\n\r");

        //In some implementations, the self line will not give
        //the number of calls for the top level function (usually main).
        //Check the number of tokens to see if we are in this case. If so,
        //by default, we assume a number of calls value of 1.
        //int numberOfTokens = st.countTokens();

        // Skip the first token.
        // Entries are numbered with consecutive integers.
        // Each function therefore has an index number, which
        // appears at the beginning of its primary line. Each
        // cross-reference to a function, as a caller or
        // subroutine of another, gives its index number as
        // well as its name. The index number guides you if
        // you wish to look for the entry for that function.
        st.nextToken();

        // This is the percentage of the total time that was
        // spent in this function, including time spent in
        // subroutines called from this function. The time
        // spent in this function is counted again for the
        // callers of this function. Therefore, adding up these
        // percentages is meaningless.
        lineData.d0 = Double.parseDouble(st.nextToken());

        // This is the total amount of time spent in this
        // function. This should be identical to the number
        // printed in the seconds field for this function in
        // the flat profile.
        lineData.d1 = 1000000.0 * Double.parseDouble(st.nextToken());

        // This is the total amount of time spent in the
        // subroutine calls made by this function. This should
        // be equal to the sum of all the self and children
        // entries of the children listed directly below this
        // function.
        String fixer = st.nextToken();
        lineData.d2 = 1000000.0 * Double.parseDouble(fixer);
        if (fixLengths) {
            /* sometimes, the data doesn't get aligned just right.
               We need to adjust the "called" column so that it starts
               just after the end of the children value.  Otherwise, the
               called column will start two columns too late, as in
               this example:
            index % time    self  children    called     name
                0.33  264.75      32/32          _start_blrts [2]
            [1]     58.4    0.33  264.75      32         main [1]
            OK:             0.00  131.96      32/32          HYPRE_StructSMGSetup [5]
            ...
            BAD:          103.71    4.01  151328/151328      hypre_CyclicReduction [8]
            ALSO BAD:       0.27    0.00 1135396/4373228     hypre_BoxGetStrideSize [56]

            */
            //            calledStart = string.indexOf(fixer) + fixer.length() + 1;
            fixLengths = false;
        }

        // This is the number of times the function was called.
        // If the function called itself recursively, there are
        // two numbers, separated by a `+'. The first number
        // counts non-recursive calls, and the second counts
        // recursive calls.

        boolean hasCalls;
        String str = st.nextToken();
        try {
            //int foo = 
            	Integer.parseInt(str);
            hasCalls = true;
        } catch (NumberFormatException nfe) {
            hasCalls = false;
        }

        if (!hasCalls) {
            // if the number of calls is absent, assume 1.
            lineData.i0 = 1;
            lineData.s0 = str;
        } else {
            if (str.indexOf("+") < 0) {
                StringTokenizer st2 = new StringTokenizer(str, "+");
                lineData.i0 = Integer.parseInt(st2.nextToken());
                // do this?
                // lineData.i0 += Integer.parseInt(st2.nextToken());
            }
            lineData.s0 = st.nextToken(); //Name
        }

        while (st.hasMoreTokens()) {
            String tmp = st.nextToken();
            if ((tmp.indexOf("[") != 0) && (!tmp.endsWith("]")))
                lineData.s0 += " " + tmp; //Name
        }
        lineData.s0 = fix(lineData.s0);
        return lineData;
    }

    private LineData getParentChildLineData(String string) {

        // first, check to see if we are missing the index, %time, and self values.
        String tmp = string.trim();
        boolean missing = false;
        if (tmp.length() < string.length() - 20) {
            missing = true;
        }

        LineData lineData = new LineData();
        StringTokenizer st = new StringTokenizer(string, " \t\n\r");

        // unlike the other line parsers, this function assumed a fixed
        // location for values.  That may be erroneous, but I think that
        // is the format for gprof output. Sample lines:
        /*
        index  % time    self  children called     name
                                             <spontaneous>
                 0.16     1.77    1/1        start [1]
        [2]    100.00    0.16     1.77    1      main [2]
                 1.77        0    1/1        a <cycle 1> [5]
        ----------------------------------------
                 1.77        0    1/1        main [2]
        [3]     91.71    1.77        0    1+5    <cycle 1 as a whole> [3]
                 1.02        0    3          b <cycle 1> [4]
                 0.75        0    2          a <cycle 1> [5]
                    0        0    6/6        c [6]
        ----------------------------------------
                                  3          a <cycle 1> [5]
        [4]     52.85    1.02        0    0      b <cycle 1> [4]
                                  2          a <cycle 1> [5]
                    0        0    3/6        c [6]
        ----------------------------------------
                 1.77        0    1/1        main [2]
                                  2          b <cycle 1> [4]
        [5]     38.86    0.75        0    1      a <cycle 1> [5]
                                  3          b <cycle 1> [4]
                    0        0    3/6        c [6]
        ----------------------------------------
                    0        0    3/6        b <cycle 1> [4]
                    0        0    3/6        a <cycle 1> [5]
        [6]      0.00       0        0    6      c [6]
        ----------------------------------------
                                  called/total       parents
                0.02        0.09    2379             hypre_SMGRelax <cycle 2> [11]
        -----------------------------------------------
        */

        String tmpStr = null;

        if (missing) {
            lineData.d0 = 0.0;
            lineData.d1 = 0.0;
        } else {
            tmpStr = st.nextToken().trim();// string.substring(selfStart, descendantsStart).trim();
            //String tmpStr = st.nextToken();
            if (tmpStr.length() > 0) {
                lineData.d0 = 1000000.0 * Double.parseDouble(tmpStr);
            } else {
                lineData.d0 = 0.0;
            }

            //        System.err.println("Error parsing file: " + currentFile + ", line: " + linenumber);
            //        System.err.println("selfStart: " + selfStart);
            //        System.err.println("descendantsStart: " + descendantsStart);
            //        System.err.println("calledStart: " + calledStart);

            try {
                tmpStr = st.nextToken().trim();//string.substring(descendantsStart, calledStart).trim();
                //tmpStr = st.nextToken();
                if (tmpStr.length() > 0) {
                    lineData.d1 = 1000000.0 * Double.parseDouble(tmpStr);
                } else {
                    lineData.d1 = 0.0;
                }
            } catch (Exception e) {
                System.err.println("Error parsing file: " + currentFile + ", line: " + linenumber);
                //            System.err.println("selfStart: " + selfStart);
                //            System.err.println("descendantsStart: " + descendantsStart);
                //            System.err.println("calledStart: " + calledStart);
                System.err.println(e.getMessage());
                throw new DataSourceException(e, currentFile);
            }
        }

        // check if the counts 'spill' into the name field.
        //        String spillTest = string.substring(calledStart, string.length()).trim();
        //        int space = spillTest.indexOf(" ");
        //        if (space > (nameStart - calledStart)) {
        //            tmpStr = spillTest.substring(0, space).trim();
        //        } else {
        //            tmpStr = string.substring(calledStart, nameStart).trim();
        //        }

        tmpStr = st.nextToken().trim();

        // check for a ratio
        if (tmpStr.indexOf("/") >= 0) {
            StringTokenizer st2 = new StringTokenizer(tmpStr, "/");
            // the number of times self was called from parent
            lineData.i0 = Integer.parseInt(st2.nextToken());
            // the total number of nonrecursive calls to self from all
            // its parents
            lineData.i1 = Integer.parseInt(st2.nextToken());
        } else {
            lineData.i0 = Integer.parseInt(tmpStr);
            lineData.i1 = lineData.i0;
        }

        // the rest is the name
        //        if (space > (nameStart - calledStart)) {
        //            int end = spillTest.lastIndexOf("[") - 1;
        //            lineData.s0 = spillTest.substring(space, end).trim();
        //        } else {
        //            int end = string.lastIndexOf("[") - 1;
        //            lineData.s0 = string.substring(nameStart, end).trim();
        //        }

        String eventName = "";
        while (st.hasMoreTokens()) {
            // skip last token
            String token = st.nextToken();
            if (st.hasMoreTokens()) {
                eventName = eventName + " " + token;
            }
        }
        eventName = eventName.trim();
        lineData.s0 = eventName;
        return lineData;
    }

//    private LineData getSummaryLineData(String string) {
//        LineData lineData = new LineData();
//        StringTokenizer st = new StringTokenizer(string, " \t\n\r");
//
//        lineData.d0 = Double.parseDouble(st.nextToken());
//        lineData.d1 = 1000.0 * Double.parseDouble(st.nextToken());
//        lineData.d2 = 1000.0 * Double.parseDouble(st.nextToken());
//        if (st.countTokens() > 5) {
//            lineData.i0 = Integer.parseInt(st.nextToken());
//            lineData.d3 = Double.parseDouble(st.nextToken());
//            lineData.d4 = Double.parseDouble(st.nextToken());
//        } else {
//            lineData.i0 = 1;
//            lineData.d3 = lineData.d2;
//            lineData.d4 = lineData.d2;
//        }
//
//        lineData.s0 = st.nextToken(); //Name
//        while (st.hasMoreTokens()) {
//            String tmp = st.nextToken();
//            if ((tmp.indexOf("[") != 0) && (!tmp.endsWith("]")))
//                lineData.s0 += " " + tmp; //Name
//        }
//        lineData.s0 = fix(lineData.s0);
//        return lineData;
//    }

    /*
     * when C and Fortran code are mixed, the C routines have to be mapped to
     * either .function or function_ . Strip the leading period or trailing
     * underscore, if it is there.
     */
    private String fix(String inString) {
        String outString = inString;
        if (fixNames) {
            if (inString.indexOf(".*") == 0)
                outString = inString.substring(2, inString.length());
            else if (inString.indexOf(".") == 0)
                outString = inString.substring(1, inString.length());
            else if (inString.endsWith("_"))
                outString = inString.substring(0, inString.length() - 1);
        }
        return outString;
    }

}