/*
 * Name: MpiPDataSource.java 
 * Author: Kevin Huck 
 * Description: Parse an mpiP data file.
 */

/*
 * To do: The mpiP data has min, mean, max values. What should be done with
 * these values? Should they be stored in a user event?
 * 
 * Support older versions of the output file
 */

package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class MpiPDataSource extends DataSource {

    private File file;
    private LineData callsiteData = new LineData();
    private String[] eventNames = null;

    private boolean levelInformationPresent;

    public MpiPDataSource(File file) {
        super();
        this.file = file;
    }

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        return 0;
    }

    public void load() throws IOException, FileNotFoundException {

        Function function = null;
        FunctionProfile functionProfile = null;

        String inputString = null;

        StringTokenizer genericTokenizer;

        //System.out.println("Processing data file, please wait ......");
        //long time = System.currentTimeMillis();

        FileInputStream fileIn = new FileInputStream(file);
        InputStreamReader inReader = new InputStreamReader(fileIn);
        BufferedReader br = new BufferedReader(inReader);

        //Set the metric name.
        String metricName = "Time";

        int metric = this.getNumberOfMetrics();
        this.addMetric(metricName);

        Date startTime = null;
        Date stopTime = null;

        SimpleDateFormat sdfInput = new SimpleDateFormat("yyyy MM dd hh:mm:ss");

        // find the start and stop time
        while ((inputString = br.readLine()) != null) {
            int colon = inputString.indexOf(":");
            
            if (inputString.startsWith("@") && (colon != -1)) {
                String key = inputString.substring(0,colon).substring(1).trim();
                String value = inputString.substring(colon+1);

                if (key.equals("MPI Task Assignment")) {
                    StringTokenizer st = new StringTokenizer(value," ");
                    String idToken = st.nextToken();
                    String machineToken = st.nextToken();
                    int id = Integer.parseInt(idToken);
                    Thread thread = addThread(id,0,0);
                    thread.getMetaData().put(key,machineToken);
                } else {
                    getMetaData().put(key,value);
                }
            }
                    
            
            if (inputString.startsWith("@ Start time")) {
                String remainder = inputString.substring(inputString.indexOf(':') + 1);

                try {
                    startTime = sdfInput.parse(remainder);
                } catch (java.text.ParseException except) {
                    System.out.println("Warning, couldn't parse date \"" + remainder + "\"");
                }
            }
            
            if (!inputString.startsWith("@")) {
                // headers over
                break;
            }
        }

        fileIn = new FileInputStream(file);
        inReader = new InputStreamReader(fileIn);
        br = new BufferedReader(inReader);

        // find the start and stop time
        while ((inputString = br.readLine()) != null) {
            if (inputString.startsWith("@ Stop time")) {

                String remainder = inputString.substring(inputString.indexOf(':') + 1);

                try {
                    stopTime = sdfInput.parse(remainder);
                } catch (java.text.ParseException except) {
                    System.out.println("Warning, couldn't parse date \"" + remainder + "\"");
                }

                // exit this while loop
                break;
            }
        }

        // start from the top in case we couldn't find start and stop time
        fileIn = new FileInputStream(file);
        inReader = new InputStreamReader(fileIn);
        br = new BufferedReader(inReader);

        int eventCount = 0;
        // find the callsite names
        while ((inputString = br.readLine()) != null) {
            if (inputString.startsWith("@--- Callsites:")) {

                // inputString is typically 
                // "@--- Callsites: 235 -------------------------------------------------------"
                // we need to parse out that number, below is a poor, but working way to do it

                genericTokenizer = new StringTokenizer(inputString, ":");
                inputString = genericTokenizer.nextToken(); // left side
                inputString = genericTokenizer.nextToken(); // right side
                genericTokenizer = new StringTokenizer(inputString, " ");
                // get the callsite count
                eventCount = Integer.parseInt(genericTokenizer.nextToken());
                // the callsite names are indexed at 1, not 0.
                eventNames = new String[eventCount + 1];
                // System.out.println(eventCount + " callsites found.");
                // ignore the next two lines
                br.readLine();
                inputString = br.readLine();

                if (inputString.startsWith(" ID Lev File")) {
                    levelInformationPresent = true;
                }

                // exit this while loop
                break;
            }
        }

        if (inputString != null) {

            inputString = br.readLine();
            while (!inputString.startsWith("-------")) {
                //getCallsiteHeaders(inputString);

                StringTokenizer st1 = new StringTokenizer(inputString, " ");

                int id, level, line = 0;
                String file, parent, mpiCall;

                id = Integer.parseInt(st1.nextToken()); // callsite id
                if (levelInformationPresent) {
                    level = Integer.parseInt(st1.nextToken()); // Level
                    file = st1.nextToken(); // Filename

                    boolean lineAndFile = true;
                    try {
                        line = Integer.parseInt(st1.nextToken()); // Line
                    } catch (NumberFormatException e) {
                        // no file/line number info
                        lineAndFile = false;
                    }

                    parent = "";
                    while (st1.hasMoreTokens()) {
                        parent = parent + " " + st1.nextToken();
                    }

                    if (level == 0) {
                        int loc = parent.lastIndexOf(" ");
                        if (loc == -1) {
                            throw new RuntimeException("Couldn't find MPI Call!");
                        }
                        mpiCall = parent.substring(loc + 1);
                        parent = parent.substring(0, loc);

                        if (lineAndFile) {
                            eventNames[id] = parent + " [file:" + file + " line:" + line + "] => " + "MPI_" + mpiCall;
                        } else {
                            eventNames[id] = parent + " [Address:" + file + "] => " + "MPI_" + mpiCall;
                        }

                    } else {
                        if (lineAndFile) {
                            eventNames[id] = parent + " [file:" + file + " line:" + line + "] => " + eventNames[id];
                        } else {
                            eventNames[id] = parent + " [Address:" + file + "] => " + eventNames[id];
                        }
                    }

                } else {
                    level = 0;
                    mpiCall = st1.nextToken();
                    parent = st1.nextToken();
                    file = st1.nextToken();
                    line = Integer.parseInt(st1.nextToken());
                    eventNames[id] = parent + " [file:" + file + " line:" + line + "] => " + "MPI_" + mpiCall;
                    eventNames[id] = eventNames[id].trim();
                }

                inputString = br.readLine();

            }

            //            
            //            // parse each of the event names
            //            for (int i = 1; i <= eventCount; i++) {
            //                inputString = br.readLine();
            //                getCallsiteHeaders(inputString);
            //                eventNames[i] = new String(callSite.parent + " => " + "MPI_" + callSite.mpiCall + " file: "
            //                        + callSite.file + " line:" + callSite.line);
            //            }
        }

        // find the callsite data
        int eventDataCount = 0;
        while ((inputString = br.readLine()) != null) {
            // 0.9
            if (inputString.startsWith("@--- Callsite statistics") || inputString.startsWith("@--- Callsite Time statistics")) {
                // exit this while loop
                // System.out.print("Found callsite data: ");
                genericTokenizer = new StringTokenizer(inputString, ":");
                // left side
                inputString = genericTokenizer.nextToken();
                // right side
                inputString = genericTokenizer.nextToken();
                genericTokenizer = new StringTokenizer(inputString, " ");
                // get the callsite count
                eventDataCount = Integer.parseInt(genericTokenizer.nextToken());
                // System.out.println(eventDataCount + " callsite data
                // lines found.");
                // ignore the next two lines
                br.readLine();
                inputString = br.readLine();

                break;
            }
        }

        if (inputString != null) {
            // parse each of the event names
            for (int i = 0; i < eventDataCount; i++) {
                inputString = br.readLine();
                while (inputString != null && (inputString.length() == 0))
                    inputString = br.readLine();
                if (inputString != null) {
					if (inputString.startsWith("--------"))
						break;
                    getCallsiteData(inputString);

                    function = this.addFunction(eventNames[callsiteData.i0], 1);

                    if (callsiteData.i1 >= 0) {
                        // get the node data
                        int nodeID = callsiteData.i1;
                        int contextID = 0;
                        int threadID = 0;
                        Node node = this.addNode(nodeID);
                        Context context = node.addContext(contextID);
                        Thread thread = context.getThread(threadID);
                        if (thread == null) {
                            thread = context.addThread(threadID);
                        }
                        functionProfile = thread.getFunctionProfile(function);
                        if (functionProfile == null) {
                            functionProfile = new FunctionProfile(function);
                            thread.addFunctionProfile(functionProfile);
                        }
                        functionProfile.setExclusive(metric, callsiteData.d5);
                        //functionProfile.setExclusivePercent(metric, callsiteData.d3);
                        functionProfile.setInclusive(metric, callsiteData.d5);
                        //functionProfile.setInclusivePercent(metric, callsiteData.d3);
                        functionProfile.setNumCalls(callsiteData.i2);
                        functionProfile.setNumSubr(0);

                    } else {
                        i = i - 1;
                    }

                }
				if (inputString.startsWith("--------"))
					break;
            }
        }

        // find the callsite data
        while ((inputString = br.readLine()) != null) {
            // 0.9
            if (inputString.startsWith("@--- Callsite Message Sent statistics")) {
                // ignore the next two lines
                br.readLine();
                inputString = br.readLine();
                // exit this while loop
                break;
            }
        }

        if (inputString != null) {
        	UserEvent userEvent = null;
        	UserEventProfile userEventProfile = null;
            // parse each of the event names
            inputString = br.readLine();
            while (inputString != null) {
            	if (inputString != null && 
					inputString.length() > 0 &&
					!inputString.startsWith("--------")) {
					System.out.println("WHILE: " + inputString);
                	getUsereventData(inputString);
                	userEvent = this.addUserEvent(eventNames[callsiteData.i0]);
                	if (callsiteData.i1 >= 0) {
                    	// get the node data
                    	int nodeID = callsiteData.i1;
                    	int contextID = 0;
                    	int threadID = 0;
                    	Node node = this.addNode(nodeID);
                    	Context context = node.addContext(contextID);
                    	Thread thread = context.getThread(threadID);
                    	if (thread == null) {
                        	thread = context.addThread(threadID);
                    	}
                    	userEventProfile = thread.getUserEventProfile(userEvent);
                    	if (userEventProfile == null) {
                        	userEventProfile = new UserEventProfile(userEvent);
                        	thread.addUserEventProfile(userEventProfile);
                    	}
                    	userEventProfile.setNumSamples(callsiteData.i2);
                    	userEventProfile.setMaxValue(callsiteData.d0);
                    	userEventProfile.setMeanValue(callsiteData.d1);
                    	userEventProfile.setMinValue(callsiteData.d2);
						// not accurate, but no way to get it.
                    	userEventProfile.setSumSquared(callsiteData.i2*(callsiteData.d1*callsiteData.d1));

                	}
           		}
            	inputString = br.readLine();
				if (inputString.startsWith("--------"))
					break;
            }
        }

        //Close the file.
        br.close();

        // Add the .MpiP.Application

        if (startTime != null && stopTime != null) {

            double inclusive = (stopTime.getTime() - startTime.getTime()) * 1000;

            function = this.addFunction(".MpiP.Application", 1);

            for (Iterator it = this.getNodes(); it.hasNext();) {
                Node node = (Node) it.next();
                for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                    Context context = (Context) it2.next();
                    for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                        edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it3.next();

                        List functions = thread.getFunctionProfiles();

                        double exclusive = inclusive;
                        double numSubroutines = 0;
                        for (Iterator e4 = functions.iterator(); e4.hasNext();) {
                            FunctionProfile fp = (FunctionProfile) e4.next();
                            if (fp != null) {
                                numSubroutines = numSubroutines + fp.getNumCalls();
                                exclusive = exclusive - fp.getExclusive(metric);
                            }
                        }

                        if (exclusive < 0 || inclusive <= 0) {
                            inclusive = Math.abs(exclusive);
                            exclusive = 0;
                        }

                        double exclusivePercent = exclusive / inclusive;

                        functionProfile = new FunctionProfile(function);

                        thread.addFunctionProfile(functionProfile);

                        functionProfile.setInclusive(metric, inclusive);
                        //functionProfile.setInclusivePercent(metric, 100);

                        functionProfile.setExclusive(metric, exclusive);
                        //functionProfile.setExclusivePercent(metric, exclusivePercent);
                        functionProfile.setNumCalls(1);
                        functionProfile.setNumSubr(numSubroutines);
                    }
                }
            }

        }

        //                time = (System.currentTimeMillis()) - time;
        //                System.out.println("Done processing data file!");
        //                System.out.println("Time to process file (in milliseconds): " + time);

        this.generateDerivedData();
        this.aggregateMetaData();

    }

    private void getCallsiteData(String string) {
        StringTokenizer st1 = new StringTokenizer(string, " ");
        callsiteData.s0 = st1.nextToken(); // MPI function
        callsiteData.i0 = Integer.parseInt(st1.nextToken()); // callsite
        // index

        String tmpString = st1.nextToken(); // rank
        if (tmpString.equals("*")) {
            callsiteData.i1 = -1;
        } else {
            callsiteData.i1 = Integer.parseInt(tmpString); // rank
        }
        callsiteData.i2 = Integer.parseInt(st1.nextToken()); // count
        callsiteData.d0 = Double.parseDouble(st1.nextToken()); // Max
        callsiteData.d1 = Double.parseDouble(st1.nextToken()) * 1000; // Mean
        callsiteData.d2 = Double.parseDouble(st1.nextToken()); // Min
        callsiteData.d3 = Double.parseDouble(st1.nextToken()); // App%
        callsiteData.d4 = Double.parseDouble(st1.nextToken()); // MPI%
        callsiteData.d5 = callsiteData.d1 * callsiteData.i2; // Total time for this node
    }

    private void getUsereventData(String string) {
        StringTokenizer st1 = new StringTokenizer(string, " ");
        callsiteData.s0 = st1.nextToken(); // MPI function
        callsiteData.i0 = Integer.parseInt(st1.nextToken()); // callsite
        // index

        String tmpString = st1.nextToken(); // rank
        if (tmpString.equals("*")) {
            callsiteData.i1 = -1;
        } else {
            callsiteData.i1 = Integer.parseInt(tmpString); // rank
        }
        callsiteData.i2 = Integer.parseInt(st1.nextToken()); // count
        callsiteData.d0 = Double.parseDouble(st1.nextToken()); // Max
        callsiteData.d1 = Double.parseDouble(st1.nextToken()); // Mean
        callsiteData.d2 = Double.parseDouble(st1.nextToken()); // Min
        callsiteData.d3 = Double.parseDouble(st1.nextToken()); // Sum
    }

}