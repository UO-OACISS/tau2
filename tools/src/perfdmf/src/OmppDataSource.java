package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;
import java.util.StringTokenizer;

public class OmppDataSource extends DataSource {

    private File file;

    public OmppDataSource(File file) {
        this.file = file;
    }

    public void cancelLoad() {
    // TODO Auto-generated method stub

    }

    public int getProgress() {
        // TODO Auto-generated method stub
        return 0;
    }

    private double getDouble(String string, int index) {
        System.out.println("string = " + string + ", index = " + index);
        StringTokenizer st = new StringTokenizer(string, ",;");
        for (int i = 0; i < index; i++) {
            st.nextToken();
        }
        String str = st.nextToken();
        return Double.parseDouble(str);
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {

        Metric time = addMetric("Time");

        FileInputStream fileIn = new FileInputStream(file);
        InputStreamReader inReader = new InputStreamReader(fileIn);
        BufferedReader br = new BufferedReader(inReader);

        String inputString = null;
        int numThreads = 0;
        while ((inputString = br.readLine()) != null) {

            if (inputString.startsWith("##BEG header")) {
                inputString = br.readLine();
                while (!inputString.startsWith("##END header")) {
                    int comma = inputString.indexOf(',');
                    
                    String key = inputString.substring(0,comma);
                    String value = inputString.substring(comma+1);
                    this.getMetaData().put(key,value);
                    if (inputString.startsWith("OMPP_CTR")) {
                        String metricName = inputString.substring(comma + 1).trim();
                        if (!metricName.equals("not set")) {
                            System.out.println("Found Metric: " + metricName);
                            Metric metric = addMetric(metricName);
                        }
                    }
                    if (inputString.startsWith("Thread Count")) {
                        String string = inputString.substring(comma + 1).trim();
                        numThreads = Integer.parseInt(string);
                        for (int i = 0; i < numThreads; i++) {
                            addThread(0, 0, i);
                        }
                    }
                    inputString = br.readLine();
                }
            }

            if (inputString.startsWith("##BEG ompp callgraph")) {
                inputString = br.readLine();
                inputString = br.readLine();
                double inclusive = getDouble(inputString, 0);
                double inclusivePercent = getDouble(inputString, 1);
                double exclusive = getDouble(inputString, 2);
                double exclusivePercent = getDouble(inputString, 3);

                String topString = inputString.substring(inputString.lastIndexOf(',') + 1).trim();
                Function function = addFunction(topString);
                for (int i = 0; i < numThreads; i++) {
                    Thread thread = getThread(0, 0, i);
                    FunctionProfile fp = new FunctionProfile(function);
                    thread.addFunctionProfile(fp);
                    fp.setInclusive(0, inclusive * 1e6);
                    fp.setExclusive(0, exclusive * 1e6);

                }

                while (!inputString.startsWith("##END ompp callgraph")) {
                    inputString = br.readLine();
                }
            }

            Function barrierFunction = addFunction("OMP Barrier");
            if (inputString.startsWith("##BEG callgraph region profiles")) {
                inputString = br.readLine();
                while (!inputString.startsWith("##END callgraph region profiles")) {

                    if (inputString.startsWith("##BEG")) {
                        inputString = br.readLine();
                        String functionName = null;
                        String leafName = "";
                        while (inputString.startsWith("[")) {
                            if (functionName == null) {
                                functionName = inputString;
                            } else {
                                functionName = functionName + " => " +inputString;
                            }
                            leafName = inputString;
                            inputString = br.readLine();
                        }
                        Function function = addFunction(functionName);
                        Function leafFunction = addFunction(leafName);
                        for (int i = 0; i < numThreads; i++) {
                            inputString = br.readLine();
                            int tid = (int) getDouble(inputString, 0);

                            Thread thread = addThread(0, 0, tid);
                            FunctionProfile fp = new FunctionProfile(function, getNumberOfMetrics());
                            FunctionProfile lfp = thread.getOrCreateFunctionProfile(leafFunction, getNumberOfMetrics());
                            thread.addFunctionProfile(fp);

                            if (leafName.indexOf("USER REGION") != -1) {
                                double execTin = getDouble(inputString, 1);
                                double execTex = getDouble(inputString, 2);
                                double execC = getDouble(inputString, 3);
                                
                                thread.addFunctionProfile(fp);
                                fp.setExclusive(0, execTex*1e6);
                                fp.setInclusive(0, execTin*1e6);
                                fp.setNumCalls(execC);
                                lfp.setExclusive(0, lfp.getExclusive(0)+execTex*1e6);
                                lfp.setInclusive(0, lfp.getInclusive(0)+execTin*1e6);
                                lfp.setNumCalls(lfp.getNumCalls()+execC);
                                
                                int idx = 4;
                                for (int m = 1; m < getNumberOfMetrics(); m++) {
                                    double metricInclusive = getDouble(inputString, idx++);
                                    double metricExclusive = getDouble(inputString, idx++);
                                    fp.setExclusive(m, metricExclusive);
                                    fp.setInclusive(m, metricInclusive);
                                    lfp.setExclusive(m, lfp.getExclusive(m)+metricExclusive);
                                    lfp.setInclusive(m, lfp.getExclusive(m)+metricInclusive);
                                }

                            } else if (leafName.indexOf("CRITICAL") != -1) {
                                double execT = getDouble(inputString, 1);
                                double execC = getDouble(inputString, 2);
                                double bodyTinclusive = getDouble(inputString, 3);
                                double bodyTexclusive = getDouble(inputString, 4);
                                double enterBarT = getDouble(inputString, 5);
                                double exitBarT = getDouble(inputString, 6);

                                fp.setExclusive(0, bodyTexclusive*1e6);
                                fp.setInclusive(0, bodyTinclusive*1e6);
                                fp.setNumCalls(execC);
                                lfp.setExclusive(0, lfp.getExclusive(0)+bodyTexclusive*1e6);
                                lfp.setInclusive(0, lfp.getInclusive(0)+bodyTinclusive*1e6);
                                lfp.setNumCalls(lfp.getNumCalls()+execC);

                                int idx = 7;
                                for (int m = 1; m < getNumberOfMetrics(); m++) {
                                    double metricInclusive = getDouble(inputString, idx++);
                                    double metricExclusive = getDouble(inputString, idx++);
                                    fp.setExclusive(m, metricExclusive);
                                    fp.setInclusive(m, metricInclusive);
                                    lfp.setExclusive(m, lfp.getExclusive(m)+metricExclusive);
                                    lfp.setInclusive(m, lfp.getExclusive(m)+metricInclusive);
                                }
                            } else {
                                String callpathBarrierString = functionName + " => OMP Barrier";
                                Function cpf = addFunction(callpathBarrierString);
                                FunctionProfile cbfp = thread.getOrCreateFunctionProfile(cpf,getNumberOfMetrics());
                                FunctionProfile bfp = thread.getOrCreateFunctionProfile(barrierFunction, getNumberOfMetrics());
                                double execT = getDouble(inputString, 1);
                                double execC = getDouble(inputString, 2);
                                double bodyTinclusive = getDouble(inputString, 3);
                                double bodyTexclusive = getDouble(inputString, 4);
                                double exitBarT = getDouble(inputString, 5);
                                double startupT = getDouble(inputString, 6);
                                double shutdwnT = getDouble(inputString, 7);

                                fp.setExclusive(0, (bodyTexclusive+startupT+shutdwnT)*1e6);
                                fp.setInclusive(0, execT*1e6);
                                fp.setNumCalls(execC);
                                cbfp.setExclusive(0,exitBarT);
                                bfp.setExclusive(0,bfp.getExclusive(0)+exitBarT);
                                lfp.setExclusive(0, lfp.getExclusive(0)+fp.getExclusive(0));
                                lfp.setInclusive(0, lfp.getInclusive(0)+fp.getInclusive(0));
                                lfp.setNumCalls(lfp.getNumCalls()+execC);

                                int idx = 8;
                                for (int m = 1; m < getNumberOfMetrics(); m++) {
                                    double metricInclusive = getDouble(inputString, idx++);
                                    double metricExclusive = getDouble(inputString, idx++);
                                    fp.setExclusive(m, metricExclusive);
                                    fp.setInclusive(m, metricInclusive);
                                    lfp.setExclusive(m, lfp.getExclusive(m)+metricExclusive);
                                    lfp.setInclusive(m, lfp.getExclusive(m)+metricInclusive);
                                }
                            }

                        }

                    }

                    inputString = br.readLine();
                }
            }

        }

        this.generateDerivedData();

    }

}
