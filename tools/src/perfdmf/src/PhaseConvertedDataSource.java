package edu.uoregon.tau.perfdmf;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Iterator;
import java.util.List;

public class PhaseConvertedDataSource extends DataSource {


    public PhaseConvertedDataSource(DataSource callpathDataSource, List phases) {

        if (callpathDataSource.getCallPathDataPresent() == false) {
            throw new IllegalArgumentException("Can't make phase profile without callpath data");
        }

        int numMetrics = callpathDataSource.getNumberOfMetrics();

        for (int i = 0; i < numMetrics; i++) {
            this.addMetric(callpathDataSource.getMetricName(i));
        }

        for (Iterator thrd = callpathDataSource.getAllThreads().iterator(); thrd.hasNext();) {
            Thread srcThread = (Thread) thrd.next();

            Node node = this.addNode(srcThread.getNodeID());
            Context context = node.addContext(srcThread.getContextID());
            Thread thread = context.addThread(srcThread.getThreadID(), numMetrics);

            for (Iterator it = srcThread.getFunctionProfileIterator(); it.hasNext();) {
                FunctionProfile srcFp = (FunctionProfile) it.next();
                if (srcFp == null) {
                    continue;
                }
                if (!srcFp.isCallPathFunction()) { // does not contain "=>", just copy over
                    Function function = this.addFunction(srcFp.getName(), numMetrics);

                    // copy the group over
                    for (Iterator gr = srcFp.getFunction().getGroups().iterator(); gr.hasNext(); ) {
                        Group srcGroup = (Group) gr.next();
                        Group group = this.addGroup(srcGroup.getName());
                        function.addGroup(group);
                    }

                    FunctionProfile fp = new FunctionProfile(function, numMetrics);
                    thread.addFunctionProfile(fp);

                    for (int i = 0; i < callpathDataSource.getNumberOfMetrics(); i++) {
                        fp.setExclusive(i, srcFp.getExclusive(i));
                        fp.setInclusive(i, srcFp.getInclusive(i));
                    }

                    fp.setNumCalls(srcFp.getNumCalls());
                    fp.setNumSubr(srcFp.getNumSubr());

                } else {
                    String name = srcFp.getName();

                    String path = UtilFncs.getAllButRightMost(name);
                    String child = UtilFncs.getRightMost(name);

                    // find the phase that this child belongs to
                    // either in the phase list, or the leftmost guy
                    while (!match(UtilFncs.getRightMost(path), phases) && path.indexOf("=>") >= 0) {
                        path = UtilFncs.getAllButRightMost(path);
                    }

                    String phase = UtilFncs.getRightMost(path);
                    Function phaseFunction = this.addFunction(phase);
                    phaseFunction.addGroup(this.addGroup("TAU_PHASE"));
                    
                    
                    String timer = phase + " => " + child;
                    Function function = this.addFunction(timer);

                    // copy the group over
                    for (Iterator gr = srcFp.getFunction().getGroups().iterator(); gr.hasNext(); ) {
                        Group srcGroup = (Group) gr.next();
                        Group group = this.addGroup(srcGroup.getName());
                        function.addGroup(group);
                    }
                    
                    
                    FunctionProfile functionProfile = thread.getFunctionProfile(function);

                    if (functionProfile == null) {
                        functionProfile = new FunctionProfile(function, numMetrics);
                        thread.addFunctionProfile(functionProfile);
                    }

                    // add to existing data since multiple paths can result in the same PHASE => CHILD
                    for (int m = 0; m < callpathDataSource.getNumberOfMetrics(); m++) {
                        functionProfile.setExclusive(m, functionProfile.getExclusive(m) + srcFp.getExclusive(m));
                        functionProfile.setInclusive(m, functionProfile.getInclusive(m) + srcFp.getInclusive(m));
                    }
                    functionProfile.setNumCalls(functionProfile.getNumCalls() + srcFp.getNumCalls());
                    functionProfile.setNumSubr(functionProfile.getNumSubr() + srcFp.getNumSubr());
                }

            }

        }

        this.generateDerivedData();
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        // TODO Auto-generated method stub

    }

    public int getProgress() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void cancelLoad() {
        // TODO Auto-generated method stub

    }

    
    private boolean match(String needle, List haystack) {
        return haystack.contains(needle);
    }

    
}
