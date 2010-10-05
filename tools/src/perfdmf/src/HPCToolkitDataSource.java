package edu.uoregon.tau.perfdmf;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Iterator;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

/**
 * Reader for HPCToolkit
 *
 *
 * @see <a href="http://www.hipersoft.rice.edu/hpctoolkit/">
 * http://www.hipersoft.rice.edu/hpctoolkit/</a> for more information about HPCToolkit
 * 
 * The data we read here is the XML form that comes from hpcquick, e.g. :
 * 
 * hpcrun -e PAPI_FP_INS -e PAPI_L1_DCM simple
 * hpcquick hpcquick -P simple.PAPI_FP_INS*
 * cat hpcquick.db/hpcquick.hpcviewer
 * 
 * TODO: We should probably make some attempt to split PAPI_FP_INS-0 and 
 *       PAPI_FP_INS-1 into different threads rather than two metrics.  This seems
 *       to be what happens to the data when you do an MPI run and combine them with
 *       hpcquick.
 * 
 * 
 * <P>CVS $Id: HPCToolkitDataSource.java,v 1.2 2006/03/29 20:14:38 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class HPCToolkitDataSource extends DataSource {

    private File file;
    private HPCToolkitXMLHandler handler = new HPCToolkitXMLHandler(this);

    /**
     * Constructor for HPCToolkitDataSource
     * @param file      file containing cube data
     */
    public HPCToolkitDataSource(File file) {
        this.file = file;
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        try {

            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);

            xmlreader.parse(new InputSource(new FileInputStream(file)));

            // now subract out children inclusive values from parent exclusive values
            for (int i = 0; i < this.getNumberOfMetrics(); i++) {
                Thread thread = this.getThread(0, 0, 0);
                for (Iterator<Function> it = this.getFunctions(); it.hasNext();) {
                    Function function = it.next();

                    FunctionProfile fp = thread.getFunctionProfile(function);

                    if (fp != null) {
                        FunctionProfile parent = getParent(thread, function);

                        if (parent != null) {
                            double newValue = parent.getExclusive(i) - fp.getInclusive(i);
                            if (newValue < 0) {
                                newValue = 0; // clamp the value to zero because this is a sample based approach and will result in negative values
                            }

                            parent.setExclusive(i, newValue);
                        }
                    }
                }
            }

            // Set flat profile data (D is combined from A=>D, B=>D, etc)
            Thread thread = this.getThread(0, 0, 0);
            for (Iterator<Function> it = this.getFunctions(); it.hasNext();) {
                Function function = it.next();
                FunctionProfile fp = thread.getFunctionProfile(function);
                if (fp != null) {
                    FunctionProfile flat = getFlatFunctionProfile(thread, function);
                    if (flat != null) {
                        for (int i = 0; i < this.getNumberOfMetrics(); i++) {
                            flat.setExclusive(i, flat.getExclusive(i) + fp.getExclusive(i));
                            flat.setInclusive(i, flat.getInclusive(i) + fp.getInclusive(i));
                        }
                    }
                }
            }
            this.setGroupNamesPresent(true);


            this.generateDerivedData();
        } catch (SAXException e) {
            throw new DataSourceException(e);
        }

    } 
    
    // retrieve the parent profile on a given thread (A=>B for A=>B=>C)
    private FunctionProfile getParent(Thread thread, Function function) {
        if (!function.isCallPathFunction()) {
            return null;
        }
        String functionName = function.getName();
        String parentName = functionName.substring(0, functionName.lastIndexOf("=>"));
        Function parentFunction = this.getFunction(parentName);
        FunctionProfile parent = thread.getFunctionProfile(parentFunction);
        return parent;
    }

    // given A => B => C, this retrieves the FP for C
    private FunctionProfile getFlatFunctionProfile(Thread thread, Function function) {
        if (!function.isCallPathFunction()) {
            return null;
        }
        String childName = function.getName().substring(function.getName().lastIndexOf("=>") + 2).trim();
        Function childFunction = this.addFunction(childName);
        FunctionProfile childFP = thread.getFunctionProfile(childFunction);
        if (childFP == null) {
            childFP = new FunctionProfile(childFunction, this.getNumberOfMetrics());
            thread.addFunctionProfile(childFP);
        }
        return childFP;
    }

    public int getProgress() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void cancelLoad() {
        // TODO Auto-generated method stub

    }

}
