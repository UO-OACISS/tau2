/**
 * 
 */
package glue;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectOutput;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cqos.WekaClassifierWrapper;
import weka.core.*;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

/**
 * @author khuck
 *
 */
public class CQoSClassifierOperation extends AbstractPerformanceOperation {

	public static final String SUPPORT_VECTOR_MACHINE = WekaClassifierWrapper.SUPPORT_VECTOR_MACHINE;
	//public static final String SUPPORT_VECTOR_MACHINE2 = WekaClassifierWrapper.SUPPORT_VECTOR_MACHINE2;
	public static final String NAIVE_BAYES = WekaClassifierWrapper.NAIVE_BAYES;
    public static final String MULTILAYER_PERCEPTRON = WekaClassifierWrapper.MULTILAYER_PERCEPTRON;
    public static final String LINEAR_REGRESSION = WekaClassifierWrapper.LINEAR_REGRESSION;
    public static final String J48 = WekaClassifierWrapper.J48;
    //public static final String AODE = WekaClassifierWrapper.AODE;
    public static final String ALTERNATING_DECISION_TREE = WekaClassifierWrapper.ALTERNATING_DECISION_TREE;
    public static final String RANDOM_TREE = WekaClassifierWrapper.RANDOM_TREE;

	private String metric = "Time";
	private Set<String> metadataFields = null;
	private String classLabel = null;
	private WekaClassifierWrapper wrapper = null;
	private String classifierType = MULTILAYER_PERCEPTRON;
	private int trainingSize = 0;
		
	/**
	 * @param inputs
	 */
	public CQoSClassifierOperation(List<PerformanceResult> inputs, String metric, 
			Set<String> metadataFields, String classLabel) {
		super(inputs);
		this.metric = metric;
		this.metadataFields = metadataFields;
		this.classLabel = classLabel;
	}

 	public List<PerformanceResult> processData() {
		try {
		// create a map to store the UNIQUE tuples
		Map<Hashtable<String,String>,PerformanceResult> tuples = 
			new HashMap<Hashtable<String,String>,PerformanceResult>();
		
		// iterate through the inputs
		for (PerformanceResult input : this.inputs) {
			// create a local Hashtable 
			Hashtable<String,String> localMeta = new Hashtable<String,String>();
			// get the input's metadata
			TrialMetadata tmp = new TrialMetadata(input);
			Hashtable<String,String> meta = tmp.getCommonAttributes();
			// create a reduced hashtable
			if (this.metadataFields != null) {
				for (String key : this.metadataFields) {
					// don't include the class label - we want just the "best" method for that parameter - not all of them
					if (!key.equals(this.classLabel)) {
						String tmpStr = meta.get(key);
						if (tmpStr == null) {
							System.out.println("NO VALUE FOUND FOR KEY: "+ key);
						} else {
							localMeta.put(key, meta.get(key));
						}
					}
				}
			// otherwise, if the user didn't specify a set of properties, use them all (?)
			} else {
				for (String key : meta.keySet()) {
					// don't include the class label - we want just the "best" method for that parameter - not all of them
					if (!key.equals(this.classLabel))
						localMeta.put(key, meta.get(key));
				}				
			}
			// check if the metric is one of the metrics, or a metadata field
			if (!input.getMetrics().contains(metric)) {
				// put the hashtable in the set: if its performance is "better" than the existing one
				// or if it doesn't exist yet.
				PerformanceResult result = tuples.get(localMeta);
				String tmpSuccess = localMeta.get("success");
				if (tmpSuccess == null || tmpSuccess.equals("1")) { // don't save this value if it didn't converge!
					if (result == null) {
						tuples.put(localMeta,input);
					} else {
						if (Double.parseDouble(meta.get(metric)) < 
								Double.parseDouble((new TrialMetadata(result)).getCommonAttributes().get(metric))) {
							tuples.put(localMeta,input);
						}
					}
				}
			} else {
				// put the hashtable in the set: if its performance is "better" than the existing one
				// or if it doesn't exist yet.
				PerformanceResult result = tuples.get(localMeta);
				if (result == null) {
					tuples.put(localMeta,input);
				} else {
					if (input.getInclusive(0, input.getMainEvent(), metric) < 
							result.getInclusive(0, result.getMainEvent(), this.metric)) {
						tuples.put(localMeta,input);					
					}
				}
			}
		}
		
		List<Map<String,String>> trainingData = new ArrayList<Map<String,String>>();

		// ok, we have the set of "optimal" methods for each unique tuple.  Convert them to Instances.
		for (Hashtable<String,String> tmp : tuples.keySet()) {
			Map<String,String> tmpMap = new HashMap<String,String>();
			
			// set the independent parameters
			if (this.metadataFields != null) {
				for (String metaKey : this.metadataFields) {
					tmpMap.put(metaKey, tmp.get(metaKey));
				}
			} else {
				for (String metaKey : tmp.keySet()) {
					tmpMap.put(metaKey, tmp.get(metaKey));
				}
			}
			tmpMap.put(this.classLabel, (new TrialMetadata(tuples.get(tmp)).getCommonAttributes().get(classLabel)));
			trainingData.add(tmpMap);
		}
		
		trainingSize  = trainingData.size();
		System.out.println("Total instances for training: " + trainingSize);
		System.out.println("Using keys: " + this.metadataFields.toString());

		try {
			this.wrapper = new WekaClassifierWrapper (trainingData, this.classLabel);
			this.wrapper.setClassifierType(classifierType);
			this.wrapper.buildClassifier();
		} catch (Exception e) {
			System.err.println(e.getMessage());
			//e.printStackTrace();			
		}
		// in case of a stack overflow
		} catch (Exception e2) {
			System.err.println(e2.getMessage());
		}
		
		return null;
	}
	
	public String getClass(Map<String,String> inputFields) {
		return wrapper.getClass(inputFields);
	}

	public double getConfidence() {
		return wrapper.getConfidence();
	}

	public void writeClassifier(String fileName) {
		WekaClassifierWrapper.writeClassifier(fileName, this.wrapper);
	}

	public String getClassifierType() {
		return this.classifierType;
	}

	public void setClassifierType(String classifierType) {
		this.classifierType = classifierType;			
	}

	public String crossValidateModel() {
		return this.wrapper.crossValidateModel(Math.max(trainingSize/100, 10));
	}

}
