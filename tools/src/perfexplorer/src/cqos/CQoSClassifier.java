/**
 * 
 */
package cqos;

import java.util.*;

/**
 * @author khuck
 *
 */
public class CQoSClassifier {

	/**
	 * 
	 */
	public CQoSClassifier() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		if (args.length < 2) {
			System.out.println("\nUsage: java -jar classifier.jar <fileName> <name:value> [<name:value>...]");
			System.err.println("Example: java -jar classifier.jar /tmp/classifier.serialized parm1:value1 parm2:'value 2'\n");
			System.exit(1);
		}
		
		// first parameter is the serialized classifier
		String fileName = args[0];
		// read in our classifier
		//System.out.print("Reading " + fileName + "...");
		WekaClassifierWrapper wrapper = WekaClassifierWrapper.readClassifier(fileName);
		//System.out.println("Done.");

		Map/*<String,String>*/ inputFields = new HashMap/*<String,String>*/();

		// the remaining parameters are name/value pairs to test the classifier.
		for (int i = 1 ; i < args.length ; i++) {
			StringTokenizer st = new StringTokenizer(args[i], ":");
			String name = st.nextToken();
			String value = st.nextToken();
			// strip the beginning and ending single quotes, if necessary
			if (value.startsWith("'") && value.startsWith("'")) {
				value = value.substring(1,value.length()-1);
			}
			inputFields.put(name, value);
		}

		//System.out.println(inputFields + ":\n" + wrapper.getClass(inputFields) + 
		System.out.println(wrapper.getClass(inputFields) + ", confidence: " + wrapper.getConfidence());
	}

}
