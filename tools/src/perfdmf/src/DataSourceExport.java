package edu.uoregon.tau.perfdmf;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.zip.GZIPOutputStream;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.common.MetaDataMap.MetaDataValue;

public class DataSourceExport {

	private static int findGroupID(Group groups[], Group group) {
		for (int i = 0; i < groups.length; i++) {
			if (groups[i] == group) {
				return i;
			}
		}
		throw new RuntimeException("Couldn't find group: " + group.getName());
	}

	public static void writeDelimited(DataSource dataSource, File file)
			throws FileNotFoundException, IOException {
		FileOutputStream out = new FileOutputStream(file);
		writeDelimited(dataSource, out);
	}

	public static void writeDelimited(DataSource dataSource, OutputStream out)
			throws IOException {
		OutputStreamWriter outWriter = new OutputStreamWriter(out);
		BufferedWriter bw = new BufferedWriter(outWriter);

		int numMetrics = dataSource.getNumberOfMetrics();

		bw.write("Node\tContext\tThread\tFunction\tNumCalls\tNumSubr");

		for (int i = 0; i < numMetrics; i++) {
			String metricName = dataSource.getMetricName(i);
			bw.write("\tInclusive " + metricName);
			bw.write("\tExclusive " + metricName);
		}

		bw.write("\tGroup");
		bw.write("\n");

		for (Iterator<edu.uoregon.tau.perfdmf.Thread> it = dataSource
				.getAllThreads().iterator(); it.hasNext();) {
			Thread thread = it.next();

			for (Iterator<FunctionProfile> it2 = thread
					.getFunctionProfileIterator(); it2.hasNext();) {
				FunctionProfile fp = it2.next();
				if (fp != null) {
					bw.write(thread.getNodeID() + "\t" + thread.getContextID()
							+ "\t" + thread.getThreadID() + "\t");
					bw.write(fp.getName() + "\t");
					bw.write(fp.getNumCalls() + "\t");
					bw.write(fp.getNumSubr() + "");
					for (int i = 0; i < numMetrics; i++) {
						bw.write("\t" + fp.getInclusive(i));
						bw.write("\t" + fp.getExclusive(i));
					}
					bw.write("\t" + fp.getFunction().getGroupString());
					bw.write("\n");
				}
			}
			bw.write("\n");
		}

		bw.write("Node\tContext\tThread\tUser Event\tNumSamples\tMin\tMax\tMean\tStdDev\n");
		for (Iterator<Thread> it = dataSource.getAllThreads().iterator(); it
				.hasNext();) {
			Thread thread = it.next();

			for (Iterator<UserEventProfile> it2 = thread.getUserEventProfiles(); it2
					.hasNext();) {
				UserEventProfile uep = it2.next();
				if (uep != null) {

					bw.write(thread.getNodeID() + "\t" + thread.getContextID()
							+ "\t" + thread.getThreadID());
					bw.write("\t" + uep.getUserEvent().getName());

					bw.write("\t" + uep.getNumSamples());
					bw.write("\t" + uep.getMinValue());
					bw.write("\t" + uep.getMaxValue());
					bw.write("\t" + uep.getMeanValue());
					bw.write("\t" + uep.getStdDev());
					bw.write("\n");

				}
			}
			bw.write("\n");
		}
		bw.close();
		outWriter.close();
		out.close();

	}

	public static void writePacked(DataSource dataSource, File file)
			throws FileNotFoundException, IOException {
		// File file = new File("/home/amorris/test.ppk");
		FileOutputStream ostream = new FileOutputStream(file);
		writePacked(dataSource, ostream);
	}

	public static void writePacked(DataSource dataSource, OutputStream ostream)
			throws IOException {
		GZIPOutputStream gzip = new GZIPOutputStream(ostream);
		BufferedOutputStream bw = new BufferedOutputStream(gzip);
		DataOutputStream p = new DataOutputStream(bw);

		//Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
		int numFunctions = dataSource.getFunctions().size();//0;

		
		
//		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
//			Function function = it.next();
//			if (!function.isGroupMember(derived)) {
//				numFunctions++;
//			}
//		}

		int numMetrics = dataSource.getNumberOfMetrics();
		int numUserEvents = dataSource.getNumUserEvents();
		int numGroups = dataSource.getNumGroups();

		// write out magic cookie
		p.writeChar('P'); // two bytes
		p.writeChar('P'); // two bytes
		p.writeChar('K'); // two bytes

		// write out version
		p.writeInt(2); // four bytes

		// write out lowest compatibility version
		p.writeInt(1); // four bytes

		// Write meta-data
		ByteArrayOutputStream headerStream = new ByteArrayOutputStream();
		DataOutputStream headerData = new DataOutputStream(headerStream);

		// future versions can put another header block here, we will skip this
		// many bytes
		headerData.writeInt(0);

		if (dataSource.getMetaData() != null) {
			// write out the trial meta-data, this data is normalized across all
			// threads (i.e. it applies to all threads)
			MetaDataMap metaData = dataSource.getMetaData();
			headerData.writeInt(metaData.size());
			for (Iterator<MetaDataKey> it2 = metaData.keySet().iterator(); it2
					.hasNext();) {
				MetaDataKey key = it2.next();
				MetaDataValue value = metaData.get(key);
				headerData.writeUTF(key.name);
				headerData.writeUTF(value.value.toString());
			}
		} else {
			headerData.writeInt(0);
		}

		headerData.writeInt(dataSource.getAllThreads().size());
		for (Iterator<Thread> it = dataSource.getAllThreads().iterator(); it
				.hasNext();) {
			Thread thread = it.next();
			MetaDataMap metaData = thread.getMetaData();
			headerData.writeInt(thread.getNodeID());
			headerData.writeInt(thread.getContextID());
			headerData.writeInt(thread.getThreadID());
			headerData.writeInt(metaData.size());
			for (Iterator<MetaDataKey> it2 = metaData.keySet().iterator(); it2
					.hasNext();) {
				MetaDataKey key = it2.next();
				MetaDataValue value = metaData.get(key);
				headerData.writeUTF(key.name);
				headerData.writeUTF(value.value.toString());
			}
		}
		headerData.close();

		p.writeInt(headerData.size());
		p.write(headerStream.toByteArray());

		// write out metric names
		p.writeInt(numMetrics);
		for (int i = 0; i < numMetrics; i++) {
			String metricName = dataSource.getMetricName(i);
			p.writeUTF(metricName);
		}

		// write out group names
		p.writeInt(numGroups);
		Group groups[] = new Group[numGroups];
		int idx = 0;
		for (Iterator<Group> it = dataSource.getGroups(); it.hasNext();) {
			Group group = it.next();
			String groupName = group.getName();
			p.writeUTF(groupName);
			groups[idx++] = group;
		}

		// write out function names
		Function functions[] = new Function[numFunctions];
		idx = 0;
		p.writeInt(numFunctions);
		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
			Function function = it.next();
			//if (!function.isGroupMember(derived)) {

				functions[idx++] = function;
				p.writeUTF(function.getName());

				List<Group> thisGroups = function.getGroups();
				if (thisGroups == null) {
					p.writeInt(0);
				} else {
					p.writeInt(thisGroups.size());
					for (int i = 0; i < thisGroups.size(); i++) {
						Group group = thisGroups.get(i);
						p.writeInt(findGroupID(groups, group));
					}
				}
			//}
		}

		// write out user event names
		UserEvent userEvents[] = new UserEvent[numUserEvents];
		idx = 0;
		p.writeInt(numUserEvents);
		for (Iterator<UserEvent> it = dataSource.getUserEventIterator(); it.hasNext();) {
			UserEvent userEvent = it.next();
			userEvents[idx++] = userEvent;
			p.writeUTF(userEvent.getName());
		}

		// write out the number of threads
		p.writeInt(dataSource.getAllThreads().size());

		// write out each thread's data
		for (Iterator<Thread> it = dataSource.getAllThreads().iterator(); it
				.hasNext();) {
			Thread thread = it.next();

			p.writeInt(thread.getNodeID());
			p.writeInt(thread.getContextID());
			p.writeInt(thread.getThreadID());

			// count (non-null) function profiles
			int count = 0;
			for (int i = 0; i < numFunctions; i++) {
				FunctionProfile fp = thread.getFunctionProfile(functions[i]);
				if (fp != null) {
					count++;
				}
			}
			p.writeInt(count);

			// write out function profiles
			for (int i = 0; i < numFunctions; i++) {
				FunctionProfile fp = thread.getFunctionProfile(functions[i]);

				if (fp != null) {
					p.writeInt(i); // which function (id)
					p.writeDouble(fp.getNumCalls());
					p.writeDouble(fp.getNumSubr());

					for (int j = 0; j < numMetrics; j++) {
						p.writeDouble(fp.getExclusive(j));
						p.writeDouble(fp.getInclusive(j));
					}
				}
			}

			// count (non-null) user event profiles
			count = 0;
			for (int i = 0; i < numUserEvents; i++) {
				UserEventProfile uep = thread
						.getUserEventProfile(userEvents[i]);
				if (uep != null) {
					count++;
				}
			}

			p.writeInt(count); // number of user event profiles

			// write out user event profiles
			for (int i = 0; i < numUserEvents; i++) {
				UserEventProfile uep = thread
						.getUserEventProfile(userEvents[i]);

				if (uep != null) {
					p.writeInt(i);
					p.writeInt((int) uep.getNumSamples());
					p.writeDouble(uep.getMinValue());
					p.writeDouble(uep.getMaxValue());
					p.writeDouble(uep.getMeanValue());
					p.writeDouble(uep.getSumSquared());
				}
			}
		}

		p.close();
		gzip.close();
		ostream.close();

	}

	private static String xmlFixUp(String string) {
		string = string.replaceAll("&", "&amp;");
		string = string.replaceAll(">", "&gt;");
		string = string.replaceAll("<", "&lt;");
		string = string.replaceAll("\n", "&#xa;");
		return string;
	}

	private static void writeXMLSnippet(BufferedWriter bw,
			MetaDataMap metaDataMap) throws IOException {
		for (Iterator<MetaDataKey> it2 = metaDataMap.keySet().iterator(); it2.hasNext();) {
			MetaDataKey key = it2.next();
			MetaDataValue value = metaDataMap.get(key);
			bw.write("<attribute><name>" + xmlFixUp(key.name) + "</name><value>"
					+ xmlFixUp(value.value.toString()) + "</value></attribute>");
		}
	}

	private static void writeMetric(File root, DataSource dataSource,
			int metricID, Function[] functions, String[] groupStrings,
			UserEvent[] userEvents, List<Thread> threads) throws IOException {

		// int numMetrics = dataSource.getNumberOfMetrics();
		int numUserEvents = dataSource.getNumUserEvents();
		// int numGroups = dataSource.getNumGroups();

		for (Iterator<Thread> it = threads.iterator(); it.hasNext();) {
			Thread thread = it.next();

			String suffix = null;
			if (thread.getNodeID() >= 0) {
				suffix = thread.getNodeID() + "." + thread.getContextID() + "."
						+ thread.getThreadID();
			} else {
				suffix = thread.toString().replace(" ", "");
			}
			File file = new File(root + "/profile." + suffix);

			FileOutputStream out = new FileOutputStream(file);
			OutputStreamWriter outWriter = new OutputStreamWriter(out);
			BufferedWriter bw = new BufferedWriter(outWriter);

			// count function profiles
			int count = 0;
			for (int i = 0; i < functions.length; i++) {
				FunctionProfile fp = thread.getFunctionProfile(functions[i]);
				if (fp != null) {
					count++;
				}
			}

			if (dataSource.getNumberOfMetrics() == 1
					&& dataSource.getMetricName(metricID).equals("Time")) {
				bw.write(count + " templated_functions\n");
			} else {
				bw.write(count + " templated_functions_MULTI_"
						+ dataSource.getMetricName(metricID) + "\n");
			}

			if (dataSource.getMetaData() != null) {
				bw.write("# Name Calls Subrs Excl Incl ProfileCalls<metadata>");
				writeXMLSnippet(bw, dataSource.getMetaData());
				writeXMLSnippet(bw, thread.getMetaData());
				bw.write("</metadata>\n");

			} else {
				bw.write("# Name Calls Subrs Excl Incl ProfileCalls\n");
			}

			// write out function profiles
			for (int i = 0; i < functions.length; i++) {
				FunctionProfile fp = thread.getFunctionProfile(functions[i]);

				if (fp != null) {
					bw.write('"' + functions[i].getName() + "\" ");
					bw.write((int) fp.getNumCalls() + " ");
					bw.write((int) fp.getNumSubr() + " ");
					bw.write(fp.getExclusive(metricID) + " ");
					bw.write(fp.getInclusive(metricID) + " ");
					bw.write("0 " + "GROUP=\"" + groupStrings[i] + "\"\n");
				}
			}

			bw.write("0 aggregates\n");

			// count user event profiles
			count = 0;
			for (int i = 0; i < numUserEvents; i++) {
				UserEventProfile uep = thread
						.getUserEventProfile(userEvents[i]);
				if (uep != null) {
					count++;
				}
			}

			if (count > 0) {
				bw.write(count + " userevents\n");
				bw.write("# eventname numevents max min mean sumsqr\n");

				// write out user event profiles
				for (int i = 0; i < numUserEvents; i++) {
					UserEventProfile uep = thread
							.getUserEventProfile(userEvents[i]);

					if (uep != null) {
						bw.write('"' + userEvents[i].getName() + "\" ");
						bw.write(uep.getNumSamples() + " ");
						bw.write(uep.getMaxValue() + " ");
						bw.write(uep.getMinValue() + " ");
						bw.write(uep.getMeanValue() + " ");
						bw.write(uep.getSumSquared() + "\n");
					}
				}
			}
			bw.close();
			outWriter.close();
			out.close();
		}

	}

	public static String createSafeMetricName(String name) {
		String ret = name.replace('/', '\\');
		return ret;
	}

	public static void writeProfiles(DataSource dataSource, File directory)
			throws IOException {
		writeProfiles(dataSource, directory, dataSource.getAllThreads());
	}

	public static void writeAggProfiles(DataSource dataSource, File directory)
			throws IOException {
		writeProfiles(dataSource, directory, dataSource.getAggThreads());
	}

	public static void writeAggMPISummary(DataSource dataSource,
			boolean suppress, boolean metadata) throws IOException {
		writeMPISummary(dataSource, dataSource.getAggThreads(), suppress,
				metadata);
	}

	public static void writeMetaDataSummary(DataSource dataSource)
			throws IOException {
		MetaDataMap m = dataSource.getMetaData();
		System.out.println("Metadata:");

		Iterator<MetaDataKey> iks = m.keySet().iterator();
		int maxKeyLength = 0;
		while (iks.hasNext()) {
			String k = iks.next().name;
			maxKeyLength = Math.max(maxKeyLength, k.length());
		}

		Set<Entry<MetaDataKey, MetaDataValue>> ms = m.entrySet();
		Iterator<Entry<MetaDataKey, MetaDataValue>> ims = ms.iterator();
		while (ims.hasNext()) {
			Entry<MetaDataKey, MetaDataValue> ems = ims.next();
			System.out.format("%-" + maxKeyLength + "s%s", ems.getKey().name, ": "
					+ ems.getValue().value.toString());
			System.out.println();
			// System.out.println(ems.getKey() + " = " + ems.getValue());
		}
	}

	public static void writeProfiles(DataSource dataSource, File directory,
			List<Thread> threads) throws IOException {

		int numMetrics = dataSource.getNumberOfMetrics();
		int numUserEvents = dataSource.getNumUserEvents();
		int numGroups = dataSource.getNumGroups();

		int idx = 0;

		//Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

		int numFunctions = dataSource.getFunctions().size();//0;
//		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
//			Function function = it.next();
//			if (function.isGroupMember(derived)) {
//				continue;
//			}
//			numFunctions++;
//		}

		// write out group names
		Group groups[] = new Group[numGroups];
		for (Iterator<Group> it = dataSource.getGroups(); it.hasNext();) {
			Group group = it.next();
			// String groupName = group.getName();
			groups[idx++] = group;
		}

		Function functions[] = new Function[numFunctions];
		String groupStrings[] = new String[numFunctions];
		idx = 0;

		// write out function names
		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
			Function function = it.next();

			//if (!function.isGroupMember(derived)) {
				functions[idx] = function;

				List<Group> thisGroups = function.getGroups();

				if (thisGroups == null) {
					groupStrings[idx] = "";
				} else {
					groupStrings[idx] = "";

					for (int i = 0; i < thisGroups.size(); i++) {
						Group group = thisGroups.get(i);
						if (i == 0) {
							groupStrings[idx] = group.getName();
						} else {
							groupStrings[idx] = groupStrings[idx] + "|"
									+ group.getName();
						}
					}

					groupStrings[idx] = groupStrings[idx].trim();
				}
				idx++;
			//}
		}

		UserEvent userEvents[] = new UserEvent[numUserEvents];
		idx = 0;
		// collect user event names
		for (Iterator<UserEvent> it = dataSource.getUserEventIterator(); it.hasNext();) {
			UserEvent userEvent = it.next();
			userEvents[idx++] = userEvent;
		}

		if (numMetrics == 1) {
			writeMetric(directory, dataSource, 0, functions, groupStrings,
					userEvents, threads);
		} else {
			for (int i = 0; i < numMetrics; i++) {
				String name = "MULTI__"
						+ createSafeMetricName(dataSource.getMetricName(i));
				boolean success = (new File(directory + File.separator + name)
						.mkdir());
				if (!success) {
					System.err.print("Failed to create directory: " + name);
				} else {
					writeMetric(new File(directory + File.separator + name),
							dataSource,
							i, functions, groupStrings, userEvents, threads);
				}
			}
		}
	}

	public static void writeMPISummary(DataSource dataSource,
			List<Thread> threads, boolean suppress, boolean metadata)
			throws IOException {

		int numMetrics = dataSource.getNumberOfMetrics();
		int numUserEvents = dataSource.getNumUserEvents();
		// int numGroups = dataSource.getNumGroups();

		int idx = 0;

		//Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

		int numFunctions = dataSource.getFunctions().size();//0;
//		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
//			Function function = it.next();
//			if (function.isGroupMember(derived)) {
//				continue;
//			}
//			numFunctions++;
//		}

		// // write out group names
		// Group groups[] = new Group[numGroups];
		// for (Iterator<Group> it = dataSource.getGroups(); it.hasNext();) {
		// Group group = it.next();
		// //String groupName = group.getName();
		// groups[idx++] = group;
		// }

		Function functions[] = new Function[numFunctions];
		String groupStrings[] = new String[numFunctions];
		idx = 0;

		// write out function names
		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
			Function function = it.next();

			//if (!function.isGroupMember(derived)) {
				functions[idx] = function;

				List<Group> thisGroups = function.getGroups();

				if (thisGroups == null) {
					groupStrings[idx] = "";
				} else {
					groupStrings[idx] = "";

					for (int i = 0; i < thisGroups.size(); i++) {
						Group group = thisGroups.get(i);
						if (i == 0) {
							groupStrings[idx] = group.getName();
						} else {
							groupStrings[idx] = groupStrings[idx] + "|"
									+ group.getName();
						}
					}

					groupStrings[idx] = groupStrings[idx].trim();
				}
				idx++;
			//}
		}

		UserEvent userEvents[] = new UserEvent[numUserEvents];
		idx = 0;
		// collect user event names
		for (Iterator<UserEvent> it = dataSource.getUserEventIterator(); it.hasNext();) {
			UserEvent userEvent = it.next();
			userEvents[idx++] = userEvent;
		}

		if (numMetrics == 1) {
			writeMPIMetric(dataSource, 0, functions, groupStrings, userEvents,
					threads, suppress);
		} else {
			for (int i = 0; i < numMetrics; i++) {
				String name = "MULTI__"
						+ createSafeMetricName(dataSource.getMetricName(i));
				System.out.print(name);
				// boolean success = (new File(name).mkdir());
				// if (!success) {
				// System.err.print("Failed to create directory: " + name);
				// } else {
				writeMPIMetric(dataSource, i, functions, groupStrings,
						userEvents, threads, suppress);
				// }
			}
		}
		if (metadata) {
			System.out.println();
			writeMetaDataSummary(dataSource);
		}
	}

	private static void writeMPIMetric(DataSource dataSource, int metricID,
			Function[] functions, String[] groupStrings,
			UserEvent[] userEvents, List<Thread> threads, boolean suppress)
			throws IOException {

		int numUserEvents = dataSource.getNumUserEvents();

		Thread[] minMaxMean = new Thread[3];
		for (int i = 0; i < threads.size(); i++) {
			if (threads.get(i).getThreadID() == Thread.MIN) {
				minMaxMean[0] = threads.get(i);
			} else if (threads.get(i).getThreadID() == Thread.MAX) {
				minMaxMean[1] = threads.get(i);
			} else if (threads.get(i).getThreadID() == Thread.MEAN) {
				minMaxMean[2] = threads.get(i);
			}
		}

		for (int i = 0; i < 3; i++) {

			if (minMaxMean[i] == null) {
				String missing = "Min";

				if (i == 1) {
					missing = "Max";
				} else if (i == 2) {
					missing = "Mean";
				}
				System.out.println("No " + missing + " data available.");
			}
		}

		String metType = dataSource.getMetric(metricID).getName();

		// ="Time";

		System.out.println();
		if (dataSource.getNumberOfMetrics() != 1) {
			System.out.print("MULTI_" + metType + "\n");
			// if(metricID!=0){
			// metType="Count";
			// }
		}
		if (metType.toLowerCase().contains("time"))
			metType = metType + " (Microseconds)";
		else
			metType = metType + " (Counts)";

		int metListLen = 15;
		String headerFormat = "%-12s%-12s%-12s%-12s%-12s%-12s%-16s%-16s%-16s%-16s%-16s%-16s%-16s%-16s%-16s";
		if (suppress) {
			metListLen = 9;
			System.out.format("%-36s%-48s%-48s%-8s", "   Calls",
					"   Inclusive " + metType, "   Bytes Transferred",
					"   Name");
			headerFormat = "%-12s%-12s%-12s%-16s%-16s%-16s%-16s%-16s%-16s";
		} else {
			System.out.format("%-24s%-24s%-48s%-48s%-48s%-8s", "   Calls",
					"   Child Calls", "   Exclusive " + metType,
					"   Inclusive " + metType, "   Bytes Transferred",
					"   Name");
		}

		System.out.println();

		String[] header = new String[metListLen];
		for (int i = 0; i < header.length; i++) {
			if (i % 3 == 0) {
				header[i] = "   Min";
			} else if (i % 3 == 1) {
				header[i] = "   Max";
			} else if (i % 3 == 2) {
				header[i] = "   Mean";
			}
		}

		System.out.format(headerFormat, (Object[]) header);
		System.out.println();

		double[] values = new double[15];

		// write out function profiles
		for (int i = 0; i < functions.length; i++) {
			String fName = "";
			for (int m = 0; m < minMaxMean.length; m++) {
				if (minMaxMean[m] == null) {

					continue;
				}

				FunctionProfile fp = minMaxMean[m]
						.getFunctionProfile(functions[i]);

				if (fp != null) {
					fName = functions[i].getName();

					values[m] = fp.getNumCalls();

					values[m + 3] = fp.getNumSubr();
					values[m + 6] = fp.getExclusive(metricID);
					values[m + 9] = fp.getInclusive(metricID);
					boolean gotComm = false;
					for (int j = 0; j < numUserEvents; j++) {
						UserEventProfile uep = minMaxMean[m]
								.getUserEventProfile(userEvents[j]);

						if (uep != null) {

							String ueName = userEvents[j].getName();

							if (ueName.contains(fName)) {

								if (m == 0) {
									values[m + 12] = uep.getMinValue();
								} else if (m == 1) {
									values[m + 12] = uep.getMaxValue();
								} else if (m == 2) {
									values[m + 12] = uep.getMeanValue();
								}

								gotComm = true;
								break;
							}
						}
					}
					if (!gotComm) {
						values[m + 12] = 0;
					}

				}
			}

			printValues(values, fName, suppress);

		}

	}

	private static void printValues(double[] values, String name,
			boolean suppress) {

		if (suppress) {
			System.out.format(
					"%12.3f%12.3f%12.3f%16.5f%16.5f%16.5f%16.5f%16.5f%16.5f",
					values[0], values[1], values[2], values[9], values[10],
					values[11], values[12], values[13], values[14]);
		} else {
			System.out
					.format("%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%16.5f%16.5f%16.5f%16.5f%16.5f%16.5f%16.5f%16.5f%16.5f",
							values[0], values[1], values[2], values[3],
							values[4], values[5], values[6], values[7],
							values[8], values[9], values[10], values[11],
							values[12], values[13], values[14]);
		}
		System.out.print("   " + name);

		System.out.println();
	}
	

}
