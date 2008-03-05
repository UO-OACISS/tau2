package edu.uoregon.tau.common;

public class VersionTester {

	public static void main(String[] args) {
		String usage = "javaVersionTester <min version>";
		String version = System.getProperty("java.version");
		if (args == null || args.length == 0) {
			System.out.println(usage);
			System.out.println("java version: " + version);
			return;
		}
		double required = Double.parseDouble(args[0]);
		double shortVersion = Double.parseDouble(version.substring(0,3));
		//System.out.println("java version: " + shortVersion);
		if (required <= shortVersion) {
			System.out.println("success");
			System.exit(0);
		} else {
			System.out.println("failed");
			System.exit(1);
		}
	}
}
