<?echo '<?xml version="1.0" encoding="utf-8"?>'?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">

<head>
<title>University of Oregon - PerfExplorer</title>
<link rel="stylesheet" type="text/css" href="new.css"/>
<link rel="icon" href="tau16x16.gif" type="image/gif"/>
<?php include("utility_functions.php"); ?>
</head>

<body>

<div style="float: left; min-width: 280px;">
<p>
<img src="tau-large.png" alt="TAU logo" height="250" width="250"/><br/>
<a href="#intro">Introduction</a><br/>
<a href="#features">New Features in PerfExplorer 2.0</a><br/>
<a href="#prereqs">Prerequisites</a><br/>
<a href="#running">Running PerfExplorer</a><br/>
<a href="#using">Using PerfExplorer</a><br/>
<a href="#bugs">Known Bugs</a><br/>
<a href="#other">Other stuff...</a><br/>
</p>
</div>

<div>
<p>
<h1>PerfExplorer</h1>
Thanks for your interest in PerfExplorer!  This simple page is hopefully only
temporary - more useful content will come at a later time.
</p>

<p>
<h2>For The Impatient</h2>
<h3><a href="perfexplorer.jnlp">Click here to run PerfExplorer!</a></h3>
</p>
</div>

<div>
<a name="intro"/>
<h2>Introduction</h2>
<p>
PerfExplorer is a framework for parallel performance data mining and knowledge
discovery. The framework architecture enables the development and integration
of data mining operations that will be applied to large-scale parallel
performance profiles.  PerfExplorer was developed as a research project by <a
href="http://www.cs.uoregon.edu/~khuck/">Kevin Huck</a>, a Doctoral Candidate
in Computer and Information Science at the University of Oregon.
</p>

<p>
The overall goal of the PerfExplorer project is to create a software framework
to integrate sophisticated data mining techniques in the analysis of
large-scale parallel performance data.  
</p>

<p>
PerfExplorer supports clustering, summarization, association, regression, and
correlation. Cluster analysis is the process of organizing data points into
logically similar groupings, called clusters. Summarization is the process of
describing the similarities within, and dissimilarities between, the discovered
clusters. Association is the process of finding relationships in the data. One
such method of association is regression analysis, the process of finding
independent and dependent correlated variables in the data. In addition,
comparative analysis extends these operations to compare results from different
experiments, for instance, as part of a parametric study.
</p>
</div>

<div>
<a name="features"/>
<h2>New Features in PerfExplorer 2.0</h2>
<ul>
<li><b>Analysis Process Automation</b>:  
PerfExplorer 2 adds analysis process automation through the use of a scripting
interface.  The interface is in Python, and scripts can be used to build
analysis workflows.  The Python scripts control the Java classes in the
application through the <a href="http://www.jython.org/">Jython
interpreter.</a>  There are two types of components which are useful in
building analysis scripts.  The first type is the PerformanceResult interface,
and the second is the PerformanceAnalysisComponent interface.  For
documentation on how to use the Java classes, see the <a
href="javadoc">javadoc</a>, and the example scripts below.</li>
<li><b>Metadata Encoding and Incorporation</b>:
In order to interpret performance results within the context of the application
experiment environment, PerfExplorer requires the context metadata.  There are 
three was to get context metadata into the performance profiles - the TAU 
measurement runtime automatically collects some hardware information, the TAU
instrumentation library provides the ability to include application context
parameters into the profile data, and metadata can be added to the performcane
data when the data is loaded into PerfDMF.  In PerfExplorer, this data can be
correlated with performance results and included in inference rules (see below)
in order to explain performance behavior.</li>
<li><b>Inference Engine</b>:  In order to reason about performance causes,
PerfExplorer has incorporated the <a
href="http://www.jboss.com/products/rules">JBoss Rules engine</a>.
PerfExplorer uses parallel computing, hardware and application rule sets to
interpret the performance results, and provide suggestions for improving
application performance.</li>
<li><b>Provenance and Persistence</b>:  Using the <a
href="http://www.hibernate.org/">Hibernate Relational Persistence for Java</a>, perfexplorer retains a provenance of analysis results, and also stores all 
intermediate results.  Analysis scripts can use the provenance object to
access the intermediate results without maintaining local references to 
results.
<i>Note:  The provenance/persistence feature is not fully supported at this
time.</i></li>
</ul>

</div>

<div>
<a name="prereqs"/>
<h2>Prerequisites</h2>
<h3>Java Runtime Environment (JRE) 5</h3>
<p>
PerfExplorer is a Java application, and therefore it requires that Java is
installed.  If you have Java, make sure the version is Java 5 or better.  
<a href="http://java.sun.com">If you need to install Java, you can get it from
Sun.</a>
</p>

<h3>PerfDMF</h3>
<p>
PerfDMF is the Performance Data Management Framework from TAU.  If you have
already installed TAU on your computer, then you already have a copy of
PerfDMF, and the appropriate configuration files.  See the <a
href="http://tau.uoregon.edu">TAU website</a> for more details.  If you aren't
hosting a PerfDMF data repository but rather wish to connect to a remote
repository, then PerfExplorer will help you configure your connection.
</p>

</div>

<div>
<a name="running"/>
<h2>Running PerfExplorer</h2>
<p>
<a href="perfexplorer.jnlp">Click here to run the PerfExplorer Java Web Start
file.</a>  You can use the JNLP file to run PerfExplorer again later - 
your platform instructions may vary (see the Java Web Start documentation for
more details).  If you are running PerfExplorer on Windows, allow Java and
PerfExplorer to communicate with the internet when the "Windows Security Alert"
window appears.
</p>
<p>
Also, you will want the Java Console open when you run PerfExplorer.  From <a
href="http://java.sun.com/developer/JDCTechTips/2001/tt0530.html">Sun Developer
Network</a>, here are the instructions for enabling the console:  
</p>
<p>
<i>The console is one of many settings that Java Web Start sets to "off" by
default. To display the console, run the Java Web Start application. Under
File->Preferences, go to the Advanced Tab and select "Show Java Console".</i>
</p>


</div>

<div>
<a name="using"/>
<h2>Using PerfExplorer</h2>
<p>
The full documentation for PerfExplorer 1.0 is located at the <a
href="http://www.cs.uoregon.edu/research/tau/docs/perfexplorer/">TAU web
site</a>.
</p>

<p>
If you want to write scripts for PerfExplorer 2.0, then you need the <a
href="javadoc">javadoc</a> for the glue package.  The glue package is the set
of analysis components and data components for scriptable analysis in
PerfExplorer 2.0.  Here are some examples of analysis scripts:
</p>
<ol>
<li>...coming soon...</li>
<li>...coming soon...</li>
<li>...coming soon...</li>
</ol>

<p>
If you want to write inference rules for PerfExplorer 2.0, then here are some
examples:
</p>
<ol>
<li>...coming soon...</li>
<li>...coming soon...</li>
<li>...coming soon...</li>
</ol>

</div>

<div>
<a name="bugs"/>
<h2>Known Bugs</h2>
<p>
There are several.  However, some of the more important ones:
</p>
<ol>
<li>The tree does not refresh correctly when a database configuration is added,
deleted or changed.  You will have to exit the application and restart in order
to see your changes.</li>
<li>Derby databases are not supported through the JNLP interface.  Don't know
why, but sometimes Derby can't open derby.log (permissions error), and the
database won't continue.  If you need to access a Derby PerfDMF database,
please contact Kevin Huck at <?php echo
javascript_ascii_email_string("khuck@cs.uoregon.edu", ""); ?>.</li>
</ol>

</div>

<div>
<a name="other"/>
<h2>Other stuff...</h2>
<p>
For any questions, comments, problems, etc., please contact Kevin Huck at
<?php echo javascript_ascii_email_string("khuck@cs.uoregon.edu", ""); ?>, or 
with more general TAU questions,
<?php echo javascript_ascii_email_string("tau-bugs@cs.uoregon.edu", ""); ?>.
</p>
</div>

<p>
<a href="http://validator.w3.org/check?uri=referer">
<img src="http://www.w3.org/Icons/valid-xhtml10"
alt="Valid XHTML 1.0 Strict" height="31" width="88" style="border:none"/></a>
</p>

<p>
<?php
// Change to the name of the file
$last_modified = filemtime("index.php");

// Display the results
// eg. Last modified Monday, 27th October, 2003 @ 02:59pm
print "Last modified " . date("l, dS F, Y @ h:ia T", $last_modified);
?>
</p>

</body>
</html>
