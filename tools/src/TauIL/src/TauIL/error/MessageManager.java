package TauIL.error;

import java.io.File;
import java.io.PrintStream;

public class MessageManager {
    private File warn_log = new File("warning.log");
    private File err_log = new File("error.log");

    private PrintStream warn_out = System.out;
    private PrintStream err_out = System.err;

    private boolean report_err = true;
    private boolean report_warn = true;

    private boolean log_err = false;
    private boolean log_warn = false;

    private String app = "";

    public MessageManager(String app) {
	this.app = app;
    }

    public void report(WarningMessage message) {
	if (report_warn)
	    warn_out.println(app + ": " + message);

	if (log_warn) ;
    }

    public void report(ErrorMessage message) {
	if (report_err)
	    err_out.println(app + ": " + message);

	if (log_err) ;
    }

    public void enableErrorLog() {
	log_err = true;
    }

    public void disableErrorLog() {
	log_err = false;
    }

    public void enableWarningLog() {
	log_warn = true;
    }

    public void disableWarningLog() {
	log_warn = false;
    }

    public void enableAllLogging() {
	enableErrorLog();
	enableWarningLog();
    }

    public void disableAllLogging() {
	disableErrorLog();
	disableWarningLog();
    }

    public void setErrorLog(File log) {
	err_log = log;
    }

    public void setWarningLog(File log) {
	warn_log = log;
    }

    public void enableErrorOutput() {
	report_err = true;
    }

    public void disableErrorOutput() {
	report_err = false;
    }

    public void enableWarningOutput() {
	report_warn = true;
    }

    public void disableWarningOutput() {
	report_warn = false;
    }

    public void enableAllOutput() {
	enableErrorOutput();
	enableWarningOutput();
    }

    public void disableAllOutput() {
	disableErrorOutput();
	disableWarningOutput();
    }

    public void setErrorOutput(PrintStream out) {
	err_out = out;
    }

    public void setWarningOutput(PrintStream out) {
	warn_out = out;
    }

    public void enableErrorReporting() {
	enableErrorLog();
	enableErrorOutput();
    }

    public void enableWarningReporting() {
	enableWarningLog();
	enableErrorOutput();
    }

    public void enableAllReporting() {
	enableWarningReporting();
	enableErrorReporting();
    }

    public void disableErrorReporting() {
	disableErrorLog();
	disableErrorOutput();
    }

    public void disableWarningReporting() {
	disableWarningLog();
	disableWarningOutput();
    }

    public void disableAllReporting() {
	disableErrorReporting();
	disableWarningReporting();
    }
}

