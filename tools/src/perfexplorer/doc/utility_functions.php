<?php

function javascript_ascii_email_string ($email, $link) {
    $js_str = "<script type=\"text/javascript\">document.write(String.fromCharCode(";
    if (!empty($link)) 
        $email  = "<a href=\"mailto:$email\">$link</a>";
    else
        $email  = "<a href=\"mailto:$email\">$email</a>";

    $len    = strlen($email);
    for ($i = 0; $i < $len; $i++) {
        $js_str .= ord($email{$i});
	if ($i < $len - 1) 
	  $js_str .= ",";
    }

    $js_str .= "));</script>";

    return $js_str;
}

function connect_to_database() {

	$user = "khuck";

	// Parse your .my.cnf file; give complete pathname to YOUR home dir
	$mysql_conf = parse_ini_file("/home/users/$user/.my.cnf", TRUE);
	// Parse the config file for this app; this one in the current directory
	$myapp_conf = parse_ini_file(".myapp.conf");

	$location = $mysql_conf['client']['host'].":".$mysql_conf['client']['port'];
	$password = $myapp_conf['password'];
	$database = $myapp_conf['database'];
	$debug    = $myapp_conf['debug'];

	// connect to the database server
	$conn = mysql_connect("$location","$user","$password") 
		or die ("Could not connect MySQL");
	if ($debug) {
		print "Connection to <b>$location</b> successful.<br />";};

	// select the database
	$selected_db=mysql_select_db($database,$conn) 
		or die ("Could not open database");
	if ($debug) {
		print "Database <b>$database</b> selected.<br />";};

	// Now you are ready to use mysql_query() and mysql_fetch()

	return ($conn);
}

function close_database($conn) {
	mysql_close($conn);
}

function DateAdd($interval, $number, $date) {

    $date_time_array = getdate($date);
    $hours = $date_time_array['hours'];
    $minutes = $date_time_array['minutes'];
    $seconds = $date_time_array['seconds'];
    $month = $date_time_array['mon'];
    $day = $date_time_array['mday'];
    $year = $date_time_array['year'];

    switch ($interval) {
    
        case 'yyyy':
            $year+=$number;
            break;
        case 'q':
            $year+=($number*3);
            break;
        case 'm':
            $month+=$number;
            break;
        case 'y':
        case 'd':
        case 'w':
            $day+=$number;
            break;
        case 'ww':
            $day+=($number*7);
            break;
    }
       $timestamp= mktime($hours,$minutes,$seconds,$month,$day,$year);
    return $timestamp;
}

function email_everyone ($user_id, $subject, $message) {
	$message = $message."\n\n---\nGo to the kitchen: http://www.cs.uoregon.edu/~khuck/kitchen/\n";
	// send an email!
	$query = "SELECT id, name, email FROM users";
	$result = mysql_query($query) or die ('Query failed: ' . mysql_error());

	$got_one = false;
	$to_list = "";
	$from = "";
	while ($line = mysql_fetch_array($result, MYSQL_ASSOC + MYSQL_RETURN_NULLS)) {
		if ($got_one) {
			$to_list = $to_list.", ";
		}
		//$to_list = $to_list."<".$line["name"]."> ".$line["email"];
		$to_list = $to_list.$line["email"];
		if ($user_id == $line["id"])
			$from = "From: ".$line["email"];
		$got_one = true;
	}
	mysql_free_result($result);

	mail ($to_list, $subject, $message, $from) or die("couldn't send mail!");

}

?>
