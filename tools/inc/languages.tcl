# TODO
#  - Allow multiple possible extensions for each language source code file.



set languages {hpc++ pc++ ansic}

set pc++(cgm_cmd)         "cgm"
set pc++(cgm_ext)         "dep"
set pc++(prog_ext)        "pc"
set pc++(template)        "lang_support/pc++/PROJ-TEMPLATE"
set pc++(compileopts)      {CXX_SWITCHES PCXX_SWITCHES}
set pc++(compileopts_desc) {"extra C++ switches:" "extra pC++ switches:"}
set pc++(tools)            {cosy fancy spiffy cagey classy racy speedy}

set hpc++(cgm_cmd)         "cppcgm"
set hpc++(cgm_ext)         "dep"
set hpc++(prog_ext)        "C"
set hpc++(template)        "lang_support/hpc++/PROJ-TEMPLATE"
set hpc++(compileopts)      {CXX_SWITCHES}
set hpc++(compileopts_desc) {"extra C++ switches:"}
set hpc++(tools)            {cosy fancy spiffy cagey classy racy}

set ansic(cgm_cmd)          "ccgm"
set ansic(cgm_ext)          "dep"
set ansic(prog_ext)         "c"
set ansic(template)         "lang_support/ansic/PROJ-TEMPLATE"
set ansic(compileopts)      {CC_SWITCHES}
set ansic(compileopts_desc) {"extra C switches:"}
set ansic(tools)            {cosy fancy spiffy cagey classy}



proc Lang_GetCGM {lang} {
    global languages; foreach l $languages {global $l}
    return [eval set ${lang}(cgm_cmd)]
}

proc Lang_GetExt {lang} {
    global languages; foreach l $languages {global $l}
    return [eval set ${lang}(cgm_ext)]
}

proc Lang_GetProgExt {lang} {
    global languages; foreach l $languages {global $l}
    return [eval set ${lang}(prog_ext)]
}

proc Lang_GetTemplateName {lang} {
    global languages; foreach l $languages {global $l}
    return [eval set ${lang}(template)]
}

proc Lang_GetCompileOpts {lang} {
    global languages; foreach l $languages {global $l}
    return [eval set ${lang}(compileopts)]
}

proc Lang_GetCompileOptsDesc {lang} {
    global languages; foreach l $languages {global $l}
    return [eval set ${lang}(compileopts_desc)]
}

proc Lang_GuessLang {filen} {
    global languages; foreach l $languages {global $l}

    set file_ext [string range [file extension $filen] 1 end]
    foreach lang $languages {
	if {$file_ext == [Lang_GetProgExt $lang]} {
	    return $lang
	}
    }
}

# returns 1 if compatible, 0 if not
proc Lang_CheckCompatibility {proj_langs tool} {
    global languages; foreach l $languages {global $l}
    if {[llength $proj_langs] == 0} { return 1; }
    foreach lang $proj_langs {
	if {[lsearch [eval set ${lang}(tools)] $tool] >= 0} {
	    return 1;
	}
    }
    return 0
}


